import os
import shutil
import subprocess
import warnings
from pathlib import Path
from typing import Dict

import fal_client
import torch
from opencc import OpenCC
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

warnings.filterwarnings("ignore")

os.environ["FAL_KEY"] = ""

# Constants
INPUT_DIR = "input/"
OUTPUT_DIR = "output/"
TARGET_LANG = "zh"

# Ensure directories exist
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Subtitle preprocessing functions
s2hk = OpenCC("s2hk").convert


# Utility functions
def convert_time_to_hms(seconds_float: float) -> str:
    hours, remainder = divmod(seconds_float, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{milliseconds:03}"


def response_to_srt(result: Dict, srt_path: str) -> None:
    with open(srt_path, "w", encoding="utf-8") as srt_file:
        for counter, chunk in enumerate(result["chunks"], 1):
            start_time = chunk.get("timestamp", [0])[0]
            end_time = chunk.get("timestamp", [0, start_time + 2.0])[1]
            start_time_hms = convert_time_to_hms(start_time)
            end_time_hms = convert_time_to_hms(end_time)
            transcript = s2hk(chunk["text"].strip())
            srt_entry = f"{counter}\n{start_time_hms} --> {end_time_hms}\n{transcript}\n\n"
            srt_file.write(srt_entry)


def extract_audio_from_video(video_file_path: Path) -> bool:
    print(f"Extracting audio from: {video_file_path}")
    filename = video_file_path.stem
    audio_file_path = Path(f"{OUTPUT_DIR}/{filename}").with_suffix(".mp3")

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_file_path), "-q:a", "0", "-map", "a", str(audio_file_path), "-hide_banner"],
            capture_output=True,
            text=True,
            check=True,
        )
        print(f"Audio extracted successfully: {audio_file_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio from {video_file_path}:")
        print(f"ffmpeg stdout: {e.stdout}")
        print(f"ffmpeg stderr: {e.stderr}")
        return False


# Transcribe


def whisper_hf_transcribe(audio_path: str):
    """
    Transcribe audio file using whisper-large-v3-turbo model with Hugging Face optimization.

    Returns:
        dict: A dictionary containing the transcription result with the following structure:
            {
                "text": str,  # Full transcribed text
                "chunks": [
                    {
                        "timestamp": Tuple[float],  # Start and end time of the chunk
                        "text": str,  # Transcribed text for this chunk
                    }
                ]
            }
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"Using device: {device} ({torch_dtype})")

    model_id = "openai/whisper-large-v3-turbo"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )
    result = pipe(audio_path)
    return result


def whisper_fal_transcribe(audio_path: str, language: str = "en") -> Dict:
    """
    Transcribe an audio file using fal-ai/wizper model.

    This function uploads the audio file, subscribes to the transcription service,
    and returns the transcription result.

    It defaults at English.

    Args:
        audio_path (str): The path to the audio file to be transcribed.
        language (str): The language of the audio file. Defaults to "en".
    Returns:
        dict: A dictionary containing the transcription result with the following structure:
            {
                "text": str,  # Full transcribed text
                "chunks": List[dict],  # List of transcription chunks
                    # Each chunk is a dictionary with:
                    {
                        "timestamp": List[float],  # Start and end time of the chunk
                        "text": str,  # Transcribed text for this chunk
                    }
            }
    """

    def on_queue_update(update):
        if isinstance(update, fal_client.InProgress):
            for log in update.logs:
                print(log["message"])

    url = fal_client.upload_file(audio_path)
    result = fal_client.subscribe(
        # "fal-ai/wizper",
        "fal-ai/whisper",
        arguments={
            "audio_url": url,
            "task": "transcribe",
            "language": language,
        },
        with_logs=True,
        on_queue_update=on_queue_update,
    )
    return result


# Main


def process_video(video_file_path: Path) -> None:
    if extract_audio_from_video(video_file_path):
        audio_file_path = Path(f"{OUTPUT_DIR}/{video_file_path.stem}").with_suffix(".mp3")
        result = whisper_fal_transcribe(audio_path=str(audio_file_path), language=TARGET_LANG)
        # result = whisper_hf_transcribe(audio_path=str(audio_file_path))
        response_to_srt(result=result, srt_path=audio_file_path.with_suffix(".srt"))
    else:
        print(f"Skipping transcription for {video_file_path} due to audio extraction failure.")


if __name__ == "__main__":
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    video_files = [Path(f"{INPUT_DIR}/{f}") for f in os.listdir(INPUT_DIR) if f != ".DS_Store"]
    if not video_files:
        print("No video files found in the input directory.")
    else:
        latest_video = max(video_files, key=os.path.getmtime)
        print(f"Processing the most recently modified video: {latest_video}")
        process_video(latest_video)
