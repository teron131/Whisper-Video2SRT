import os
import shutil
import subprocess
import warnings
from pathlib import Path

from dotenv import load_dotenv

from utils import response_to_srt
from whisper import whisper_transcribe

load_dotenv()
warnings.filterwarnings("ignore")


# Constants
INPUT_DIR = "input/"
OUTPUT_DIR = "output/"
TARGET_LANG = "zh"

# Ensure directories exist
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_audio_from_video(video_file_path: Path) -> bool:
    """Extract audio from video file using ffmpeg."""
    print(f"Extracting audio from: {video_file_path}")
    filename = video_file_path.stem
    output_folder = Path(OUTPUT_DIR) / filename
    os.makedirs(output_folder, exist_ok=True)
    audio_file_path = output_folder / f"{filename}.mp3"

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


def process_video(video_file_path: Path) -> None:
    """Process a video file by extracting audio and generating subtitles."""
    if extract_audio_from_video(video_file_path):
        filename = video_file_path.stem
        output_folder = Path(OUTPUT_DIR) / filename
        audio_file_path = output_folder / f"{filename}.mp3"
        srt_file_path = audio_file_path.with_suffix(".srt")

        print(f"Transcribing {audio_file_path}")
        result = whisper_transcribe(str(audio_file_path), whisper_model="fal", language=TARGET_LANG)
        # Get SRT content as string
        srt_content = response_to_srt(result)
        # Write to file
        srt_file_path.write_text(srt_content, encoding="utf-8")
        print(f"Transcription completed for {srt_file_path}")
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
