#!/usr/bin/env python
# coding: utf-8

import os
import sys
import shutil
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from tqdm import tqdm
import opencc

import warnings
warnings.filterwarnings("ignore")


# # Folders Creation

def get_base_path():
    if getattr(sys, 'frozen', False):
        # Running in a bundle (executable created by PyInstaller)
        return os.path.dirname(sys.executable)
    elif '__file__' in globals():
        # Running in a script or packaged environment
        return os.path.dirname(os.path.abspath(__file__))
    else:
        # Running in a Jupyter notebook
        return os.getcwd()

base_path = get_base_path()

video_dir = base_path + "/video_input/"
audio_dir = base_path + "/audio_output/"
srt_dir = base_path + "/srt_output/"
source_lang = "chinese"

os.makedirs(audio_dir, exist_ok=True)
os.makedirs(srt_dir, exist_ok=True)


# # Video to Audio: FFmpeg

def extract_audio_from_video(video_dir, audio_dir):
    """
    Extracts audio from any file in the specified directory using ffmpeg. If the file 
    is not a video, ffmpeg will return an error. The extracted audio files are saved 
    in FLAC format in another directory.
    
    Args:
    - video_dir (str): Directory containing the input files.
    - audio_dir (str): Directory where the extracted audio files will be saved.

    Returns:
    - None
    """
    
    # Delete the directory and its contents if the directory exists
    if os.path.exists(audio_dir):
        shutil.rmtree(audio_dir)

    # Create the output directory if it doesn't exist
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)

    video_files = [f for f in os.listdir(video_dir) if f != '.DS_Store']
    for filename in tqdm(video_files, desc="Extracting audio"):
        print(filename)
        video_file_path = os.path.join(video_dir, filename)
        audio_file_name = os.path.splitext(filename)[0] + '.flac'
        audio_file_path = os.path.join(audio_dir, audio_file_name)

        # Convert video to audio using ffmpeg
        os.system(f'ffmpeg -y -i "{video_file_path}" -q:a 0 -map a "{audio_file_path}" -hide_banner -loglevel error')

        print(f"Extracted audio from video: {filename}")


extract_audio_from_video(video_dir=video_dir, audio_dir=audio_dir)


# # Transcription: Whisper

def convert_time_to_srt(seconds_float):
    """Converts a time in seconds to 'hh:mm:ss,ms' format for SRT."""
    hours, remainder = divmod(seconds_float, 3600)
    minutes, seconds = divmod(remainder, 60)
    seconds, milliseconds = divmod(seconds, 1)
    milliseconds = int(milliseconds * 1000)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{milliseconds:03}"

def whisper_transcribe(input_directory, output_directory):
    # Load HuggingFace Whisper model
    device = "cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"Using device: {device} ({torch_dtype})")

    model_id = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)

    # Setup pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )

    # Process audio files
    audio_files = [file for file in os.listdir(input_directory) if file.endswith('.flac')]  # Adjust the extension if needed
    for audio_file in tqdm(audio_files, desc="Transcribing audio files"):
        audio_path = os.path.join(input_directory, audio_file)
        result = pipe(audio_path, return_timestamps=True, generate_kwargs={"language": "chinese"})  # Adjust language if needed

        # Process and write SRT content
        srt_content = []
        counter = 1
        for chunk in result['chunks']:
            start_time_srt = convert_time_to_srt(chunk['timestamp'][0])

            # Assign a default duration if the end timestamp is None
            if chunk['timestamp'][1] is not None:
                end_time_srt = convert_time_to_srt(chunk['timestamp'][1])
            else:
                # Assuming a default duration of 2 seconds
                end_time_srt = convert_time_to_srt(chunk['timestamp'][0] + 2.0)
            
            transcript = chunk['text'].strip()
            srt_content.append(f"{counter}\n{start_time_srt} --> {end_time_srt}\n{transcript}")
            counter += 1

        srt_output = "\n\n".join(srt_content)
        srt_filename = os.path.splitext(audio_file)[0] + '.srt'
        srt_path = os.path.join(output_directory, srt_filename)

        # Convert simplified Chinese to traditional Chinese
        converter = opencc.OpenCC('s2hk')
        srt_output = converter.convert(srt_output)
        
        # Save to SRT file
        with open(srt_path, 'w', encoding='utf-8') as srt_file:
            srt_file.write(srt_output)

        print(f"Generated SRT: {srt_filename}")


whisper_transcribe(input_directory=audio_dir, output_directory=srt_dir)


# # Delete Extra Files

# Delete the directory and its contents if the directory exists
if os.path.exists(audio_dir):
    shutil.rmtree(audio_dir)

