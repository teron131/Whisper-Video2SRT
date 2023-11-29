#!/usr/bin/env python
# coding: utf-8

import os
import sys
import shutil
import whisper
from tqdm import tqdm

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
source_lang = "Cantonese"

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
    model = whisper.load_model("large-v3")

    audio_files = [file for file in os.listdir(input_directory) if file.endswith('.flac')]  # Adjust the extension if needed

    for audio_file in tqdm(audio_files, desc="Transcribing"):
        audio_path = os.path.join(input_directory, audio_file)
        result = model.transcribe(audio_path, task="transcribe", word_timestamps=True)

        # Process and write SRT content
        srt_content = []
        counter = 1
        for segment in result['segments']:
            start_time_srt = convert_time_to_srt(segment['start'])
            end_time_srt = convert_time_to_srt(segment['end'])
            transcript = segment['text'].strip()  # Strip whitespace from the transcript
            srt_content.append(f"{counter}\n{start_time_srt} --> {end_time_srt}\n{transcript}")
            counter += 1

        srt_output = "\n\n".join(srt_content)  # Join segments with two newlines
        srt_filename = os.path.splitext(audio_file)[0] + '.srt'
        srt_path = os.path.join(output_directory, srt_filename)

        with open(srt_path, 'w', encoding='utf-8') as srt_file:
            srt_file.write(srt_output)
        
        print(f"Generated SRT: {srt_filename}")


whisper_transcribe(input_directory=audio_dir, output_directory=srt_dir)


# # Delete Extra Files

# Delete the directory and its contents if the directory exists
if os.path.exists(audio_dir):
    shutil.rmtree(audio_dir)

