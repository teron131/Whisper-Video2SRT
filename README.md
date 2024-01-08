# Whisper Video2SRT

The Whisper Video2SRT repository focuses on converting video files (supported by FFmpeg) into .srt subtitle files. It incorporates the OpenAI Whisper large-v3 model via the HuggingFace pipeline (https://huggingface.co/openai/whisper-large-v3), enabling optimization capabilities. While the main GitHub version provides timestamping and SRT file output functionalities, they are not integrated, and this version lacks optimization features.

# Usage

1.  Place your video files into the `video_input` folder.

2.  Run the `video2srt.ipynb` or `video2srt.py` script.

# Requirements

## Installing FFmpeg

### On macOS:

1. **Install Homebrew** (if not already installed):
   Run this in Terminal:
   ```
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Add Homebrew to PATH** (if needed):
- For Zsh (default shell):
  ```
  echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zshrc
  eval "$(/opt/homebrew/bin/brew shellenv)"
  ```
- For Bash:
  ```
  echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.bash_profile
  ```

3. **Install FFmpeg**:
    ```
    brew install ffmpeg
    ```

### On Windows:

1. **Download FFmpeg**:
- Visit [FFmpeg's official website](https://ffmpeg.org/download.html).
- Choose and download the Windows build.

2. **Extract and Place FFmpeg**:
- Extract the downloaded file to a folder, for example, `C:\\FFmpeg`.

3. **Add FFmpeg to PATH**:
- Type "Environment Variables" in the search bar on the Start menu.
- Click on "Edit the system environment variables" or "Edit environment variables for your account".
- Under 'System Variables', find and select the 'Path' variable, then click 'Edit'.
- Click 'New' and add the path to the `bin` folder inside your FFmpeg folder (e.g., `C:\\FFmpeg\\bin`).
- Click 'OK' to close all dialog boxes.

Remember to restart your IDE after installing FFmpeg.

## Install Python Packages

Run this command:
```
pip install -r requirements.txt
```