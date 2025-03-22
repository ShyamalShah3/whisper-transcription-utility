# Whisper Transcription Utility

This repository contains code to recursively transcribe audio files to text using OpenAI's Whisper model with GPU acceleration.

## Requirements

- Python 3.8+
- NVIDIA GPU with CUDA support (will fall back to CPU if unavailable)
- **FFmpeg installed on your system** (REQUIRED)

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. **IMPORTANT**: Install FFmpeg on your system:
   - On Ubuntu/Debian: `sudo apt-get install ffmpeg`
   - On macOS: `brew install ffmpeg`
   - On Windows: Download from [FFmpeg website](https://ffmpeg.org/download.html) and add to PATH

## Usage

Run the script with the following parameters:

```bash
python transcribe.py <file_extension> <input_directory>
```

**Parameters:**
* `file_extension` - The audio file extension to search for (without the dot)
  * Examples: `m4a`, `mp3`, `wav`, `flac`
* `input_directory` - Directory to recursively search for audio files
  * Can be relative (e.g., `./recordings`) or absolute (e.g., `/home/user/recordings`)

**Process:**
1. Recursively finds all subdirectories in the input directory
2. For each directory (including the root), finds all audio files with the specified extension
3. Transcribes all files in each directory in alphanumeric order
4. Saves all transcriptions from a single directory into one text file named after that directory

## Example

```bash
python transcribe.py m4a ./recordings
```

This will:
- Find all directories containing `.m4a` files in the `./recordings` folder
- Transcribe all `.m4a` files in each directory
- For example, if you have files `./recordings/interview1.m4a` and `./recordings/interview2.m4a`, 
  they will be transcribed into `./recordings/recordings.txt`
- If you have files `./recordings/meeting1/audio1.m4a` and `./recordings/meeting1/audio2.m4a`, 
  they will be transcribed into `./recordings/meeting1/meeting1.txt`

## Notes

- The script uses Whisper's "large" model which requires approximately 10GB of VRAM
- First-time use will download the model weights (~3GB)
- Each directory's transcriptions are saved in a single file named after the directory
- Files are transcribed in alphanumeric order and separated by dividers in the output file
