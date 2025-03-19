# Whisper Transcription Utility

This repository contains code to transcribe audio files to text using OpenAI's Whisper model with GPU acceleration.

## Requirements

- Python 3.8+
- NVIDIA GPU with CUDA support
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

Run the script with the input audio file and output directory:

```bash
python transcribe.py /path/to/audio/file.mp3 /path/to/output/directory
```

The transcription will be saved as a text file in the specified output directory, with the same base name as the input file.

## Example

```bash
python transcribe.py sample.mp3 transcriptions/
```

This will create a file `transcriptions/sample.txt` containing the transcription.

## Notes

- The script uses Whisper's "large" model which requires approximately 10GB of VRAM
- First-time use will download the model weights (~3GB)
