#!/usr/bin/env python3

import argparse
import os
import sys
import subprocess
import torch
import whisper
from pathlib import Path

def transcribe_audio(input_file, output_dir):
    # Check if CUDA is available
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("CUDA not available. Using CPU.")
    
    # Load the largest Whisper model
    model = whisper.load_model("large", device=device)
    print("Model loaded successfully.")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the base filename without extension
    input_path = Path(input_file)
    base_filename = input_path.stem
    
    # Transcribe the audio file
    print(f"Transcribing {input_file}...")
    result = model.transcribe(input_file)
    
    # Write transcription to output file
    output_file = os.path.join(output_dir, f"{base_filename}.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(result["text"])
    
    print(f"Transcription saved to {output_file}")
    return output_file

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        # Check if ffmpeg is installed
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        print("Error: ffmpeg is not installed or not in PATH.")
        print("Please install ffmpeg:")
        print("  - Ubuntu/Debian: sudo apt-get install ffmpeg")
        print("  - macOS: brew install ffmpeg")
        print("  - Windows: Download from https://ffmpeg.org/download.html")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio files using OpenAI's Whisper model")
    parser.add_argument("input_file", help="Path to the input audio file")
    parser.add_argument("output_dir", help="Directory where the transcription will be saved")
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.isfile(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist.")
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    try:
        transcribe_audio(args.input_file, args.output_dir)
    except Exception as e:
        print(f"Error during transcription: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
