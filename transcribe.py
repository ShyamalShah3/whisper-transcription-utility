#!/usr/bin/env python3

import argparse
import os
import sys
import subprocess
import torch
import whisper
from pathlib import Path
import glob

def transcribe_audio(input_file, model):
    """Transcribe a single audio file using Whisper model"""
    print(f"Transcribing {input_file}...")
    result = model.transcribe(str(input_file))
    return result["text"]

def process_directory(input_dir, file_extension, model):
    """Process all directories recursively and transcribe audio files"""
    # Get all subdirectories including the root
    all_dirs = [input_dir] + [str(p) for p in Path(input_dir).glob('**/') if p != Path(input_dir)]
    
    for dir_path in all_dirs:
        # Get all audio files with the specified extension in the current directory (non-recursive)
        audio_files = sorted(Path(dir_path).glob(f'*.{file_extension}'))
        
        if not audio_files:
            continue
            
        # Create output file path based on the directory name
        dir_name = os.path.basename(os.path.normpath(dir_path))
        output_file = os.path.join(dir_path, f"{dir_name}.txt")
        
        print(f"Found {len(audio_files)} .{file_extension} files in {dir_path}")
        
        # Open the output file in write mode (will overwrite if exists)
        with open(output_file, "w", encoding="utf-8") as f:
            for audio_file in audio_files:
                # Transcribe the audio file and append to the output
                transcription = transcribe_audio(audio_file, model)
                f.write(f"File: {audio_file.name}\n\n")
                f.write(transcription)
                f.write("\n\n" + "-" * 80 + "\n\n")
                
        print(f"Transcription for {dir_path} saved to {output_file}")

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
    parser = argparse.ArgumentParser(description="Recursively transcribe audio files using OpenAI's Whisper model")
    parser.add_argument("file_extension", help="Audio file extension to search for (e.g., 'm4a', 'mp3', 'wav')")
    parser.add_argument("input_dir", help="Directory to search for audio files recursively")
    args = parser.parse_args()
    
    # Check if directory exists
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    try:
        # Check if CUDA is available
        if torch.cuda.is_available():
            device = "cuda"
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            print("CUDA not available. Using CPU.")
        
        # Load the largest Whisper model (once for all files)
        model = whisper.load_model("large", device=device)
        print("Model loaded successfully.")
        
        # Process all directories and transcribe audio files
        process_directory(args.input_dir, args.file_extension, model)
        
    except Exception as e:
        print(f"Error during transcription: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
