from pathlib import Path

import config
import librosa
import glob

import pandas as pd
import soundfile as sf

WAV_48K_DIR = f"{config.EVAL_DATA_DIR}/wav48"
WAV_16K_DIR = f"{config.EVAL_DATA_DIR}/wav16"


def create_wav_48k(output_dir: str, input_dir: str, extensions: list[str] = ["flac", "wav"]):
    for extension in extensions:
        for audio_path in glob.iglob(f"**/*.{extension}", root_dir=input_dir, recursive=True): 
            audio_path_obj = Path(audio_path) 
            output_filepath = f"{output_dir}/{audio_path_obj.parent}/{audio_path_obj.stem}.wav" # create file path for the output file
            Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)                     # assure the directory exists
            
            x, sr = librosa.load(f"{input_dir}/{audio_path}", sr=None)
            sf.write(output_filepath, x, sr)
            
def create_wav_16k(output_dir: str, input_dir: str, extensions: list[str] = ["flac", "wav"]):
    for extension in extensions:
        for audio_path in glob.iglob(f"**/*.{extension}", root_dir=input_dir, recursive=True): 
            audio_path_obj = Path(audio_path) 
            output_filepath = f"{output_dir}/{audio_path_obj.parent}/{audio_path_obj.stem}.wav"
            Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)
            
            x, sr = librosa.load(f"{input_dir}/{audio_path}", sr=None)
            x_res = librosa.resample(x, orig_sr=sr, target_sr=16000)
            sf.write(output_filepath, x_res, 16000)
            
if __name__ == "__main__":
    create_wav_16k("data/output/wav16", "data/output/wav48", ["wav"])
    create_wav_16k("data/input/wav16", "data/input/wav48", ["wav"])
            
