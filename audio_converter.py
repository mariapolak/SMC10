from pathlib import Path
from tqdm import tqdm

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
            output_filepath = f"{output_dir}/{audio_path_obj.parent}/{audio_path_obj.stem}.{extension}" # create file path for the output file
            Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)                     # assure the directory exists
            
            x, sr = librosa.load(f"{input_dir}/{audio_path}", sr=None)
            sf.write(output_filepath, x, sr)
            
def create_wav_16k(output_dir: str, input_dir: str, extensions: list[str] = ["flac", "wav"]):
    for extension in extensions:
        for audio_path in tqdm(glob.glob(f"**/*.{extension}", root_dir=input_dir, recursive=True)): 
            audio_path_obj = Path(audio_path) 
            output_filepath = f"{output_dir}/{audio_path_obj.parent}/{audio_path_obj.stem}.{extension}"
            Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)
            
            x, sr = librosa.load(f"{input_dir}/{audio_path}", sr=None)
            x_res = librosa.resample(x, orig_sr=sr, target_sr=16000)
            sf.write(output_filepath, x_res, 16000)
            
if __name__ == "__main__":
    # create_wav_16k("data/output/wav16", "data/output/wav48", ["flac"])
    # create_wav_16k("data/input/SpeechDatasets/VCTK/wav16_silence_trimmed", "data/input/SpeechDatasets/VCTK/wav48_silence_trimmed", ["flac"])
    create_wav_16k("conversationGeneration/generations/audio/train_16k_no_aug", "conversationGeneration/generations/audio/train_48k_no_aug", ["wav"])
            
