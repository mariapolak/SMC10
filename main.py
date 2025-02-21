from tsm import harmonice_percussive_separation, noise_morphing, phase_vocoder
from ps import noise_morphing_ps
from modules import plotting

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

import librosa
import os

OUTPUT_DIR = "data/output"

configuration = {
    "tsm_factors": [0.5, 0.75, 0.9, 1.1, 1.25, 2.],
    "ps_factors": [-12, -7, -2, 2, 7, 12],
}

tsm_algorithms = [harmonice_percussive_separation.time_stretch, phase_vocoder.time_stretch]
ps_algorithms = [noise_morphing_ps]

def run_time_stretch_test(x: np.ndarray, sr: int, filename: str, tsm_algorithm: callable):
    algorithm_name = tsm_algorithm.__name__ 
    
    for tsm_factor in configuration["tsm_factors"]:
        output_filepath = f"{OUTPUT_DIR}/tsm/{algorithm_name}/{filename}_{tsm_factor}"
        
        y = tsm_algorithm(x, sr, tsm_factor)
        plotting.plot_audio_comparison(x, y, sr, f"Time Stretch {tsm_factor}x", save=True, filepath=f"{output_filepath}.png")
        sf.write(f"{output_filepath}.flac", y, sr)

def run_pitch_shift_test(x: np.ndarray, sr: int, filename: str, ps_algorithm: callable):
    algorithm_name = ps_algorithm.__name__
    
    for ps_factor in configuration["ps_factors"]:
        output_filepath = f"{OUTPUT_DIR}/ps/{algorithm_name}/{filename}_{ps_factor}"
        
        y = ps_algorithm(x, sr, ps_factor)
        plotting.plot_audio_comparison(x, y, sr, f"Pitch Shift {ps_factor} st", save=True, filepath=f"{output_filepath}.png")
        sf.write(f"{output_filepath}.flac", y, sr)

def run_batch_tsm_test(input_dir: str):
    for filename in os.listdir(input_dir):
        if filename.endswith(".wav"):
            filepath = os.path.join(input_dir, filename)
            x, sr = librosa.load(filepath, sr=None)
            
            for tsm_algorithm in tsm_algorithms:
                run_time_stretch_test(x, sr, filename, tsm_algorithm)


def run_batch_ps_test(input_dir: str):
    for filename in os.listdir(input_dir):
        if filename.endswith(".wav"):
            filepath = os.path.join(input_dir, filename)
            x, sr = librosa.load(filepath, sr=None)
            
            for ps_algorithm in ps_algorithms:
                run_pitch_shift_test(x, sr, filename, ps_algorithm)

if __name__ == "__main__":
    x, sr = librosa.load("test.wav", sr=None)
    
    run_time_stretch_test(x, sr, "test.wav", harmonice_percussive_separation.time_stretch)
    run_time_stretch_test(x, sr, "test.wav", phase_vocoder.time_stretch)
    





