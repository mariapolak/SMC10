from tsm.time_stretch_base import TimeStretchBase
from tsm import harmonic_percussive_separation, noise_morphing, phase_vocoder

from ps.pitch_shift_base import PitchShiftBase
from ps import noise_morphing_ps
from modules import plotting

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

import librosa
import os

INPUT_DIR = "data/input"
OUTPUT_DIR = "data/output"

CONFIG = {
    "tsm_factors": [0.5, 0.75, 0.9, 1.1, 1.25, 2.],
    "ps_factors": [-12, -7, -2, 2, 7, 12],
}

TSM_ALGORITHMS = [harmonic_percussive_separation.HPS(), phase_vocoder.PV()]
PS_ALGORITHMS = [noise_morphing_ps.NoiseMorphingPS()]

def run_time_stretch_test(x: np.ndarray, sr: int, filename: str, tsm: TimeStretchBase):
    """
    Time stretches the input audio signal with the given TSM algorithm by each factor in the CONFIG dictionary and saves the outputs to files.
    """
    for tsm_factor in CONFIG["tsm_factors"]:
        output_filepath = f"{OUTPUT_DIR}/tsm/{tsm.name}/{filename}_{tsm_factor}"
        
        y = tsm.time_stretch(x, sr, tsm_factor)
        plotting.plot_audio_comparison(x, y, sr, f"Time Stretch {tsm_factor}x", save=True, filepath=f"{output_filepath}.png")
        sf.write(f"{output_filepath}.flac", y, sr)

def run_pitch_shift_test(x: np.ndarray, sr: int, filename: str, ps: PitchShiftBase):  
    """
    Pitch shifts the input audio signal with the given PS algorithm by each factor in the CONFIG dictionary and saves the outputs to files.
    """
    for ps_factor in CONFIG["ps_factors"]:
        output_filepath = f"{OUTPUT_DIR}/ps/{ps.name}/{filename}_{ps_factor}"
        
        y = ps.pitch_shift(x, sr, ps_factor)
        plotting.plot_audio_comparison(x, y, sr, f"Pitch Shift {ps_factor} st", save=True, filepath=f"{output_filepath}.png")
        sf.write(f"{output_filepath}.flac", y, sr)

def run_batch_tsm_test(input_dir: str):
    for filename in os.listdir(input_dir):
        if filename.endswith(".flac"):                      # find all flac files in the input directory
            filepath = os.path.join(input_dir, filename)    # get the full path of the file
            x, sr = librosa.load(filepath, sr=None)         # load the audio file
            
            for tsm_algorithm in TSM_ALGORITHMS:            # test each time-stretching algorithm on the audio file
                run_time_stretch_test(x, sr, filename, tsm_algorithm)


def run_batch_ps_test(input_dir: str):
    for filename in os.listdir(input_dir):
        if filename.endswith(".flac"):                      # find all flac files in the input directory
            filepath = os.path.join(input_dir, filename)    # get the full path of the file
            x, sr = librosa.load(filepath, sr=None)         # load the audio file
            
            for ps_algorithm in PS_ALGORITHMS:              # test each pitch-shifting algorithm on the audio file
                run_pitch_shift_test(x, sr, filename, ps_algorithm)

if __name__ == "__main__":
    # run_batch_tsm_test(INPUT_DIR)

    filename = f"{INPUT_DIR}/p227_002_mic1.flac"
    x, sr = librosa.load(filename, sr=None)
    
    run_time_stretch_test(x, sr, f"{INPUT_DIR}/p227_002_mic1.flac", harmonic_percussive_separation.HPS())
    run_time_stretch_test(x, sr, f"{INPUT_DIR}/p227_002_mic1.flac", phase_vocoder.PV())
    





