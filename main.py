from tsm.time_stretch_base import TimeStretchBase
from tsm import harmonic_percussive_separation, noise_morphing, phase_vocoder

from ps.pitch_shift_base import PitchShiftBase
from ps import noise_morphing_ps
from modules import plotting

from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import performence_eval as perf

import librosa
import glob
import pandas as pd

INPUT_DIR = "data/input"
OUTPUT_DIR = "data/output"
OUTPUT_EVAL_OBJ_DIR = "evaluation/objective"
OUTPUT_EVAL_SUBJ_DIR = "evaluation/subjective"

SPEED_ITERATIONS = 10
MEMORY_ITERATIONS = 2

ALGORITHM_FACTORS = {
    "tsm_factors": [0.5, 2.],
    "ps_factors": [-12, 12],
}

TIMESTAMP = datetime.now().strftime("%y%m%d%H%M")
TSM_ALGORITHMS = [harmonic_percussive_separation.HPS(), phase_vocoder.PV()]
PS_ALGORITHMS = [noise_morphing_ps.NoiseMorphingPS()]

def run_time_stretch_test(x: np.ndarray, sr: int, file_path_out: str, tsm: TimeStretchBase):
    """
    Time stretches the input audio signal with the given TSM algorithm by each factor in the CONFIG dictionary and saves the outputs to files.
    """
    for tsm_factor in ALGORITHM_FACTORS["tsm_factors"]:
        
        y = tsm.time_stretch(x, sr, tsm_factor)
        plotting.plot_audio_comparison(x, y, sr, f"Time Stretch {tsm_factor}x", save=True, filepath=f"{file_path_out}_{tsm_factor}.png")
        sf.write(f"{file_path_out}_{tsm_factor}.flac", y, sr)

def run_pitch_shift_test(x: np.ndarray, sr: int, file_path_out: str, ps: PitchShiftBase):  
    """
    Pitch shifts the input audio signal with the given PS algorithm by each factor in the CONFIG dictionary and saves the outputs to files.
    """
    for ps_factor in ALGORITHM_FACTORS["ps_factors"]:
        
        y = ps.pitch_shift(x, sr, ps_factor)
        plotting.plot_audio_comparison(x, y, sr, f"Pitch Shift {ps_factor} st", save=True, filepath=f"{file_path_out}_{ps_factor}.png")
        sf.write(f"{file_path_out}_{ps_factor}.flac", y, sr)

def run_batch_tsm_test(input_dir: str):
    for audio_path in glob.iglob('**/*.flac', root_dir=input_dir, recursive=True):  # find all flac files in the input directory
        x, sr = librosa.load(f"{input_dir}/{audio_path}", sr=None)                  # load the audio file
        
        for tsm_algorithm in TSM_ALGORITHMS:            # test each time-stretching algorithm on the audio file
            output_filepath, filename = get_output_path_and_filename("tsm", tsm_algorithm.name, audio_path)
            run_time_stretch_test(x, sr, f"{output_filepath}/{filename}", tsm_algorithm)


def run_batch_ps_test(input_dir: str):
    for audio_path in glob.iglob('**/*.flac', root_dir=input_dir, recursive=True):  # find all flac files in the input directory
        x, sr = librosa.load(f"{input_dir}/{audio_path}", sr=None)                  # load the audio file
        
        for ps_algorithm in PS_ALGORITHMS: # test each pitch-shifting algorithm on the audio file
            output_filepath, filename = get_output_path_and_filename("ps", ps_algorithm.name, audio_path)
            run_pitch_shift_test(x, sr, f"{output_filepath}/{filename}", ps_algorithm)


def create_directories(input_dir: str):
    for audio_path in glob.iglob('**/*.flac', root_dir=input_dir, recursive=True):
        for tsm_algorithm in TSM_ALGORITHMS:
            output_filepath, _ = get_output_path_and_filename("tsm", tsm_algorithm.name, audio_path)
            Path(output_filepath).mkdir(parents=True, exist_ok=True)
        for ps_algorithm in PS_ALGORITHMS:
            output_filepath, _ = get_output_path_and_filename("ps", ps_algorithm.name, audio_path)
            Path(output_filepath).mkdir(parents=True, exist_ok=True)

def get_output_path_and_filename(mode: str, algorithm: str, input_file_path: str) -> tuple[str, str]:
    input_file_path_dir = Path(input_file_path).parent
    return f"{OUTPUT_DIR}_{TIMESTAMP}/{mode}/{algorithm}/{input_file_path_dir}", Path(input_file_path).stem

def run_performance_test(): # output_csv: str, test_audio: str, iterations: int
    output_csv = f"{OUTPUT_EVAL_OBJ_DIR}/performance_{TIMESTAMP}.csv"
    test_audio = next(glob.iglob('**/*.flac', root_dir=INPUT_DIR, recursive=True))
    x, sr = librosa.load(f"{INPUT_DIR}/{test_audio}", sr=None)
    audio_length = len(x) / sr
    tsm_factor = 2
    ps_factor = 2

    data = []

    for tsm_algorithm in TSM_ALGORITHMS:
        speed = perf.measure_speed(SPEED_ITERATIONS, tsm_algorithm.time_stretch, x, sr, tsm_factor)
        memory = perf.measure_memory(MEMORY_ITERATIONS, tsm_algorithm.time_stretch, x, sr, tsm_factor)
        data.append({
            "algorithm": tsm_algorithm.name, "type": "TSM", "speed": speed, 
            "speed_iterations": SPEED_ITERATIONS, "memory": memory, "memory_iterations": MEMORY_ITERATIONS,
            "audio_length": audio_length, "sample_rate": sr, "factor": tsm_factor
        })
        
    for ps_algorithm in PS_ALGORITHMS:
        speed = perf.measure_speed(SPEED_ITERATIONS, ps_algorithm.pitch_shift, x, sr, ps_factor)
        memory = perf.measure_memory(MEMORY_ITERATIONS, ps_algorithm.pitch_shift, x, sr, ps_factor)
        data.append({
            "algorithm": ps_algorithm.name, "type": "PS", "speed": speed, 
            "speed_iterations": SPEED_ITERATIONS, "memory": memory, "memory_iterations": MEMORY_ITERATIONS,
            "audio_length": audio_length, "sample_rate": sr, "factor": ps_factor
        })

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    ### Audio and plot measurments
    # create_directories(INPUT_DIR)
    # run_batch_tsm_test(INPUT_DIR)
    # run_batch_ps_test(INPUT_DIR)    

    ### Performance measurments
    run_performance_test()






