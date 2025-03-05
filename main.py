from tsm.time_stretch_base import TimeStretchBase
from ps.pitch_shift_base import PitchShiftBase

from modules import plotting
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import performence_eval as perf
import quality_eval_speech as qual

import config
import librosa
import glob


def run_time_stretch_test(x: np.ndarray, sr: int, file_path_out: str, tsm: TimeStretchBase):
    """
    Time stretches the input audio signal with the given TSM algorithm by each factor in the CONFIG dictionary and saves the outputs to files.
    """
    for tsm_factor in config.ALGORITHM_FACTORS["tsm_factors"]:
        
        y = tsm.time_stretch(x, sr, tsm_factor)
        # plotting.plot_audio_comparison(x, y, sr, f"Time Stretch {tsm_factor}x", save=True, filepath=f"{file_path_out}_{tsm_factor}.png")
        sf.write(f"{file_path_out}_{tsm_factor}.wav", y, sr)

def run_pitch_shift_test(x: np.ndarray, sr: int, file_path_out: str, ps: PitchShiftBase):  
    """
    Pitch shifts the input audio signal with the given PS algorithm by each factor in the CONFIG dictionary and saves the outputs to files.
    """
    for ps_factor in config.ALGORITHM_FACTORS["ps_factors"]:
        
        y = ps.pitch_shift(x, sr, ps_factor)
        # plotting.plot_audio_comparison(x, y, sr, f"Pitch Shift {ps_factor} st", save=True, filepath=f"{file_path_out}_{ps_factor}.png")
        sf.write(f"{file_path_out}_{ps_factor}.wav", y, sr)

def run_batch_tsm_test(input_dir: str, extension: str = "flac"):
    for audio_path in glob.iglob(f"**/*.{extension}", root_dir=input_dir, recursive=True):  # find all flac files in the input directory
        x, sr = librosa.load(f"{input_dir}/{audio_path}", sr=None)                  # load the audio file
        
        for tsm_algorithm in config.TSM_ALGORITHMS:            # test each time-stretching algorithm on the audio file
            output_filepath, filename = get_output_path_and_filename("tsm", tsm_algorithm.name, audio_path)
            run_time_stretch_test(x, sr, f"{output_filepath}/{filename}", tsm_algorithm)


def run_batch_ps_test(input_dir: str, extension: str = "flac"):
    for audio_path in glob.iglob(f"**/*.{extension}", root_dir=input_dir, recursive=True):  # find all flac files in the input directory
        x, sr = librosa.load(f"{input_dir}/{audio_path}", sr=None)                  # load the audio file
        
        for ps_algorithm in config.PS_ALGORITHMS: # test each pitch-shifting algorithm on the audio file
            output_filepath, filename = get_output_path_and_filename("ps", ps_algorithm.name, audio_path)
            run_pitch_shift_test(x, sr, f"{output_filepath}/{filename}", ps_algorithm)


def create_directories(input_dir: str, extension: str = "flac"):
    for audio_path in glob.iglob(f"**/*.{extension}", root_dir=input_dir, recursive=True):
        for tsm_algorithm in config.TSM_ALGORITHMS:
            output_filepath, _ = get_output_path_and_filename("tsm", tsm_algorithm.name, audio_path)
            Path(output_filepath).mkdir(parents=True, exist_ok=True)
        for ps_algorithm in config.PS_ALGORITHMS:
            output_filepath, _ = get_output_path_and_filename("ps", ps_algorithm.name, audio_path)
            Path(output_filepath).mkdir(parents=True, exist_ok=True)

def get_output_path_and_filename(mode: str, algorithm: str, input_file_path: str) -> tuple[str, str]:
    input_file_path_dir = Path(input_file_path).parent
    return f"{config.OUTPUT_DIR}_{config.TIMESTAMP}/{mode}/{algorithm}/{input_file_path_dir}", Path(input_file_path).stem



if __name__ == "__main__":
    ## Audio and plot measurments
    input_dir = f"{config.INPUT_DIR}/wav48"

    create_directories(input_dir, "wav")
    run_batch_tsm_test(input_dir, "wav")
    run_batch_ps_test(input_dir, "wav")

    # ## Performance measurments
    # perf.run_performance_test()

    # ## Quality measurments
    # qual.run_speech_metrics()
    # qual.run_audio_metrics()






