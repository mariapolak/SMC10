from tsm.time_stretch_base import TimeStretchBase
from ps.pitch_shift_base import PitchShiftBase

from modules import plotting
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import performence_eval as perf
import quality_eval_speech as qual
import audio_converter as ac

import config
import librosa
import glob

import logging

logger = logging.getLogger(__name__)

def run_batch_tsm_test(input_dir: str, extension: str = "flac"):
    for audio_path in glob.iglob(f"**/*.{extension}", root_dir=input_dir, recursive=True):  # find all flac files in the input directory
        x, sr = librosa.load(f"{input_dir}/{audio_path}", sr=None)                          # load the audio file
        
        for tsm_algorithm in config.TSM_ALGORITHMS:            # test each time-stretching algorithm on the audio file
            for tsm_factor in config.ALGORITHM_FACTORS["tsm_factors"]:
                output_filepath, filename = get_output_path_and_filename("tsm", tsm_algorithm.name, tsm_factor, audio_path)
                try:
                    if tsm_factor == "rt_up":
                        y_tmp = tsm_algorithm.time_stretch(x, sr, 2)
                        y = tsm_algorithm.time_stretch(y_tmp, sr, 0.5)
                    else: 
                        y = tsm_algorithm.time_stretch(x, sr, tsm_factor)
                except librosa.ParameterError:
                    logger.error(f"ParameterError: {tsm_algorithm.name} {tsm_factor} {audio_path}")
                except RuntimeError as e:
                    logger.error(f"RuntimeError: {tsm_algorithm.name} {tsm_factor} {audio_path}")
                sf.write(f"{output_filepath}/{filename}.wav", y, sr)


def run_batch_ps_test(input_dir: str, extension: str = "flac"):
    for audio_path in glob.iglob(f"**/*.{extension}", root_dir=input_dir, recursive=True):  # find all flac files in the input directory
        x, sr = librosa.load(f"{input_dir}/{audio_path}", sr=None)                  # load the audio file
        
        for ps_algorithm in config.PS_ALGORITHMS: # test each pitch-shifting algorithm on the audio file
            for ps_factor in config.ALGORITHM_FACTORS["ps_factors"]:
                output_filepath, filename = get_output_path_and_filename("ps", ps_algorithm.name, ps_factor, audio_path)
                try:
                    if ps_factor == "rt_up":
                        y_tmp = ps_algorithm.pitch_shift(x, sr, 12)
                        y = ps_algorithm.pitch_shift(y_tmp, sr, -12)
                    else:  
                        y = ps_algorithm.pitch_shift(x, sr, ps_factor)
                except librosa.ParameterError:
                    logger.error(f"ParameterError: {ps_algorithm.name} {ps_factor} {audio_path}")
                except RuntimeError as e:
                    logger.error(f"RuntimeError: {ps_algorithm.name} {ps_factor} {audio_path}")
                sf.write(f"{output_filepath}/{filename}.wav", y, sr)


def create_directories(input_dir: str, extension: str = "flac"):
    for audio_path in glob.iglob(f"**/*.{extension}", root_dir=input_dir, recursive=True):
        for tsm_algorithm in config.TSM_ALGORITHMS:
            for tsm_factor in config.ALGORITHM_FACTORS["tsm_factors"]:
                output_filepath, _ = get_output_path_and_filename("tsm", tsm_algorithm.name, tsm_factor, audio_path)
                Path(output_filepath).mkdir(parents=True, exist_ok=True)
        for ps_algorithm in config.PS_ALGORITHMS:
            for ps_factor in config.ALGORITHM_FACTORS["ps_factors"]:
                output_filepath, _ = get_output_path_and_filename("ps", ps_algorithm.name, ps_factor, audio_path)
                Path(output_filepath).mkdir(parents=True, exist_ok=True)

def get_output_path_and_filename(mode: str, algorithm: str, factor: int, input_file_path: str) -> tuple[str, str]:
    input_file_path_dir = Path(input_file_path).parent
    return f"{config.OUTPUT_DIR}_{config.TIMESTAMP}/wav48/{mode}/{algorithm}/{factor}/{input_file_path_dir}", Path(input_file_path).stem


def sanity_check():    
    import sounddevice as sd
    
    x, sr = librosa.load(f"{config.INPUT_DIR}/wav48/p225/p225_039.wav", sr=None)
    
    for tsm_algorithm in config.TSM_ALGORITHMS:
        for tsm_factor in config.ALGORITHM_FACTORS["tsm_factors"]:
            if tsm_factor == "rt_up":
                y_tmp = tsm_algorithm.time_stretch(x, sr, 2)
                print("y_tmp", y_tmp.shape)
                y = tsm_algorithm.time_stretch(y_tmp, sr, 0.5)
                print("y", y.shape)
            else:
                y = tsm_algorithm.time_stretch(x, sr, tsm_factor)
        
    # for ps_algorithm in config.PS_ALGORITHMS:
    #     for ps_factor in config.ALGORITHM_FACTORS["ps_factors"]:
    #         y = ps_algorithm.pitch_shift(x, sr, ps_factor)

if __name__ == "__main__":
    ### Audio and plot measurments
    input_dir = f"{config.INPUT_DIR}/wav48"

    create_directories(input_dir, "wav")
    logging.basicConfig(filename=f'{config.OUTPUT_DIR}_{config.TIMESTAMP}/run_tsm_ps.log', encoding='utf-8', level=logging.WARNING)
    
    print("Running batch time-stretching")
    run_batch_tsm_test(input_dir, "wav")
    
    print("Running batch pitch-shifting")
    run_batch_ps_test(input_dir, "wav")
    
    ac.create_wav_16k(f"{config.OUTPUT_DIR}_{config.TIMESTAMP}/wav16", f"{config.OUTPUT_DIR}_{config.TIMESTAMP}/wav48", ["wav"])
    # sanity_check()






