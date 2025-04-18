from pathlib import Path
from tqdm import tqdm
from datetime import datetime

import soundfile as sf
import numpy as np
import matplotlib as plt

import config
import librosa
import glob
import logging
from tsm import phase_vocoder, resampling_tsm
from ps import phase_vocoder_ps
import csv

logger = logging.getLogger(__name__)

def augment_with_pvs(input_dir, output_dir):
    logger.info("Starting augmentation with phase vocoder")
    tsm_pv = phase_vocoder.PV()
    ps_pv = phase_vocoder_ps.PV()

    csv_file = f"{output_dir}/augmented_files.csv"
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Filename", "Pitch_Shift_Factor", "Time_Stretch_Factor"])

        for directory in tqdm(glob.glob(f"./*/", root_dir=input_dir, recursive=True)):
            logger.info(f"Processing directory: {directory}")
            for audio_path in glob.iglob(f"**/*.flac", root_dir=f"{input_dir}/{directory}", recursive=True):
                output_audio_directory = Path(f"{output_dir}/{directory}/{audio_path}").parent
                filename = Path(audio_path).stem
                output_audio_directory.mkdir(parents=True, exist_ok=True)

                x, sr = librosa.load(f"{input_dir}/{directory}/{audio_path}", sr=None)  # load the audio file
                output_file = f"{output_audio_directory}/{filename}.flac"
                sf.write(output_file, x, sr)
                writer.writerow([output_file, 0, 0])

                for i in range(3):
                    try:
                        ps_factor = np.clip(np.random.normal(0, 0.8), -1, 1)
                        ps_factor = ps_factor * 3 if ps_factor < 0 else ps_factor * 5
                        y_tmp = ps_pv.pitch_shift(x, sr, ps_factor)
                    except librosa.ParameterError:
                        logger.error(f"ParameterError: {ps_pv.name} {ps_factor} {audio_path}")
                        continue
                    except RuntimeError as e:
                        logger.error(f"RuntimeError: {ps_pv.name} {ps_factor} {audio_path}")
                        continue

                    try:
                        tsm_factor = (np.clip(np.random.normal(0, 0.8), -1, 1) + 1) / 2
                        tsm_factor = tsm_factor * 0.65 + 0.75

                        y = tsm_pv.time_stretch(y_tmp, sr, tsm_factor)
                    except librosa.ParameterError:
                        logger.error(f"ParameterError: {tsm_pv.name} {tsm_factor} {audio_path}")
                        continue
                    except RuntimeError as e:
                        logger.error(f"RuntimeError: {tsm_pv.name} {tsm_factor} {audio_path}")
                        continue
                
                    try:
                        output_file = f"{output_audio_directory}/{filename}_{i}.flac"
                        sf.write(output_file, y, sr)
                        writer.writerow([output_file, ps_factor, tsm_factor])

                    except Exception as e:
                        logger.error(f"Error writing file {output_audio_directory}/{filename}_{i}.flac: {e}")
                        continue

def augment_with_resampling(input_dir, output_dir):
    logger.info("Starting augmentation with resampling")
    resampler = resampling_tsm.ResamplingTSM()

    csv_file = f"{output_dir}/augmented_files.csv"
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Filename", "Resampling_Factor"])

        for directory in tqdm(glob.glob(f"./*/", root_dir=input_dir, recursive=True)):
            logger.info(f"Processing directory: {directory}")

            for audio_path in glob.iglob(f"**/*.flac", root_dir=f"{input_dir}/{directory}", recursive=True):
                output_audio_directory = Path(f"{output_dir}/{directory}/{audio_path}").parent
                filename = Path(audio_path).stem
                output_audio_directory.mkdir(parents=True, exist_ok=True)

                x, sr = librosa.load(f"{input_dir}/{directory}/{audio_path}", sr=None)  # load the audio file
                output_file = f"{output_audio_directory}/{filename}.flac"
                sf.write(output_file, x, sr)
                writer.writerow([output_file, 0])
                for i in range(3):
                    try:
                        rs_factor = (np.clip(np.random.normal(0, 0.8), -1, 1) + 1) / 2
                        rs_factor = rs_factor * 0.65 + 0.75
                        y = resampler.time_stretch(x, sr, rs_factor)
                    except librosa.ParameterError:
                        logger.error(f"ParameterError: {resampler.name} {rs_factor} {audio_path}")
                        continue
                    except RuntimeError as e:
                        logger.error(f"RuntimeError: {resampler.name} {rs_factor} {audio_path}")
                        continue
                
                    try:
                        output_file = f"{output_audio_directory}_{i}/{filename}.flac"
                        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
                        sf.write(output_file, y, sr)
                        writer.writerow([output_file, rs_factor])

                    except Exception as e:
                        logger.error(f"Error writing file {output_audio_directory}/{filename}_{i}.flac: {e}")
                        continue

def test():
    ps_factors = []
    tsm_factors = []
    for i in range(100):
        ps_factor = np.clip(np.random.normal(0, 0.8), -1, 1)
        ps_factor = ps_factor * 3 if ps_factor < 0 else ps_factor * 5
        ps_factors.append(ps_factor)

        tsm_factor = (np.clip(np.random.normal(0, 0.8), -1, 1) + 1) / 2
        tsm_factor = tsm_factor * 0.65 + 0.75
        tsm_factors.append(tsm_factor)

    plt.figure(figsize=(10, 5))
    # generate a scatter plot with 0:100 as x-axis and ps_factors as y-axis
    plt.scatter(range(100), ps_factors, label='Pitch Shift Factors')
    plt.xlabel('Sample Number')
    plt.ylabel('Factor Value')
    plt.title('Randomly Generated Factors')
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.scatter(range(100), tsm_factors, label='Time Stretch Factors')
    plt.xlabel('Sample Number')
    plt.ylabel('Factor Value')
    plt.title('Randomly Generated Factors')
    plt.show()



if __name__ == "__main__":
    input_dir = f"{config.INPUT_DIR}/SpeechDatasets/VCTK/wav16_silence_trimmed" 
    output_dir = f"{config.OUTPUT_DIR}_{config.TIMESTAMP}/SpeechDatasets/VCTK/wav16_silence_trimmed"
    Path(output_dir).mkdir(parents=True, exist_ok=True) 

    logging.basicConfig(
        filename=f'{config.OUTPUT_DIR}_{config.TIMESTAMP}/run_tsm_ps.log',
        encoding='utf-8',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    augment_with_pvs(input_dir, output_dir)
    logger.info("Augmentation with phase vocoder completed")

    timestamp = datetime.now().strftime("%y%m%d%H%M")
    output_dir_2 = f"{config.OUTPUT_DIR}_{timestamp}/SpeechDatasets/VCTK/wav16_silence_trimmed"
    Path(output_dir_2).mkdir(parents=True, exist_ok=True) 

    augment_with_resampling(input_dir, output_dir_2)
    logger.info("Augmentation with resampling completed")


    
            






