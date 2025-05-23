import numpy as np
import librosa
import glob
import os
import soundfile as sf
import matplotlib.pyplot as plt
import logging 
from pathlib import Path
import config
# for each directory in "wav/tests"
# read all the wav files
# and pad them with 0 to the same length (max length)
# save overwrite them
def pad_wav_files_in_directory(directory):
    # Get all wav files in the directory
    wav_files = glob.glob(f"{directory}/*.wav")
    print(f"Found {len(wav_files)} wav files in {directory}")
    # Read all wav files and find the maximum length
    max_length = 0
    audio_data = []
    
    sr = None
    for wav_file in wav_files:
        print(f"Reading {wav_file}")
        y, sr = librosa.load(wav_file, sr=None)
        audio_data.append(y)
        max_length = max(max_length, len(y))
    
    print(f"Maximum length: {max_length} / {max_length / sr } seconds")
    
    # # plot all audio data on the same plot with same time-scale (max_length)
    # plt.figure(figsize=(15, 10))
    # for i, y in enumerate(audio_data):
    #     plt.subplot(len(audio_data), 1, i + 1)
    #     plt.plot(np.linspace(0, max_length / sr, num=max_length), y)
    #     plt.title(f"File {i + 1}")
    #     plt.xlabel("Time (s)")
    #     plt.ylabel("Amplitude")
    
    # plt.suptitle(f"Audio files in {directory}")
    # plt.tight_layout()
    # plt.show()
    
    # Pad each wav file to the maximum length
    for i, wav_file in enumerate(wav_files):
        y = audio_data[i]
        if len(y) < max_length:
            y = np.pad(y, (0, max_length - len(y)), 'constant')
        
        # Save the padded wav file
        print(f"Writing {wav_file}")
        sf.write(wav_file, y, sr)

def assert_length_input_output():
    Path("evaluation/objective/").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=f'evaluation/audio_padding_{config.TIMESTAMP}.log',
        encoding='utf-8',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    REF_DIR_48k = "data/input/wav48"
    DEG_PARENT_DIR_48k = "data/output/wav48"

    for audio_path in glob.iglob(f"**/*.wav", root_dir=REF_DIR_48k, recursive=True): 
        for tsm_algorithm in config.TSM_ALGORITHMS:
            for tsm_factor in config.REFERENCE_FACTORS["tsm_factors"]:
                DEG_TSM_DIR_48k = f"{DEG_PARENT_DIR_48k}/tsm/{tsm_algorithm.name}/{tsm_factor}"

                infile = f"{REF_DIR_48k}/{audio_path}"
                outfile = f"{DEG_TSM_DIR_48k}/{audio_path}"

                x, sr = librosa.load(infile, sr=None)
                y, sr = librosa.load(outfile, sr=None)

                if len(x) == len(y):
                    continue

                logging.info(f" TSM Algo {tsm_algorithm.name} | Tsm factor {tsm_factor} | Padding {outfile} | Original length: {len(y)} samples | Target length: {len(x)} samples")
                if len(x) < len(y):
                    y = y[:len(x)]
                elif len(x) > len(y):
                    y = np.pad(y, (0, len(x) - len(y)), 'constant')
                
                sf.write(outfile, y, sr)

# for each directory in "wav/tests"
# print("Processing directories...")
# for dir in glob.glob("assets/wav/trainig/*"):
#     # check if the directory is a directory
#     if os.path.isdir(dir):
#         # pad the wav files in the directory
#         pad_wav_files_in_directory(dir)
#         print(f"Processed directory: {dir}")
#     else:
#         print(f"Skipped non-directory: {dir}")
        
# for dir in glob.glob("assets/wav/tests/*"):
#     # check if the directory is a directory
#     if os.path.isdir(dir):
#         # pad the wav files in the directory
#         pad_wav_files_in_directory(dir)
#         print(f"Processed directory: {dir}")
#     else:
#         print(f"Skipped non-directory: {dir}")



