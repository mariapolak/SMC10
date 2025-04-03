import numpy as np
import librosa
import glob
import os
import soundfile as sf
import matplotlib.pyplot as plt

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
        
# for each directory in "wav/tests"
print("Processing directories...")
for dir in glob.glob("assets/wav/trainig/*"):
    # check if the directory is a directory
    if os.path.isdir(dir):
        # pad the wav files in the directory
        pad_wav_files_in_directory(dir)
        print(f"Processed directory: {dir}")
    else:
        print(f"Skipped non-directory: {dir}")
        
for dir in glob.glob("assets/wav/tests/*"):
    # check if the directory is a directory
    if os.path.isdir(dir):
        # pad the wav files in the directory
        pad_wav_files_in_directory(dir)
        print(f"Processed directory: {dir}")
    else:
        print(f"Skipped non-directory: {dir}")
