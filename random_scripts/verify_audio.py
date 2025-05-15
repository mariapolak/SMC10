import glob 
import os
# import wave
import librosa
import soundfile as sf
import torchaudio as ta
import torch
from tqdm import tqdm

def is_wav_faulty(wav_path):
    try:
        wav, _ = ta.load(str(wav_path))
        return False
    except Exception:
        print(f"Error loading {wav_path}")
        return True

base_dir = './../no_aug/train'  # Change this to your target directory
wav_files = ["spk1.wav", "spk2.wav", "spk3.wav"]

energies = {
    "spk1": [],
    "spk2": [],
    "spk3": []
}

files = glob.glob(os.path.join(base_dir, '*'))
for dir_path in tqdm(files, desc="Checking WAV files"):
    if not os.path.isdir(dir_path):
        continue
    
    # for each dir open the wav files and calulate their energy
    for wav_file in wav_files:
        wav_path = os.path.join(dir_path, wav_file)
        if not os.path.exists(wav_path):
            print(f"File {wav_path} does not exist.")
            continue
        
        # Check if the WAV file is faulty
        if is_wav_faulty(wav_path):
            print(f"Faulty WAV file: {wav_path}")
            continue

        # Load the audio file
        try:
            audio, sr = ta.load(wav_path)
            energy = torch.sum(audio ** 2).item()
            # Append the energy to the corresponding list
            if wav_file == "spk1.wav":
                energies["spk1"].append(energy)
            elif wav_file == "spk2.wav":
                energies["spk2"].append(energy)
            elif wav_file == "spk3.wav":
                energies["spk3"].append(energy)
            
        except Exception as e:
            print(f"Error processing {wav_file}: {e}")

    
# calculate the mean and std of the energies
for key in energies:
    if len(energies[key]) > 0:
        mean_energy = sum(energies[key]) / len(energies[key])
        std_energy = (sum((x - mean_energy) ** 2 for x in energies[key]) / len(energies[key])) ** 0.5
        print(f"{key} - Mean Energy: {mean_energy}, Std Energy: {std_energy}")
    else:
        print(f"No valid energy values for {key}.")