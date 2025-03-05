from pathlib import Path
from modules.decomposeSTN import decomposeSTN as STN

import config
import librosa
import glob
import json

import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

def run_audio_metrics():
    ### Objective measure from AES
    pass

def plot_fuzzy_energy(input: np.array, outputs: np.ndarray, outputs_names: list[str], sr: int):
    def get_fuzzy_energy(x: np.array, sr: int):
        # TODO figure out how to get dB
        X = librosa.stft(x)
        Xn = np.abs(X)
        Y = np.sum(Xn**2, axis=0)
        return Y

    nWin1 = 8192 # samples
    nWin2 = 512 # samples

    [xs, xt, xn] = STN.decSTN(input, sr, nWin1, nWin2)

    Xs = get_fuzzy_energy(xs, sr)
    Xt = get_fuzzy_energy(xt, sr)
    Xn = get_fuzzy_energy(xn, sr)

    fuzzy_energy = {
        "sines":[],
        "transients":[],
        "noise":[],
        "name": []
    }

    for output, name in zip(outputs, outputs_names):
        [ys, yt, yn] = STN.decSTN(output, sr, nWin1, nWin2)
        fuzzy_energy["sines"].append(get_fuzzy_energy(ys, sr))
        fuzzy_energy["transients"].append(get_fuzzy_energy(yt, sr))
        fuzzy_energy["noise"].append(get_fuzzy_energy(yn, sr))
        fuzzy_energy["name"].append(name)

    len_fuzzy_energy = len(fuzzy_energy["sines"][0])
    # interpolate Xs, Xt, Xn to match the length of the outputs
    zoom_factor = len_fuzzy_energy / len(Xs)
    Xs = zoom(Xs, zoom_factor)
    Xt = zoom(Xt, zoom_factor)
    Xn = zoom(Xn, zoom_factor)

    fuzzy_energy["sines"].append(Xs)
    fuzzy_energy["transients"].append(Xt)
    fuzzy_energy["noise"].append(Xn)
    fuzzy_energy["name"].append("Original")

    df = pd.DataFrame(fuzzy_energy)

    time = np.linspace(0, len(input) / sr, len(df['sines'][0]))

    plt.figure()
    plt.subplot(3, 1, 1)
    for signal, name in zip(df['sines'], df['name']):
        plt.plot(time, signal, label=name)
    plt.ylabel('Sines')
    plt.legend()

    plt.subplot(3, 1, 2)
    for signal, name in zip(df['transients'], df['name']):
        plt.plot(time, signal, label=name)
    plt.ylabel('Transients')
    plt.legend()

    plt.subplot(3, 1, 3)
    for signal, name in zip(df['noise'], df['name']):
        plt.plot(time, signal, label=name)
    plt.ylabel('Noise')
    plt.legend()

    plt.xlabel('Time (s)')
    plt.show()
    

# Metric for pitch shifting:
def decay_rate_deviation(input: np.array, outputs: np.ndarray, outputs_names: list[str], sr: int):
    # Compute RMS envelope for input
    frame_length = 500
    hop_length = frame_length // 2
    x_rms = librosa.feature.rms(y=input, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(x_rms)), sr=sr, hop_length=hop_length)
    
    decay_rates = {
        "name": [],
        "decay_rate_deviation": []
    }

    plt.figure(figsize=(10, 5))
    plt.plot(times, x_rms, color="r", label="RMS In Envelope")

    for output, name in zip(outputs, outputs_names):
        y_rms = librosa.feature.rms(y=output, frame_length=frame_length, hop_length=hop_length)[0]
        R_dr_L1 = np.linalg.norm((y_rms / x_rms) - 1, ord=2) # TODO find correct formula for decay rate deviation bc this is bollocks
        decay_rates["name"].append(name)
        decay_rates["decay_rate_deviation"].append(R_dr_L1)
        plt.plot(times, y_rms, label=f"RMS Out Envelope - {name}")

    # Plot original audio in the background
    plt.plot(np.linspace(0, len(input) / sr, len(input)), input, color="gray", alpha=0.5, label="Original Audio")

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("RMS Envelope of the Signal")
    plt.legend()
    plt.show()

    for name, rate in zip(decay_rates["name"], decay_rates["decay_rate_deviation"]):
        print(f"Decay Rate Deviation (L1 Norm) for {name}: {rate}")

def spectral_envelopes_mse(input: np.array, outputs: np.ndarray, outputs_names: list[str], sr: int):
    # nice and all but if we just feed a not working algorithm that doesn't change the output at all, the mse will be 0
    def calculate_spectral_envelope(signal: np.array, sr: int):
        S = np.abs(librosa.stft(signal))
        envelope = np.mean(S, axis=0)
        return envelope

    original_envelope = calculate_spectral_envelope(input, sr)

    mse_values = {
        "name": [],
        "mse": []
    }

    plt.figure(figsize=(10, 5))
    freqs = np.linspace(0, sr / 2, len(original_envelope))
    plt.plot(freqs, original_envelope, color="r", label="Original Spectral Envelope")

    for output, name in zip(outputs, outputs_names):
        output_envelope = calculate_spectral_envelope(output, sr)
        mse = np.mean((original_envelope - output_envelope)**2)
        mse_values["name"].append(name)
        mse_values["mse"].append(mse)
        plt.plot(freqs, output_envelope, label=f"Spectral Envelope - {name}")

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title("Spectral Envelopes")
    plt.legend()
    plt.show()

    for name, mse in zip(mse_values["name"], mse_values["mse"]):
        print(f"Mean Squared Error for {name}: {mse}")



if __name__ == "__main__":
    x, sr = librosa.load(f"data\input\p227\p227_001_mic1.flac", sr=None)
    y, sr = librosa.load(f"data\output_2502270954\ps\PSOLA\p227\p227_001_mic1_12.flac", sr=None)
    y1, sr = librosa.load(f"data\output_2502270954\ps\PSOLA\p227\p227_001_mic1_-12.flac", sr=None)

    spectral_envelopes_mse(x, [y,y1], ["PSOLA", "PSOLA2"], sr)