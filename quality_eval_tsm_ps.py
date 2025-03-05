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
    ### Fuzzy Energy
    ### Estimate spectral envelopes
    ### decay rate deviation (preservation of spectral characteristics)
    ### Objective measure from AES
    pass

def get_fuzzy_energy(x: np.array, sr: int):
    X = librosa.stft(x)
    Xn = np.abs(X)
    Y = np.sum(Xn**2, axis=0)
    return Y

def plot_fuzzy_energy(input: np.array, outputs: np.ndarray, outputs_names: list[str], sr: int):
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
    
if __name__ == "__main__":
    x, sr = librosa.load(f"data\input\p227\p227_001_mic1.flac", sr=None)
    y, sr = librosa.load(f"data\output_2502270954\\tsm\HPS\p227\p227_001_mic1_2.0.flac", sr=None)

    plot_fuzzy_energy(x, [y], ["HPS"], sr)