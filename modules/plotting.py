import matplotlib.pyplot as plt
import numpy as np
import librosa

def plot_waveform(x : np.ndarray, sr : int, title : str):
    librosa.display.waveshow(x, sr=sr)
    plt.title(title)

def plot_spectrogram(x : np.ndarray, sr : int, title : str):
    D = librosa.stft(x)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    librosa.display.specshow(S_db, x_axis='time', y_axis='hz', sr=sr)
    plt.colorbar(format="%+2.f dB")
    plt.title(title)
    
def plot_audio(x: np.ndarray, sr: int, title: str):
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plot_waveform(x, sr, title)
    plt.subplot(1, 2, 2)
    plot_spectrogram(x, sr, title)
    plt.show()
    
def plot_audio_comparison(x: np.ndarray, y: np.ndarray, sr: int, title: str, save: bool = False, filepath: str = None):
    if save and filepath is None:
        raise ValueError("Filename must be provided if saving the plot.")
    
    if x.shape != y.shape:
        # If the shapes are different we pad the shorter signal with zeros to the same length
        max_len = max(x.shape[0], y.shape[0])
        x = librosa.util.fix_length(x, size=max_len)
        y = librosa.util.fix_length(y, size=max_len)
    
    plt.figure(figsize=(14, 10))
    plt.suptitle(title)
    
    plt.subplot(2, 2, 1)
    plot_waveform(x, sr, f"Original")
    plt.subplot(2, 2, 2)
    plot_spectrogram(x, sr, f"Original")
    
    plt.subplot(2, 2, 3)
    plot_waveform(y, sr, f"Modified")
    plt.subplot(2, 2, 4)
    plot_spectrogram(y, sr, f"Modified")
    
    plt.tight_layout()
    
    if save:
        plt.savefig(filepath)
    else:
        plt.show()