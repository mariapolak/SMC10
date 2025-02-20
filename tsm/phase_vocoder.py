import pytsmod as tsm
import numpy as np
import librosa

def time_stretch(input: np.array, sr: int, stretch_factor: float) -> np.array:
    y = tsm.phase_vocoder(input, stretch_factor, phase_lock=True)
    return y
