from modules.NM.nm import noise_stretching
from modules.decomposeSTN import decomposeSTN as STN
from .time_stretch_base import TimeStretchBase

import pytsmod as tsm
import numpy as np
import librosa

"""
Implementation of the noise morphing algorithm for time-stretching noise in an audio signal. 
Eloi Moliner et al., “Noise Morphing for Audio Time Stretching,” IEEE Signal Processing Letters 31 (2024): 1144–1148, accessed September 25, 2024, https://ieeexplore.ieee.org/document/10494355/.
"""

class NoiseMorphing(TimeStretchBase):
    def __init__(self):
        self.nWin1 = 8192 # samples
        self.nWin2 = 512 # samples

    def time_stretch(self, x: np.array, sr: int, stretch_factor: float) -> np.array:
        [xs, xt, xn] = STN.decSTN(x, sr, self.nWin1, self.nWin2)
        
        xs_stretched = sines_stretching(xs, stretch_factor)
        xt_stretched = transient_stretching(xt, sr, stretch_factor)
        xn_stretched = noise_stretching(xn, stretch_factor)

        return xs_stretched + xt_stretched + xn_stretched

    @property
    def name(self):
        return "NM"

def sines_stretching(x: np.array, stretch_factor: float) -> np.array:
    y = tsm.phase_vocoder(x, stretch_factor, phase_lock=True)
    return y

def transient_stretching(x: np.array, sr: int, stretch_factor: float) -> np.array:
    onsets = librosa.onset.onset_detect(y=x, sr=sr, units='samples', backtrack=True)
    onsets = np.append(onsets, len(x)) # add the end of the signal

    y = np.zeros(int(np.ceil(len(x) * stretch_factor)))

    pad_before = int(2e-3 * sr) # 2ms of padding in the beginning of the transient
    pad_after = int(10e-3 * sr) # 10ms of padding at the end of the transient

    for i in range(len(onsets) - 1):
        start = onsets[i]
        end = onsets[i+1]
        length = end - start
        
        # window for the transient (rectangular window with triangular edges 2ms before 10 ms after)
        window = np.array([])
        if length < pad_before + pad_after: # if the transient is shorter than the padding - should not happen
            window = np.ones(length)
        else:
            transient_length = length - pad_before - pad_after
            window = np.linspace(0, 1, pad_before)
            window = np.append(window, np.ones(transient_length))
            window = np.append(window, np.linspace(1, 0, pad_after))    
        
        inv_window = 1 - window # inverse window

        transient = x[start:end] * window # cut the transient out

        start_stretched = int(start * stretch_factor)
        end_stretched = min(start_stretched + length, len(y))
        length_stretched = end_stretched - start_stretched

        y[start_stretched:end_stretched] *= inv_window[:length_stretched] # apply inverse window to the area where the transient will be placed
        y[start_stretched:end_stretched] += transient[:length_stretched] # place the transient
    
    return y
