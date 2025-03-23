from modules.NM.nm import noise_stretching
from modules.decomposeSTN import decomposeSTN as STN
from .pitch_shift_base import PitchShiftBase

import numpy as np
import librosa

class NoiseMorphingPS(PitchShiftBase):
    def __init__(self):
        self.nWin1 = 8192   # samples
        self.nWin2 = 512    # samples
        
    def pitch_shift(self, input: np.array, sr: int, shift_factor_st: float) -> np.array:
        [xs, xt, xn] = STN.decSTN(input, sr, self.nWin1, self.nWin2)
        xs = align_length(len(input), xs)
        xt = align_length(len(input), xt)
        xn = align_length(len(input), xn)
        
        shift_factor = self.pitch_factor_st_to_linear(shift_factor_st)
        
        xs_shifted = sines_shifting(xs, sr, shift_factor_st)
        xt_shifted = transient_shifting(xt, sr, shift_factor_st)
        xn_shifted = noise_shifting(xn, sr, shift_factor)

        return xs_shifted + xt_shifted + xn_shifted

    @property
    def name(self):
        return "NMPS"

def sines_shifting(x: np.array, sr: int, shift_factor_st: float) -> np.array:
    y = librosa.effects.pitch_shift(x, sr=sr, n_steps=shift_factor_st)
    return y

def transient_shifting(x: np.array, sr: int, shift_factor_st: float) -> np.array:
    return x

def noise_shifting(x: np.array, sr: int, shift_factor: float) -> np.array:
    xn_stretched = noise_stretching(x, shift_factor)
    xn_shifted = librosa.resample(xn_stretched, orig_sr=sr*shift_factor, target_sr=sr)
    
    xn_shifted = align_length(len(x), xn_shifted) # trim the noise signal to the length of the input signal
    
    return xn_shifted

def align_length(length: int, x: np.array):
    if len(x) > length:
        return x[:length]
    else:
        return np.pad(x, (0, length - len(x)))