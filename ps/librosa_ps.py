from .pitch_shift_base import PitchShiftBase

import numpy as np
import librosa

class LibrsaPS(PitchShiftBase):
    def pitch_shift(self, input: np.array, sr: int, shift_factor_st: float) -> np.array:
        y = librosa.effects.pitch_shift(input, sr=sr, n_steps=shift_factor_st)
        return y

    @property
    def name(self):
        return "LibrosaPS"
    