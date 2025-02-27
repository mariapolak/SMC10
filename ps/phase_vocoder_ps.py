from .pitch_shift_base import PitchShiftBase

import pytsmod as tsm
import numpy as np
import librosa

class PV(PitchShiftBase):
    def pitch_shift(self, input: np.array, sr: int, shift_factor_st: float) -> np.array:
        pass

    @property
    def name(self):
        return "PV"
