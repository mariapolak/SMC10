from .time_stretch_base import TimeStretchBase

import pytsmod as tsm
import numpy as np
import librosa

class PV(TimeStretchBase):
    def time_stretch(self, input: np.array, sr: int, stretch_factor: float) -> np.array:
        y = tsm.phase_vocoder(input, stretch_factor, phase_lock=True)
        return y

    @property
    def name(self):
        return "PV"
