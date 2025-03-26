from .time_stretch_base import TimeStretchBase

import numpy as np
import librosa

class ResamplingTSM(TimeStretchBase):
    def time_stretch(self, input: np.array, sr: int, stretch_factor: float) -> np.array:
        y = librosa.resample(input, orig_sr=sr, target_sr=stretch_factor*sr)
        return y

    @property
    def name(self):
        return "RES"
