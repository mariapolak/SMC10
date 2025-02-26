from .time_stretch_base import TimeStretchBase

import pytsmod as tsm
import numpy as np
import librosa


class HPS(TimeStretchBase):
    def time_stretch(self, input: np.array, sr: int, stretch_factor: float) -> np.array:
        y = tsm.hptsm(input, stretch_factor)
        return y

    @property
    def name(self):
        return "HPS"



if __name__ == '__main__':
    hps = HPS()
    
    # Load an example audio file
    x, sr = librosa.load()
    
    # Stretch the audio signal by a factor of 2
    y = hps.time_stretch(x, sr, 2)

    # Save the stretched audio signal to a file
    librosa.output.write_wav('trumpet_stretched.wav', y, sr)