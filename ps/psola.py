from .pitch_shift_base import PitchShiftBase

import pytsmod as tsm
import numpy as np
import crepe
import librosa

class TDPSOLA(PitchShiftBase):
    def pitch_shift(self, input: np.array, sr: int, shift_factor_st: float) -> np.array:
        shift_factor = self.pitch_factor_st_to_linear(shift_factor_st)
        # _, f0, _, _ = crepe.predict(input, sr, viterbi=True, step_size=10)
        f0, _, _ = librosa.pyin(input, sr=sr, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C8'), fill_na=None)
        y = tsm.tdpsola(input, sr, f0, beta=shift_factor, p_hop_size=441, p_win_size=1470)

        return y

    @property
    def name(self):
        return "PSOLA"
    
if __name__ == "__main__":
    import sounddevice as sd
    import matplotlib.pyplot as plt

    x, sr = librosa.load(f"..\data\input\p227_001\p227_001_mic1.flac", sr=None) # Load audio
    psola = TDPSOLA() # Initialize PSOLA object
    y = psola.pitch_shift(x, sr, 12) # Pitch shift

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    
    D_1 = librosa.stft(x)
    S_db_1 = librosa.amplitude_to_db(np.abs(D_1), ref=np.max)
    librosa.display.specshow(S_db_1, x_axis='time', y_axis='hz', sr=sr)

    plt.subplot(1, 2, 2)
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    librosa.display.specshow(S_db, x_axis='time', y_axis='hz', sr=sr)
    plt.show()