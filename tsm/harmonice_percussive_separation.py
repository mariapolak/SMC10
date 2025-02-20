import pytsmod as tsm
import numpy as np
import librosa

def time_stretch(input: np.array, sr: int, stretch_factor: float) -> np.array:
    y = tsm.hptsm(input, stretch_factor)
    return y


if __name__ == '__main__':
    # Load an example audio file
    x, sr = librosa.load()

    # Stretch the audio signal by a factor of 2
    y = time_stretch(x, sr, 2)

    # Save the stretched audio signal to a file
    librosa.output.write_wav('trumpet_stretched.wav', y, sr)