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

    def time_stretch(self, input: np.array, sr: int, stretch_factor: float) -> np.array:
        [xs, xt, xn] = STN.decSTN(input, sr, self.nWin1, self.nWin2)

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

    y = np.zeros(int(len(x) * stretch_factor))

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

def noise_stretching(x: np.array, stretch_factor: float) -> np.array:
    """
    Stretch the noise in an audio signal by a given factor.

    Parameters:
    x (np.array): Input audio signal.
    stretch_factor (float): Factor by which to stretch the noise in the signal.

    Returns:
    np.array: The noise-stretched audio signal.
    """
    fft_size = window_size = 2048
    hop = fft_size // 2
    window = np.hanning(window_size)

    # Compute the STFT of x
    X = librosa.stft(x, n_fft=fft_size, hop_length=hop, window=window)

    Xn_db = 10 * np.log10(np.abs(X)) # log-magnitude spectrum
    Xn_stretched_db = frames_interpolation(Xn_db, stretch_factor) # interpolate log-magnitude spectrum
    Xn_stretched = 10**(Xn_stretched_db / 10) # inverse log-magnitude spectrum

    Yn_stretched = noise_morphing(
        Xn_stretched, 
        original_signal_len=len(x), 
        n_fft=fft_size, 
        hop_length=hop, 
        window=window, 
        stretch_factor=stretch_factor) # morph with white noise

    # Compute the ISTFT of the stretched noise
    y = librosa.istft(Yn_stretched, n_fft=fft_size, hop_length=hop, window=window)
    return y

def frames_interpolation(X: np.ndarray, stretch_factor: float) -> np.ndarray:
    """ Linearly interpolates the log-magnitude spectrum is then according to the stretching factor based on the two neighboring spectra, occurring before and after the interpolation point.

    Args:
        X (np.ndarray): log-magnitude spectrum
        stretch_factor (float, optional): stretching factor

    Returns:
        np.ndarray: Stretched log-magnitude spectrum
    """

    x_len = X.shape[1] # number of frames
    x_stretched_len = int(np.ceil(x_len * stretch_factor)) # number of frames after stretching 
    
    if stretch_factor > 1: # fixing stft padding
        x_stretched_len -= 1

    x_stretched = np.zeros((X.shape[0], x_stretched_len)) # output array

    # Boundries
    x_stretched[:,0] = X[:,0] 
    x_stretched[:,-1] = X[:,-1]

    # streteched original indices -> [0,1,2,3,...,N] -> 1.5 -> [0, 1.5, 3, 4.5,..., N*1.5]
    stretched_indices = np.arange(x_len) * stretch_factor
    j = 1
    
    prev_idx = 0
    prev_xn = X[:,0] 

    next_idx = stretched_indices[j]
    next_xn = X[:,j]

    prev_next_dist = next_idx - prev_idx

    # Interpolation
    for i in range(1, x_stretched_len-1):
        while i > next_idx:
            j += 1

            prev_idx = next_idx
            prev_xn = next_xn

            next_idx = stretched_indices[j]
            next_xn = X[:,j]

            prev_next_dist = next_idx - prev_idx
        
        prev_dist = i - prev_idx
        next_dist = next_idx - i

        x_stretched[:,i] = prev_xn * (next_dist / prev_next_dist) + next_xn * (prev_dist / prev_next_dist)

    return x_stretched

def noise_morphing(x: np.ndarray, original_signal_len: int, n_fft: int, window: np.ndarray, hop_length: int,  stretch_factor: float):
    """ Interpolated magnitude spectra are modulated by the white noise spectral frames via element-wise multiplication.

    Args:
        x (np.ndarray): magnitude spectrum
        original_signal_len (int): length of the original signal
        n_fft (int): FFT length
        window (np.ndarray): FFT windowing function
        hop_length (int): FFT hop length
        stretch_factor (float, optional): stretch factor

    Returns:
        np.ndarray: Modulated magnitude spectra
    """

    white_noise = np.random.normal(0, 1, int(np.ceil(original_signal_len*stretch_factor))) # generate white noise
    white_noise = white_noise / np.max(np.abs(white_noise)) # normalize it

    E = librosa.stft(white_noise, n_fft=n_fft, hop_length=hop_length, window=window) # run STFT on it 
    assert E.shape[1] == x.shape[1]

    E = E / np.sqrt(np.sum(window**2)) # normalize by the window energy to ensure spectral magnitude equals 1

    # multiply each frame of the white noise by the interpolated frame of the input's noise (x)
    return x * E # element wise multiplication

