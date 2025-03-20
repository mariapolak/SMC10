from modules.decomposeSTN import decomposeSTN as STN

import numpy as np
import librosa

def noise_stretching(x: np.array, stretch_factor: float) -> np.array:
    """
    Stretch the noise in an audio signal by a given factor.

    Parameters:
    x (np.array): Input audio signal.
    stretch_factor (float): Factor by which to stretch the noise in the signal.

    Returns:
    np.array: The noise-stretched audio signal.
    """
    y_len = int(np.ceil(len(x) * stretch_factor))
    fft_size = window_size = 2048
    hop = fft_size // 2
    window = np.hanning(window_size)

    # Compute the STFT of x
    X = librosa.stft(x, n_fft=fft_size, hop_length=hop, window=window, center=False)

    Xn_db = 10 * np.log10(np.abs(X)) # log-magnitude spectrum
    Xn_stretched_db = frames_interpolation(Xn_db, stretch_factor, y_len=y_len, n_fft=fft_size, hop_size=hop) # interpolate log-magnitude spectrum
    Xn_stretched = 10**(Xn_stretched_db / 10) # inverse log-magnitude spectrum

    Yn_stretched = noise_morphing(
        Xn_stretched, 
        original_signal_len=len(x), 
        n_fft=fft_size, 
        hop_length=hop, 
        window=window, 
        stretch_factor=stretch_factor) # morph with white noise

    # Compute the ISTFT of the stretched noise
    y = librosa.istft(Yn_stretched, n_fft=fft_size, hop_length=hop, window=window, center=False, length=y_len)
    return y

def frames_interpolation(X: np.ndarray, stretch_factor: float, y_len: int, n_fft: int, hop_size: int) -> np.ndarray:
    """ Linearly interpolates the log-magnitude spectrum is then according to the stretching factor based on the two neighboring spectra, occurring before and after the interpolation point.

    Args:
        X (np.ndarray): log-magnitude spectrum
        stretch_factor (float, optional): stretching factor

    Returns:
        np.ndarray: Stretched log-magnitude spectrum
    """
    print(stretch_factor)
    x_len = X.shape[1] # number of frames

    x_stretched_len = 1 + int((y_len - n_fft) / hop_size) # number of frames in the stretched signal
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
        while i > next_idx and j < len(stretched_indices) - 1:
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

    E = librosa.stft(white_noise, n_fft=n_fft, hop_length=hop_length, window=window, center=False) # run STFT on it 
    print("whte noise shape", white_noise.shape)
    print(E.shape)
    print(x.shape)
    
    assert E.shape[1] == x.shape[1]

    E = E / np.sqrt(np.sum(window**2)) # normalize by the window energy to ensure spectral magnitude equals 1

    # multiply each frame of the white noise by the interpolated frame of the input's noise (x)
    return x * E # element wise multiplication