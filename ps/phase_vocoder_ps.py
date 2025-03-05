from .pitch_shift_base import PitchShiftBase
from dataclasses import dataclass

import pytsmod as tsm
import numpy as np
import librosa
import heapq

class PV(PitchShiftBase):
    """
    Based on Phase Vocoder Done Right pseudo code https://www.researchgate.net/publication/319503719_Phase_Vocoder_Done_Right
    Stores the track info locally and calculates the expected phase delta for the hop and frame size
    :param info: A TrackInfo object providing information about the track and the transformation

    Implementation based on https://github.com/BlackspireAudio/ba21_loma_2_py/
    """

    frame_size: int
    frame_size_padded: int
    frame_size_nyquist: int
    frame_size_resampling: int
    hop_size_factor: int
    hop_size_analysis: int
    hop_size_synthesis: int
    half_tone_steps_to_shift: int
    pitch_shift_factor: float
    time_stretch_ratio: float
    windowType: str
    normalize: bool

    phase_delta_target: np.array
    phase_delta_prev: np.array
    phase_analysis_prev: np.array
    phase_synthesis: np.array
    mid_range: slice
    magnitude_min_factor: float
    max_magnitude: float
    magnitude_prev: np.array

    def setup(self, sample_rate, shift_factor_st, magnitude_min_factor=10**-6):
        self.frame_size = int(2 ** (np.round(np.log2(sample_rate / 20))))
        self.pitch_shift_factor = self.pitch_factor_st_to_linear(shift_factor_st)
        self.hop_size_synthesis = int(self.frame_size / self.hop_size_factor)
        self.hop_size_analysis = int(self.hop_size_synthesis / self.pitch_shift_factor)
        self.time_stretch_ratio = self.hop_size_synthesis / self.hop_size_analysis
        self.frame_size_resampling = int(np.floor(self.frame_size * self.hop_size_analysis / self.hop_size_synthesis))
        self.frame_size_padded = self.frame_size
        self.frame_size_nyquist = int(self.frame_size_padded / 2 + 1)
        self.phase_delta_target = (2.0 * np.pi * self.info.hop_size_analysis) * np.array(range(0, self.info.frame_size_nyquist)) / self.info.frame_size_padded
        self.phase_analysis_prev = np.zeros(self.info.frame_size_nyquist)
        self.phase_synthesis = np.zeros(self.info.frame_size_nyquist)
        self.mid_range = slice(0, self.info.frame_size_nyquist)

        self.frame_index = 0

        self.magnitude_min_factor = magnitude_min_factor
        self.max_magnitude = 0
        self.magnitude_prev = np.zeros(self.info.frame_size_nyquist)
        self.phase_delta_prev = np.zeros(self.info.frame_size_nyquist)

    def pitch_shift(self, input: np.array, sr: int, shift_factor_st: float) -> np.array:
        """
        Calculates the expected phase shift using the significant magnitudes from the current and last frame and placing them into a self sorting max heap.
        Calculating the partial derivatives the phase influenced by the magnitudes in the heap are then propagated in a vertical direction.
        :param frame_fft: a frame in the frequency domain spectrum
        :return: the transformed phase for each bin
        """
        self.setup(sr, shift_factor_st)
        np.fft.rfft(input)
        X = librosa.stft(input, n_fft=self.frame_size)
        phase_transformed = self.phase_shifter.process(X)
        frame_fft_transposed = (abs(X) * np.exp(1j * phase_transformed))
        y = librosa.istft(frame_fft_transposed, n_fft=self.frame_size)

        y_resampled = librosa.resample(y, sr, sr * self.time_stretch_ratio)
        return y_resampled

    def process(self, frame_fft: list[complex]) -> list[float]:
        """
        Calculates the expected phase shift using the significant magnitudes from the current and last frame and placing them into a self sorting max heap.
        Calculating the partial derivatives the phase influenced by the magnitudes in the heap are then propagated in a vertical direction.
        :param frame_fft: a frame in the frequency domain spectrum
        :return: the transformed phase for each bin
        """
        magnitude = abs(frame_fft)
        # get imaginary values from fft
        phase_analysis = np.angle(frame_fft)

        # calculate the diff between last and current phase vector)
        phase_delta = self.phase_delta_target + princarg(phase_analysis - self.phase_analysis_prev - self.phase_delta_target)
        phase_delta = phase_delta * self.info.time_stretch_ratio

        mid_range = self.mid_range

        self.max_magnitude = max(max(magnitude), self.max_magnitude)
        min_magnitude = self.magnitude_min_factor * self.max_magnitude

        significant_magnitudes = {i: magnitude[i] for i in range(0, self.info.frame_size_nyquist)[mid_range] if magnitude[i] > min_magnitude}
        max_heap = [HeapBin(i, -1, self.magnitude_prev[i], 0) for i in significant_magnitudes.keys()]
        heapq.heapify(max_heap)

        # perform simple horizontal phase propagation for bins with insignificant magnitude
        for i in range(0, self.info.frame_size_nyquist)[mid_range]:
            if i not in significant_magnitudes.keys():
                self.phase_synthesis[i] = self.phase_synthesis[i] + phase_delta[i]

        while len(significant_magnitudes) > 0 and len(max_heap) > 0:
            max_bin = heapq.heappop(max_heap)
            bin_index = max_bin.bin_index
            time_index = max_bin.time_index
            if time_index < 0 and bin_index in significant_magnitudes.keys():
                # bin has been taken from the last frame and horizontal phase propagation (using backwards phase time derivative and trapezoidal integration) is needed
                self.phase_synthesis[bin_index] = self.phase_synthesis[bin_index] + (self.phase_delta_prev[bin_index] + phase_delta[bin_index]) / 2
                # add the current bin of the current frame to the heap for further processing
                heapq.heappush(max_heap, HeapBin(bin_index, 0, significant_magnitudes.get(bin_index), princarg(self.phase_synthesis[bin_index] - phase_analysis[bin_index])))
                # remove the processed bin from the set
                significant_magnitudes.pop(bin_index)

            if time_index >= 0:
                # the bin is from the current frame and vertical phase propagation (potentially in both directions) is needed
                for bin_index_other in (bin_index - 1, bin_index + 1):
                    # check if the surrounding two bins have significant magnitudes
                    if bin_index_other in significant_magnitudes.keys():
                        self.phase_synthesis[bin_index_other] = phase_analysis[bin_index_other] + max_bin.phase_rotation
                        # add the next / prev bin to the heap for further processing
                        heapq.heappush(max_heap, HeapBin(bin_index_other, 0, magnitude[bin_index_other], max_bin.phase_rotation))
                        # remove the processed bin from the set
                        significant_magnitudes.pop(bin_index_other)

        self.phase_analysis_prev = np.copy(phase_analysis)
        self.phase_delta_prev = np.copy(phase_delta)
        self.magnitude_prev = np.copy(magnitude)
        self.frame_index += 1

        return self.phase_synthesis

    def phase_reset(self, phase):
        """"
        Resets the phases according to the phase reset type and sets the current mid range
        :param phase: untransformed phase of the current frame
        :return: returns the default mid_range
        """
        self.phase_synthesis = phase
        return self.mid_range

    @property
    def name(self):
        return "PV"

def princarg(phase):
    return phase - 2 * np.pi * np.round(phase / (2*np.pi)).astype(int)

@dataclass
class HeapBin:
    bin_index: int
    time_index: int
    magnitude: float
    phase_rotation: float

    def __lt__(self, other): return self.magnitude > other.magnitude

    def __eq__(self, other): return self.magnitude == other.magnitude

    def __str__(self): return str(self.magnitude)