from abc import ABC, abstractmethod
import numpy as np

class PitchShiftBase(ABC):

    @abstractmethod
    def pitch_shift(self, input: np.array, sr: int, shift_factor: float) -> np.array:
        pass

    @property
    @abstractmethod
    def name(self):
        pass