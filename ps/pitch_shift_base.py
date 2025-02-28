from abc import ABC, abstractmethod
import numpy as np

class PitchShiftBase(ABC):

    @abstractmethod
    def pitch_shift(self, input: np.array, sr: int, shift_factor_st: float) -> np.array:
        pass

    def pitch_factor_st_to_linear(self, shift_factor_st: float) -> float:
        return pow(2, shift_factor_st/12)

    @property
    @abstractmethod
    def name(self):
        pass
    
    @property
    def type(self):
        return "PS"