from abc import ABC, abstractmethod
import numpy as np

class TimeStretchBase(ABC):

    @abstractmethod
    def time_stretch(self, input: np.array, sr: int, stretch_factor: float) -> np.array:
        pass

    @property
    @abstractmethod
    def name(self):
        pass