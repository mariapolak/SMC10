from .tsm_net.tsmnet import Stretcher
from .time_stretch_base import TimeStretchBase
from pathlib import Path

import os
import time
import torch
import torchaudio

import noisereduce as nr
import numpy as np

class TSMNET(TimeStretchBase):
    def __init__(self):
        self.model_weight = f'{Path(__file__).parent}/tsm_net/weights/speech.pt'
        self.model = Stretcher(self.model_weight)
        self.processing_sample_rate = 22050
        super().__init__()
    
    
    def time_stretch(self, input: np.array, sr: int, stretch_factor: float) -> np.array:
        x = torch.from_numpy(input) # Convert the NumPy array to a PyTorch tensor
        
        if len(x.shape) == 1:  # If the audio is mono, add a channel dimension
            x = x.unsqueeze(0)
        
        speed = 1 / stretch_factor
        
        x = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.processing_sample_rate)(x) # Resample the audio
        y = self.model(x, speed).cpu() # Perform time-stretching
        y = torchaudio.transforms.Resample(orig_freq=self.processing_sample_rate, new_freq=sr)(y)

        # perform noise reduction
        y = y.numpy()
        y = y.squeeze(0)
        y = nr.reduce_noise(y=y, sr=sr)
        
        y = align_length(y, int(np.ceil(len(input) * stretch_factor)))
        
        return y

    @property
    def name(self):
        return "TSMNET"
    
def align_length(x: np.array, desired_length: int) -> np.array:
    if len(x) < desired_length:
        x = np.pad(x, (0, desired_length - len(x)))
    elif len(x) > desired_length:
        x = x[:desired_length]
    return x
