from ps.pitch_shift_base import PitchShiftBase

import librosa
import numpy as np

from ps.pv_dr.common.enums import WindowType
from ps.pv_dr.dsp.phase import *
from ps.pv_dr.dsp.resample import *
from ps.pv_dr.dsp.transform import *
from ps.pv_dr.dsp.source import AudioSource
from ps.pv_dr.dsp.wrapper import PitchShiftWrapper


class PV(PitchShiftBase):
    """
    Based on Phase Vocoder Done Right pseudo code https://www.researchgate.net/publication/319503719_Phase_Vocoder_Done_Right
    Implementation based on https://github.com/BlackspireAudio/ba21_loma_2_py/
    """
    
    def pitch_shift(self, input: np.array, sr: int, shift_factor_st: float) -> np.array:
        track_info = TrackInfo()
        track_info.sample_rate = sr
        track_info.hop_size_factor = 4
        track_info.normalize = False
        track_info.windowType = WindowType.hann.name
        track_info.half_tone_steps_to_shift = shift_factor_st
        
        audio_source = AudioSource(track_info, input, sr)
        track = audio_source.get_track()
        
        wrapper = PitchShiftWrapper(PitchShifter(track.info,
                                                PhaseLockedDynamicShifter(track.info, magnitude_min_factor=10 ** -6), 
                                                LibrosaResampler(track.info)))
        
        y = wrapper.process(track)
        return y
    
    
    @property
    def name(self):
        return "PVPS"
    