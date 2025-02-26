from tsm.time_stretch_base import TimeStretchBase
from tsm import harmonic_percussive_separation, noise_morphing, phase_vocoder

from ps.pitch_shift_base import PitchShiftBase
from ps import noise_morphing_ps

from datetime import datetime

INPUT_DIR = "data/input"
OUTPUT_DIR = "data/output"
OUTPUT_EVAL_OBJ_DIR = "evaluation/objective"
OUTPUT_EVAL_SUBJ_DIR = "evaluation/subjective"

SPEED_ITERATIONS = 10
MEMORY_ITERATIONS = 2

ALGORITHM_FACTORS = {
    "tsm_factors": [0.5, 2.],
    "ps_factors": [-12, 12],
}

TIMESTAMP = datetime.now().strftime("%y%m%d%H%M")
TSM_ALGORITHMS = [harmonic_percussive_separation.HPS(), phase_vocoder.PV()]
PS_ALGORITHMS = [noise_morphing_ps.NoiseMorphingPS()]