from tsm import harmonic_percussive_separation, noise_morphing, phase_vocoder, tsm_net_wrapper, resampling_tsm
from ps import noise_morphing_ps, phase_vocoder_ps, psola, librosa_ps

from datetime import datetime

INPUT_DIR = "data/input"
OUTPUT_DIR = "data/output"
EVAL_OBJ_DIR = "evaluation/objective"
EVAL_SUBJ_DIR = "evaluation/subjective"
EVAL_DATA_DIR = "data/input"

SPEED_ITERATIONS = 10
MEMORY_ITERATIONS = 2

ALGORITHM_FACTORS = {
    "tsm_factors": [0.5, "rt_up", "rt_down", 1.5, 2], # [0.75,0.85,1.35,1.25,1.5,2]
    "ps_factors": [-12, "rt_up", "rt_down", 7, 12] # [-7,-1,1,3,7,12]
}

TIMESTAMP = datetime.now().strftime("%y%m%d%H%M")
TSM_ALGORITHMS = [harmonic_percussive_separation.HPS(), phase_vocoder.PV(), tsm_net_wrapper.TSMNET(), noise_morphing.NoiseMorphing(), resampling_tsm.ResamplingTSM()]
PS_ALGORITHMS = [psola.TDPSOLA(), phase_vocoder_ps.PV(), noise_morphing_ps.NoiseMorphingPS(), librosa_ps.LibrosaPS()]