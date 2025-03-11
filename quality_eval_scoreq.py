import scoreq
from pathlib import Path

import config
import librosa
import glob
import json

import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt

import audio_converter as ac

# won't be used
def run_scoreq(output_file: str, algorithm_type: str, algorithm_name: str,
               root_dir_deg: str = ac.WAV_16K_DIR, root_dir_ref: str = config.INPUT_DIR):
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    results = []

    nr_scoreq = scoreq.Scoreq(data_domain='natural', mode='nr') # Predict quality of natural speech in NR mode
    for audio_path in glob.iglob(f"**/*.wav", root_dir=root_dir_ref, recursive=True): 
        deg_path = f"{root_dir_deg}/{audio_path}"
        pred_mos = nr_scoreq.predict(test_path=deg_path, ref_path=None)
        results.append({"file": audio_path, "mode": "no_ref", "type": algorithm_type, "algo": algorithm_name , "pred_mos": pred_mos})

    ref_scoreq = scoreq.Scoreq(data_domain='natural', mode='ref') # Predict quality of natural speech in REF mode
    for audio_path in glob.iglob(f"**/*.wav", root_dir=root_dir_ref, recursive=True):
        ref_path = f"{root_dir_ref}/{audio_path}"
        deg_path = f"{root_dir_deg}/{audio_path}"
        pred_distance = ref_scoreq.predict(test_path=deg_path, ref_path=ref_path)
        results.append({"file": audio_path, "mode": "ref", "type": algorithm_type, "algo": algorithm_name , "pred_distance": pred_distance})
        
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
