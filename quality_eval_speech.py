from pathlib import Path
from pesq import pesq
from pystoi import stoi
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio

import config
import librosa
import glob
import json
import torch

import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt

import audio_converter as ac

def run_stoi(output_file: str, algorithm_type: str, algorithm_name: str,
             root_dir_deg: str = ac.WAV_48K_DIR, root_dir_ref: str = config.INPUT_DIR):
    """ STOI - Short-Time Objective Intelligibility measure. 
    48k, reference needed, reference of the same length as the degraded signal

    Args:
        output_file (str): Name of the output file
        algorithm_type (str): PS/TSM
        algorithm_name (str): Name of PS/TSM algorithm
        root_dir_deg (str, optional): Root directory with transformed signals. Defaults to ac.WAV_48K_DIR.
        root_dir_ref (str, optional): Root directory of reference signals. Defaults to config.INPUT_DIR.
    """
    
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    results = []
    fs = 48000
    
    for audio_path in glob.iglob(f"**/*.wav", root_dir=root_dir_ref, recursive=True):
        ref, _ = librosa.load(f"{root_dir_ref}/{audio_path}", sr=None)
        deg, _ = librosa.load(f"{root_dir_deg}/{audio_path}", sr=None)
        
        d = stoi(ref, deg, fs, extended=False) # d = stoi(clean, denoised, fs, extended=False) Clean and den should have the same length, and be 1D
        results.append({"file": audio_path, "mode": "ref", "type": algorithm_type, "algo": algorithm_name , "pred_distance": d})
    
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)

def run_pesq(output_file: str, algorithm_type: str, algorithm_name: str,
             root_dir_deg: str = ac.WAV_16K_DIR, root_dir_ref: str = config.INPUT_DIR):
    """ PESQ - Perceptual Evaluation of Speech Quality
    16k, reference needed, reference of the same length as the degraded signal

    Args:
        output_file (str): Name of the output file
        algorithm_type (str): PS/TSM
        algorithm_name (str): Name of PS/TSM algorithm
        root_dir_deg (str, optional): Root directory with transformed signals. Defaults to ac.WAV_16K_DIR.
        root_dir_ref (str, optional): Root directory of reference signals. Defaults to config.INPUT_DIR.
    """
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    results = []

    for audio_path in glob.iglob(f"**/*.wav", root_dir=root_dir_ref, recursive=True): 
        ref, _ = librosa.load(f"{root_dir_ref}/{audio_path}", sr=None)
        deg, _ = librosa.load(f"{root_dir_deg}/{audio_path}", sr=None)
        
        pesq_score = pesq(16000, ref, deg, 'wb')
        results.append({"file": audio_path, "type": algorithm_type, "algo": algorithm_name , "pesq_score": pesq_score})

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)

def prepare_audio_aesthetics_json(output_file: str, root_dir: str = ac.WAV_48K_DIR):
    """ Prepare the Audio Aesthetics JSON file for the evaluation
    48k, no referrence needed

    Args:
        output_file (str): Name of the output jsonl file
        root_dir (str, optional): Root directory with transformed signals. Defaults to ac.WAV_48K_DIR.
    """
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        for audio_path in glob.iglob(f"**/*.wav", root_dir=root_dir, recursive=True): 
            f.write(f'{{"path":"{root_dir}/{audio_path}"}}\n')


def prepare_nisqa_csv(output_file: str, root_dir: str = ac.WAV_48K_DIR):
    """ Prepare the NISQA csv file for the evaluation
    48k, no referrence needed

    Args:
        output_file (str): Name of the output csv file
        root_dir (str, optional): Root directory with transformed signals. Defaults to ac.WAV_48K_DIR.
    """
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        f.write("path\n")
        for audio_path in glob.iglob(f"**/*.wav", root_dir=root_dir, recursive=True): 
            f.write(f"{root_dir}/{audio_path}\n")
         
# VISQOL
def prepare_visqol_csv(output_file: str, root_dir_deg: str = ac.WAV_16K_DIR, root_dir_ref: str = f"{config.INPUT_DIR}/wav16"):
    """ Prepare the VISQOL csv file for the evaluation
    16k, reference needed, reference of the same length as the degraded signal

    Args:
        output_file (str): Name of the output csv file
        root_dir_deg (str, optional): Root directory with transformed signals. Defaults to ac.WAV_16K_DIR.
        root_dir_ref (str, optional): Root directory of reference signals. Defaults to config.INPUT_DIR/wav16.
    """
    
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        f.write("path\n")
        for audio_path in glob.iglob(f"**/*.wav", root_dir=root_dir_ref, recursive=True): 
            f.write(f"/{root_dir_ref}/{audio_path},/{root_dir_deg}/{audio_path}\n")

def plot_audio_aesthetics_results(jsonl_path: str):
    # plot average value for each of the metrics
    ce_values = []
    cu_values = []
    pc_values = []
    pq_values = []

    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            ce_values.append(data["CE"])
            cu_values.append(data["CU"])
            pc_values.append(data["PC"])
            pq_values.append(data["PQ"])

    metrics = {
        "Content Enjoyment (CE)": ce_values,
        "Content Usefulness (CU)": cu_values,
        "Production Complexity (PC)": pc_values,
        "Production Quality (PQ)": pq_values
    }

    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}

    plt.figure(figsize=(10, 5))
    plt.bar(avg_metrics.keys(), avg_metrics.values())
    plt.xlabel('Metrics')
    plt.ylabel('Average Value')
    plt.title('Average Audio Aesthetics Metrics')
    plt.show()

def run_sisnr(output_file: str, algorithm_type: str, algorithm_name: str,
             root_dir_deg: str = ac.WAV_48K_DIR, root_dir_ref: str = config.INPUT_DIR):
    """ Scale Invariant Signal Noise Ratio
    48k, reference needed, reference of the same length as the degraded signal

    Args:
        output_file (str): Name of the output file
        algorithm_type (str): PS/TSM
        algorithm_name (str): Name of PS/TSM algorithm
        root_dir_deg (str, optional): Root directory of transformend signals. Defaults to ac.WAV_48K_DIR.
        root_dir_ref (str, optional): Root directory of reference signals. Defaults to config.INPUT_DIR.
    """
    
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    results = []
    si_snr = ScaleInvariantSignalNoiseRatio()
    for audio_path in glob.iglob(f"**/*.wav", root_dir=root_dir_ref, recursive=True): 
        ref, _ = librosa.load(f"{root_dir_ref}/{audio_path}", sr=None)
        deg, _ = librosa.load(f"{root_dir_deg}/{audio_path}", sr=None)
        
        score = si_snr(torch.tensor(ref), torch.tensor(deg)) 
        results.append({"file": audio_path, "type": algorithm_type, "algo": algorithm_name , "sisnr_score": float(score)})

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    
# TODO: run all on what I already have and then write scripts for plotting all the results

if __name__ == "__main__":
    REF_DIR_16k = "data/evaluation_short/input/16k"
    REF_DIR_48k = "data/evaluation_short/input/48k"
    
    DEG_TSM_DIR_16k = "data/evaluation_short/output_tsm/16k"
    DEG_TSM_DIR_48k = "data/evaluation_short/output_tsm/48k"
    
    DEG_PS_DIR_16k = "data/evaluation_short/output_ps/16k"
    DEG_PS_DIR_48k = "data/evaluation_short/output_ps/48k"
    
    sinsr_output_file = "evaluation/objective/sisnr/sisnr.csv"
    pesq_output_file = "evaluation/objective/pesq/pesq.csv"
    stoi_output_file = "evaluation/objective/stoi/stoi.csv"
    
    audio_aeaesthetics_input_file = "evaluation/objective/audio_aesthetics/audio_aesthetics_input.jsonl"
    visqol_input_file = "evaluation/objective/visqol/visqol_input.csv"
    nisqa_input_file = "evaluation/objective/nisqa/nisqa_input.csv"

    ### PS 
    print(" ===================== Running PS Evaluation ===================== ")
    print("Preparing Audio Aesthetics JSON")
    prepare_audio_aesthetics_json(audio_aeaesthetics_input_file, DEG_PS_DIR_48k)
    print("Preparing VISQOL CSV")
    prepare_visqol_csv(visqol_input_file, DEG_PS_DIR_16k, REF_DIR_16k)
    print("Preparing NISQA CSV")
    prepare_nisqa_csv(nisqa_input_file, DEG_PS_DIR_48k)
    print("SISNR")
    run_sisnr(sinsr_output_file, "ps", "PSOLA", DEG_PS_DIR_48k, REF_DIR_48k)
    print("PESQ")
    run_pesq(pesq_output_file, "ps", "PSOLA", DEG_PS_DIR_16k, REF_DIR_16k)
    print("STOI")
    run_stoi(stoi_output_file, "ps", "PSOLA", DEG_PS_DIR_48k, REF_DIR_48k)
    
    ### TSM
    print(" ===================== Running TSM Evaluation ===================== ")
    print("Preparing Audio Aesthetics JSON")
    prepare_audio_aesthetics_json(audio_aeaesthetics_input_file, DEG_TSM_DIR_48k)
    print("Preparing NISQA CSV")
    prepare_nisqa_csv(nisqa_input_file, DEG_TSM_DIR_48k)

            