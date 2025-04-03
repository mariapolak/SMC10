from pathlib import Path
from pesq import pesq
from pystoi import stoi
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio

import config
import librosa
import glob
import json
import torch
import subprocess

import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt

import audio_converter as ac

def run_stoi(output_file: str, algorithm_type: str, algorithm_name: str, factor: int,
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
        results.append({"file": audio_path, "mode": "ref", "type": algorithm_type, "algo": algorithm_name , "factor": factor, "pred_distance": d})
    
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)

def run_pesq(output_file: str, algorithm_type: str, algorithm_name: str, factor: int,
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
        results.append({"file": audio_path, "type": algorithm_type, "algo": algorithm_name , "factor": factor, "pesq_score": pesq_score})

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

def run_audio_aesthetics(input_file:str, output_file: str, root_dir: str = ac.WAV_48K_DIR):
    """ Run the Audio Aesthetics evaluation
    48k, no referrence needed
    
    Args:
        output_file (str): Name of the output jsonl file
        root_dir (str, optional): Root directory with transformed signals. Defaults to ac.WAV_48K_DIR.
    """
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    prepare_audio_aesthetics_json(input_file, root_dir)
    
    # audio-aes evaluation/objective/audio_aesthetics/audio_aesthetics_input.jsonl --batch-size 100 > evaluation/objective/audio_aesthetics/output.jsonl 
    with open(output_file, "w") as out_f:
        subprocess.run(["audio-aes", input_file, "--batch-size", "70"], stdout=out_f)
        
    # Read JSON lines files
    df_1 = pd.read_json(output_file, lines=True)
    df_2 = pd.read_json(input_file, lines=True)

    # Concatenate the dataframes
    df = pd.concat([df_2, df_1], axis=1)

    # Save the combined dataframe to a new JSON lines file
    df.to_json(output_file, orient='records', lines=True)
    
    Path(input_file).unlink() # delete the input file

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

def run_nisqa(input_file:str, output_file_tts: str, output_file: str, root_dir: str = ac.WAV_48K_DIR):
    """ Run the NISQA evaluation
    48k, no referrence needed
    
    Args:
        input_file (str): Name of the input csv file
        output_file (str): Name of the output csv file
        root_dir (str, optional): Root directory with transformed signals. Defaults to ac.WAV_48K_DIR.
    """
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    Path(output_file_tts).parent.mkdir(parents=True, exist_ok=True)
    
    prepare_nisqa_csv(input_file, root_dir)

    # python evaluation/NISQA/run_predict.py --mode predict_csv --pretrained_model evaluation/NISQA/weights/nisqa_tts.tar --csv_file evaluation/objective/nisqa/nisqa_input.csv --csv_deg path --num_workers 0 --bs 10 --output_dir evaluation/objective/nisqa/
    # TTS
    subprocess.run(["python", "evaluation/NISQA/run_predict.py", "--mode", "predict_csv", "--pretrained_model", "evaluation/NISQA/weights/nisqa_tts.tar", 
                    "--csv_file", input_file, "--csv_deg", "path", "--num_workers", "0", "--bs", "10", "--output_dir", "evaluation/objective/nisqa/"], stdout = subprocess.DEVNULL)
    
    # rename results.csv to output_file_tts.csv
    Path("evaluation/objective/nisqa/NISQA_results.csv").rename(output_file_tts)
    
    # python evaluation/NISQA/run_predict.py --mode predict_csv --pretrained_model evaluation/NISQA/weights/nisqa.tar --csv_file evaluation/objective/nisqa/nisqa_input.csv --csv_deg path --num_workers 0 --bs 10 --output_dir evaluation/objective/nisqa/
    # REF DEG
    subprocess.run(["python", "evaluation/NISQA/run_predict.py", "--mode", "predict_csv", "--pretrained_model", "evaluation/NISQA/weights/nisqa.tar",
                    "--csv_file", input_file, "--csv_deg", "path", "--num_workers", "0", "--bs", "10", "--output_dir", "evaluation/objective/nisqa/"], stdout = subprocess.DEVNULL)
    Path("evaluation/objective/nisqa/NISQA_results.csv").rename(output_file)
        
    Path(input_file).unlink() # delete the input file 
        
# VISQOL
def prepare_visqol_csv(output_file: str, root_dir_deg: str = ac.WAV_16K_DIR, root_dir_ref: str = ac.WAV_16K_DIR):
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

def run_visqol(input_file:str, output_file: str, root_dir_deg: str = ac.WAV_16K_DIR, root_dir_ref: str = ac.WAV_16K_DIR):
    """ Run the VISQOL evaluation
    16k, reference needed, reference of the same length as the degraded signal
    
    Args:
        output_file (str): Name of the output jsonl file
        root_dir (str, optional): Root directory with transformed signals. Defaults to ac.WAV_48K_DIR.
    """
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    prepare_visqol_csv(input_file, root_dir_deg, root_dir_ref)
    # docker run -it -v ./data:/data -v ./evaluation:/evaluation mubtasimahasan/visqol:v3 \                                                              (smc10) 
                                        #   --batch_input_csv /evaluation/objective/visqol/visqol_input.csv \
                                        #   --results_csv /evaluation/objective/visqol/visqol_results.csv \
                                        #   --use_speech_mode
    with open(output_file, "w") as out_f:
        subprocess.run(["docker", "run", "--gpus=all", "-v", "./data:/data", "-v", "./evaluation:/evaluation", "mubtasimahasan/visqol:v3",
                        "--batch_input_csv", f"/{input_file}", "--results_csv", f"/{output_file}", "--use_speech_mode"], stdout = subprocess.DEVNULL)
        
    Path(input_file).unlink() # delete the input file

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

def run_sisnr(output_file: str, algorithm_type: str, algorithm_name: str, factor: int,
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
        results.append({"file": audio_path, "type": algorithm_type, "algo": algorithm_name, "factor": factor, "sisnr_score": float(score)})

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)

def run_all():
    REF_DIR_16k = "data/input/wav16"
    REF_DIR_48k = "data/input/wav48"
    
    DEG_PARENT_DIR_16k = "data/output/wav16"
    DEG_PARENT_DIR_48k = "data/output/wav48"
    
    print(" ===================== Running TSM Evaluation ===================== ")
    
    for tsm_algorithm in config.TSM_ALGORITHMS:
        ## audio_aesthetics and nisqa no ref
        for tsm_factor in config.NO_REFERENCE_FACTORS["tsm_factors"]:
            print(f"Running Evaluation of TSM: {tsm_algorithm.name} with factor: {tsm_factor}")
            DEG_TSM_DIR_16k = f"{DEG_PARENT_DIR_16k}/tsm/{tsm_algorithm.name}/{tsm_factor}"
            DEG_TSM_DIR_48k = f"{DEG_PARENT_DIR_48k}/tsm/{tsm_algorithm.name}/{tsm_factor}"
            
            audio_aeaesthetics_input_file = f"evaluation/objective/audio_aesthetics/audio_aesthetics_{tsm_algorithm.name}_{tsm_factor}_input.jsonl"
            audio_aeaesthetics_output_file = f"evaluation/objective/audio_aesthetics/audio_aesthetics_{tsm_algorithm.name}_{tsm_factor}.jsonl"
                    
            nisqa_input_file = f"evaluation/objective/nisqa/nisqa_input_{tsm_algorithm.name}_{tsm_factor}.csv"
            nisqa_output_file_tts = f"evaluation/objective/nisqa/tts/nisqa_{tsm_algorithm.name}_{tsm_factor}.csv"
            nisqa_output_file = f"evaluation/objective/nisqa/std/nisqa_{tsm_algorithm.name}_{tsm_factor}.csv"
            
            print("Preparing Audio Aesthetics JSON")
            run_audio_aesthetics(audio_aeaesthetics_input_file, audio_aeaesthetics_output_file, DEG_TSM_DIR_48k)
            
            print("Preparing NISQA CSV")
            run_nisqa(nisqa_input_file, nisqa_output_file_tts, nisqa_output_file, DEG_TSM_DIR_48k)
            
        ## sisnr, pesq, stoi, visqol with ref
        for tsm_factor in config.REFERENCE_FACTORS["tsm_factors"]:
            print(f"Running Evaluation of TSM: {tsm_algorithm.name} with factor: {tsm_factor}")
            
            DEG_TSM_DIR_16k = f"{DEG_PARENT_DIR_16k}/tsm/{tsm_algorithm.name}/{tsm_factor}"
            DEG_TSM_DIR_48k = f"{DEG_PARENT_DIR_48k}/tsm/{tsm_algorithm.name}/{tsm_factor}"
            
            sinsr_output_file = f"evaluation/objective/sisnr/sisnr_{tsm_algorithm.name}_{tsm_factor}.csv"
            pesq_output_file = f"evaluation/objective/pesq/pesq_{tsm_algorithm.name}_{tsm_factor}.csv"
            stoi_output_file = f"evaluation/objective/stoi/stoi_{tsm_algorithm.name}_{tsm_factor}.csv"
            
            visqol_input_file = f"evaluation/objective/visqol/visqol_input_{tsm_algorithm.name}_{tsm_factor}.csv"
            visqol_output_file = f"evaluation/objective/visqol/visqol_{tsm_algorithm.name}_{tsm_factor}.csv"

            print("Preparing VISQOL CSV")
            run_visqol(visqol_input_file, visqol_output_file, DEG_TSM_DIR_16k, REF_DIR_16k)
            print("SISNR")
            run_sisnr(sinsr_output_file, "tsm", tsm_algorithm.name, tsm_factor, DEG_TSM_DIR_48k, REF_DIR_48k)
            print("PESQ")
            run_pesq(pesq_output_file, "tsm", tsm_algorithm.name, tsm_factor, DEG_TSM_DIR_16k, REF_DIR_16k)
            print("STOI")
            run_stoi(stoi_output_file, "tsm", tsm_algorithm.name, tsm_factor, DEG_TSM_DIR_48k, REF_DIR_48k)
    
    print(" ===================== Running PS Evaluation ===================== ")
          
    for ps_algorithm in config.PS_ALGORITHMS:
        ## audio_aesthetics and nisqa no ref
        for ps_factor in config.NO_REFERENCE_FACTORS["ps_factors"]:
            print(f"Running Evaluation of PS: {ps_algorithm.name} with factor: {ps_factor}")
            
            DEG_PS_DIR_16k = f"{DEG_PARENT_DIR_16k}/ps/{ps_algorithm.name}/{ps_factor}"
            DEG_PS_DIR_48k = f"{DEG_PARENT_DIR_48k}/ps/{ps_algorithm.name}/{ps_factor}"
                     
            audio_aeaesthetics_input_file = f"evaluation/objective/audio_aesthetics/audio_aesthetics_{ps_algorithm.name}_{ps_factor}_input.jsonl"
            audio_aeaesthetics_output_file = f"evaluation/objective/audio_aesthetics/audio_aesthetics_{ps_algorithm.name}_{ps_factor}.jsonl"
            
            nisqa_input_file = f"evaluation/objective/nisqa/nisqa_input_{ps_algorithm.name}_{ps_factor}.csv"
            nisqa_output_file_tts = f"evaluation/objective/nisqa/tts/nisqa_{ps_algorithm.name}_{ps_factor}.csv"
            nisqa_output_file = f"evaluation/objective/nisqa/std/nisqa_{ps_algorithm.name}_{ps_factor}.csv"
                  
            print("Preparing Audio Aesthetics JSON")
            run_audio_aesthetics(audio_aeaesthetics_input_file, audio_aeaesthetics_output_file, DEG_PS_DIR_48k)
            
            print("Preparing NISQA CSV")
            run_nisqa(nisqa_input_file, nisqa_output_file_tts, nisqa_output_file, DEG_PS_DIR_48k)
                 
        ## sisnr, pesq, stoi, visqol with ref  
        for ps_factor in config.REFERENCE_FACTORS["ps_factors"]:
            print(f"Running Evaluation of PS: {ps_algorithm.name} with factor: {ps_factor}")
            
            DEG_PS_DIR_16k = f"{DEG_PARENT_DIR_16k}/ps/{ps_algorithm.name}/{ps_factor}"
            DEG_PS_DIR_48k = f"{DEG_PARENT_DIR_48k}/ps/{ps_algorithm.name}/{ps_factor}"
            
            sinsr_output_file = f"evaluation/objective/sisnr/sisnr_{ps_algorithm.name}_{ps_factor}.csv"
            pesq_output_file = f"evaluation/objective/pesq/pesq_{ps_algorithm.name}_{ps_factor}.csv"
            stoi_output_file = f"evaluation/objective/stoi/stoi_{ps_algorithm.name}_{ps_factor}.csv"

            visqol_input_file = f"evaluation/objective/visqol/visqol_input_{ps_algorithm.name}_{ps_factor}.csv"
            visqol_output_file = f"evaluation/objective/visqol/visqol_{ps_algorithm.name}_{ps_factor}.csv"

            print("Preparing VISQOL CSV")
            run_visqol(visqol_input_file, visqol_output_file, DEG_PS_DIR_16k, REF_DIR_16k)
            print("SISNR")
            run_sisnr(sinsr_output_file, "ps", ps_algorithm.name, ps_factor, DEG_PS_DIR_48k, REF_DIR_48k)
            print("PESQ")
            run_pesq(pesq_output_file, "ps", ps_algorithm.name, ps_factor, DEG_PS_DIR_16k, REF_DIR_16k)
            print("STOI")
            run_stoi(stoi_output_file, "ps", ps_algorithm.name, ps_factor, DEG_PS_DIR_48k, REF_DIR_48k)
            

if __name__ == "__main__":
    run_all()

            