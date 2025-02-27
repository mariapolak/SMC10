from pathlib import Path
from pesq import pesq
from pystoi import stoi

import config
import librosa
import glob
import json
import scoreq

import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt

def run_audio_metrics():
    ### Fuzzy Energy
    ### Estimate spectral envelopes
    ### decay rate deviation (preservation of spectral characteristics)
    ### Objective measure from AES
    pass

def run_stoi():
    # TODO for each wav file in the directory, run the predictions and save them to a csv file

    clean, fs = sf.read('path/to/clean/audio')
    denoised, fs = sf.read('path/to/denoised/audio')

    # Clean and den should have the same length, and be 1D
    d = stoi(clean, denoised, fs, extended=False)

def run_scoreq():
    # Predict quality of natural speech in NR mode
    nr_scoreq = scoreq.Scoreq(data_domain='natural', mode='nr')
    # TODO for each wav file in the directory, run the predictions and save them to a csv file
    # TODO use 16 kHz wavs 
    # TODO understand which of the metrics will be useful for the evaluation

    pred_mos = nr_scoreq.predict(test_path='./data/opus.wav', ref_path=None)

    # Predict quality of natural speech in REF mode
    ref_scoreq = scoreq.Scoreq(data_domain='natural', mode='ref')
    pred_distance = ref_scoreq.predict(test_path='./data/opus.wav', ref_path='./data/ref.wav')

    # Predict quality of synthetic speech in NR mode
    nr_scoreq = scoreq.Scoreq(data_domain='synthetic', mode='nr')
    pred_mos = nr_scoreq.predict(test_path='./data/opus.wav', ref_path=None)

    # Predict quality of synthetic speech in REF mode
    ref_scoreq = scoreq.Scoreq(data_domain='synthetic', mode='ref')
    pred_distance = ref_scoreq.predict(test_path='./data/opus.wav', ref_path='./data/ref.wav')



# TODO modify the function so it takes algorithm type and algorithm name as arguments and saves it correctly to csv
def run_pesq(output_file: str, root_dir_deg: str = config.OUTPUT_DIR, root_dir_ref: str = config.INPUT_DIR, extensions: list[str] = ["flac", "wav"]):
    results = []

    for extension in extensions:
        for audio_path in glob.iglob(f"**/*.{extension}", root_dir=root_dir_ref, recursive=True): 
            x, sr = librosa.load(f"{root_dir_ref}/{audio_path}", sr=None)
            x_res = librosa.resample(x, sr, 16000)

            y, sr = librosa.load(f"{root_dir_deg}/{audio_path}", sr=None)
            y_res = librosa.resample(y, sr, 16000)
            
            pesq_score = pesq(16000, x_res, y_res, 'wb')
            results.append({"file": audio_path, "pesq_score": pesq_score, "algorthim": root_dir_deg})

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)

# TODO update once saving is done somewhere else
def prepare_audio_aesthetics_json(output_dir: str, root_dir: str = config.INPUT_DIR, extensions: list[str] = ["flac", "wav"]):
    """
    Prepare the audio aesthetics jsonl file for the audio aesthetics evaluation
    jsonl format:
    {"path":"/path/to/a.wav"}
    {"path":"/path/to/b.wav"}
    {"path":"/path/to/z.wav"}
    """
    (Path(output_dir) / "wav").mkdir(parents=True, exist_ok=True)

    with open(f"{output_dir}/audio_aesthetics.jsonl", "w") as f:
        for extension in extensions:
            for audio_path in glob.iglob(f"**/*.{extension}", root_dir=root_dir, recursive=True): 
                x, sr = librosa.load(f"{root_dir}/{audio_path}", sr=None)
                output_filepath = f"{output_dir}/wav/{Path(audio_path).stem}.wav"
                sf.write(output_filepath, x, sr)
                f.write(f'{{"path":"{output_filepath}"}}\n')


def plot_audio_aesthetics_results(jsonl_path: str):
    # JSONL Description:
    # Output file will contain the same number of rows as input.jsonl. Each row contains 4 axes of prediction with a JSON-formatted dictionary. 
    # Check the following table for more info: Axes name | Full name |---|---| CE | Content Enjoyment CU | Content Usefulness PC | Production Complexity PQ | Production Quality
    # Output line example: 
    # {"CE": 5.146, "CU": 5.779, "PC": 2.148, "PQ": 7.220}

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

# TODO: function to change all the files to wav format and save them in a new directory
# TODO: function to change all the files from 48kHz to 16kHz and save them in a new directory

# TODO: run all on what I already have and then write scripts for plotting all the results

if __name__ == "__main__":
    prepare_audio_aesthetics_json("evaluation/objective/aa") # run evaluation by `audio-aes input.jsonl --batch-size 100 > output.jsonl``


            