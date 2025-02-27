from pathlib import Path
from pesq import pesq

import config
import librosa
import glob
import json

import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt

def run_speech_metrics():
    ### PESQ
    ### STOI
    ### NISQA # python run_predict.py --mode predict_dir --pretrained_model weights/nisqa.tar --data_dir /path/to/folder/with/wavs --num_workers 0 --bs 10 --output_dir /path/to/dir/with/results
    ### SCOREQ
    pass

def run_audio_metrics():
    ### Fuzzy Energy
    ### Estimate spectral envelopes
    ### decay rate deviation (preservation of spectral characteristics)
    ### Objective measure from AES
    pass


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

def prepare_audio_aesthetics_json(output_dir: str, root_dir: str = config.INPUT_DIR, extensions: list[str] = ["flac", "wav"]):
    """
    Prepare the audio aesthetics jsonl file for the audio aesthetics evaluation
    jsonl format:
    {"path":"/path/to/a.wav"}
    {"path":"/path/to/b.wav"}
    {"path":"/path/to/z.wav"}
    """
    output_dir = f"{output_dir}_{config.TIMESTAMP}"
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

if __name__ == "__main__":
    prepare_audio_aesthetics_json("evaluation/objective/aa") # run evaluation by `audio-aes input.jsonl --batch-size 100 > output.jsonl``


            