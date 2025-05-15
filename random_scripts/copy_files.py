import glob 
import os
from tqdm import tqdm
from pathlib import Path
import shutil

root_dir = "/Volumes/Hub/Conversations/data/output/wav48/"
for audio_path in tqdm(glob.glob(f"**/*/p[0-9][0-9][0-9]_020.wav", root_dir=root_dir, recursive=True)): 
    input_file = f"{root_dir}/{audio_path}"
    output_dir = "/Users/mariapolak/Documents/AAU/Semester_4/SMC10/data/output"
    output_file = os.path.join(output_dir, audio_path)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # copy file
    shutil.copy2(input_file, output_file)