# Pitch-Shifting and Time-Stretching for Speech Data Augmentation

### Master's Thesis With GN Store Nord A/S

This repository contains the code, data, and resources for the master’s thesis project "Pitch-Shifting and Time-Stretching in Speech Data Augmentation" completed at Aalborg University in Spring 2025.

## Research Goals

- How do the most popular algorithms for PS and TSM compare in terms of quality?  
  – What are effective methods for evaluating the quality of PS and TSM?  
  – Do any of the tested objective evaluation metrics correlate with subjective evaluation scores?  
- How do these algorithms compare in terms of performance?  
- Does higher-quality or more natural-sounding PS/TSM positively impact model training?  
    – Is it beneficial to use state-of-the-art methods for data augmentation?  
    – Is it justified to use slower methods that require more computational resources?


## Project Overview

This project investigates the role of pitch-shifting (PS) and time-scale modification (TSM) in speech processing and machine learning. It evaluates a range of classical and modern PS/TSM algorithms in terms of:

- Perceptual audio quality
- Computational performance
- Objective evaluation metrics (speech and PS/TS oriented)
- Impact on deep learning models for speech separation

The central hypothesis: *the choice of algorithm matters—not just for audio quality, but also for downstream ML performance.*

## Key Features

- Implementation and benchmarking of 9 PS/TSM algorithms
- Subjective listening tests using MUSHRA-style evaluation (This is not included in this repository)
- Objective evaluation using metrics
- Integration with [Demucs](https://github.com/facebookresearch/demucs) for speech separation experiments (see details in the [model repo](https://github.com/mp-smc23/demucs-v2-ps-ts/))
- Data augmentation pipelines for speech datasets

## Structure

The repository is organized as follows:

- `evaluation/`: Scripts and results of evaluations
- `ps/`: Implementations of pitch-shifting algorithms
- `tsm/`: Implementation of time-stretching algorithms
- `modules/`: Modules supporting implementations
- `plots/`: notebooks for plotting data and results
- `demucs-v2-ps-ts/`: submodule containing models and training scripts


## Running the project

To run the project install all python dependencies using:
```
pip install -r requirements.txt
```
`batch_ps_tsm.py` applies PS and TSM methods to data found in `data/` directory. It is possible to change the default behavior by modifying `config.py` file.  
`augment_data.py` applies augmentation pipeline to given data.