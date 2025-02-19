# SMC10

### Master's Thesis With GN Store Nord A/S

## Initial Problem Statement

The goal of this project is to evaluate different pitch shift and time stretch algorithms as ML data augmentation methods for speech denosing. The assessment of the algorithms will focus on the realism of the transformation and their implementation performance (execution speed and memory usage).

Machine learning models for speech denoising require a pair of recordings: the clean speech (target) and the noisy speech (which includes background noises and reverberation). Often, the noisy speech is constructed using a speech dataset, noise recordings, and impulse responses. Typical speech datasets contain recordings from a set of speakers, with limited variability in terms of pronunciation speed or pitch. This might result in poor model's performance when encountering unusual speech speeds or pitches. To avoid this, it is necessary to introduce pitch-shifting and time-stretching during data construction for each voice used in training.

Ralism of the voice transformation, as well as the processing time and memory usage, are of high importance and should be the main focus of the project. The processing should be fast enough for the purposes of an online training (where the noisy speech is constructed during training).

## Research Goals:

1. Does higher quality/natural sounding PS/TSM have positive impact on the model training?
2. Does it make sense to use SOTA methods for data augmentation?
3. Does it make sense to use a slower method that is more heavy on resources?
4. [Optional] Does variable PS/TSM rate benefit the training?
