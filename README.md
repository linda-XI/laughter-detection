### Towards Efficient Laughter Detection with Convolutional Neural Networks

This repo is based on the laughter detection model by privous students [Lasse Wolter](https://github.com/LasseWolter/laughter-detection-icsi) and retrains it on the
[ICSI Meeting corpus](https://ieeexplore.ieee.org/abstract/document/1198793)

The data pipeline uses [Lhotse](https://github.com/lhotse-speech/lhotse), a new Python library for speech and audio data preparation.

This repository consists of three main parts:
1. Evaluation Pipeline
2. Data Pipeline
3. Training Code

The following list outlines which parts of the repository belong to each of them and classifies the parts/files as one of three types:
1. `from scratch`: entirely written by myself
2. `adapted`: code taken from [Lasse Wolter](https://github.com/LasseWolter/laughter-detection-icsi) and adapted
3. `unmodified`: code taken from [Lasse Wolter](https://github.com/LasseWolter/laughter-detection-icsi) and not adapted or modified

- **Evalation Pipeline** (adapted): 
    - `analysis`
        - `transcript_parsing/parse.py` +`preprocess.py(adapted)`: parsing and preprocessing the ICSI transcripts
        - `analyse.py(adapted)`: main function, that parses and evaluates predictions from .TextGrid files output by the model
    - `visualise.py(adapted)`: functions for visualising model performance (incl. prec-recall curve and confusion matrix)
    - `flops.py(from scratch)`: functions to calculate the FLOPs of models
    - `inference_time.py(from scratch)`: functions to calculate the inference time of models
    - `rftPricision.py(from scratch)`: functions to draw diagram for accuracy and speed metrics

- **Data Pipeline** (adapted) 
    - `compute_features(adapted)`:  computes feature representing the whole corpus and specific subsets of the ICSI corpus
    - `create_data_df.py(adapted)`: creates a dataframe representing training, development and test-set 

- **Training Code**(adapted):
    - `models.py(adapted)` : defines the model architecture
    - `model_utils.py(from scratch)`: defines model architecture
    - `train.py(adapted)` : main training code
    - `segment_laughter.py(adapted)` + `laugh_segmenter.py(unmodified)` : inference code to run laughter detection on audio files
    - `datasets.py(unmodified)` + `load_data.py(adapted)` : the new LAD (Laugh Activity Detection) Dataset + new inference Dataset and code for their creation

- **Misc**:
    - `config.py(adapted)` : configurations for different parts of the pipeline
    - `results.zip` (N/A): contains the model predictions from experiments presented in my thesis

