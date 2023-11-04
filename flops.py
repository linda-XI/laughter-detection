import config
import models
import time
from distutils.util import strtobool
from functools import partial
from torch import optim, nn
import laugh_segmenter
import os
import sys
import pickle
import time
import librosa
import argparse
import torch
import numpy as np
import pandas as pd
import scipy.io.wavfile
from tqdm import tqdm
import tgt
import load_data
from thop import profile, clever_format
sys.path.append('./utils/')
import torch_utils
import audio_utils
from torchstat import stat

parser = argparse.ArgumentParser()

parser.add_argument('--model_path', type=str,
                    default='checkpoints/in_use/resnet_with_augmentation')
parser.add_argument('--config', type=str, default='resnet_with_augmentation')


args = parser.parse_args()


config = config.MODEL_MAP[args.config]


# Turn comma-separated parameter strings into list of floats


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device {device}")

# Load the Model
Total_params = 0
Trainable_params = 0
NonTrainable_params = 0

model = config['model'](
    dropout_rate=0.0, linear_layer_size=config['linear_layer_size'], filter_sizes=config['filter_sizes'])
model.set_device(device)

imputa = torch.randn(32,1,100,44)
print('la')
flops,total_params = profile(model, inputs = (imputa,))
macs, params = clever_format([flops,total_params]) 
#print('FLOPs =' +str(flops/1000**3)+'G')
#print(stat(model,(1,100,44)))
print(macs,params)

