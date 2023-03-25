# Example usage:
# python segment_laughter.py --input_audio_file=tst_wave.wav --output_dir=./tst_wave --save_to_textgrid=False --save_to_audio_files=True --min_length=0.2 --threshold=0.5

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

sys.path.append('./utils/')
import torch_utils
import audio_utils

parser = argparse.ArgumentParser()

parser.add_argument('--model_path', type=str,
                    default='checkpoints/in_use/resnet_with_augmentation')
parser.add_argument('--config', type=str, default='resnet_with_augmentation')
parser.add_argument('--output_dir', type=str, default=None)
parser.add_argument('--model_name', type=str, default='')
args = parser.parse_args()

model_path = args.model_path
config = config.MODEL_MAP[args.config]
output_dir = args.output_dir
model_name = args.model_name
# Turn comma-separated parameter strings into list of floats 


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device {device}")

# Load the Model

model = config['model'](
    dropout_rate=0.0, linear_layer_size=config['linear_layer_size'], filter_sizes=config['filter_sizes'])
model.set_device(device)

if os.path.exists(model_path):
    # if device == 'cuda':
    if torch.cuda.is_available():
        print(model_path + '/best.pth.tar')
        torch_utils.load_checkpoint(model_path + '/best.pth.tar', model)
    else:
        # Different method needs to be used when using CPU
        # see https://pytorch.org/tutorials/beginner/saving_loading_models.html for details
        checkpoint = torch.load(
            model_path + '/best.pth.tar', lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
    model.eval()
else:
    raise Exception(f"Model checkpoint not found at {model_path}")


# Load the audio file and features


def load_and_pred(audio_path):
    '''
    Input: audio_path for audio to predict 
    Output: time taken to predict (excluding the generation of output files)
    Loads audio, runs prediction and outputs results according to flag-settings (e.g. TextGrid or Audio)
    '''
    start_time = time.time()  # Start measuring time
    #starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    batch_time_list = []
    inference_generator = load_data.create_time_dataloader(audio_path)
    preprocessing_time = time.time() - start_time  # stop measuring time

    probs = []
    for model_inputs in tqdm(inference_generator):
        # x = torch.from_numpy(model_inputs).float().to(device)
        # Model inputs from new inference generator are tensors already
        model_inputs = model_inputs[:, None, :, :]  # add additional dimension
        x = model_inputs.float().to(device)

        #starter.record()
        starter = time.time()
        preds = model(x).cpu().detach().numpy().squeeze()
        #ender.record()
        #torch.cuda.synchronize()
        ender = time.time()
        #curr_time = starter.elapsed_time(ender)
        curr_time = ender - starter
        batch_time_list.append(curr_time)

        if len(preds.shape) == 0:
            preds = [float(preds)]
        else:
            preds = list(preds)
        probs += preds
    probs = np.array(probs)

    file_length = audio_utils.get_audio_length(audio_path)

    # Removed because it can output probs < 0
    # probs = laugh_segmenter.lowpass(probs)

    # Get a list of instance for each setting passed in  

    return sum(batch_time_list), preprocessing_time, file_length


output_file = model_name + '_' + 'inference_time.csv'
output_time_dir = os.path.join(output_dir, output_file)

tot_list = []
iterate = 10
total_rft = 0
total_preprocessing = 0
total_audio_len = 0
print('la')
for i in range(iterate):
    
    total_inference_time, preprocessing_time, audio_len = load_and_pred('./data/icsi/speech/Bed002/chan1.sph')
    rtf = total_inference_time / (audio_len * 1000)
    total_audio_len = total_audio_len + audio_len
    total_rft = total_rft + rtf
    total_preprocessing = total_preprocessing + preprocessing_time


sub_list = [total_rft/iterate, total_preprocessing/iterate, total_audio_len/iterate, iterate]
tot_list.append(sub_list)


cols = ['rtf', 'preprocessing time', 'audio length', 'iter']
df = pd.DataFrame(tot_list, columns=cols)
df.to_csv(output_time_dir, index=False)
# load_and_pred('./data/icsi/speech/Bed002/chan1.sph','./overfit/test/Bed002')
