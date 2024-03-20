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
import json

parser = argparse.ArgumentParser()

parser.add_argument('--model_path', type=str,
                    default='checkpoints/in_use/resnet_with_augmentation')
parser.add_argument('--config', type=str, default='resnet_with_augmentation')
parser.add_argument('--thresholds', type=str, default='0.5',
                    help='Single value or comma-separated list of thresholds to evaluate')
parser.add_argument('--min_lengths', type=str, default='0.2',
                    help='Single value or comma-separated list of min_lengths to evaluate')
# parser.add_argument('--input_audio_file', required=True, type=str)
parser.add_argument('--input_audio_file', type=str)
parser.add_argument('--output_dir', type=str, default=None)
parser.add_argument('--save_to_audio_files', type=str, default='True')
parser.add_argument('--save_to_textgrid', type=str, default='False')
parser.add_argument('--input_dir', type=str, default='./data/icsi/speech')
parser.add_argument('--audio_names', type=str, default='Bmr021')
args = parser.parse_args()

model_path = args.model_path
config = config.MODEL_MAP[args.config]
audio_path = args.input_audio_file
save_to_audio_files = bool(strtobool(args.save_to_audio_files))
save_to_textgrid = bool(strtobool(args.save_to_textgrid))
output_dir = args.output_dir
input_dir = args.input_dir
audio_names = [x for x in args.audio_names.split(',')]

# Turn comma-separated parameter strings into list of floats 
thresholds = [float(t) for t in args.thresholds.split(',')]
min_lengths = [float(l) for l in args.min_lengths.split(',')]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device {device}")

# Load the Model
if args.config.startswith('mobile'):
    model = config['model'](dropout_rate=0.0,
                            linear_layer_size=config['linear_layer_size'], filter_sizes=config['filter_sizes'],
                            inverted_residual_setting=config['inverted_residual_setting'])
else:
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

def fix_over_underflow(prob):
    ''' 
    Fixes probability that is out of the range (0,1) and sets them to
    
    1 or slightly larger than 0 because threshold 0 shouldn't rule them out
    This seems to be a bug in the code taken from Gillick et al.

    '''
    if prob > 1: 
        print('WARN: Fixed probability > 1')
        return 1
    # <= to count also create predictions for threshold=0 when prob is 0
    if prob <= 0: 
        print('WARN: Fixed probability <= 0')
        return 0.0000001
    else: return prob
# Load the audio file and features


def load_and_pred(audio_path, full_output_dir):
    '''
    Input: audio_path for audio to predict 
    Output: time taken to predict (excluding the generation of output files)
    Loads audio, runs prediction and outputs results according to flag-settings (e.g. TextGrid or Audio)
    '''
    start_time = time.time()  # Start measuring time
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    batch_time_list = []
    inference_generator = load_data.create_inference_dataloader(audio_path)
    preprocessing_time = time.time() - start_time  # stop measuring time

    probs = []
    for model_inputs in tqdm(inference_generator):
        ## x = torch.from_numpy(model_inputs).float().to(device)
        # Model inputs from new inference generator are tensors already
        model_inputs = model_inputs[:, None, :, :]  # add additional dimension
        x = model_inputs.float().to(device)

        starter.record()
        preds = model(x).cpu().detach().numpy().squeeze()

        # print(preds.shape)=32

        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        batch_time_list.append(curr_time)

        if len(preds.shape) == 0:
            preds = [float(preds)]
        else:
            preds = list(preds)
        probs += preds
    probs = np.array(probs)

    # get probchan.json

    # components = audio_path.split('/')
    # meet = components[-2]
    # chan = components[-1].split('.')[0]
    # outpath = 'sample/probchan/'
    # if not os.path.isdir(outpath):
    #     os.makedirs(outpath)
    # with open(outpath + 'probs'+meet+chan+'.json', 'w') as filehandle:
            
    #         json.dump(probs.tolist(), filehandle)

    file_length = audio_utils.get_audio_length(audio_path)

    fps = len(probs) / float(file_length)

    # Removed because it can output probs < 0
    # probs = laugh_segmenter.lowpass(probs)

    # Get a list of instance for each setting passed in  
    # compare predict with labels
    instance_dict = laugh_segmenter.get_laughter_instances(
        probs, thresholds=thresholds, min_lengths=min_lengths, fps=fps)

    time_taken = time.time() - start_time  # stop measuring time
    time_avg = sum(batch_time_list) / len(probs)
    # time_avg = 0
    print(f'GPU time for inference per batch: {time_avg:.2f}s')

    for setting, instances in instance_dict.items():
        print(f"Found {len(instances)} laughs for threshold {setting[0]} and min_length {setting[1]}.")
        instance_output_dir = os.path.join(full_output_dir, f't_{setting[0]}', f'l_{setting[1]}')
        save_instances(instances, instance_output_dir, save_to_audio_files, save_to_textgrid, audio_path)

    return sum(batch_time_list), preprocessing_time, file_length
    # return 1, 1, file_length


def save_instances(instances, output_dir, save_to_audio_files, save_to_textgrid, full_audio_path):
    '''
    Saves given instances to disk in a form that is specified by the passed parameters. 
    Possible forms:
        1. as audio file
        2. as textgrid file
    '''
    os.system(f"mkdir -p {output_dir}")
    if len(instances) > 0:
        if save_to_audio_files:
            full_res_y, full_res_sr = librosa.load(full_audio_path, sr=44100)
            wav_paths = []
            maxv = np.iinfo(np.int16).max
            if output_dir is None:
                raise Exception(
                    "Need to specify an output directory to save audio files")
            else:
                for index, instance in enumerate(instances):
                    laughs = laugh_segmenter.cut_laughter_segments(
                        [instance], full_res_y, full_res_sr)
                    wav_path = output_dir + "/laugh_" + str(index) + ".wav"
                    scipy.io.wavfile.write(
                        wav_path, full_res_sr, (laughs * maxv).astype(np.int16))
                    wav_paths.append(wav_path)
                print(laugh_segmenter.format_outputs(instances, wav_paths))

        if save_to_textgrid:
            laughs = [{'start': i[0], 'end': i[1]} for i in instances]
            tg = tgt.TextGrid()
            laughs_tier = tgt.IntervalTier(name='laughter', objects=[
                tgt.Interval(l['start'], l['end'], 'laugh') for l in laughs])
            tg.add_tier(laughs_tier)
            fname = os.path.splitext(os.path.basename(full_audio_path))[0]
            tgt.write_to_file(tg, os.path.join(
                output_dir, fname + '.TextGrid'))

            print('Saved laughter segments in {}'.format(
                os.path.join(output_dir, fname + '.TextGrid')))


def i_pred():
    """
    Interactive Prediction Shell running until interrupted
    """
    print('Model loaded. Waiting for file input...')
    while True:
        audio_path = input()
        if os.path.isfile(audio_path):
            audio_length = audio_utils.get_audio_length(audio_path)
            print(audio_length)
            load_and_pred(audio_path)
        else:
            print("audio_path doesn't exist. Try again...")


def calc_real_time_factor(audio_path, iterations):
    """
    Calculates realtime factor by reading 'audio_path' and running prediction 'iteration' times 
    """
    if os.path.isfile(audio_path):
        audio_length = audio_utils.get_audio_length(audio_path)
        print(f"Audio Length: {audio_length}")
    else:
        raise ValueError(f"Audio_path doesn't exist. Given path {audio_path}")

    sum_time = 0
    for i in range(0, iterations):
        print(f'On iteration {i + 1}')
        sum_time += load_and_pred(audio_path)

    av_time = sum_time / iterations
    # Realtime factor is the 'time taken for prediction' / 'duration of input audio'
    av_real_time_factor = av_time / audio_length
    print(
        f"Average Realtime Factor over {iterations} iterations: {av_real_time_factor:.2f}")



output_time_dir = os.path.join(output_dir, 'inference_time.csv')

tot_list = []
for meet_name in audio_names:
    full_path = os.path.join(input_dir, meet_name)
    full_output_dir = os.path.join(output_dir, meet_name)
    for sph_file in os.listdir(full_path):

        full_sph_file = os.path.join(full_path, sph_file)
        print(full_sph_file)
        total_inference_time, preprocessing_time, audio_len = load_and_pred(full_sph_file, full_output_dir)
        rtf = total_inference_time / (audio_len * 1000)
        sub_list = [meet_name, sph_file, rtf, preprocessing_time, audio_len]
        tot_list.append(sub_list)

cols = ['meeting_id', 'chan', 'rtf', 'preprocessing time', 'audio length']
df = pd.DataFrame(tot_list, columns=cols)
df.to_csv(output_time_dir, index=False)
#load_and_pred('./data/icsi/speech/Bed002/chan1.sph','./overfit/test/Bed002')
