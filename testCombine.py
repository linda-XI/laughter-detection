import os
import pandas as pd
from typing import List, Optional, Tuple
from pydantic import BaseModel
from strenum import StrEnum
from lxml import etree
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import ANALYSIS as cfg

# laugh_only_df = pd.DataFrame(tot_laugh_only_segs)
# invalid_df = pd.DataFrame(tot_invalid_segs )
# speech_df = pd.DataFrame(tot_speech_segs)
# noise_df = pd.DataFrame(tot_noise_segs)
laugh_only_df = pd.read_csv('./sample/laugh_only_df.csv')


def textgrid_to_list(full_path, params):
    # There are more recorded channels than participants
    # thus, not all channels are mapped to a participant
    # We focus on those that are mapped to a participant
    
    if params['chan_id'] not in parse.chan_to_part[params['meeting_id']].keys():
        return []

    # If file is empty -> no predictions
    if os.stat(full_path).st_size == 0:
        print(f"WARNING: Found an empty .TextGrid file. This usually shouldn't happen.  \
        Did something go wrong with evaluation for {params['meeting_id']: params['chan_id']}")
        return []

    interval_list = []
    part_id = parse.chan_to_part[params['meeting_id']][params['chan_id']]
    
    print(full_path)
    grid = textgrids.TextGrid(full_path)
    for interval in grid['laughter']:
        
        # TODO: Change for breath laugh?!
        if str(interval.text) == 'laugh':
            seg_length = interval.xmax - interval.xmin
            interval_list.append([params['meeting_id'], part_id, params['chan_id'], interval.xmin,
                                  interval.xmax, seg_length, str(interval.text), str(interval.text)])
    return interval_list

def get_params_from_path(path):
    '''
    Input: path
    Output: dict of parameters
    '''
    params = {}
    path = os.path.normpath(path)
    # First cut of .TextGrid
    # then split for to get parameters which are given by dir-names
    params_list = path.replace('.TextGrid', '').split('/')
    # Adjustment because current files are stored as chanN_laughter.TextGrid
    # TODO: change back to old naming convention
    chan_id = params_list[-1].split('_')[0]
    # Check if filename follows convention -> 'chanN.TextGrid'
    if not chan_id.startswith('chan'):
        raise NameError(
            "Did you follow the naming convention for channel .TextGrid-files -> 'chanN.TextGrid'")

    params['chan_id'] = chan_id
    params['min_len'] = params_list[-2]

    # Strip the 't_' prefix and turn threshold into float
    thr = params_list[-3].replace('t_', '')
    params['threshold'] = float(thr)
    meeting_id = params_list[-4]
    # Check if meeting ID is valid -> B**NNN
    if not len(meeting_id) == 6:  # All IDs are 6 chars long
        raise NameError(
            "Did you follow the required directory structure? all chanN.TextGrid files \
            need to be in a directory with its meeting ID as name -> e.g. B**NNN")

    params['meeting_id'] = meeting_id
    return params


def textgrid_to_df(file_path):
    '''
    convert textgrid of a meeting into dataframe
    '''
    tot_list = []
    for filename in os.listdir(file_path):
        if filename.endswith('.TextGrid'):
            full_path = os.path.join(file_path, filename)
            
            params = get_params_from_path(full_path)
            tot_list += textgrid_to_list(full_path,
                                         params)

    cols = ['meeting_id', 'part_id', 'chan', 'start',
            'end', 'length', 'type', 'laugh_type']
    df = pd.DataFrame(tot_list, columns=cols)
    return df

def update_laugh_only_df(path, out_path, use_cache=False):
    """
    Creates a dataframe summarising evaluation metrics per meeting for each parameter-set
    """
    # if use_cache and os.path.isfile(f'{os.path.dirname(__file__)}/.cache/eval_df.csv'):
    #     print("-----------------------------------------")
    #     print("NO NEW EVALUATION - USING CACHED VERSION")
    #     print("-----------------------------------------")
    #     eval_df = pd.read_csv(out_path)
    # else:
    # all_evals = []
    print('Calculating metrics for every meeting for every parameter-set:')
    for meeting in os.listdir(path):
        #print(f'Evaluating meeting {meeting}...')
        meeting_path = os.path.join(path, meeting)
        # meeting_id = meeting_path.split("/")[-1]
        threshold_dir = os.path.join(meeting_path, 't_0.8')
        #textgrid_dir: dir that contains all channel textgrid of a meeting
        textgrid_dir = os.path.join(threshold_dir, 't_0.2')
        for filename in os.listdir(textgrid_dir):
            if filename.endswith('.TextGrid'):
                full_path = os.path.join(textgrid_dir, filename)
                #one channel one df
                textgrid_df = textgrid_to_df(full_path)

                i, j = 0, 0  
                while i < len(laugh_only_df) and j < len(textgrid_df):
                    # 获取当前行的 'start' 值
                    start_total = laugh_only_df.iloc[i]['start']
                    start_new = textgrid_df.iloc[j]['start']

                    # 计算 'start' 之差
                    diff = abs(start_new - start_total)

                    # 如果 'start_new' 与 'start_total' 之差小于 0.2，且不是同一个人发出的。则添加新的行到总的 DataFrame 中
                    if ((diff < 0.2) 
                        and ((textgrid_df.iloc[j]['meeting_id'] != laugh_only_df.iloc[j]['meeting_id']) 
                        or (textgrid_df.iloc[j]['chan'] != laugh_only_df.iloc[j]['chan']))):
                            
                        new_row = pd.DataFrame({
                            'meeting_id': textgrid_df.iloc[j]['meeting_id'],
                            'part_id': textgrid_df.iloc[j]['part_id'],
                            'chan': textgrid_df.iloc[j]['chan'],
                            'start': laugh_only_df.iloc[i]['start'],
                            'end': laugh_only_df.iloc[j]['end'],
                            'length': laugh_only_df.iloc[j]['length'],
                            'type': laugh_only_df.iloc[i]['type'],
                            'laugh_type': laugh_only_df.iloc[i]['laugh_type']
                        }, index=[0])
                        laugh_only_df = pd.concat([laugh_only_df, new_row], ignore_index=True)
                        j += 1
                    elif start_new < start_total:
                        j += 1
                    else:
                        i += 1
            
                    # thr_val = threshold.replace('t_', '')
                    # min_len_val = min_length.replace('l_', '')
                    # out = eval_preds(pred_laughs, meeting_id, thr_val, min_len_val)
                    # all_evals.append(out)
                    # Log progress
#---tot_fp_speech_time: actual speech, predict laugh---#
        # cols = ['meeting', 'threshold', 'min_len', 'precision', 'recall',
        #         'corr_pred_time', 'tot_pred_time', 'tot_transc_laugh_time', 'num_of_pred_laughs', 'valid_pred_laughs', 'num_of_transc_laughs',
        #         'tot_fp_speech_time', 'tot_fp_noise_time', 'tot_fp_silence_time']
        #cols = ['meeting', 'threshold', 'min_len', 'precision', 'recall',
         #       'corr_pred_time', 'tot_pred_time', 'tot_transc_laugh_time', 'num_of_pred_laughs', 'valid_pred_laughs', 'num_of_transc_laughs',
          #      'tot_fp_speech_time', 'tot_fp_noise_time', 'tot_fp_silence_time','tot_fp_false_silence_time']
        # if len(cols) != len(all_evals[0]):
            # raise Exception(
                # f'List returned by eval_preds() has wrong length. Expected length: {len(cols)}. Found: {len(all_evals[0])}.')
        # eval_df = pd.DataFrame(all_evals, columns=cols)
        # if not os.path.isdir(f'{os.path.dirname(__file__)}/.cache'):
            # subprocess.run(['mkdir', '.cache'])
    laugh_only_df.to_csv(out_path, index=False)

    return laugh_only_df




if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Usage: python analyse.py <preds_dir>')
        exit(1) 
    #path is ./sample/textgrid
    #outpath is ./sample/testOutput/new_laugh_only.csv
    path = sys.argv[1]
    out_path = sys.argv[2]
    update_laugh_only_df(path, out_path)