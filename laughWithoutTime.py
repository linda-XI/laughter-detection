# precision recall for laugh event
#TP: predlaugh overlap with actual laugh
#FP: predlaugh in actual non-laugh
#FN: no predlaugh overlap with this actual laugh
#TN: no predlaugh in this actual non-laugh

from matplotlib.ticker import MaxNLocator
import textgrids
import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import subprocess
import numpy as np
import portion as P
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import ANALYSIS as cfg
from analysis.transcript_parsing import parse
import analysis.preprocess as prep
import analysis.utils as utils

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
                                  interval.xmax, seg_length, params['threshold'], params['min_len'], str(interval.text)])
    return interval_list


def textgrid_to_df(file_path):
    tot_list = []
    for filename in os.listdir(file_path):
        if filename.endswith('.TextGrid'):
            full_path = os.path.join(file_path, filename)
            
            params = get_params_from_path(full_path)
            tot_list += textgrid_to_list(full_path,
                                         params)

    cols = ['meeting_id', 'part_id', 'chan', 'start',
            'end', 'length', 'threshold', 'min_len', 'laugh_type']
    df = pd.DataFrame(tot_list, columns=cols)
    return df


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
####################################################################################################

def eval_preds(pred_per_meeting_df, meeting_id, threshold, min_len, print_stats=False):
    """
    Calculate evaluation metrics for a particular meeting for a certain parameter set
    """
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    if pred_per_meeting_df.size != 0:

        # Group predictions by participant
        group_by_part = pred_per_meeting_df.groupby(['part_id'])

        for part_id, part_df in group_by_part:#iter textgrid
            # convert prep.index to list
            part_pred_frames = P.empty()
            for _, row in part_df.iterrows():
                # Create interval representing the predicted laughter defined by this row
                pred_start_frame = utils.to_frames(row['start'])
                pred_end_frame = utils.to_frames(row['end'])
                pred_laugh = P.openclosed(pred_start_frame, pred_end_frame)

                # Append interval to total predicted frames for this participant
                part_pred_frames = part_pred_frames | pred_laugh

            laugh_list = list(prep.laugh_index[meeting_id][part_id])
            non_laugh_list = list(prep.speech_index[meeting_id][part_id]) +  list(prep.noise_index[meeting_id][part_id]) + list(prep.silence_index[meeting_id][part_id]) 
            part_pred_list = list(part_pred_frames)
            for laugh in laugh_list:
                hit = False
                for predlaugh in part_pred_list:
                    if predlaugh.overlaps(laugh):
                        hit = True
                        break
                if hit == True:
                    TP += 1
                else: FN += 1
            for nonlaugh in non_laugh_list:
                hit = False
                for predlaugh in part_pred_list:
                    if predlaugh in nonlaugh:
                        hit = True
                        break
                if hit == True:
                    FP += 1
                else: TN += 1

            
    # If there is no predicted laughter time for this meeting -> precision=1
    if TP+FP == 0:
        prec = 1
    else:
        prec = TP/(TP+FP)
    if TP + FN == 0:
        # If there is no positive data (no laughs in this meeting)
        # the recall doesn't mean anything -> thus, NaN
        recall = float('NaN')
    else:
        recall = TP/(TP+FN)

    if(print_stats):
        # print(f'total transcribed time: {tot_transc_laugh_time:.2f}\n'
        #       f'total predicted time: {tot_predicted_time:.2f}\n'
        #       f'correct: {tot_corr_pred_time:.2f}\n'
        #       f'incorrect: {tot_incorr_pred_time:.2f}\n')

        print(f'Meeting: {meeting_id}\n'
              f'Threshold: {threshold}\n'
              f'Precision: {prec:.4f}\n'
              f'Recall: {recall:.4f}\n')
    return[meeting_id, threshold, min_len, prec, recall]
        
def create_evaluation_df(path, out_path):
    """
    Creates a dataframe summarising evaluation metrics per meeting for each parameter-set
    """

    all_evals = []
    print('Calculating metrics for every meeting for every parameter-set:')
    for meeting in os.listdir(path):
        #print(f'Evaluating meeting {meeting}...')
        meeting_path = os.path.join(path, meeting)
        meeting_id = meeting_path.split("/")[-1]
        for threshold in os.listdir(meeting_path):
            threshold_dir = os.path.join(meeting_path, threshold)
            for min_length in os.listdir(threshold_dir):
                print(f'Meeting:{meeting_id}, Threshold:{threshold}, Min-Length:{min_length}')
                textgrid_dir = os.path.join(threshold_dir, min_length)
                pred_laughs = textgrid_to_df(textgrid_dir)
            
                thr_val = threshold.replace('t_', '')
                min_len_val = min_length.replace('l_', '')
                out = eval_preds(pred_laughs, meeting_id, thr_val, min_len_val)
                all_evals.append(out)
    cols = ['meeting', 'threshold', 'min_len', 'precision', 'recall']

    if len(cols) != len(all_evals[0]):
        raise Exception(
            f'List returned by eval_preds() has wrong length. Expected length: {len(cols)}. Found: {len(all_evals[0])}.')
    eval_df = pd.DataFrame(all_evals, columns=cols)
    eval_df.to_csv(out_path, index=False)

def calc_sum_stats(eval_df):
    """
    Calculate summary statistics across all meetings per parameter-set
    """
    # Old version - not weighted
    # - problem with different length meetings
    # sum_stats = eval_df.groupby('threshold')[
    #     ['precision', 'recall', 'valid_pred_laughs']].agg(['mean', 'median']).reset_index()

    # New version - calculating metrics once for the whole corpus 
    # - solves problem with different length meetings
    sum_vals = eval_df.groupby(['min_len', 'threshold'])[['corr_pred_time','tot_pred_time','tot_transc_laugh_time']].agg(['sum']).reset_index()

    # Flatten Multi-index to Single-index
    sum_vals.columns = sum_vals.columns.map('{0[0]}'.format) 

    sum_vals['precision'] = sum_vals['corr_pred_time'] / sum_vals['tot_pred_time']
    # If tot_pred_time was zero set precision to 1
    sum_vals.loc[sum_vals.tot_pred_time == 0, 'precision'] = 1 

    sum_vals['recall'] = sum_vals['corr_pred_time'] / sum_vals['tot_transc_laugh_time']
    sum_stats = sum_vals[['threshold', 'min_len', 'precision', 'recall']]
    # Filter thresholds
    #sum_stats = sum_stats[sum_stats['threshold'].isin([0.2,0.4,0.6,0.8])]
    return sum_stats

def analyse(preds_dir):
    '''
    Analyse the predictions in the passed dir by comparing it to a the transcribed laughter events.

    preds_dir: Path that contains all predicted laughs in separate dirs for each parameter
    '''
    print(f'Analysing {preds_dir}')
    force_analysis = True 

    preds_path = Path(preds_dir)
    split = preds_path.name
    sum_stats_cache_file = 'sum_stats_noTimeStamp.csv'
    eval_df_cache_file = 'eval_df_noTimeStamp.csv'
    sum_stats_out_path = (preds_path.parent / f"{split}_{sum_stats_cache_file}")
    eval_df_out_path = (preds_path.parent / f"{split}_{eval_df_cache_file}")
    if not force_analysis and os.path.isfile(sum_stats_out_path):
        print('========================\nLOADING STATS FROM DISK\n')
        sum_stats = pd.read_csv(sum_stats_out_path)
    else:
        # Then create or load eval_df -> stats for each meeting
        eval_df = create_evaluation_df(preds_dir, eval_df_out_path)
        # stats_for_different_min_length(preds_path)
        sum_stats = calc_sum_stats(eval_df)
        print('\nWeighted summary stats across all meetings:')
        print(sum_stats)
        sum_stats.to_csv(sum_stats_out_path, index=False)
        print(f'\nWritten evaluation outputs to: {sum_stats_out_path}')


    
    # Create plots for different thresholds
    # for t in [.2, .4, .6, .8]:
    #     plot_aggregated_laughter_length_dist(eval_df, t, save_dir='./imgs/')
    #     plot_agg_pred_time_ratio_dist(eval_df, t, save_dir='./imgs/')


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python analyse.py <preds_dir>')
        exit(1) 
    preds_path = sys.argv[1]
    analyse(preds_path)