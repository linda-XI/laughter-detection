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

##################################################
# PARSE TEXTGRID
##################################################



##############################################################################################################################
def textgrid_to_list_notLaugh(full_path, params):
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

        if not str(interval.text) == 'laugh' and not str(interval.text) == 'breath laugh':
            seg_length = interval.xmax - interval.xmin
            interval_list.append([params['meeting_id'], part_id, params['chan_id'], interval.xmin,
                                  interval.xmax, seg_length, params['threshold'], params['min_len'], str(interval.text)])
        print(params['meeting_id'])
        print(params['chan_id'])
        print(interval_list)

    return interval_list
##############################################################################################################

def textgrid_to_df(file_path):
    tot_list_notLaugh = []
    for filename in os.listdir(file_path):
        if filename.endswith('.TextGrid'):
            full_path = os.path.join(file_path, filename)
            
            params = get_params_from_path(full_path)
            tot_list_notLaugh += textgrid_to_list_notLaugh(full_path,
                                         params)
            

    cols = ['meeting_id', 'part_id', 'chan', 'start',
            'end', 'length', 'threshold', 'min_len', 'laugh_type']
    df_notLaugh = pd.DataFrame(tot_list_notLaugh, columns=cols)
    return df_notLaugh


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

##################################################
# ANALYSE
##################################################

def seg_index_overlap(index, segment, meeting_id, part_id):
    '''
    Returns the time [in s] the passed segment overlaps with the given index
    Returns 0 if no overlap.

    params: segment: an interval as defined by the portion library
    '''

    if part_id not in index[meeting_id].keys():
        # No segments transcribed for this participant => no overlap 
        return 0

    # Get overlap by taking the intersection (&)
    overlap = index[meeting_id][part_id] & segment
    overlap_time = utils.to_sec(utils.p_len(overlap))

    return overlap_time

def not_laugh_match(pred_notlaugh, meeting_id, part_id):
    '''
    Checks if a predicted laugh events for a particular meeting overlap with the
    transcribed laugh events for that meeting
    Input: P.Interval (Union of all laughter intervals for particular participant in one meeting)
    Returns: (time_predicted_correctly, time_predicted_falsely)
    '''

    invalid_mismatch = seg_index_overlap(prep.invalid_index, pred_notlaugh, meeting_id, part_id)
    if part_id in prep.invalid_index[meeting_id].keys():
        # Remove laughter occurring in mixed settings because we don't evaluate them
        pred_notlaugh = pred_notlaugh - prep.invalid_index[meeting_id][part_id]

    pred_length = utils.to_sec(utils.p_len(pred_notlaugh))

    correct = 0 
    incorrect = pred_length
    if part_id in prep.laugh_index[meeting_id].keys():
        correct = seg_index_overlap(prep.laugh_index, pred_notlaugh, meeting_id, part_id)
        incorrect = pred_length - correct

    # Get type of misclassification 
    speech_mismatch = seg_index_overlap(prep.speech_index, pred_notlaugh, meeting_id, part_id)
    silence_mismatch = seg_index_overlap(prep.silence_index, pred_notlaugh, meeting_id, part_id)
    noise_mismatch = seg_index_overlap(prep.noise_index, pred_notlaugh, meeting_id, part_id)
    remain_mismatch = incorrect - speech_mismatch - silence_mismatch - noise_mismatch
    
    incorrect = incorrect - remain_mismatch
   # assert remain_mismatch < 0.001, f"Accumulated false positives don't match the total incorrect time. Difference: {remain_mismatch}"

    return(correct, incorrect, speech_mismatch, noise_mismatch, silence_mismatch)
    #return(correct, incorrect, speech_mismatch, noise_mismatch, silence_mismatch, remain_mismatch)


def eval_preds(pred_per_meeting_df, meeting_id, threshold, min_len, print_stats=False):
    """
    Calculate evaluation metrics for a particular meeting for a certain parameter set
    """
    tot_predicted_time, tot_corr_pred_time, tot_incorr_pred_time = 0, 0, 0
    # Keep track of types misclassified (False-Positives) times for confusion matrix
    tot_tp_speech_time = 0
    tot_tp_noise_time = 0 
    tot_tp_silence_time = 0 
    tot_fp_false_silence_time = 0
    tot_transc_laugh_time = prep.laugh_index[meeting_id]['tot_len']
    num_of_tranc_laughs = parse.laugh_only_df[parse.laugh_only_df['meeting_id']
                                              == meeting_id].shape[0]
    num_of_pred_laughs = pred_per_meeting_df.shape[0]

    # Count by
    num_of_VALID_pred_notlaughs = 0

    if pred_per_meeting_df.size != 0:

        # Group predictions by participant
        group_by_part = pred_per_meeting_df.groupby(['part_id'])

        for part_id, part_df in group_by_part:
            part_pred_frames = P.empty()
            for _, row in part_df.iterrows():
                # Create interval representing the predicted not laughter defined by this row
                pred_start_frame = utils.to_frames(row['start'])
                pred_end_frame = utils.to_frames(row['end'])
                pred_not_laugh = P.openclosed(pred_start_frame, pred_end_frame)

                # If the there are no invalid frames for this participant at all
                # or if the laugh frame doesn't lie in an invalid section -> increase num of valid predictions
                if part_id not in prep.invalid_index[meeting_id].keys() or \
                        not prep.invalid_index[meeting_id][part_id].contains(pred_not_laugh):
                    num_of_VALID_pred_notlaughs += 1

                # Append interval to total predicted frames for this participant
                part_pred_frames = part_pred_frames | pred_not_laugh

            corr, incorr, speech, noise, silence =  not_laugh_match(part_pred_frames, meeting_id, part_id)
           # corr, incorr, speech, noise, silence, false_silence =  laugh_match(part_pred_frames, meeting_id, part_id)
            tot_corr_pred_time += corr
            tot_incorr_pred_time += incorr
            tot_tp_speech_time += speech
            tot_tp_noise_time += noise
            tot_tp_silence_time += silence
            #tot_fp_false_silence_time += false_silence

    tot_predicted_time = tot_corr_pred_time + tot_incorr_pred_time
    # If there is no predicted laughter time for this meeting -> precision=1
    if tot_predicted_time == 0:
        prec = 1
    else:
        prec = tot_corr_pred_time/tot_predicted_time
    if tot_transc_laugh_time == 0:
        # If there is no positive data (no laughs in this meeting)
        # the recall doesn't mean anything -> thus, NaN
        recall = float('NaN')
    else:
        recall = tot_corr_pred_time/tot_transc_laugh_time

    if(print_stats):
        print(f'total transcribed time: {tot_transc_laugh_time:.2f}\n'
              f'total predicted time: {tot_predicted_time:.2f}\n'
              f'correct: {tot_corr_pred_time:.2f}\n'
              f'incorrect: {tot_incorr_pred_time:.2f}\n')

        print(f'Meeting: {meeting_id}\n'
              f'Threshold: {threshold}\n'
              f'Precision: {prec:.4f}\n'
              f'Recall: {recall:.4f}\n')

    #return[meeting_id, threshold, min_len, prec, recall, tot_corr_pred_time, tot_predicted_time,
     #      tot_transc_laugh_time, num_of_pred_laughs, num_of_VALID_pred_laughs, num_of_tranc_laughs, 
      #     tot_fp_speech_time, tot_fp_noise_time, tot_fp_silence_time, tot_fp_false_silence_time]
    return[meeting_id, threshold, min_len, prec, recall, tot_corr_pred_time, tot_predicted_time,
           tot_transc_laugh_time, num_of_pred_laughs, num_of_VALID_pred_notlaughs, num_of_tranc_laughs,
           tot_tp_speech_time, tot_tp_noise_time, tot_tp_silence_time]

def create_evaluation_df(path, out_notLaugh_path, use_cache=False):
    """
    Creates a dataframe summarising evaluation metrics per meeting for each parameter-set
    """
    if use_cache and os.path.isfile(f'{os.path.dirname(__file__)}/.cache/eval_df.csv') and os.path.isfile(f'{os.path.dirname(__file__)}/.cache/eval_notLaugh_df.csv'):
        print("-----------------------------------------")
        print("NO NEW EVALUATION - USING CACHED VERSION")
        print("-----------------------------------------")
        eval_notLaugh_df = pd.read_csv(out_notLaugh_path)
    else:
        all_evals_notLaugh = []
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
                    
                    pred_notLaughs = textgrid_to_df(textgrid_dir)
                
                    thr_val = threshold.replace('t_', '')
                    min_len_val = min_length.replace('l_', '')
                    
                
                    
                    out_notLaugh = eval_preds(pred_notLaughs, meeting_id, thr_val, min_len_val)
                    
                    all_evals_notLaugh.append(out_notLaugh)
                    # Log progress

        #---fp_laugh: actual laught, predict notlaugh; tot_tp_speech_time:actual speech, predict not laugh---#
        cols_notLaugh = ['meeting', 'threshold', 'min_len', 'precision', 'recall',
                'fp_laugh', 'tot_pred_time', 'tot_transc_notLaugh_time', 'num_of_pred_notLaughs', 'valid_pred_notLaughs', 'num_of_transc_notLaughs',
                'tot_tp_speech_time', 'tot_tp_noise_time', 'tot_tp_silence_time']

        if (len(cols_notLaugh) != len(cols_notLaugh[0])) :
            raise Exception(
                f'List returned by eval_preds() has wrong length.  Expected length for laugh: {len(cols_notLaugh)}. Found: {len(all_evals_notLaugh[0])}.')
        
        eval_notLaugh_df = pd.DataFrame(all_evals_notLaugh, columns=cols_notLaugh)
        if not os.path.isdir(f'{os.path.dirname(__file__)}/.cache'):
            subprocess.run(['mkdir', '.cache'])
        
        eval_notLaugh_df.to_csv(out_notLaugh_path, index=False)
        print(f'\nWritten evaluation outputs to: {out_notLaugh_path}')

    return  eval_notLaugh_df





##################################################
# OTHER
##################################################
MIN_LENGTH = cfg['model']['min_length']


def laugh_df_to_csv(df):
    """
    Used to generate .csv file of a subset of laughter events (e.g. breath-laughs)
    e.g. for generating .wav-files using
    ./output_processing/laughs_to_wav.py from this .csv
    """
    df = df[df['laugh_type'] == 'breath-laugh']
    df.to_csv('breath_laugh.csv')


# TODO: rewrite this function (if needed) to work with new structure
def stats_for_different_min_length(preds_path):
    global MIN_LENGTH

    # Rounding to compensate np.arrange output inaccuracy (e.g.0.600000000001)
    lengths = list(np.arange(5.2, 8.2, 0.2).round(1))

    # This will contain each df with summary stats for different min_length values
    df_list = []

    for min_length in lengths:
        MIN_LENGTH = min_length
        print(f"Using min_laugh_length: {MIN_LENGTH}")

        # Need to recreate laughter indices and eval_df because min_length was changed
        # First create laughter segment indices
        # Mixed laugh index needs to be created first (see implementation of laugh_index)
        # NEED TO CHANGED THE FOLLOWING TWO LINES
        # create_mixed_laugh_index(parse.invalid_df)
        # create_laugh_index(parse.laugh_only_df)

        # Then create or load eval_df -> stats for each meeting
        eval_notLaugh_df = create_evaluation_df(preds_path)

        

        # Print out the number of laughter events left for this min_length
        acc_len = 0
        acc_ev = 0
        for meeting in prep.laugh_index.keys():
            acc_len += prep.laugh_index[meeting]['tot_len']
            acc_ev += prep.laugh_index[meeting]['tot_events']
        print(f'Agg. laugh length: {acc_len:.2f}')
        print(f'Total laugh events: {acc_ev}')
        # Print number of invalid laughter events for this min_length
        acc_len = 0
        acc_ev = 0
        for meeting in prep.invalid_index.keys():
            acc_len += prep.invalid_index[meeting]['tot_len']
            acc_ev += prep.invalid_index[meeting]['tot_events']
        print(f'Agg. invalid laugh length: {acc_len:.2f}')
        print(f'Total invalid laugh events: {acc_ev}')

    tot_df = pd.concat(df_list)
    tot_df.to_csv('sum_stats_for_different_min_lengths.csv')


def create_csvs_for_meeting(meeting_id, preds_path):
    """
    Writes 2 csv files to disk:
        1) containing the transcribed laughter events for this meeting
        2) containing all predicted laughter events (for threshholds: 0.2, 0.4, 0.6, 0.8)
            - thus, duplicates are possible -> take this into account when analysing

    Can be used for analysing predictions vs. transcriptions with external software/in different ways 
    """
    tranc_laughs = parse.laugh_only_df[parse.laugh_only_df['meeting_id'] == meeting_id]
    tranc_laughs.to_csv(f'{meeting_id}_transc.csv')

    meeting_path = os.path.join(preds_path, meeting_id)
    # Get predictions for different threshholds
    df1 = textgrid_to_df(
        f'{meeting_path}/t_0.2/l_0.2')
    df2 = textgrid_to_df(
        f'{meeting_path}/t_0.4/l_0.2')
    df3 = textgrid_to_df(
        f'{meeting_path}/t_0.6/l_0.2')
    df4 = textgrid_to_df(
        f'{meeting_path}/t_0.8/l_0.2')
    # Concat them and write them to file
    result = pd.concat([df1, df2, df3, df4])
    result.to_csv(f'{meeting_id}_preds.csv')


def analyse(preds_dir):
    '''
    Analyse the predictions in the passed dir by comparing it to a the transcribed laughter events.

    preds_dir: Path that contains all predicted laughs in separate dirs for each parameter
    '''
    print(f'Analysing {preds_dir}')
    force_analysis = False 

    preds_path = Path(preds_dir)
    split = preds_path.name
    #TODO:change not laugh path
    eval_df_notlaugh_out_path = (preds_path.parent / f"{split}_{cfg['eval_notLaugh_df_cache_file']}")
    if  force_analysis or not os.path.isfile(eval_df_notlaugh_out_path):
        
        # Then create or load eval_df -> stats for each meeting
        eval_notLaugh_df = create_evaluation_df(preds_dir, eval_df_notlaugh_out_path, use_cache=True)
        # stats_for_different_min_length(preds_path)
        
        #TODO:sum stats for not laugh
        
        


    
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