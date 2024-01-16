import os
import pandas as pd
from typing import List, Optional, Tuple
from pathlib import Path
import sys
import textgrids
from analysis.transcript_parsing import parse
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import ANALYSIS as cfg
import portion as P
import analysis.utils as utils
from pydantic import BaseModel
from strenum import StrEnum

# laugh_only_df = pd.DataFrame(tot_laugh_only_segs)
# invalid_df = pd.DataFrame(tot_invalid_segs )
# speech_df = pd.DataFrame(tot_speech_segs)
# noise_df = pd.DataFrame(tot_noise_segs)
laugh_only_df = pd.read_csv('./sample/laugh_only_df.csv')

class SegmentType(StrEnum):
    '''
    Describes the type of data that was transcribed in this segment. 
    For detailed information: https://www1.icsi.berkeley.edu/Speech/mr/icsimc_doc/trans_guide.txt
    '''
    INVALID = 'invalid'  # e.g. laughter segments occurring next to speech or other noise
    SPEECH = 'speech' 
    LAUGH = 'laugh' 
    OTHER_VOCAL = 'other_vocal'  # segments containing a single VocalSound that's not laughter
    NON_VOCAL = 'non_vocal' # segments containing a single NonVocalSound (e.g. 'mic noise')
    MIXED = 'mixed'  # contains some mixture of speech / noise and silence (but no laughter)

class Segment(BaseModel):
    """Represent a Transcription segment from ICSI transcripts"""

    meeting_id: str
    part_id: str
    chan_id: str
    start: float
    end: float
    length: float
    type: SegmentType
    laugh_type: Optional[str]

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


def textgrid_to_df(full_path):
    '''
    convert textgrid of a meeting into dataframe
    '''           
    params = get_params_from_path(full_path)
    tot_list = textgrid_to_list(full_path,
                                    params)

    cols = ['meeting_id', 'part_id', 'chan_id', 'start',
            'end', 'length', 'type', 'laugh_type']
    df = pd.DataFrame(tot_list, columns=cols)
    return df

def update_laugh_only_df(path, use_cache=False):
    '''
        a channel can heard oter channels'laugh. 
        use update_laugh_only_df to update the laugh_only_df to add these extra laugh for each channel
        Note that this will cause overlaps of laughter for each channel.
    '''

    global laugh_only_df
    temp_laugh_df = pd.DataFrame() 
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
    for meeting in os.listdir(path, out_path):
        #print(f'Evaluating meeting {meeting}...')
        meeting_path = os.path.join(path, meeting)
        # meeting_id = meeting_path.split("/")[-1]
        threshold_dir = os.path.join(meeting_path, 't_0.8')
        #textgrid_dir: dir that contains all channel textgrid of a meeting
        textgrid_dir = os.path.join(threshold_dir, 'l_0.2')
        for filename in os.listdir(textgrid_dir):
            if filename.endswith('.TextGrid'):
                full_path = os.path.join(textgrid_dir, filename)
                #one channel one df
                textgrid_df = textgrid_to_df(full_path)

                i, j = 0, 0  
                while i < len(laugh_only_df) and j < len(textgrid_df):
                    if(textgrid_df.iloc[j]['meeting_id'] == laugh_only_df.iloc[i]['meeting_id']):
                        # 获取当前行的 'start' 值
                        start_total = laugh_only_df.iloc[i]['start']
                        start_new = textgrid_df.iloc[j]['start']

                        # 计算 'start' 之差
                        diff = abs(start_new - start_total)
                        print(diff)

                        # 如果 'start_new' 与 'start_total' 之差小于 0.2，且属于同一个meeting，且不是同一个人发出的。则添加新的行到总的 DataFrame 中
                        if ((diff < 1)  
                            and (textgrid_df.iloc[j]['chan_id'] != laugh_only_df.iloc[i]['chan_id'])):
                            
                            new_row = pd.DataFrame({
                                'meeting_id': textgrid_df.iloc[j]['meeting_id'],
                                'part_id': textgrid_df.iloc[j]['part_id'],
                                'chan_id': textgrid_df.iloc[j]['chan_id'],
                                'start': laugh_only_df.iloc[i]['start'],
                                'end': laugh_only_df.iloc[i]['end'],
                                'length': laugh_only_df.iloc[i]['length'],
                                'type': laugh_only_df.iloc[i]['type'],
                                'laugh_type': textgrid_df.iloc[j]['laugh_type']
                            }, index=[0])
                            print(new_row)
                            temp_laugh_df = pd.concat([temp_laugh_df, new_row], ignore_index=True)
                            j += 1
                        elif start_new < start_total:
                            j += 1
                        else:
                            i += 1
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
    laugh_only_df = pd.concat([laugh_only_df, temp_laugh_df], ignore_index=True)
    laugh_only_df.sort_values(by=['meeting_id', 'start'], inplace=True)
    laugh_only_df.to_csv(out_path, index=False)
    

    return laugh_only_df


def interval_to_seg(meeting_id, part_id, chan_id, interval) -> Optional[Segment]:
    """
    Input: xml laughter segment as etree Element, meeting id
    Output: list of features representing this laughter segment:
        - Format: [meeting_id, part_id, chan_id, start, end, length, l_type]
        - returns [] if no valid part_id was found
    """
    
    new_seg = Segment(
        meeting_id=meeting_id,
        part_id=part_id,
        chan_id=chan_id,
        start=interval.lower,
        end=interval.upper,
        length=interval.upper - interval.lower,
        type='laugh',
        laugh_type='laugh',
    )
        
    return new_seg

def delete_from_df(non_laugh_df, laugh_portion):
    for _, row in non_laugh_df.iterrows():
        # Create interval representing the predicted laughter defined by this row
        start = row['start']
        end = row['end']
        duration = P.openclosed(start, end)
        portion = portion | duration
    portion = portion - laugh_portion
    return portion

def refine_laugh_df(out_path):
    '''refine each channel's df to delete the overlap between rows
       input: laugh_only_df
    '''
    global laugh_only_df
    laugh_only_list: List[Segment] = []
    speech_only_list: List[Segment] = []
    laugh_group_by = laugh_only_df.groupby(['meeting_id','part_id', 'chan_id'])
    # speech_group_by = speech_only_df.groupby(['meeting_id','part_id', 'chan_id'])

    for id, part_df in laugh_group_by:
        meeting_id = id[0]
        part_id = id[1]
        chan_id = id[2]
        laugh_portion = P.empty()
        for _, row in part_df.iterrows():
            # Create interval representing the predicted laughter defined by this row
            start = row['start']
            end = row['end']
            laugh_duration = P.openclosed(start, end)
            laugh_portion = laugh_portion | laugh_duration
        #get speech df of a channel, convert it into portion, then remove the laugh portion from the speech protion
        # part_speech_df = speech_group_by.get_group(id)
        # speech_portion = delete_from_df(part_speech_df, laugh_portion)

        #>>> list(P.open(10, 11) | P.closed(0, 1) | P.closed(20, 21))
        #[[0,1], (10,11), [20,21]]
        #add one channel's laugh into df
        for interval in list(laugh_portion):
            print(interval)
            seg = interval_to_seg(meeting_id, part_id, chan_id, interval)
            laugh_only_list.append(seg.dict())

        # for interval in list(speech_portion):
            # seg = interval_to_seg(meeting_id, part_id, chan_id, interval)
            # speech_only_list.append(seg.dict())

    laugh_only_df = pd.DataFrame(laugh_only_list)
    laugh_only_df.sort_values(by=['meeting_id', 'start'], inplace=True)
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
    # refine_laugh_df(out_path)
