from typing import List, Optional, Tuple
from pydantic import BaseModel
from strenum import StrEnum
from lxml import etree
from pathlib import Path
import sys
import portion as P
import textgrids
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import ANALYSIS as cfg

# Using lxml instead of xml.etree.ElementTree because it has full XPath support
# xml.etree.ElementTree only supports basic XPath syntax
import os
import pandas as pd

chan_to_part = {}  # index mapping channel to participant per meeting
part_to_chan = {}  # index mapping participant to channel per meeting

# Dataframes containing different types of segments - one per row
laugh_only_df = pd.DataFrame()  
invalid_df = pd.DataFrame() 
noise_df = pd.DataFrame()    
speech_df = pd.DataFrame()   

# Dataframe containing total length and audio_path of each channel
info_df = pd.DataFrame()


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


def parse_preambles(path):
    global chan_to_part, part_to_chan
    """
    Creates 2 id mappings
    1) Dict: (meeting_id) -> (dict(chan_id -> participant_id))
    2) Dict: (meeting_id) -> (dict(participant_id -> chan_id))
    """
    dirname = os.path.dirname(__file__)
    preambles_path = os.path.join(dirname, path)
    chan_to_part = {}
    tree = etree.parse(preambles_path)
    meetings = tree.xpath("//Meeting")
    for meeting in meetings:
        id = meeting.get("Session")
        part_map = {}
        # Make sure that both Name and Channel attributes exist
        for part in meeting.xpath(
            "./Preamble/Participants/Participant[@Name and @Channel]"
        ):
            part_map[part.get("Channel")] = part.get("Name")

        chan_to_part[id] = part_map

    part_to_chan = {}
    for meeting_id in chan_to_part.keys():
        part_to_chan[meeting_id] = {
            part_id: chan_id for (chan_id, part_id) in chan_to_part[meeting_id].items()
        }


def xml_to_segment(xml_seg, meeting_id: str) -> Optional[Segment]:
    """
    Input: xml laughter segment as etree Element, meeting id
    Output: list of features representing this laughter segment:
        - Format: [meeting_id, part_id, chan_id, start, end, length, l_type]
        - returns [] if no valid part_id was found
    """
    part_id = xml_seg.get("Participant")
    # If participant doesn't have a corresponding audio channel
    # discard this segment
    if part_id not in part_to_chan[meeting_id].keys():
        return None

    chan_id = part_to_chan[meeting_id][part_id]
    start = float(xml_seg.get("StartTime"))
    end = float(xml_seg.get("EndTime"))
    length = end - start

    seg_type, laugh_type = _get_segment_type(xml_seg)

    new_seg = Segment(
        meeting_id=meeting_id,
        part_id=part_id,
        chan_id=chan_id,
        start=start,
        end=end,
        length=length,
        type=seg_type,
        laugh_type=laugh_type,
    )
    return new_seg


def _get_segment_type(xml_seg) -> Tuple[SegmentType, str]:
    """
    Determine the segment type of passed xml-segment - if laughter also return type of laughter
    Input: xml-segment of the shape of ICSI transcript segments
    Output: [segment_type, laugh_type]
    """
    children = xml_seg.getchildren()
    laugh_type  = None
    seg_type = SegmentType.MIXED

    if len(children) == 0:
        seg_type = SegmentType.SPEECH
    elif len(children) == 1:
        child = children[0]
        if child.tag == "VocalSound":
            if "laugh" in child.get("Description"):
                # Check that there is no text in any sub-element of this tag
                # which meant speech occurring next to laughter
                if "".join(xml_seg.itertext()).strip() == "":
                    seg_type = SegmentType.LAUGH
                    laugh_type = child.get("Description")
                else:
                    seg_type = SegmentType.INVALID

            else:
                seg_type = SegmentType.OTHER_VOCAL

        elif child.tag == "NonVocalSound":
            seg_type = SegmentType.NON_VOCAL

        else:
            # This is because there are also tags like <Comment>
            seg_type = SegmentType.SPEECH
    else:
        # Track laughter next to speech or noise to discard these segments from evaluation
        # If laughter occurs next to speech we can properly track it but it's still laughter
        # Thus a prediction on such a segment shouldn't be considered wrong but just be ignored.
        laughs = xml_seg.xpath("./VocalSound[contains(@Description, 'laugh')]")
        
        # If one of VocalSound or NonVocalSound tags appear, classify as mixed
        tag_types = list(map(lambda x: x.tag, children))
        if laughs != []:
            seg_type = SegmentType.INVALID
        elif "NonVocalSound" in tag_types or "VocalSound" in tag_types: 
            seg_type = SegmentType.MIXED
        else:
            seg_type = SegmentType.SPEECH

    return (seg_type, laugh_type)


def get_segment_list(filename, meeting_id):
    """
    Returns four lists of Segment objects represented as dict:
        1) List containing segments laughter only (no text or other sounds surrounding it)
        2) List containing invalid segments (e.g. laughter surrounding by other sounds)
        3) List containing speech segments
        4) List containing noise segments
    """
    # Comment shows which types of segments each list will hold
    invalid_list: List[Segment] = []  # INVALID
    laugh_only_list: List[Segment] = []  # LAUGH
    speech_list: List[Segment] = []  # SPEECH
    noise_list: List[Segment] = []  # MIXED, NON_VOCAL, OTHER_VOCAL

    # Get all segments that contain some kind of laughter (e.g. 'laugh', 'breath-laugh')
    # xpath_exp = "//Segment[VocalSound[contains(@Description,'laugh')]]"
    tree = etree.parse(filename)
    # laugh_segs = tree.xpath(xpath_exp)
    all_segs = tree.xpath("//Segment")

    # For each laughter segment classify it as laugh only or mixed laugh
    # mixed laugh means that the laugh occurred next to speech or any other sound
    for xml_seg in all_segs:
        seg = xml_to_segment(xml_seg, meeting_id)
        if (seg==None): # Skip segment without audio chan
            continue
        if seg.type == SegmentType.LAUGH:
            laugh_only_list.append(seg.dict())
        elif seg.type == SegmentType.SPEECH:
            speech_list.append(seg.dict())
        elif seg.type == SegmentType.INVALID:
            invalid_list.append(seg.dict())
        else:
            noise_list.append(seg.dict())

    return invalid_list, speech_list, laugh_only_list, noise_list


def general_info_to_list(filename, meeting_id):
    general_info_list = []
    tree = etree.parse(filename)
    # Get End-Timestamp of last transcription of the meeting
    meeting_len = tree.find("//Transcript").get("EndTime")
    for chan_id, part_id in chan_to_part[meeting_id].items():
        path = os.path.join(meeting_id, f"{chan_id}.sph")
        general_info_list.append([meeting_id, part_id, chan_id, meeting_len, path])

    return general_info_list


def get_transcripts(path):
    """
    Parse meeting transcripts and store laughs in laugh_df
    """

    files = []
    # If a directory is given take all .mrt files
    # otherwise only take given file
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, path)

    # Match particular file or all .mrt files
    if os.path.isdir(path):
        for filename in os.listdir(path):
            # All icsi meetings have a 6 letter ID (-> split strips the .mrt extension)
            if filename.endswith(".mrt") and len(filename.split(".")[0]) == 6:
                files.append(filename)
    else:
        if path.endswith(".mrt"):
            files.append(path)

    return files
#--------------------deal with extra laugh----------------------------------#
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

def textgrid_to_list(full_path, params):
    # There are more recorded channels than participants
    # thus, not all channels are mapped to a participant
    # We focus on those that are mapped to a participant
    
    if params['chan_id'] not in chan_to_part[params['meeting_id']].keys():
        return []

    # If file is empty -> no predictions
    if os.stat(full_path).st_size == 0:
        print(f"WARNING: Found an empty .TextGrid file. This usually shouldn't happen.  \
        Did something go wrong with evaluation for {params['meeting_id']: params['chan_id']}")
        return []

    interval_list = []
    part_id = chan_to_part[params['meeting_id']][params['chan_id']]
    
    print(full_path)
    grid = textgrids.TextGrid(full_path)
    for interval in grid['laughter']:
        
        # TODO: Change for breath laugh?!
        if str(interval.text) == 'laugh':
            seg_length = interval.xmax - interval.xmin
            interval_list.append([params['meeting_id'], part_id, params['chan_id'], interval.xmin,
                                  interval.xmax, seg_length, str(interval.text), str(interval.text)])
    return interval_list

        
    
def delete_from_df(non_laugh_df, laugh_portion):
    '''
    delete extra laugh from speech only df and so on
    '''
    for _, row in non_laugh_df.iterrows():
        # Create interval representing the predicted laughter defined by this row
        start = row['start']
        end = row['end']
        duration = P.open(start, end)
        portion = portion | duration
    portion = portion - laugh_portion
    return portion


def update_laugh_only_df(path):
    '''
        Add extra laugh into laugh_only_df
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
    for meeting in os.listdir(path):
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
                            # print(new_row)
                            temp_laugh_df = pd.concat([temp_laugh_df, new_row], ignore_index=True)
                            j += 1
                        elif start_new < start_total:
                            j += 1
                        else:
                            i += 1
                    else:
                        i += 1
            
        # if not os.path.isdir(f'{os.path.dirname(__file__)}/.cache'):
            # subprocess.run(['mkdir', '.cache'])
    laugh_only_df = pd.concat([laugh_only_df, temp_laugh_df], ignore_index=True)
    laugh_only_df.sort_values(by=['meeting_id', 'start'], inplace=True)

def refine_laugh_df(out_path):
    '''refine each channel's df to delete the overlap of laughter between rows using portion library.
       remove extra laugh from speech only df and so on.
    '''
    global laugh_only_df, invalid_df, noise_df, speech_df

    laugh_only_list: List[Segment] = []
    speech_only_list: List[Segment] = []
    invalid_only_list: List[Segment] = []
    noise_only_list: List[Segment] = []

    laugh_group_by = laugh_only_df.groupby(['meeting_id','part_id', 'chan_id'])
    speech_group_by = speech_df.groupby(['meeting_id','part_id', 'chan_id'])
    invalid_group_by = invalid_df.groupby(['meeting_id','part_id', 'chan_id'])
    noise_group_by = noise_df.groupby(['meeting_id','part_id', 'chan_id'])

    for id, part_df in laugh_group_by:
        meeting_id = id[0]
        part_id = id[1]
        chan_id = id[2]
        laugh_portion = P.empty()
        for _, row in part_df.iterrows():
            # Create interval representing the predicted laughter defined by this row
            start = row['start']
            end = row['end']
            laugh_duration = P.open(start, end)
            laugh_portion = laugh_portion | laugh_duration
        #get speech df of a channel, convert it into portion, then remove the laugh portion from the speech protion
        part_speech_df = speech_group_by.get_group(id)
        part_invalid_df = invalid_group_by.get_group(id)
        part_noise_df = noise_group_by.get_group(id)

        speech_portion = delete_from_df(part_speech_df, laugh_portion)
        invalid_portion = delete_from_df(part_invalid_df, laugh_portion)
        noise_portion = delete_from_df(part_noise_df, laugh_portion)

        #>>> list(P.open(10, 11) | P.closed(0, 1) | P.closed(20, 21))
        #[[0,1], (10,11), [20,21]]
        #add one channel's laugh into df
        for interval in list(laugh_portion):
            # print(interval)
            seg = interval_to_seg(meeting_id, part_id, chan_id, interval)
            laugh_only_list.append(seg.dict())

        for interval in list(speech_portion):
            seg = interval_to_seg(meeting_id, part_id, chan_id, interval)
            speech_only_list.append(seg.dict())

        for interval in list(invalid_portion):
            seg = interval_to_seg(meeting_id, part_id, chan_id, interval)
            invalid_only_list.append(seg.dict())

        for interval in list(noise_portion):
            seg = interval_to_seg(meeting_id, part_id, chan_id, interval)
            noise_only_list.append(seg.dict())

    laugh_only_df = pd.DataFrame(laugh_only_list)
    laugh_only_df.sort_values(by=['meeting_id', 'start'], inplace=True)
    laugh_only_df.to_csv(out_path + '/test_laugh_only_df.csv.csv', index=False)

    speech_df = pd.DataFrame(speech_only_list)
    speech_df.sort_values(by=['meeting_id', 'start'], inplace=True)
    speech_df.to_csv(out_path + '/test_speech_df.csv', index=False)

    invalid_df = pd.DataFrame(invalid_only_list)
    invalid_df.sort_values(by=['meeting_id', 'start'], inplace=True)
    invalid_df.to_csv(out_path + '/test_invalid_df.csv', index=False)

    noise_df = pd.DataFrame(noise_only_list)
    noise_df.sort_values(by=['meeting_id', 'start'], inplace=True)
    noise_df.to_csv(out_path + '/test_noise_df.csv', index=False)

    return laugh_only_df, speech_df, noise_df, speech_df


#----------------------------------------------------------------------------#

def create_dfs(file_dir, files):
    """
    Creates four segment-dataframes and one info-dataframe:
        1) laugh_only_df: dataframe containing laughter only snippets
        2) invalid_df: containing snippets with laughter next to speech (discarded from evaluation)
        3) speech_df: contains speech
        4) noise_df: containing all other segments
            - NOTE: this doesn't include silence as silence happens mostly BETWEEN transcription segments

        5) general_info_df: dataframe containing total length and audio_path of each channel

    segment_dataframes columns: ['meeting_id', 'part_id', 'chan_id', 'start', 'end', 'length', 'type', 'laugh_type']

    info_df columns: ['meeting_id', 'part_id', 'chan_id', 'length', 'path']

    """
    global laugh_only_df, invalid_df, info_df, noise_df, speech_df

    # Define lists holding all the rows for those dataframes
    tot_invalid_segs = []
    tot_speech_segs= []
    tot_laugh_only_segs = []
    tot_noise_segs = []

    general_info_list = []
    # Iterate over all .mrt files
    for filename in files:
        # Get meeting id by getting the basename and stripping the extension
        basename = os.path.basename(filename)
        meeting_id = os.path.splitext(basename)[0]
        full_path = os.path.join(file_dir, filename)

        general_info_sublist = general_info_to_list(full_path, meeting_id)
        general_info_list += general_info_sublist

        invalid, speech, laugh_only, noise = get_segment_list(full_path, meeting_id)
        tot_invalid_segs += invalid
        tot_speech_segs += speech 
        tot_laugh_only_segs += laugh_only
        tot_noise_segs += noise

    laugh_only_df = pd.DataFrame(tot_laugh_only_segs)
    invalid_df = pd.DataFrame(tot_invalid_segs )
    speech_df = pd.DataFrame(tot_speech_segs)
    noise_df = pd.DataFrame(tot_noise_segs)

    #TODO 根据textgrid处理dataframe

    # Create info_df with specified columns and dtypes
    info_dtypes = {"length": "float"}
    info_cols = ["meeting_id", "part_id", "chan_id", "length", "path"]
    info_df = pd.DataFrame(general_info_list, columns=info_cols)
    info_df = info_df.astype(dtype=info_dtypes)


def parse_transcripts(path):
    """
    Function executed on import of this module.
    Parses transcripts (including preamble.mrt) and creates:
        - chan_to_part: index mapping channel to participant per meeting
        - part_to_chan: index mapping participant to channel per meeting
        - laugh_only_df: dataframe containing transcribed laugh only events
        - invalid_df: dataframe containing invalid segments (e.g. mixed laugh and speech)
    """
    parse_preambles(os.path.join(path, "preambles.mrt"))

    transc_files = get_transcripts(path)
    create_dfs(path, transc_files)
    update_laugh_only_df(cfg['extra_laugh_dir'])
    refine_laugh_df(cfg['test_df_dir'])


def _print_stats(df):
    """
    Print stats of laugh_df - for information/debugging only
    """
    print(df)
    if df.size == 0:
        print("Empty DataFrame")
        return
    print("avg-snippet-length: {:.2f}s".format(df["length"].mean()))
    print("Number of snippets: {}".format(df.shape[0]))
    tot_dur = df["length"].sum()
    print(
        "Accumulated segment duration in three formats: \n- {:.2f}h \n- {:.2f}min \n- {:.2f}s".format(
            (tot_dur / 3600), (tot_dur / 60), tot_dur
        )
    )


def main():
    """
    Main executed when file is called directly
    NOT on import
    """
    file_path = os.path.join(os.path.dirname(__file__), "data")
    parse_transcripts(file_path)

    print("\n----INALID SEGMENTS-----")
    _print_stats(invalid_df)

    print("\n----SPEECH SEGMENTS-----")
    _print_stats(speech_df)
    
    print("\n----LAUGHTER ONLY-----")
    _print_stats(laugh_only_df)

    print("\n----NOISE SEGMENTS-----")
    _print_stats(noise_df)

    print("\n----INFO DF-----")
    print(info_df)


if __name__ == "__main__":
    main()


#############################################
# EXECUTED ON IMPORT
#############################################
# Parse transcripts on import
parse_transcripts(cfg['transcript_dir'])