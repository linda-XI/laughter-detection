import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from analysis.transcript_parsing import parse
#from analysis.preprocess import create_index_from_df
import portion as P
import analysis.utils as utils

def append_to_index(index, row, meeting_id, part_id):
    start = utils.to_frames(row["start"])
    end = utils.to_frames(row["end"])
    seg_as_interval = P.openclosed(start,end)
                    # Append to existing intervals or create new dict entry
    if part_id in index[meeting_id].keys():
        index[meeting_id][part_id] = index[meeting_id][part_id] | seg_as_interval 
    else:
        index[meeting_id][part_id] = seg_as_interval 
    seg_len = utils.to_sec(utils.p_len(seg_as_interval))

    index[meeting_id]["tot_len"] += seg_len
    index[meeting_id]["tot_events"] += 1
    return index

def create_index_from_df(df, laugh_index):
    index = {}
    meeting_groups = df.groupby(["meeting_id"])
    for meeting_id, meeting_df in meeting_groups:
        index[meeting_id] = {}
        index[meeting_id]["tot_len"] = 0
        index[meeting_id]["tot_events"] = 0
        part_groups = meeting_df.sort_values("start").groupby(["part_id"])
        for part_id, part_df in part_groups:
            for _, row in part_df.iterrows():
                index = append_to_index(index, row, meeting_id, part_id)
               # index[meeting_id][part_id] = index[meeting_id][part_id] - create_meeting_laugh_seg(meeting_id, laugh_index)
    return index

def get_seg_from_index(index, meeting_id):
    if meeting_id in index.keys():
        union = P.empty()
        for part_id in index[meeting_id].keys():
            print(part_id)
            if (part_id != 'tot_len') & (part_id != 'tot_events'):
                union = union | index[meeting_id].get(part_id, P.empty())
        return union
    return P.empty()

#get union laugh set of all part_id  for a given meeting id
def create_meeting_laugh_seg(meeting_id, laugh_index):
    if meeting_id in index.keys():
        whole_laugh = P.empty()
        for part_id in laugh_index[meeting_id].keys:
             if (part_id != 'tot_len') & (part_id != 'tot_events'):
                 whole_laugh = whole_laugh | laugh_index[meeting_id].get(part_id, P.empty())
        return whole_laugh
    return P.empty()
print(parse.laugh_only_df)
df = parse.laugh_only_df
#df2 = df[(df['meeting_id']!='Bmr021')&(df['meeting_id']!='Bns001')&(df['meeting_id']!='Bmr013')&(df['meeting_id']!='Bmr018')&(df['meeting_id']!='Bro021')]
df =  df.iloc[:5]
print(df)        
index = create_index_from_df(df)
print(index)
print(get_seg_from_index(index,'Bns003'))

