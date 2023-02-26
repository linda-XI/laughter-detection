from lhotse import CutSet
from lhotse import Recording, SupervisionSegment, MonoCut
import pandas as pd
import os

#dfs = pd.read_csv(os.path.join(data_dfs_dir, f'{split}_df.csv'))

#cutset=CutSet.from_jsonl(os.path.join('splitFeat/cutsets/train_feats.jsonl'))
#row_track = cutset[0]
#row_cut = row_track.truncate(offset=0, duration=1).pad(duration=1, preserve_id=True)
#print(row_cut)
#print(row_cut.id)
#sup = SupervisionSegment(id=f'sup_{row_cut.id}', recording_id=row_track.recording.id, start=1.6,
#                                             duration=1.5, channel=3, custom={'is_laugh': 1})
#row_cut.tracks[0].cut.supervisions.append(sup)
#print(row_cut.tracks[0].cut.supervisions)
###########################################################################
cuts = CutSet.from_file('test_output/cutsets/train_cutset_with_feats.jsonl')
sups = [SupervisionSegment(id='sup1', recording_id='rec1', start=0, duration=3.37, text='Hey, Matt!')]
for c in cuts:
    if len(c.supervisions) == 0:
        c.tracks[0].cut.supervisions.append(sups[0])

#        c.supervisions.append(sups)
        print(c)
     #   print(c.tracks[0].cut.supervisions)
        print(c.supervisions)
