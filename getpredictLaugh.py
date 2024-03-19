# 根据probchan.json，比对laugh only df， 选出5个天选laughter
#Bns001
import json
import pandas as pd

true_path = 'seedModel/extraDF/0.8_0.2/test_laugh_only_df.csv'
with open('probschan3.json', 'r') as f:
  data = json.load(f)
# print(data)
data = list(data)
meet = 'Bns001'
chan = 'chan3'
threshold =0.5

godChoose = []
laugh_df = pd.read_csv(true_path)
sub_laugh = laugh_df[(laugh_df['meeting_id'] == meet) & (laugh_df['chan_id'] == chan)]
for index_true, row_true in sub_laugh.iterrows():
   laugh_start_frame = int(row_true['start']*100)
   laugh_end_frame = int(row_true['end']*100)
#取出真笑声+-50的frame, 100个frame一秒
   start = laugh_start_frame - 50
   end = laugh_end_frame + 50
   godChoose.append(data[start:end])
print(godChoose)
