# 根据probchan.json，比对laugh only df， 选出5个天选laughter
#Bns001
import json
import pandas as pd
import matplotlib.pyplot as plt
import os

true_path = 'seedModel/extraDF/0.8_0.2/test_laugh_only_df.csv'
with open('probschan3.json', 'r') as f:
  data = json.load(f)
# print(data)
data = list(data)
meet = 'Bns001'
chan = 'chan3'
threshold =0.5
#save plot as sample/God/Bns001chan3.jpg
outpath = 'sample/God'
plot_file = meet+chan

godChoose = []
godTime = []
laugh_df = pd.read_csv(true_path)
sub_laugh = laugh_df[(laugh_df['meeting_id'] == meet) & (laugh_df['chan_id'] == chan)]
for index_true, row_true in sub_laugh.iterrows():
   laugh_start_frame = int(row_true['start']*100)
   laugh_end_frame = int(row_true['end']*100)
#取出真笑声+-50的frame的probability, 100个frame一秒
   start = laugh_start_frame - 50
   end = laugh_end_frame + 50
   
   godChoose.append([format(num,'.1f') for num in data[start:end]])
   godTime.append([x for x in range(start, end)])
print(godChoose[:5])

plt.figure(1)
plt.subplots(figsize=(6, 3))
plt.subplot(311)
# hm = sns.heatmap(conf_ratio_by_rows, yticklabels=['laugh', 'not laugh'], annot=show_annotations, cmap="YlGnBu")
plt.plot(godTime[0], godChoose[0])
plt.title('sample1')
plt.xlabel('frame')
plt.ylabel('predict probability')
plt.gca().invert_yaxis()
# Plotting the second histogram
plt.subplot(312)
plt.plot(godTime[1], godChoose[1])
plt.title('sample2')
plt.xlabel('frame')
plt.ylabel('predict probability')
plt.gca().invert_yaxis()

plt.subplot(313)
plt.plot(godTime[2], godChoose[2])
plt.title('sample3')
plt.xlabel('frame')
plt.ylabel('predict probability')
plt.gca().invert_yaxis()
# Adjust layout to prevent overlapping
plt.tight_layout()

if not os.path.isdir(outpath):
    os.makedirs(outpath)
plot_file = os.path.join(outpath, plot_file )
print(plot_file)

plt.savefig(plot_file)
