# 根据probchan.json，比对laugh only df， 选出5个天选laughter
#Bns001
import json
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
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
   
   godChoose.append([round(num,1) for num in data[start:end]])
   godTime.append([x for x in range(start, end)])
print(godChoose[:5])

plt.figure(1)
plt.subplots(figsize=(6, 10))
fig, ((ax1, ax2,ax3)) = plt.subplots(nrows=3, ncols=1)
# plt.subplot(311)
# hm = sns.heatmap(conf_ratio_by_rows, yticklabels=['laugh', 'not laugh'], annot=show_annotations, cmap="YlGnBu")
ax1.plot(godTime[0], godChoose[0])
ax1.set_title('sample1')
ax1.set_xlabel('frame')
ax1.set_ylabel('predict probability')
ax3.set_yticks(np.arange(0, 1.0, 0.1))
# Plotting the second histogram
# plt.subplot(312)
ax2.plot(godTime[1], godChoose[1])
ax2.set_title('sample2')
ax2.set_xlabel('frame')
ax2.set_ylabel('predict probability')
ax3.set_yticks(np.arange(0, 1.0, 0.1))

# plt.subplot(313)
ax3.plot(godTime[3], godChoose[3])
ax3.set_title('sample3')
ax3.set_xlabel('frame')
ax3.set_ylabel('predict probability')
ax3.set_yticks(np.arange(0, 1.0, 0.1))

# Adjust layout to prevent overlapping
plt.tight_layout()

if not os.path.isdir(outpath):
    os.makedirs(outpath)
plot_file = os.path.join(outpath, plot_file )
print(plot_file)

plt.savefig(plot_file)
plt.cla()
