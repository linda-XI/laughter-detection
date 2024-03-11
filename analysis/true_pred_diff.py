import analyse
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path
# print the histogram of difference between true laughter and predict laughter 
parser = argparse.ArgumentParser()
# eval/extraLaugh/resnet18_small_0200/test
# Bns001/t_0.8/l_0.2/chan0.TextGrid
parser.add_argument('--textgrid_path', type=str, required=True)
# seedModel/extraonlyDF/0.8_0.2/test_laugh_only_df.csv
parser.add_argument('--laugh_only_df_path', type=str, required=True)

parser.add_argument('--thre', type=str, required=True)
parser.add_argument('--minlen', type=str, required=True)
parser.add_argument('--meet', type=str, required=True)
parser.add_argument('--chan', type=str, required=True)
parser.add_argument('--outpath', type=str, required=True)
parser.add_argument('--model', type=str, required=True)
args = parser.parse_args()

textgrid_path = args.textgrid_path
true_path = args.laugh_only_df_path
threpath = args.thre
minlenpath = args.minlen
meet = args.meet
chan = args.chan
outpath = args.outpath
model = args.model

threpath = 't_'+ threpath
minlenpath = 'l_' + minlenpath
chanpath = chan + '.TextGrid'

textgrid_path = os.path.join(textgrid_path, meet, threpath, minlenpath, chanpath )
true_df = pd.read_csv(true_path)


def textgrid_to_df(file_path):
    tot_list = []
    if file_path.endswith('.TextGrid'):
        
        
        params = analyse.get_params_from_path(file_path)
        tot_list += analyse.textgrid_to_list(file_path,
                                        params)
    cols = ['meeting_id', 'part_id', 'chan', 'start',
            'end', 'length', 'threshold', 'min_len', 'laugh_type']
    df = pd.DataFrame(tot_list, columns=cols)
    return df



sub_true = true_df[(true_df['meeting_id'] == meet) & (true_df['chan_id'] == chan)]
df_predict = textgrid_to_df(textgrid_path)
startDiff = []
endDiff = []
for index_true, row_true in sub_true.iterrows():
    for index_pred, row_pred in df_predict.iterrows():
        trueStart =  row_true['start']
        trueEnd = row_true['end']
        predStart = row_pred['start']
        predEnd = row_pred['end']
        # begining of true laugh inside 
        if trueStart>= predStart and trueStart <=predEnd:
            startDiff.append(trueStart - predStart)
            endDiff.append(trueEnd - predEnd)

plt.hist(startDiff)

plt.title('Histogram Example')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)

# hm = sns.heatmap(conf_ratio, yticklabels=sum_vals['threshold'], annot=show_annotations, cmap="YlGnBu")
# hm.set_yticklabels(sum_vals['threshold'], size = 11)
# hm.set_xticklabels(labels, size = 12)
# plt.ylabel('threshold', size=12)
# plt.xticks(rotation=0)
# plt.yticks(rotation=0)
# plt.tight_layout()
# plot_file = os.path.join(cfg.ANALYSIS['plots_dir'], sub_dir, 'conf_matrix', f'{name}.png')
# Path(plot_file).parent.mkdir(exist_ok=True, parents=True)
# plt.savefig(plot_file)


plt.figure(1)
plt.subplot(211)
# hm = sns.heatmap(conf_ratio_by_rows, yticklabels=['laugh', 'not laugh'], annot=show_annotations, cmap="YlGnBu")
plt.hist(startDiff, bins = 30)
plt.title('start diff')
plt.xlabel('time difference')
plt.ylabel('Frequency')

# Plotting the second histogram
plt.subplot(212)
plt.hist(endDiff, bins = 30)
plt.title('end diff')
plt.xlabel('time difference')
plt.ylabel('Frequency')

# Adjust layout to prevent overlapping
plt.tight_layout()

# hm.set_yticklabels(['laugh', 'not laugh'], size = 11)
# hm.set_xticklabels(labels, size = 12)
# plt.ylabel('predicted class', size=12)
# plt.xticks(rotation=0)
# plt.yticks(rotation=0)
# plt.tight_layout()
# plt.gca().set_title('normalize by rows')
# plt.subplot(212)
# # hm = sns.heatmap(conf_ratio_by_cols, yticklabels=['laugh', 'not laugh'], annot=show_annotations, cmap="YlGnBu")
# plt.hist(endDiff)
# hm.set_yticklabels(['laugh', 'not laugh'], size = 11)
# hm.set_xticklabels(labels, size = 12)
# plt.ylabel('predicted class', size=12)
# plt.xticks(rotation=0)
# plt.yticks(rotation=0)
# plt.tight_layout(pad=3.0)
# plt.gca().set_title('normalize by columns')

# plot_file = os.path.join(cfg.ANALYSIS['plots_dir'], sub_dir, 'conf_matrix', f'{name}.png')
# Path(plot_file).parent.mkdir(exist_ok=True, parents=True)
if not os.path.isdir(outpath):
    os.mkdir(outpath)
thremin = args.thre.replace('.', '') + args.minlen.replace('.', '')
plot_file = os.path.join(outpath, model, (meet+'_'+chan+'_'+thremin) )

plt.savefig(plot_file)