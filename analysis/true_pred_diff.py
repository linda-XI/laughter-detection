import analysis.analyse as analyse
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--textgrid_path', type=str, required=True)
parser.add_argument('--laugh_only_df_path', type=str, required=True)
args = parser.parse_args()

textgrid_path = args.textgrid_path
true_path = args.laugh_only_df_path
textgrid_df = analyse.textgrid_to_df(textgrid_path)


meeting_id = 'Bdb001'
chan_id = 'chan1'
df_true = pd.DataFrame()
sub_true = df_true[(df_true['meeting_id'] == meeting_id) & (df_true['chan_id'] == chan_id)]
df_predict = pd.DataFrame()
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

labels = ['laugh', 'speech', 'silence', 'noise']

plt.figure(1)
plt.subplot(211)
# hm = sns.heatmap(conf_ratio_by_rows, yticklabels=['laugh', 'not laugh'], annot=show_annotations, cmap="YlGnBu")
plt.hist(startDiff)


hm.set_yticklabels(['laugh', 'not laugh'], size = 11)
hm.set_xticklabels(labels, size = 12)
plt.ylabel('predicted class', size=12)
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.tight_layout()
plt.gca().set_title('normalize by rows')
plt.subplot(212)
# hm = sns.heatmap(conf_ratio_by_cols, yticklabels=['laugh', 'not laugh'], annot=show_annotations, cmap="YlGnBu")
plt.hist(endDiff)
hm.set_yticklabels(['laugh', 'not laugh'], size = 11)
hm.set_xticklabels(labels, size = 12)
plt.ylabel('predicted class', size=12)
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.tight_layout(pad=3.0)
plt.gca().set_title('normalize by columns')

plot_file = os.path.join(cfg.ANALYSIS['plots_dir'], sub_dir, 'conf_matrix', f'{name}.png')
Path(plot_file).parent.mkdir(exist_ok=True, parents=True)
plt.savefig(plot_file)