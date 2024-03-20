import json
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn import metrics
from sklearn.metrics import auc


true_path = '/content/drive/MyDrive/MINFsample/test_laugh_only_df.csv'
laugh_df = pd.read_csv(true_path)
pres = []
acts = []

base = '/content/drive/MyDrive/MINFsample'
for meet in ['Bns001']:
  subpath = os.path.join(base,meet)
  files = os.listdir(subpath)
  # chan_list = []
  for f in files:#iterate each channel
    # chan_list.append((f.split('.')[0].split('s')[-1]))
    subpress = []
    chan = (f.split('.')[0].split('s')[-1])
    fullpath = os.path.join(subpath,f)
    print(fullpath)
    with open(fullpath, 'r') as f:
      subpress = json.load(f)

    subact = [0] * len(subpress)
    subdf = laugh_df[(laugh_df['meeting_id'] == meet) & (laugh_df['chan_id'] == chan)]
    for index_true, row_true in subdf.iterrows(): 
      laugh_start_frame = int(row_true['start']*100)
      laugh_end_frame = int(row_true['end']*100)
      for i in range(laugh_start_frame, laugh_end_frame):
        subact[i] = 1
    pres = pres + subpress
    acts = acts + subact
    act = np.array(acts)

    
pre = np.array(pres)
FPR, TPR, thresholds = metrics.roc_curve(act, pre)
AUC = auc(FPR, TPR)
print('AUC:',AUC)
plt.plot(FPR,TPR,label="AUC={:.2f}" .format(AUC),marker = 'o',color='b',linestyle='--')
plt.legend(loc=4, fontsize=10)
plt.title('ROC curve',fontsize=20)
plt.xlabel('FPR',fontsize=14)
plt.ylabel('TPR',fontsize=14)
plt.show()
plt.clf()
