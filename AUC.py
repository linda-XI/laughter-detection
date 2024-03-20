import json
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn import metrics
from sklearn.metrics import auc


acts = [1,1,1,1,1,0,0,1,1,1]

pres = [1,0,1,1,1,1,0,1,1,0]

act = np.array(acts)
pre = np.array(pres)
FPR, TPR, thresholds = metrics.roc_curve(act, pre)
AUC = auc(FPR, TPR)
print('AUC:',AUC)
plt.rc('font', family='Arial Unicode MS', size=14)
plt.plot(FPR,TPR,label="AUC={:.2f}" .format(AUC),marker = 'o',color='b',linestyle='--')
plt.legend(loc=4, fontsize=10)
plt.title('ROC曲线',fontsize=20)
plt.xlabel('FPR',fontsize=14)
plt.ylabel('TPR',fontsize=14)