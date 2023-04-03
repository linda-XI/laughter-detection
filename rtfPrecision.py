from matplotlib import pyplot as plt
import pandas as pd
from scipy.interpolate import BSpline, make_interp_spline
import numpy as np
import config as cfg
import os
import seaborn as sns
from pathlib import Path

x1=[0.001563, 0.00315, 0.00609, 0.006183, 0.0042, 0.001063, 0.002702, 0.001433, 0.001175, 0.002094, 0.026296, 0.03346]
y1=[0.56, 0.55, 0.6, 0.6, 0.587, 0.59, 0.54, 0.551, 0.433, 0.455, 0.56, 0.557]
# x2=[31,52,73,92,101,112,126,140,153,175,186,196,215,230,240,270,288,300]
# y2=[48,48,48,48,49,89,162,237,302,378,443,472,522,597,628,661,690,702]
# x3=[30,50,70,90,105,114,128,137,147,159,170,180,190,200,210,230,243,259,284,297,311]
# y3=[48,48,48,48,66,173,351,472,586,712,804,899,994,1094,1198,1360,1458,1578,1734,1797,1892]
# x=np.arange(20,350)
l1=plt.plot(x1,y1,'r--',label='type1')
# l2=plt.plot(x2,y2,'g--',label='type2')
# l3=plt.plot(x3,y3,'b--',label='type3')
# plt.plot(x1,y1,'ro-',x2,y2,'g+-',x3,y3,'b^-')
plt.plot(x1,y1,'ro-')
plt.title('CPU rtf vs F1')
plt.xlabel('CPU rtf')
plt.ylabel('F1')
plt.legend()
plt.tight_layout()
plot_file = os.path.join('./plots', 'rtfF1' )
Path(plot_file).parent.mkdir(exist_ok=True, parents=True)
plt.savefig(plot_file)
