from visualise import plot_conf_matrix
from visualise import plot_conf_matrix2
input_dirs = ['eval/320/resnet18_small_dw', 'eval/320/baseline', 'eval/320/resnet18_dw', 'eval/320/mobilenet', 'eval/320/resnet10', 'eval/320/resnet18_big']

for i in input_dirs:
    name = i.split('/')[-1]
    plot_conf_matrix(i,'test',thresholds=[0.2,0.4,0.6,0.8],min_len=0.2, name = name, sub_dir = 'MINF2')

#plot_conf_matrix(input_dir,'dev',thresholds=[0.2,0.4,0.6,0.8],min_len=0.2, name = 'res_dw18_dev')


