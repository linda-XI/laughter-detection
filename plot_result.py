from visualise import plot_conf_matrix
from visualise import plot_conf_matrix2
input_dir = 'eval/320/dw'
plot_conf_matrix(input_dir,'test',thresholds=[0.2,0.4,0.6,0.8],min_len=0.2, name = 'dw', sub_dir = 'MINF2')
#plot_conf_matrix(input_dir,'dev',thresholds=[0.2,0.4,0.6,0.8],min_len=0.2, name = 'res_dw18_dev')


