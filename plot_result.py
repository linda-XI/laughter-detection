from visualise import plot_conf_matrix
from visualise import plot_conf_matrix2
input_dir = 'eval/eval_dw'
plot_conf_matrix(input_dir,'test',thresholds=[0.2,0.4,0.6,0.8],min_len=0.2, name = 'res_dw18_test')
plot_conf_matrix(input_dir,'dev',thresholds=[0.2,0.4,0.6,0.8],min_len=0.2, name = 'res_dw18_dev')


