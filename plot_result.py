from visualise import plot_conf_matrix
from visualise import plot_conf_matrix2

plot_conf_matrix2('eval/eval_base_on_fix','dev',thresholds=[0.2,0.4,0.6,0.8],min_len=0.2, name = 'baselineonfix__conf_matrix_dev')
