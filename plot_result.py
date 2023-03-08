from visualise import plot_conf_matrix

plot_conf_matrix('data/eval_baseline','dev',thresholds=[0.2,0.4,0.6,0.8],min_len=0.2, name = 'baseline_conf_matrix', sub_dir = 'data/visual')
