#!/usr/bin/env bash
python train.py --config resnet_base --checkpoint_dir ./checkpoints/resnet_tst --data_root ./ --lhotse_dir ./raw_output  --data_dfs_dir ./data/icsi/data_dfs --num_epochs 40 && python segment_laughter.py --config resnet_base --save_to_textgrid=True --save_to_audio_files=False --input_dir=./data/icsi/speech/ --audio_names Bmr021,Bns001,Bmr013,Bmr018,Bro021 --output_dir=./data/eval_output --model_path=./checkpoints/resnet_tst --thresholds=0.2,0.4,0.6,0.8 --min_lengths=0,0.1,0.2