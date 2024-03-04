import sys, numpy as np
sys.path.append('./utils/')
import models
from functools import partial
from pathlib import Path

MODEL_MAP = {}

MODEL_MAP['resnet_base'] = {
    'batch_size': 32,
    'model': models.ResNetBigger,
    'val_data_text_path': './data/switchboard/val/switchboard_val_data.txt',
    'log_frequency': 900,
    'linear_layer_size': 48, # for new features of shape (40,100) 
    # 'linear_layer_size': 64, # original value for features of shape (44,128)
    'filter_sizes': [64,32,16,16],
}

MODEL_MAP['resnet_dw'] = {
    'batch_size': 32,
    'model': models.ResNetBigger_DW,
    'val_data_text_path': './data/switchboard/val/switchboard_val_data.txt',
    'log_frequency': 900,
    'linear_layer_size': 48, # for new features of shape (40,100) 
    # 'linear_layer_size': 64, # original value for features of shape (44,128)
    'filter_sizes': [64,32,16,16],
}

MODEL_MAP['resnet8_dw'] = {
    'batch_size': 32,
    'model': models.ResNet8_DW,
    'val_data_text_path': './data/switchboard/val/switchboard_val_data.txt',
    'log_frequency': 900,
    'linear_layer_size': 48, # for new features of shape (40,100)
    # 'linear_layer_size': 64, # original value for features of shape (44,128)
    'filter_sizes': [64,32,16,16],
}

MODEL_MAP['resnet18_dw'] = {
    'batch_size': 32,
    'model': models.ResNet18_DW,
    'val_data_text_path': './data/switchboard/val/switchboard_val_data.txt',
    'log_frequency': 900,
    'linear_layer_size': 48, # for new features of shape (40,100)
    # 'linear_layer_size': 64, # original value for features of shape (44,128)
    'filter_sizes': [64,128,256,512],
}

MODEL_MAP['resnet18_small_dw'] = {
    'batch_size': 32,
    'model': models.ResNet18_DW,
    'val_data_text_path': './data/switchboard/val/switchboard_val_data.txt',
    'log_frequency': 900,
    'linear_layer_size': 48, # for new features of shape (40,100)
    # 'linear_layer_size': 64, # original value for features of shape (44,128)
    'filter_sizes': [64,32,16,16],
}


MODEL_MAP['resnet8_baseline'] = {
    'batch_size': 32,
    'model': models.ResNet8_baseline,
    'val_data_text_path': './data/switchboard/val/switchboard_val_data.txt',
    'log_frequency': 900,
    'linear_layer_size': 960, # for new features of shape (40,100)
    # 'linear_layer_size': 64, # original value for features of shape (44,128)
    'filter_sizes': [32,16,16,16],
}

MODEL_MAP['resnet10'] = {
    'batch_size': 32,
    'model': models.ResNet10,
    'val_data_text_path': './data/switchboard/val/switchboard_val_data.txt',
    'log_frequency': 900,
    'linear_layer_size': 48, # for new features of shape (40,100)
    # 'linear_layer_size': 64, # original value for features of shape (44,128)
    # use to change width of model
    'filter_sizes': [64,128,256,512],
}

MODEL_MAP['resnet18'] = {
    'batch_size': 32,
    'model': models.ResNet18,
    'val_data_text_path': './data/switchboard/val/switchboard_val_data.txt',
    'log_frequency': 900,
    'linear_layer_size': 48, # for new features of shape (40,100)
    # 'linear_layer_size': 64, # original value for features of shape (44,128)
    # use to change width of model
    'filter_sizes': [64,128,256,512],
}

MODEL_MAP['resnet18_small'] = {
    'batch_size': 32,
    'model': models.ResNet18,
    'val_data_text_path': './data/switchboard/val/switchboard_val_data.txt',
    'log_frequency': 900,
    'linear_layer_size': 48, # for new features of shape (40,100)
    # 'linear_layer_size': 64, # original value for features of shape (44,128)
    # use to change width of model
    'filter_sizes': [64,32,16,16],
}

MODEL_MAP['resnet18_big'] = {
    'batch_size': 32,
    'model': models.ResNet18,
    'val_data_text_path': './data/switchboard/val/switchboard_val_data.txt',
    'log_frequency': 900,
    'linear_layer_size': 48, # for new features of shape (40,100)
    # 'linear_layer_size': 64, # original value for features of shape (44,128)
    # use to change width of model
    'filter_sizes': [64,128,512,512],
}

MODEL_MAP['resnet32'] = {
    'batch_size': 32,
    'model': models.ResNet32,
    'val_data_text_path': './data/switchboard/val/switchboard_val_data.txt',
    'log_frequency': 900,
    'linear_layer_size': 48, # for new features of shape (40,100)
    # 'linear_layer_size': 64, # original value for features of shape (44,128)
    'filter_sizes': [64,32,16,16],
}

MODEL_MAP['resnet50'] = {
    'batch_size': 32,
    'model': models.ResNet50,
    'val_data_text_path': './data/switchboard/val/switchboard_val_data.txt',
    'log_frequency': 900,
    'linear_layer_size': 48, # for new features of shape (40,100)
    # 'linear_layer_size': 64, # original value for features of shape (44,128)
    'filter_sizes': [64,32,16,16],
}

MODEL_MAP['resnet_with_augmentation'] = {
    'batch_size': 32,
    'model': models.ResNetBigger,
    'val_data_text_path': './data/switchboard/val/switchboard_val_data.txt',
    'log_frequency': 200,
    'linear_layer_size': 128,
    'filter_sizes': [128,64,32,32],
}

MODEL_MAP['mobilenet_v2'] = {
    'batch_size': 32,
    'model': models.MobileNetV2,
    'val_data_text_path': './data/switchboard/val/switchboard_val_data.txt',
    'log_frequency': 200,
    'num_classes': 1,
    'width_mult': 1.0,
    'inverted_residual_setting': [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ],
    'round_nearest': 8,
    'block': None,
    'norm_layer': None,
    'dropout_rate': 0.2,
    #useless params
    'linear_layer_size': 128,
    'filter_sizes': [128,64,32,32],
}

MODEL_MAP['mobilenet_small_n'] = {
    'batch_size': 32,
    'model': models.MobileNetV2,
    'val_data_text_path': './data/switchboard/val/switchboard_val_data.txt',
    'log_frequency': 200,
    'num_classes': 1,
    'width_mult': 1.0,
    'inverted_residual_setting': [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 2, 2],
                [6, 64, 2, 2],
                [6, 96, 2, 1],
                [6, 160, 1, 2],
                [6, 320, 1, 1],
            ],
    'round_nearest': 8,
    'block': None,
    'norm_layer': None,
    'dropout_rate': 0.2,
    #useless params
    'linear_layer_size': 128,
    'filter_sizes': [128,64,32,32],
}

MODEL_MAP['mobilenet_small_t'] = {
    'batch_size': 32,
    'model': models.MobileNetV2,
    'val_data_text_path': './data/switchboard/val/switchboard_val_data.txt',
    'log_frequency': 200,
    'num_classes': 1,
    'width_mult': 1.0,
    'inverted_residual_setting': [
                # t, c, n, s
                [1, 16, 1, 1],
                [2, 24, 2, 2],
                [2, 32, 3, 2],
                [2, 64, 4, 2],
                [2, 96, 3, 1],
                [2, 160, 3, 2],
                [2, 320, 1, 1],
            ],
    'round_nearest': 8,
    'block': None,
    'norm_layer': None,
    'dropout_rate': 0.2,
    #useless params
    'linear_layer_size': 128,
    'filter_sizes': [128,64,32,32],
}

MODEL_MAP['efficientnet_b0'] = {
    'batch_size': 32,
    'model': models.EfficientNet_B0,
    'val_data_text_path': './data/switchboard/val/switchboard_val_data.txt',
    'log_frequency': 900,

    'dropout': 0.2,
    'stochastic_depth_prob':  0.2,
    'num_classes':  1,

    #un-use params
    'linear_layer_size': 48, # for new features of shape (40,100)
    'filter_sizes': [64,32,16,16],
}

FEAT = {
    "num_samples": 100,
    "num_filters": 44
}

root_path = Path(__file__).absolute().parent
ANALYSIS= {
    "transcript_dir": str(root_path / 'data/icsi/transcripts'),
    "speech_dir": str(root_path / 'data/icsi/speech'),
    "plots_dir": 'plots',
    "eval_df_cache_file": "eval_df_per_meeting.csv",
    "eval_notLaugh_df_cache_file": "eval_notLaugh_df_per_meeting.csv",
    "sum_stats_cache_file": "sum_stats.csv",

    # Indices are loaded from disk if possible. This option forces re-computation 
    # If True analyse.py will take a lot longer
    "force_index_recompute": False,
    #store all the predict laugh from seed model 
    "extra_laugh_dir": str(root_path / 'seedModel/newPredTrain'),
    "test_df_dir": str(root_path / 'seedModel/extraDF'),
    # "extra_laugh_sample": str(root_path / 'sample/extra_laugh_sample'),
    "extra_laugh_sample": str(root_path / 'seedModel/extraLaughSample'),
    # adding extra laugh into laugh_only_df with given threshold and minLen
    "threshold": 0.6,
    "minLen": 0.0,
    # dataframe are loaded from disk if possible. This option forces re-computation 
    # If True analyse.py will take a lot longer
    "force_df_recompute": False,
    "cache_file" : ".cache/preprocessed_indices_0600.pkl"
}

ANALYSIS['model'] = {
    # Min-length used for parsing the transcripts
    "min_length": 0.2,

    # Frame duration used for parsing the transcripts
    "frame_duration": 1  # in ms
}

ANALYSIS['train'] = {
    # How long each sample for training should be 
    "subsample_duration": 1.0,  # in s
    "random_seed": 23,

    # Used in creation of train, val and test df in 'create_data_df'
    "float_decimals": 2,  # number of decimals to round floats to
    # Test uses the remaining fraction
    "train_val_test_split": [0.8, 0.1],
}
