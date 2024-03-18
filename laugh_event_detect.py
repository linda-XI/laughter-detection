import load_data
from tqdm import tqdm
import torch
import argparse
import config
import os
import torch_utils
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str,
                    default='checkpoints/in_use/resnet_with_augmentation')
parser.add_argument('--config', type=str, default='resnet_with_augmentation')
parser.add_argument('--output_dir', type=str, default='output/0802_output')
args = parser.parse_args()


config = config.MODEL_MAP[args.config]
model_path = args.model_path

cutset_dir = args.output_dir
print(cutset_dir)
test_loader = load_data.create_training_dataloader(cutset_dir, 'test', shuffle=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device {device}")

# Load the Model
if args.config.startswith('mobile'):
    model = config['model'](dropout_rate=0.0,
                            linear_layer_size=config['linear_layer_size'], filter_sizes=config['filter_sizes'],
                            inverted_residual_setting=config['inverted_residual_setting'])
else:
    model = config['model'](
    dropout_rate=0.0, linear_layer_size=config['linear_layer_size'], filter_sizes=config['filter_sizes'])
model.set_device(device)

if os.path.exists(model_path):
    # if device == 'cuda':
    if torch.cuda.is_available():
        print(model_path + '/best.pth.tar')
        torch_utils.load_checkpoint(model_path + '/best.pth.tar', model)
    else:
        # Different method needs to be used when using CPU
        # see https://pytorch.org/tutorials/beginner/saving_loading_models.html for details
        checkpoint = torch.load(
            model_path + '/best.pth.tar', lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
    model.eval()
else:
    raise Exception(f"Model checkpoint not found at {model_path}")

def _calc_metrics(trgs, preds):
    '''
    Calculates accuracy, precision and recall and returns them in that order
    '''
    acc = torch.sum(preds == trgs).float()/len(trgs)

    # Calculate necessary numbers for prec and recall calculation
    # '==' operator on tensors is applied element-wise
    # '*' exploits the fact that True*True = 1
    corr_pred_laughs = torch.sum((preds == trgs) * (preds == 1)).float()
    total_trg_laughs = torch.sum(trgs == 1).float()
    total_pred_laughs = torch.sum(preds == 1).float()

    total_trg_non_laughs = torch.sum(trgs == 0).float()

    # if total_pred_laughs == 0:
    #     prec = torch.tensor(1.0) 
    # else:
    #     prec = corr_pred_laughs/total_pred_laughs

    # recall = corr_pred_laughs/total_trg_laughs

    # # Returns only the content of the torch tensor
    return corr_pred_laughs.item(), total_trg_laughs.item(), total_pred_laughs.item()

corr_pred_laughs = 0
total_trg_laughs = 0
total_pred_laughs = 0
for i, batch in tqdm(enumerate(test_loader)):
    with torch.no_grad():
            #seqs, labs = batch
            segs = batch['inputs']
            labs = batch['is_laugh']

            src = torch.from_numpy(np.array(segs)).float().to(device)
            src = src[:, None, :, :]  # add additional dimension

            trgs = torch.from_numpy(np.array(labs)).float().to(device)
            output = model(src).squeeze()
            print(output)
            print(output.shape)
            print(labs)

            preds = torch.round(output)
            # sum(preds==trg).float()/len(preds)

            # Allows to evaluate several batches together for logging
            # Used to avoid lots of precision=1 because no predictions were made
            

            corr_pred_laughs_batch, total_trg_laughs_batch, total_pred_laughs_batch = _calc_metrics(trgs, preds)
            corr_pred_laughs +=  corr_pred_laughs_batch
            total_trg_laughs += total_trg_laughs_batch
            total_pred_laughs += total_pred_laughs_batch

prec = corr_pred_laughs/total_pred_laughs
recall = corr_pred_laughs/total_trg_laughs

print("prec is {} ; recall is {} ".format(prec, recall))