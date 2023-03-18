from segment_laughter import load_and_pred
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = config.MODEL_MAP['resnet_dw']
model = config['model'](
            dropout_rate=0.0, linear_layer_size=config['linear_layer_size'], filter_sizes=config['filter_sizes'])
model.set_device(device)
model_path = './checkpoints/dw_fix_tst'
print(model_path)
if os.path.exists(model_path):
        # if device == 'cuda':
    if torch.cuda.is_available():
        print(model_path + '/best.pth.tar')
        torch_utils.load_checkpoint(model_path + '/best.pth.tar', model)
    else:
                                                # Different method needs to be used when using CPU
                                                        # see https://pytorch.org/tutorials/beginner/saving_loading_models.html for details
        checkpoint = torch.load(model_path + '/best.pth.tar', lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
    model.eval()
else:
    raise Exception(f"Model checkpoint not found at {model_path}")
load_and_pred('./data/icsi/speech/Bed002/chan0.sph','./eval_debug' )
