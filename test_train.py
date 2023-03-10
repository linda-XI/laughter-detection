import load_data
import numpy as np
from lhotse import CutSet, Fbank, FbankConfig, Recording, MonoCut
cutset_dir  = './raw_output/cutsets'

dev_loader = load_data.create_inference_dataloader('./data/icsi/speech/Bmr006/chan0.sph')
val = iter(dev_loader)
batch = val.next()
arr = np.array(batch)
rec = Recording.from_file('./data/icsi/speech/Bmr021/chan0.sph')
print(rec.duration)
print(arr.shape)
#segs = batch['inputs']
#labs=batch['is_laugh']
#print(segs)
#print(labs)
print(len(batch))
