import load_data
cutset_dir  = './raw_output/cutsets'

dev_loader = load_data.create_training_dataloader(cutset_dir, 'dev')
val = iter(dev_loader)
batch = val.next()
segs = batch['inputs']
labs=batch['is_laugh']
#print(segs)
#print(labs)
print(batch)
