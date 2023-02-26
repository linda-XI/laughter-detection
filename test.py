from lhotse import CutSet, Fbank
from lhotse.dataset.input_strategies import PrecomputedFeatures, OnTheFlyFeatures
from lhotse.features.io import NumpyFilesWriter
from compute_features import create_manifest
from utils.utils import get_feat_extractor


#icsi_manifest = create_manifest('./data/icsi/speech', './data/icsi/transcripts','./manifest')
#split_cutset = CutSet.from_manifests(recordings=icsi_manifest['train']['recordings'])
#extractor = get_feat_extractor(num_samples=100, num_filters=44, use_kaldi=True)
#split_feat_cuts = split_cutset.compute_and_store_features_batch(
 #                       extractor=extractor,
  #                                      storage_path='./splitFeat',
   #                                                     num_workers=1
    #                                                                )
###############################

cuts = CutSet.from_file('splitFeat/cutsets/train_feats.jsonl')
cuts = cuts.subset(first=1)
cuts = cuts.drop_features()
cuts = cuts.compute_and_store_features(
  extractor=Fbank(),
  storage_path="test_feats",
  num_jobs=1,
)

input_strategy = PrecomputedFeatures()
for cut in cuts:
    print(cut, input_strategy([cut]))
