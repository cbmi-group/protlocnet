import os

import numpy as np

from tqdm import tqdm
from PIL import Image
from skimage import measure
from xgboost import XGBClassifier
from xgboost import XGBRegressor

from multiprocessing import Pool
from dinov2.eval.reconstruction.dci import (
    compute_importance_matrix,
    compute_disentanglement,
    compute_completeness,
)

DIR = '/home/wangt/dinov2/train_dir/protnet/training/prot_large+full'


def process_parameter(channel):
  props = measure.regionprops(measure.label(channel > 0.5))
  region = max(props, key=lambda x: x.area) if len(props) > 0 else None
  if region is not None:
    area = region.area
    eccentricity = region.eccentricity
  else:
    area = 0.0
    eccentricity = 0.0
  return area, eccentricity


def process_morphology(args):
  i, filename = args
  image = np.array(Image.open(filename).convert("RGB")).astype(np.float32) / 255.0
  nuc, seg = image[:, :, 1], image[:, :, 2]
  nuc_parameters = process_parameter(nuc)
  seg_parameters = process_parameter(seg)
  return i, np.array(nuc_parameters + seg_parameters, dtype=np.float32)


def main():
  print(f'Loading the embeddings from {DIR}')
  embeddings = np.load(os.path.join(DIR, 'embeddings.npz'))
  cls_tokens = embeddings['cls_tokens']
  localizations = embeddings['targets']
  filenames = embeddings['filenames']
  del embeddings

  # ensg id classifier
  ensgs = [f.split('/')[-2].strip() for f in filenames]
  unique_ensgs = list(set(ensgs))
  ensgids = np.array([unique_ensgs.index(e) for e in ensgs], dtype=np.int32)
  ensgids = ensgids.reshape(-1, 1)  # reshape to (N, 1) for DCI computation

  # localizations
  zero_cls = np.all(localizations == 0, 0)
  localizations = localizations[:, ~zero_cls]

  # area, eccentricity, and solidity
  morphology_labels = np.zeros((len(filenames), 4), dtype=np.float32)
  with Pool(processes=8) as pool:
    results = pool.imap(process_morphology, [(i, filename) for i, filename in enumerate(filenames)])
    for i, result in tqdm(results, total=len(filenames), desc='Processing morphology'):
      morphology_labels[i, :] = result    

  shapes = [ensgids.shape[1], localizations.shape[1]] + [1] * morphology_labels.shape[1]
  labels = np.concat([ensgids, localizations, morphology_labels], axis=1)
  
  # prepare the model and args
  model_cls = [XGBClassifier] + [XGBClassifier] * (localizations.shape[1]) + [XGBRegressor] * (morphology_labels.shape[1])
  ensg_args = {
      'objective': 'multi:softprob',
      'num_class': len(unique_ensgs),
      'eval_metric': 'merror',
      'importance_type': 'gain',
      'tree_method': 'hist',
  }
  loc_args = {
      'objective': 'binary:logistic',
      'eval_metric': 'logloss',
      'importance_type': 'gain',
      'tree_method': 'hist',
  }
  mor_args = {
      'objective': 'reg:squarederror',
      'eval_metric': 'rmse',
      'importance_type': 'gain',
      'tree_method': 'hist',
  }
  model_args = [ensg_args] + [loc_args] * (localizations.shape[1]) + [mor_args] * (morphology_labels.shape[1])
  r = compute_importance_matrix(
      cls_tokens, labels, test_ratio=0.3, random_state=42,
      model_cls=model_cls, model_args=model_args,
  )
  importance_matrix = r['importance_matrix']

  # grouping the importance matrix into protein, morphology, and localization factors
  print(f'Importance matrix shape: {importance_matrix.shape}')
  # importance_matrix = np.concatenate([
  #     importance_matrix[:, 0:1],  # protein
  #     importance_matrix[:, 1:1+localizations.shape[1]].mean(axis=1, keepdims=True),  # localization
  #     importance_matrix[:, 1+localizations.shape[1]:],  # morphology
  # ], axis=1)
  
  print(f'Computing disentanglement score')
  dzp = compute_disentanglement(importance_matrix[:512])
  dzm = compute_disentanglement(importance_matrix[512:])
  print(f'Disentanglement score for protein: {dzp:.4f}, for morphology: {dzm:.4f}')

  czp = compute_completeness(importance_matrix[:512])
  czm = compute_completeness(importance_matrix[512:])
  print(f'Completeness score for protein: {czp:.4f}, for morphology: {czm:.4f}')

  weighted_importance_matrix = np.asarray(importance_matrix, dtype=np.float64).copy()
  weighted_importance_matrix = weighted_importance_matrix / (weighted_importance_matrix.sum(axis=0, keepdims=True) + 1e-12)
  weights = np.zeros((importance_matrix.shape[1],), dtype=np.float64)
  weights[0] = 1.0 / 3.0
  weights[1:1+localizations.shape[1]] = 1.0 / 3.0 / localizations.shape[1]
  weights[1+localizations.shape[1]:] = 1.0 / 3.0 / morphology_labels.shape[1]
  weighted_importance_matrix = weighted_importance_matrix * weights[None, :]
  
  print(f'Computing weighted disentanglement score')
  dzp = compute_disentanglement(weighted_importance_matrix[:512])
  dzm = compute_disentanglement(weighted_importance_matrix[512:])
  print(f'Weighted disentanglement score for protein: {dzp:.4f}, for morphology: {dzm:.4f}')
  czp = compute_completeness(weighted_importance_matrix[:512])
  czm = compute_completeness(weighted_importance_matrix[512:])
  print(f'Weighted completeness score for protein: {czp:.4f}, for morphology: {czm:.4f}')
  
  # factor group attribution
  p = importance_matrix[:, 0]
  l = importance_matrix[:, 1:1+localizations.shape[1]].mean(axis=1)
  m = importance_matrix[:, 1+localizations.shape[1]:].mean(axis=1)
  plm = [p, l, m]
  for i, name in enumerate(['protein', 'localization', 'morphology']):
    print(f'Factor group {name} importance: {plm[i][:512].sum():.4f}, {plm[i][512:].sum():.4f}')


if __name__ == "__main__":
  main()
