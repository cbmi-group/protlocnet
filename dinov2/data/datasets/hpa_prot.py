from typing import Any, Callable, Optional, Tuple
from enum import Enum
import os
import numpy as np
import pandas as pd

from torchvision.transforms import Compose
from torchvision.datasets import ImageFolder

import logging
logger = logging.getLogger("dinov2")


PROTEIN_LOCALIZATION = [
    'Plasma membrane',  # 0
    'Centrosome',
    'Nucleoli rim',
    'Nuclear bodies',
    'Microtubule ends',
    'Nucleoli',
    'Lysosomes',
    'Peroxisomes',
    'Centriolar satellite',
    'Actin filaments',
    'Cell Junctions', # 10
    'Nucleoli fibrillar center',
    'Mitotic chromosome',
    'Lipid droplets',
    'Endoplasmic reticulum',
    'Microtubules',
    'Cytoplasmic bodies',
    'Cytokinetic bridge',
    'Mitochondria',
    'Mitotic spindle',
    'Cytosol',  # 20
    'Rods & Rings',
    'Focal adhesion sites',
    'Intermediate filaments',
    'Nuclear speckles',
    'Nuclear membrane',
    'Kinetochore',
    'Golgi apparatus',
    'Endosomes',
    'Vesicles',
    'Midbody ring', # 30
    'Midbody',
    'Nucleoplasm',
    'Aggresome',
] # 34 classes


PROTEIN_LOCALIZATION_19 = [
    'Negative', # 0
    'Nucleoplasm',
    'Nuclear membrane',
    'Nucleoli',
    'Nucleoli fibrillar center',
    'Nuclear speckles', # 5
    'Nuclear bodies',
    'Endoplasmic reticulum',
    'Golgi apparatus',
    # 'Intermediate filaments',
    'Actin filaments', # 10
    'Microtubules',
    # 'Mitotic spindle',
    'Centrosome',
    'Plasma membrane',
    'Mitochondria', # 15
    # 'Aggresome',
    'Cytosol',
    'Vesicles and punctate cytosolic patterns', # 18
]


NAME_MAPPING_34_TO_19 = {
  'Nucleoplasm': 'Nucleoplasm',
  'Nuclear membrane': 'Nuclear membrane',
  'Nucleoli': 'Nucleoli',
  'Nucleoli fibrillar center': 'Nucleoli fibrillar center',
  'Nuclear speckles': 'Nuclear speckles',
  'Nuclear bodies': 'Nuclear bodies',
  'Endoplasmic reticulum': 'Endoplasmic reticulum',
  'Golgi apparatus': 'Golgi apparatus',
  # 'Intermediate filaments': 'Intermediate filaments',
  'Actin filaments': 'Actin filaments',
  'Focal adhesion sites': 'Actin filaments',
  'Microtubules': 'Microtubules',
  # 'Mitotic spindle': 'Mitotic spindle',
  'Centrosome': 'Centrosome',
  'Centriolar satellite': 'Centrosome',
  'Plasma membrane': 'Plasma membrane',
  'Cell Junctions': 'Plasma membrane',
  'Mitochondria': 'Mitochondria',
  # 'Aggresome': 'Aggresome',
  'Cytosol': 'Cytosol',
  'Vesicles': 'Vesicles and punctate cytosolic patterns',
  'Peroxisomes': 'Vesicles and punctate cytosolic patterns',
  'Endosomes': 'Vesicles and punctate cytosolic patterns',
  'Lysosomes': 'Vesicles and punctate cytosolic patterns',
  'Lipid droplets': 'Vesicles and punctate cytosolic patterns',
  'Cytoplasmic bodies': 'Vesicles and punctate cytosolic patterns'
}


OPENCELL_LOCALIZATION = [
    'nucleoplasm', # 0
    'vesicles',
    'nucleolus_gc',
    'cytoskeleton',
    'golgi',
    'big_aggregates', # 5
    'cytoplasmic',
    'nuclear_membrane',
    'focal_adhesions',
    'er',
    'chromatin', # 10
    'nuclear_punctae',
    'nucleolus_fc_dfc',
    'membrane',
    'cell_contact',
    'centrosome', # 15
    'mitochondria',
]


class _Mode(Enum):
  PROTEIN_LOCALIZATION = "protein_localization"
  PROTEIN_LOCALIZATION_19 = "protein_localization_19"
  OPENCELL_LOCALIZATION = "opencell_localization"
  PROTEIN_TYPE = "protein_type"


class _WildCard(Enum):
    NONE = "none"
    SEPARATECHANNELS = "separate_channels"  # each channel from each image is treated as an independent sample, overrides chosen channel configuration


class SubcellLocTransform:
  def __init__(self, root, classes, mode=_Mode.PROTEIN_LOCALIZATION):
    self.ensg_locs = {}
    locations = pd.read_csv(os.path.join(root, 'subcellulars.txt'), header=0)
    LOCALIZATION = (
        PROTEIN_LOCALIZATION if mode == _Mode.PROTEIN_LOCALIZATION.value else
        PROTEIN_LOCALIZATION_19 if mode == _Mode.PROTEIN_LOCALIZATION_19.value else
        OPENCELL_LOCALIZATION if mode == _Mode.OPENCELL_LOCALIZATION.value else
        None
    )
    if LOCALIZATION is None:
      raise ValueError(f"Unsupported mode: {mode}. Supported modes are: {[m.value for m in _Mode]}")

    for _, row in locations.iterrows():
      ensg = row['Ensembl']
      locs = row['Subcellular location'].strip().split(';')
      locs_onehot = np.zeros(len(LOCALIZATION), dtype=np.int32)
 
      for loc in locs:
        if mode == _Mode.PROTEIN_LOCALIZATION_19.value:
          loc = NAME_MAPPING_34_TO_19.get(loc, 'Negative')  # Map to 19 classes, default to 'Negative' if not found

        if loc in LOCALIZATION:
          loc_idx = LOCALIZATION.index(loc)
          locs_onehot[loc_idx] = 1
        else:
          logger.warning(f"Unknown subcellular location '{loc}' for Ensembl ID '{ensg}'. Skipping.")
      self.ensg_locs[ensg] = locs_onehot
      logger.info(f"Loaded subcellular locations for Ensembl ID '{ensg}': {locs_onehot}")
    self.classes = classes

  def __call__(self, target):
    return self.ensg_locs[self.classes[target]]

  def __repr__(self) -> str:
    return f"{self.__class__.__name__}()"


class ProtHPADataset(ImageFolder):
  def __init__(
      self,
      root: str,
      split: str,
      mode: _Mode = _Mode.PROTEIN_LOCALIZATION,
      wildcard: _WildCard = _WildCard.NONE,
      transform: Optional[Callable] = None,
      target_transform: Optional[Callable] = None
  ):
    image_folder = os.path.join(root, split)
    classes, _ = self.find_classes(image_folder)
    self._targets = np.array(list(range(len(classes))))
    self.mode = mode
    self.wildcard = wildcard
    logger.info(f"Initialized ProtHPADataset with mode {self.mode}, wildcard {self.wildcard}, and {len(classes)} classes.")

    if mode in [_Mode.PROTEIN_LOCALIZATION.value, _Mode.PROTEIN_LOCALIZATION_19.value, _Mode.OPENCELL_LOCALIZATION.value]:
      sub_loc_trans = SubcellLocTransform(root, classes, mode=mode)
      if target_transform is not None:
        target_transform = Compose([sub_loc_trans, target_transform])
      else:
        target_transform = sub_loc_trans
    elif mode == _Mode.PROTEIN_TYPE.value:
      print("Using protein type mode for ProtHPADataset.")

    super().__init__(
        image_folder,
        transform,
        target_transform
    )

    if wildcard == _WildCard.SEPARATECHANNELS or wildcard == "SEPARATE_CHANNELS":
      # repeat the dataset for each channel and adjust the targets accordingly
      self.samples = self.samples * 3
      self.targets = np.repeat(self.targets, 3)
      logger.info(f"Applied separate channels wildcard. Dataset now has {len(self.samples)} samples.")

  def __getitem__(self, index: int) -> Tuple[Any, Any]:
    if self.wildcard == _WildCard.SEPARATECHANNELS or self.wildcard == "SEPARATE_CHANNELS":
      # adjust the index to account for the repeated samples
      channel_idx = index % 3
      base_index = index // 3
      path, target = self.samples[base_index]
      sample = self.loader(path)
      sample = np.array(sample)  # convert to numpy array for channel selection
      sample = sample[:, :, channel_idx:channel_idx+1]  # select the appropriate channel
      if self.transform is not None:
          sample = self.transform(sample)
      if self.target_transform is not None:
          target = self.target_transform(target)
      return sample, target
    else:
      return super().__getitem__(index)

  def get_targets(self):
    if self.mode == _Mode.PROTEIN_LOCALIZATION.value:
      return np.arange(len(PROTEIN_LOCALIZATION), dtype=np.int32)
    elif self.mode == _Mode.PROTEIN_LOCALIZATION_19.value:
      return np.arange(len(PROTEIN_LOCALIZATION_19), dtype=np.int32)
    elif self.mode == _Mode.OPENCELL_LOCALIZATION.value:
      return np.arange(len(OPENCELL_LOCALIZATION), dtype=np.int32)
    return np.array(self._targets, dtype=np.int32)
