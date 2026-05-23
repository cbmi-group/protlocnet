import os
from typing import Optional
import numpy as np
import pandas as pd
from PIL import Image

import torch.utils.data as data
from torchvision import transforms

from dinov2.data.transforms import (
  MaybeToTensor,
  RandomBrightnessExceptSegmentation,
  RandomContrastExceptSegmentation,
  make_normalize_transform,
)


def pil_loader(path):
  return Image.open(path)


class HPAPairedDataset(data.Dataset):
  def __init__(
      self,
      data_root,
      pair_list_csv: Optional[str] = None,
      image_size=[224, 224],
      interpolation=transforms.InterpolationMode.BICUBIC,
      loader=pil_loader,
  ):
    super().__init__()
    self.image_size = (image_size[0], image_size[1])
    self.random_pair = (pair_list_csv is None)

    if pair_list_csv is not None:
      # paired dataset from csv list
      self.pairs = []
      df = pd.read_csv(pair_list_csv)
      for _, row in df.iterrows():
        row_id = row['Pair_ID']
        path_A = os.path.join(data_root, row['Image_A_Path'])
        path_B = os.path.join(data_root, row['Image_B_Path'])
        self.pairs.append((row_id, path_A, path_B))
    else:
      self.classes = sorted(os.listdir(data_root))
      self.class_to_images = {}
      for cls in self.classes:
        cls_path = os.path.join(data_root, cls)
        if os.path.isdir(cls_path):
          self.class_to_images[cls] = [os.path.join(cls_path, img) for img in os.listdir(cls_path)]
      self.images = [img for imgs in self.class_to_images.values() for img in imgs]

    if self.random_pair:
      self.tfs = transforms.Compose([
          transforms.RandomRotation(degrees=(0, 180)),
          transforms.RandomResizedCrop(
              image_size, scale=(0.32, 1.6), interpolation=interpolation
          ),
          MaybeToTensor(),
          RandomBrightnessExceptSegmentation(),
          RandomContrastExceptSegmentation(),
      ])
    else:
      self.tfs = transforms.Compose([
          transforms.Resize(image_size, interpolation=interpolation),
          MaybeToTensor(),
      ])
    self.norm = make_normalize_transform()
    self.loader = loader
    self.image_size = image_size

  def __len__(self):
    if self.random_pair:
      return len(self.images)
    else:
      return len(self.pairs)

  def __getitem__(self, i):
    if self.random_pair:
      # random pair from same class
      path_A = self.images[i]
      cls = os.path.basename(os.path.dirname(path_A))
      images = [p for p in self.class_to_images[cls] if p != path_A]
      path_B = np.random.choice(images)
      row_id = os.path.basename(path_A).split('.')[0]
    else:
      row_id, path_A, path_B = self.pairs[i]

    dst_image = self.tfs(self.loader(path_A))
    src_image = self.tfs(self.loader(path_B))
    ground_truth = dst_image[0].clone()

    ret = {}
    ret['reference'] = self.norm(src_image)
    ret['target'] = self.norm(dst_image)
    ret['ground_truth'] = ground_truth
    ret['path'] = f'{row_id}.png'
    return ret