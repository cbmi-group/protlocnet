import os
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torchvision.datasets import ImageFolder

from dinov2.utils.config import setup
from dinov2.utils.utils import load_pretrained_weights
from dinov2.models import build_model_from_cfg
from dinov2.eval.reconstruction.model import ReconstructionModel
from dinov2.data.transforms import make_normalize_transform, MaybeToTensor

def setup_and_build_model(args, pretrained_weights=None):
  cudnn.benchmark = True
  config = setup(args)
  model, _ = build_model_from_cfg(config, only_teacher=True)
  if pretrained_weights is not None:
    load_pretrained_weights(model, pretrained_weights, "teacher")
  # freeze the backbone and only finetune the head
  model.eval()
  model = ReconstructionModel(backbone=model)
  model.cuda()
  return model


class HPADataset(ImageFolder):
  def __init__(
      self,
      root,
      crop_size: int = 224,
      split: str = 'train',
      interpolation=transforms.InterpolationMode.BICUBIC,
  ):
    super().__init__(root=os.path.join(root, split))
    if split == 'train':
      self.transform_resize = transforms.Compose([
          transforms.RandomResizedCrop(crop_size, interpolation=interpolation),
          MaybeToTensor(),
          transforms.RandomHorizontalFlip(p=0.5),
      ])
      self.transform_norm = make_normalize_transform()
    else:
      self.transform_resize = transforms.Compose([
          transforms.Resize((crop_size, crop_size), interpolation=interpolation),
          MaybeToTensor(),
      ])
      self.transform_norm = make_normalize_transform()

  def __getitem__(self, index):
    path, target = self.samples[index]
    sample = self.loader(path)
    sample = self.transform_resize(sample)
    target = sample.clone()  # Clone the sample to use as the target
    sample = self.transform_norm(sample)
    return sample, target
