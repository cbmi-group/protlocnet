import argparse
import sys
from pathlib import Path

import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from dinov2.data.transforms import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from dinov2.eval.generation.data import HPAPairedDataset
from dinov2.models import build_model_from_cfg


class ConfigNode(dict):
  def __getattr__(self, item):
    try:
      return self[item]
    except KeyError as exc:
      raise AttributeError(item) from exc

  def __setattr__(self, key, value):
    self[key] = value


def to_config_node(value):
  if isinstance(value, dict):
    node = ConfigNode()
    for key, item in value.items():
      node[key] = to_config_node(item)
    return node
  if isinstance(value, list):
    return [to_config_node(item) for item in value]
  return value


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Generate protein images from paired HPA samples.")
  parser.add_argument("--data-root", type=str, required=True, help="Root directory containing the paired images.")
  parser.add_argument("--pair-list-csv", type=str, required=True, help="CSV with Pair_ID, Image_A_Path and Image_B_Path.")
  parser.add_argument("--checkpoint-dir", type=str, required=True, help="Directory containing teacher_checkpoint.pth and config.yaml.")
  parser.add_argument("--save-dir", type=str, required=True, help="Directory to save gen/ and gt/ outputs.")
  parser.add_argument("--batch-size", type=int, default=8, help="Inference batch size.")
  parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers.")
  return parser.parse_args()


def load_config_from_checkpoint_dir(checkpoint_dir: Path):
  config_path = checkpoint_dir / "config.yaml"
  if not config_path.is_file():
    raise FileNotFoundError(f"Missing config file: {config_path}")
  with open(config_path, "r", encoding="utf-8") as f:
    cfg_dict = yaml.safe_load(f)
  if not isinstance(cfg_dict, dict):
    raise RuntimeError(f"Invalid config format in {config_path}")
  return to_config_node(cfg_dict)


def denorm_first_channel(x: torch.Tensor) -> torch.Tensor:
  mean = torch.tensor(IMAGENET_DEFAULT_MEAN[0], device=x.device, dtype=x.dtype)
  std = torch.tensor(IMAGENET_DEFAULT_STD[0], device=x.device, dtype=x.dtype)
  return (x * std + mean).clamp(0.0, 1.0)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
  checkpoint = torch.load(checkpoint_path, map_location=device)
  if isinstance(checkpoint, dict) and "teacher" in checkpoint and isinstance(checkpoint["teacher"], dict):
    state_dict = checkpoint["teacher"]
  if isinstance(checkpoint, dict) and "student" in checkpoint and isinstance(checkpoint["student"], dict):
    state_dict = checkpoint["student"]
  elif isinstance(checkpoint, dict):
    if "model_state_dict" in checkpoint:
      state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
      state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
      state_dict = checkpoint["model"]
    else:
      state_dict = checkpoint
  else:
    state_dict = checkpoint

  model_state_dict = model.state_dict()
  cleaned_state_dict = {}
  for key, value in state_dict.items():
    clean_key = key[len("module."):] if key.startswith("module.") else key
    if clean_key.startswith("backbone."):
      clean_key = clean_key[len("backbone."):]
    if clean_key in model_state_dict and model_state_dict[clean_key].shape == value.shape:
      cleaned_state_dict[clean_key] = value

  load_result = model.load_state_dict(cleaned_state_dict, strict=True)
  if len(cleaned_state_dict) == 0:
    raise RuntimeError(f"No matching weights loaded from checkpoint: {checkpoint_path}")
  if load_result.missing_keys:
    print(f"Warning: missing keys when loading checkpoint: {len(load_result.missing_keys)}")
  if load_result.unexpected_keys:
    print(f"Warning: unexpected keys when loading checkpoint: {len(load_result.unexpected_keys)}")
  print(f"Loaded {len(cleaned_state_dict)} parameters from checkpoint: {checkpoint_path}")


def main() -> None:
  args = parse_args()
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  checkpoint_dir = Path(args.checkpoint_dir)
  checkpoint_path = checkpoint_dir / "checkpoint_epoch_10.pth"

  cfg = load_config_from_checkpoint_dir(checkpoint_dir)
  if "train" not in cfg:
    cfg.train = {}
  if not cfg.train.get("generation", False):
    cfg.train.generation = True

  image_size = int(cfg.crops.global_crops_size)

  dataset = HPAPairedDataset(
    data_root=args.data_root,
    pair_list_csv=args.pair_list_csv,
    image_size=[image_size, image_size],
  )
  loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=torch.cuda.is_available(),
  )

  model, _ = build_model_from_cfg(cfg, only_teacher=True)
  model = model.to(device)
  load_checkpoint(model, checkpoint_path, device)
  model.eval()

  save_dir = Path(args.save_dir)
  gen_dir = save_dir / "gen"
  gt_dir = save_dir / "gt"
  gen_dir.mkdir(parents=True, exist_ok=True)
  gt_dir.mkdir(parents=True, exist_ok=True)

  with torch.no_grad():
    for batch in tqdm(loader, desc="Generating"):
      reference = batch["reference"].to(device, non_blocking=True)
      target = batch["target"].to(device, non_blocking=True)
      gt = batch["ground_truth"].numpy() # numpy array, shape (B, H, W, C), uint8
      file_names = batch["path"]

      target_contour = target[:, 1:]
      pt = model.generate(reference, target_contour)
      pt = pt.permute(0, 2, 3, 1).cpu().numpy()
      pt = np.clip(pt * 255.0, 0, 255).astype("uint8") # numpy array, shape (B, H, W, C), uint8
      pt = pt[..., 0]
      gt = np.clip(gt * 255.0, 0, 255).astype("uint8") # ensure gt is uint8

      for index, file_name in enumerate(file_names):
        Image.fromarray(pt[index], mode='L').save(gen_dir / file_name)
        Image.fromarray(gt[index], mode='L').save(gt_dir / file_name)


if __name__ == "__main__":
  main()

