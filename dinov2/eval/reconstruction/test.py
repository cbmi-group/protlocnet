import os
import sys
import argparse
from PIL import Image

import tqdm
import torch

from dinov2.eval.reconstruction.utils import setup_and_build_model, HPADataset

import logging
logger = logging.getLogger("dinov2")

def get_args_parser():
  parser = argparse.ArgumentParser("ProtLocNet reconstruction test.")
  parser.add_argument("--pretrained-weights", type=str, required=True, help="Path to the pretrained weights")
  parser.add_argument("--output-dir", type=str, default="./test_dir", help="Directory to save predictions")
  parser.add_argument("opts", help="Extra configuration options", default=[], nargs=argparse.REMAINDER)
  # data specific arguments
  parser.add_argument("--config-file", type=str, help="Model configuration file")
  parser.add_argument("--data-root", type=str, required=True, help="Root directory of the dataset")
  return parser


def save_image(tensor, path):
  image = tensor.cpu().numpy().transpose(1, 2, 0) * 255
  image = Image.fromarray(image.astype('uint8')).convert("RGB")
  image.save(path)


def do_test(model, data_root, output_dir):
  dataset = HPADataset(root=data_root, split="test")

  for f in ['p_pred', 'c_pred', 'real']:
    os.makedirs(os.path.join(output_dir, f), exist_ok=True)

  for i, (image, real) in enumerate(tqdm.tqdm(dataset, desc="Testing")):
    image = image.cuda(non_blocking=True)
    real = real.cuda(non_blocking=True)
    with torch.no_grad():
      p_pred, c_pred = model(image[None, ...])  # Add batch dimension
    save_image(p_pred.squeeze(0), os.path.join(output_dir, 'p_pred', f"{i:05d}.png"))
    save_image(c_pred.squeeze(0), os.path.join(output_dir, 'c_pred', f"{i:05d}.png"))
    save_image(real, os.path.join(output_dir, 'real', f"{i:05d}.png"))
  print(f"Saved predictions to {output_dir}")


def main(args):
  model = setup_and_build_model(args)
  state_dict = torch.load(args.pretrained_weights, map_location="cpu")
  msg = model.load_state_dict(state_dict['model'], strict=False)
  logger.info("Pretrained weights found at {} and loaded with msg: {}".format(args.pretrained_weights, msg))
  model.eval()
  do_test(model, args.data_root, args.output_dir)


if __name__ == "__main__":
  args = get_args_parser().parse_args()
  sys.exit(main(args))
