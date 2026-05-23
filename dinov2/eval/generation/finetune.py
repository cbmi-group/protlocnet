from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Dict, Iterable, Optional

import torch
from torch import nn

import dinov2.distributed as distributed
from dinov2.data import SamplerType, make_data_loader
from dinov2.utils.config import setup
from dinov2.utils.utils import load_pretrained_weights
from dinov2.models import build_model_from_cfg
from dinov2.eval.generation.data import HPAPairedDataset
from dinov2.logging import MetricLogger

torch.backends.cuda.matmul.allow_tf32 = True
logger = logging.getLogger("generation")


def get_args_parser():
  parser = argparse.ArgumentParser("ProtLocNet generation finetune")
  parser.add_argument("--pretrained-weights", type=str, required=True, help="Path to the pretrained weights")
  parser.add_argument("--output-dir", type=str, default="./train_dir", help="Directory to save checkpoints and logs")
  parser.add_argument("opts", help="Extra configuration options", default=[], nargs=argparse.REMAINDER)
  # data specific arguments
  parser.add_argument("--config-file", type=str, help="Model configuration file")
  parser.add_argument("--data-root", type=str, required=True, help="Root directory of the dataset")
  parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
  parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for data loading")
  # training specific arguments
  parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
  parser.add_argument("--base-lr", type=float, default=1e-4, help="Base learning rate for finetuning")
  parser.add_argument("--weight-decay", type=float, default=1e-3, help="Weight decay for optimizer")
  return parser


def setup_and_build_model(args, pretrained_weights):
  config = setup(args)
  model, _ = build_model_from_cfg(config, only_teacher=True)
  load_pretrained_weights(model, pretrained_weights, "teacher")
  # freeze the backbone and only finetune the head
  model.eval()
  for name, param in model.named_parameters():
    if any(name.startswith(n) for n in ['injector', 'decoder', 'protein_pred', 'contour_pred']):
      param.requires_grad = True
      logger.info(f"Finetuning parameter: {name}")
  model.cuda()
  return model, config


def scale_lr(learning_rates, batch_size):
    lr = learning_rates * (batch_size * distributed.get_global_size()) / 256.0
    logger.info(f"Scaled learning rate: {lr:.2e} (base_lr={learning_rates}, batch_size={batch_size}, world_size={distributed.get_global_size()})")
    return lr


def do_train(
    model,
    data_root,
    batch_size,
    output_dir,
    num_workers=4,
    epochs=10,
    base_lr=1e-4,
    weight_decay=1e-4,
):
  # configure data loader
  sampler_type = SamplerType.SHARDED_INFINITE
  dataset = HPAPairedDataset(data_root=data_root)
  data_loader = make_data_loader(
      dataset=dataset,
      batch_size=batch_size,
      num_workers=num_workers,
      shuffle=True,
      seed=0,  # TODO: Fix this -- cfg.train.seed
      sampler_type=sampler_type,
      sampler_advance=0,  # TODO(qas): fix this -- start_iter * cfg.train.batch_size_per_gpu,
      drop_last=True,
      collate_fn=None,
    )
  steps_per_epoch = len(dataset) // (batch_size * distributed.get_global_size())

  metric_logger = MetricLogger(delimiter='  ')
  max_iter = epochs * steps_per_epoch
  header = "Training"

  # configure optimizer and scheduler
  optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=scale_lr(base_lr, batch_size),
    weight_decay=weight_decay
  )
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iter, eta_min=0)

  for batch in metric_logger.log_every(data_loader, 10, header, max_iter):
    ref_image = batch['reference'].cuda(non_blocking=True)
    tar_image = batch['target'].cuda(non_blocking=True)
    ground_truth = batch['ground_truth'].cuda(non_blocking=True)
    ground_truth = torch.tensor(ground_truth, dtype=ref_image.dtype)
    contour = tar_image[:, 1:]

    predict = model.generate(ref_image, contour)
    predict = predict[:, 0]

    loss = nn.functional.mse_loss(predict, ground_truth)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    if distributed.get_global_size() > 1:
      # average loss across processes for logging
      reduced_loss = loss.detach()
      torch.distributed.all_reduce(reduced_loss, op=torch.distributed.ReduceOp.SUM)
      reduced_loss /= distributed.get_global_size()
    else:
      reduced_loss = loss.item()
    metric_logger.update(loss=reduced_loss)
    metric_logger.update(lr=optimizer.param_groups[0]["lr"])
  save_checkpoint(model, optimizer, epoch=epochs, output_dir=output_dir)


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, output_dir: str):
  os.makedirs(output_dir, exist_ok=True)
  torch.save(
    {
      "epoch": epoch,
      "model": model.state_dict(),
      "optimizer": optimizer.state_dict(),
    },
    os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pth"),
  )


def main(args) -> None:
  model, _ = setup_and_build_model(args, pretrained_weights=args.pretrained_weights)
  do_train(
    model,
    data_root=args.data_root,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    epochs=args.epochs,
    base_lr=args.base_lr,
    weight_decay=args.weight_decay,
    output_dir=args.output_dir,
  )


if __name__ == "__main__":
  args = get_args_parser().parse_args()
  sys.exit(main(args))