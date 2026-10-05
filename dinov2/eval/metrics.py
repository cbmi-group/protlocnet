# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from enum import Enum
import logging
from typing import Any, Dict, Optional

import torch
from torch import Tensor
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MultilabelF1Score
from torchmetrics.utilities.data import dim_zero_cat, select_topk
from torchmetrics import ConfusionMatrix


logger = logging.getLogger("dinov2")


class MetricType(Enum):
    MEAN_ACCURACY = "mean_accuracy"
    MEAN_PER_CLASS_ACCURACY = "mean_per_class_accuracy"
    PER_CLASS_ACCURACY = "per_class_accuracy"
    IMAGENET_REAL_ACCURACY = "imagenet_real_accuracy"
    MEAN_PER_CLASS_MULTICLASS_F1 = "mean_per_class_multiclass_f1"
    MEAN_PER_CLASS_MULTILABEL_F1 = "mean_per_class_multilabel_f1"
    CONFUSION_MATRIX = 'confusion_matrix'
    MULTILABEL_CONFUSION_MATRIX = 'multilabel_confusion_matrix'
    KNN_AGREEMENT = 'knn_agreement'

    @property
    def accuracy_averaging(self):
        return getattr(AccuracyAveraging, self.name, None)

    def __str__(self):
        return self.value


class AccuracyAveraging(Enum):
    MEAN_ACCURACY = "micro"
    MEAN_PER_CLASS_ACCURACY = "macro"
    PER_CLASS_ACCURACY = "none"

    def __str__(self):
        return self.value


def build_metric(metric_type: MetricType, *, num_classes: int, ks: Optional[tuple] = None):
    if metric_type.accuracy_averaging is not None:
        return build_topk_accuracy_metric(
            average_type=metric_type.accuracy_averaging,
            num_classes=num_classes,
            ks=(1, 5) if ks is None else ks,
        )
    elif metric_type == MetricType.IMAGENET_REAL_ACCURACY:
        return build_topk_imagenet_real_accuracy_metric(
            num_classes=num_classes,
            ks=(1, 5) if ks is None else ks,
        )
    elif metric_type == MetricType.MEAN_PER_CLASS_MULTILABEL_F1:
        return MetricCollection({"top-1": MultilabelF1Score(num_labels=int(num_classes), average="macro")})
    elif metric_type == MetricType.MEAN_PER_CLASS_MULTICLASS_F1:
        return MetricCollection({"top-1": MulticlassF1Score(num_classes=int(num_classes), average="macro")})
    elif metric_type == MetricType.CONFUSION_MATRIX:
        return build_confusion_matrix_metric(
            num_classes=num_classes,
            task='multiclass',
        )
    elif metric_type == MetricType.MULTILABEL_CONFUSION_MATRIX:
        return build_confusion_matrix_metric(
            num_classes=num_classes,
            task='multilabel',
        )
    elif metric_type == MetricType.KNN_AGREEMENT:
        return MetricCollection({
            "knn_agreement": KNNLocalizationAgreement(),
            "top-1": MultilabelF1Score(num_labels=int(num_classes), average="macro")
        })
    raise ValueError(f"Unknown metric type {metric_type}")


def build_topk_accuracy_metric(average_type: AccuracyAveraging, num_classes: int, ks: tuple = (1, 5)):
    metrics: Dict[str, Metric] = {
        f"top-{k}": MulticlassAccuracy(top_k=k, num_classes=int(num_classes), average=average_type.value) for k in ks
    }
    return MetricCollection(metrics)


def build_topk_imagenet_real_accuracy_metric(num_classes: int, ks: tuple = (1, 5)):
    metrics: Dict[str, Metric] = {f"top-{k}": ImageNetReaLAccuracy(top_k=k, num_classes=int(num_classes)) for k in ks}
    return MetricCollection(metrics)


class ImageNetReaLAccuracy(Metric):
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    def __init__(
        self,
        num_classes: int,
        top_k: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.top_k = top_k
        self.add_state("tp", [], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        # preds [B, D]
        # target [B, A]
        # preds_oh [B, D] with 0 and 1
        # select top K highest probabilities, use one hot representation
        preds_oh = select_topk(preds, self.top_k)
        # target_oh [B, D + 1] with 0 and 1
        target_oh = torch.zeros((preds_oh.shape[0], preds_oh.shape[1] + 1), device=target.device, dtype=torch.int32)
        target = target.long()
        # for undefined targets (-1) use a fake value `num_classes`
        target[target == -1] = self.num_classes
        # fill targets, use one hot representation
        target_oh.scatter_(1, target, 1)
        # target_oh [B, D] (remove the fake target at index `num_classes`)
        target_oh = target_oh[:, :-1]
        # tp [B] with 0 and 1
        tp = (preds_oh * target_oh == 1).sum(dim=1)
        # at least one match between prediction and target
        tp.clip_(max=1)
        # ignore instances where no targets are defined
        mask = target_oh.sum(dim=1) > 0
        tp = tp[mask]
        self.tp.append(tp)  # type: ignore

    def compute(self) -> Tensor:
        tp = dim_zero_cat(self.tp)  # type: ignore
        return tp.float().mean()


def build_confusion_matrix_metric(num_classes: int, task: str = 'multiclass'):
  metrics: Dict[str, Metric] = {
      'cm': ConfusionMatrix(
          num_classes=num_classes,
          num_labels=num_classes,
          task=task
      ),
      'top-1': MultilabelF1Score(
          num_labels=int(num_classes),
          average="macro"
      ) if task == 'multilabel' else MulticlassAccuracy(
          num_classes=int(num_classes),
          average="macro"
      ),
  }
  return MetricCollection(metrics)


class ConfusionMatrixResult:
  def __init__(self, metric_type, metrics: Dict[str, Tensor]):
    if 'cm' not in metrics:
      raise ValueError('ConfusionMatrixResult requires confusion matrix in metrics with key "cm"')

    confusion_matrix = metrics.pop('cm')
    confusion_matrix = confusion_matrix.detach().cpu()
    if metric_type == MetricType.CONFUSION_MATRIX:
      tp = torch.diag(confusion_matrix)
      fp = confusion_matrix.sum(dim=0) - tp
      fn = confusion_matrix.sum(dim=1) - tp
    elif metric_type == MetricType.MULTILABEL_CONFUSION_MATRIX:
      tp = confusion_matrix[:, 1, 1]
      fp = confusion_matrix[:, 0, 1]
      fn = confusion_matrix[:, 1, 0]
    else:
      raise ValueError(
          f'Unknown metric type {metric_type},'
          f' expected {MetricType.CONFUSION_MATRIX} or {MetricType.MULTILABEL_CONFUSION_MATRIX}'
      )

    p = tp / (tp + fp + 1e-8)
    r = tp / (tp + fn + 1e-8)
    support = tp + fn
    support_prob = support / support.sum()

    f1 = 2 * p * r / (p + r + 1e-8)
    f1_macro = f1.mean()
    f1_micro = 2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum() + 1e-8)
    f1_weighted = (f1 * support_prob).sum()

    p_macro = p.mean()
    p_weighted = (p * support_prob).sum()

    r_macro = r.mean()
    r_weighted = (r * support_prob).sum()
    self._result = {
        'f1@macro': f1_macro.item(),
        'f1@micro': f1_micro.item(),
        'f1@weighted': f1_weighted.item(),
        'precision@macro': p_macro.item(),
        'precision@weighted': p_weighted.item(),
        'recall@macro': r_macro.item(),
        'recall@weighted': r_weighted.item(),
    }
    self._result.update({k: v.item() for k, v in metrics.items()})

  @property
  def dict(self) -> Dict[str, float]:
    return self._result

  @property
  def accuracy(self) -> float:
    return self._result['f1@macro']

  def __repr__(self):
    lines = ['**** Confusion Matrix Result ****']
    lines.extend(
        [f'** {key}: {value:.4f} **' for key, value in self.dict.items()]
    )
    lines.append('*******************************\n')
    return '\n'.join(lines)


class KNNLocalizationAgreement(Metric):
    full_state_update = False

    def __init__(self, eps: float = 1e-8):
       super().__init__()
       self.eps = eps
       self.add_state('agreement_sum', default=torch.tensor(0.0), dist_reduce_fx="sum")
       self.add_state('num_samples', default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        # preds [B, D]
        # targets [B, D]
        preds = preds.float().clamp(0.0, 1.0)
        target = target.float()

        intersection = (preds * target).sum(dim=1)
        union = (preds + target - preds * target).sum(dim=1)

        agreement = intersection / (union + self.eps)
        self.agreement_sum += agreement.sum()
        self.num_samples += preds.size(0)

    def compute(self) -> Tensor:
        return self.agreement_sum / (self.num_samples.clamp_min(1))
