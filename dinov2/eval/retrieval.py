import os
import sys
import argparse
from tqdm import tqdm
from omegaconf import OmegaConf

import torch
import numpy as np

from dinov2.data.datasets.hpa_prot import ProtHPADataset
from dinov2.data.loaders import SamplerType, make_data_loader
from dinov2.data.transforms import make_classification_eval_transform
from dinov2.data.cell_dino.transforms import make_classification_eval_cell_transform
from dinov2.eval.setup import build_model_for_eval
from dinov2.data.datasets.hpa_prot import PROTEIN_LOCALIZATION


def get_args_parser(description=""):
  parser = argparse.ArgumentParser(description=description)
  parser.add_argument(
      "--root-dir",
      type=str,
      required=True,
      help="Root directory to save embeddings and aggregated features."
  )
  parser.add_argument(
      "--dataset-root",
      type=str,
      required=True,
      help="Root directory of the dataset."
  )
  parser.add_argument(
      "--batch-size",
      type=int,
      default=8,
      help="Batch size for feature extraction."
  )
  parser.add_argument(
      "--transform-type",
      type=str,
      default="default",
      choices=["none", "default", "cell"],
      help="Type of transform to apply to the images."
  )
  parser.add_argument(
      "--query-gene",
      type=str,
      default=None,
      help="Gene to query for retrieval."
  )
  parser.add_argument(
      "--topk",
      type=int,
      default=5,
      help="Number of top-k results to return."
  )
  parser.add_argument(
      "--token",
      type=str,
      default=None,
      help="Token type for intermediate layer extraction (if applicable)."
  )
  return parser


def format_target(target):
  t = ''
  for i, name in enumerate(PROTEIN_LOCALIZATION):
    if target[i] == 1:
      t += name + ', '
  return t.strip(', ')


class ProtDatasetWithFileName(ProtHPADataset):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def __getitem__(self, index):
    image, target = super().__getitem__(index)
    data = {
        "image": image,
        "target": target,
        "filename": self.samples[index][0]  # Assuming samples is a list of (filename, target) tuples
    }
    return data


def extract_features(
    root_dir,
    dataset_root,
    batch_size=8,
    transform=None
):
  if not os.path.exists(root_dir):
    raise ValueError(f"Root directory {root_dir} does not exist.")

  embeddings_path = os.path.join(root_dir, "embeddings.npz")
  if os.path.exists(embeddings_path):
    print(f"Embeddings already exist at {embeddings_path}. Loading...")
    features = np.load(embeddings_path)
    return dict(cls_tokens=features["cls_tokens"], targets=features["targets"], filenames=features["filenames"])

  config_yaml = os.path.join(root_dir, "config.yaml")
  pretrained_weights = os.path.join(root_dir, "teacher_checkpoint.pth")
  config = OmegaConf.load(config_yaml)
  model = build_model_for_eval(config, pretrained_weights)

  dataset = ProtDatasetWithFileName(
      root = dataset_root,
      split = "test",
      mode = "protein_localization",
      transform = transform,
  )
  data_loader = make_data_loader(
      dataset=dataset,
      batch_size=batch_size,
      num_workers=1,
      sampler_type=SamplerType.EPOCH,
      drop_last=False,
      shuffle=False,
      persistent_workers=False,
  )

  targets = []
  cls_tokens = []
  filenames = []

  for data in tqdm(data_loader, desc="Extracting features"):
    image = data["image"]
    target = data["target"]
    filenames.extend(data["filename"])

    image = image.cuda(non_blocking=True)
    with torch.no_grad():
      result = model.get_intermediate_layers(image, return_class_token=True)
    cls_token = [f[1].cpu().numpy() for f in result]
    targets.extend(target)
    cls_tokens.extend(cls_token)
  cls_tokens = np.concatenate(cls_tokens, axis=0)
  targets = np.array(targets)
  print(f"Extracted features: cls_tokens shape={cls_tokens.shape}, targets shape={targets.shape}")
  if not os.path.exists(root_dir):
    os.makedirs(root_dir)
  np.savez(embeddings_path, cls_tokens=cls_tokens, targets=targets, filenames=filenames)
  return dict(cls_tokens=cls_tokens, targets=targets, filenames=filenames)


def l2_normalize(x, axis=1, eps=1e-10):
  norm = np.linalg.norm(x, axis=axis, keepdims=True)
  return x / (norm + eps)


def aggregate_gene_features(features_dict):
  cls_tokens = np.asarray(features_dict["cls_tokens"], dtype=np.float32)
  targets = np.asarray(features_dict["targets"])
  gene_ids = np.asarray(features_dict["gene_ids"])

  if cls_tokens.ndim == 3:
    cls_tokens = cls_tokens[:, 0]

  cls_tokens = l2_normalize(cls_tokens, axis=1)
  gene_features, gene_targets, gene_counts = [], [], []
  unique_genes = np.unique(gene_ids)

  for gene in unique_genes:
    mask = gene_ids == gene

    feat = cls_tokens[mask].mean(0)
    feat = l2_normalize(feat[None])[0]
    gene_features.append(feat)

    y = targets[mask]
    if y.ndim > 1:
      gene_targets.append(np.max(y, axis=0))
    else:
      values, counts = np.unique(y, return_counts=True)
      gene_targets.append(values[np.argmax(counts)])

    gene_counts.append(mask.sum())

  return {
      "cls_tokens": np.stack(gene_features),
      "targets": np.asarray(gene_targets),
      "gene_ids": unique_genes,
      "counts": np.asarray(gene_counts),
  }


def retrieve_gene(gene_data, query_gene, topk=5):
  features = l2_normalize(gene_data["cls_tokens"], axis=1)
  gene_ids = gene_data["gene_ids"]

  idx = np.where(gene_ids == query_gene)[0]
  if len(idx) == 0:
      raise ValueError(f"{query_gene} not found.")

  idx = idx[0]
  sims = features @ features[idx]
  sims[idx] = -np.inf

  indices = np.argsort(sims)[::-1][:topk]

  return [
      {
          "rank": rank,
          "gene_id": gene_ids[i],
          "similarity": float(sims[i]),
          "target": gene_data["targets"][i],
          "count": int(gene_data["counts"][i]),
      }
      for rank, i in enumerate(indices, 1)
  ]


def label_jaccard(y1, y2):
  y1, y2 = np.asarray(y1), np.asarray(y2)
  inter = np.logical_and(y1, y2).sum()
  union = np.logical_or(y1, y2).sum()
  return inter / union if union > 0 else 0.0


def evaluate_dataset_retrieval(gene_data, topk=5):
  features = l2_normalize(np.asarray(gene_data["cls_tokens"], dtype=np.float32), axis=1)
  targets = np.asarray(gene_data["targets"])
  gene_ids = np.asarray(gene_data["gene_ids"])

  similarity = features @ features.T
  np.fill_diagonal(similarity, -np.inf)

  agreements = []
  precisions = []
  hits = []
  query_results = {}

  for i in range(len(gene_ids)):
    top_idx = np.argsort(similarity[i])[::-1][:topk]
    pair_scores = [label_jaccard(targets[i], targets[j]) for j in top_idx]
    relevant = np.array([score > 0 for score in pair_scores])

    agreement = np.mean(pair_scores)
    precision = relevant.mean()
    hit = float(any(score > 0 for score in pair_scores))
    agreements.append(agreement)
    precisions.append(precision)
    hits.append(hit)
    query_results[gene_ids[i]] = {
        'agreement': agreement,
        'precision': precision,
        'hit': hit,
        'retrieved_genes': gene_ids[top_idx],
        'similarities': similarity[i, top_idx],
        'pair_agreements': np.array(pair_scores),
    }

  return dict(
      agreement=float(np.mean(agreements)),
      precision=float(np.mean(precisions)),
      hit_rate=float(np.mean(hits)),
      query_results=query_results,
  )


def evaluate_constituent_retrieval(features_dict, mito_idx, cyto_idx, nuc_idx, topk=5):
  features = l2_normalize(np.asarray(features_dict['cls_tokens'], dtype=np.float32), axis=1)
  targets = np.asarray(features_dict['targets'], dtype=np.bool_)
  gene_ids = np.asarray(features_dict['gene_ids'])

  similarity = features @ features.T
  np.fill_diagonal(similarity, -np.inf)

  groups = {
      "Mito.+Cyto.": (mito_idx, cyto_idx),
      "Mito.+Nuc.": (mito_idx, nuc_idx),
  }

  results = {}
  for group_name, (a, b) in groups.items():
    query_indices = np.where(targets[:, a] & targets[:, b])[0]
    query_genes = np.unique(gene_ids[query_indices])
    gene_scores = []

    for gene in query_genes:
      gene_query_indices = query_indices[gene_ids[query_indices] == gene]
      image_agreements = []
      image_a_rates = []
      image_b_rates = []

      for idx in gene_query_indices:
        sims = similarity[idx].copy()
        sims[gene_ids == gene_ids[idx]] = -np.inf
        valid_sum = np.isfinite(sims).sum()
        if valid_sum < topk:
          continue

        neighbors = np.argsort(sims)[::-1][:topk]
        neighbor_targets = targets[neighbors]
        image_agreements.append(np.mean([label_jaccard(targets[idx], neighbor_targets[j]) for j in range(topk)]))
        image_a_rates.append(np.mean(neighbor_targets[:, a]))
        image_b_rates.append(np.mean(neighbor_targets[:, b]))

      if len(image_agreements) == 0:
        continue

      gene_scores.append({
          'gene_id': gene,
          'agreement': float(np.mean(image_agreements)),
          'a_rate': float(np.mean(image_a_rates)),
          'b_rate': float(np.mean(image_b_rates)),
          'num_images': len(image_agreements),
      })

    if len(gene_scores) == 0:
      results[group_name] = {
          'agreement': float('nan'),
          'a_rate': float('nan'),
          'b_rate': float('nan'),
          'num_images': 0,
          'num_genes': 0,
      }
      continue

    results[group_name] = {
        'agreement': float(np.mean([g['agreement'] for g in gene_scores])),
        'a_rate': float(np.mean([g['a_rate'] for g in gene_scores])),
        'b_rate': float(np.mean([g['b_rate'] for g in gene_scores])),
        'num_genes': len(gene_scores),
        'num_images': sum(x["num_images"] for x in gene_scores),
        'gene_scores': gene_scores,
    }

  return results


def main(args):
  if args.transform_type == 'none':
    transform = make_classification_eval_transform(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))
  elif args.transform_type == 'default':
    transform = make_classification_eval_transform()
  elif args.transform_type == 'cell':
    transform = make_classification_eval_cell_transform(resize_size=256, crop_size=224)
  else:
    raise ValueError(f"Unknown transform type: {args.transform_type}")

  features = extract_features(
      root_dir=args.root_dir,
      dataset_root=args.dataset_root,
      batch_size=args.batch_size,
      transform=transform
  )

  if args.token == 'px':
    features["cls_tokens"] = np.split(features["cls_tokens"], 2, axis=1)[0]
  elif args.token == 'cx':
    features["cls_tokens"] = np.split(features["cls_tokens"], 2, axis=1)[1]

  features["gene_ids"] = np.asarray([
      str(x).replace("\\", "/").rstrip("/").split("/")[-2]
      for x in features["filenames"]
  ])

  gene_data = aggregate_gene_features(features)

  print(f"Images: {len(features['gene_ids'])}, "
        f"Genes: {len(gene_data['gene_ids'])}")

  if args.query_gene is not None:
    results = retrieve_gene(gene_data, args.query_gene, args.topk)

    print(f"\nQuery: {args.query_gene}")
    for x in results:
        print(
            f"{x['rank']:>2}. {x['gene_id']} | "
            f"{x['similarity']:.4f} | "
            f"{format_target(x['target'])} | "
            f"n={x['count']}"
        )

  else:
    print("\nEvaluating retrieval performance on the entire dataset...")
    for k in (1, 5, 10, 20):
      eval_results = evaluate_dataset_retrieval(gene_data, topk=k)
      print(f"\nEvaluation with top-{k} retrieval:")
      print(f"  Agreement: {eval_results['agreement']:.4f}")
      print(f"  Precision: {eval_results['precision']:.4f}")
      print(f"  Hit rate: {eval_results['hit_rate']:.4f}")


    print("\nEvaluating constituent retrieval performance...")
    mito_idx = PROTEIN_LOCALIZATION.index("Mitochondria")
    cyto_idx = PROTEIN_LOCALIZATION.index("Cytosol")
    nuc_idx = PROTEIN_LOCALIZATION.index("Nucleoplasm")
    results = evaluate_constituent_retrieval(
        features_dict=features,
        mito_idx=mito_idx,
        cyto_idx=cyto_idx,
        nuc_idx=nuc_idx,
        topk=5,
    )

    for group, x in results.items():
      print(
          f"{group:15s} | "
          f"Agreement@5={x['agreement']:.4f} | "
          f"A={x['a_rate']:.4f} | "
          f"B={x['b_rate']:.4f} | "
          f"Genes={x['num_genes']} | "
          f"Images={x['num_images']}"
      )

  return 0


if __name__ == "__main__":
  description = "ProtLocNet retrieval evaluation script."
  args_parser = get_args_parser(description=description)
  args = args_parser.parse_args()
  sys.exit(main(args))
