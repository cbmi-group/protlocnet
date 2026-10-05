import os
import sys
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

# Limit implicit multithreading in underlying C libraries (like OpenBLAS/MKL used by NumPy)
# This prevents severe CPU contention when using multiprocessing (ProcessPoolExecutor).
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import tqdm
import numpy as np
import pandas as pd
from PIL import Image
from skimage import measure


def get_args_parser():
    parser = argparse.ArgumentParser("ProtLocNet reconstruction metrics.")
    parser.add_argument("--folder", type=str, required=True, help="Path to the folder containing the predictions and ground truth images")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(32, (os.cpu_count() or 1) + 4),
        help="Number of worker processes used to process images in parallel.",
    )
    return parser


def mae(pred, real, cell_mask=None):
    if cell_mask is not None:
        pred = pred[cell_mask]
        real = real[cell_mask]
    return np.mean(np.abs(pred - real))


def mse(pred, real, cell_mask=None):
    if cell_mask is not None:
        pred = pred[cell_mask]
        real = real[cell_mask]
    return np.mean((pred - real) ** 2)


def psnr(pred, real, cell_mask=None):
    err = mse(pred, real, cell_mask).clip(min=1e-10)  # Avoid division by zero
    return 20 * np.log10(255.0 / np.sqrt(err))


def pearson_corr(pred, real, cell_mask=None):
    if cell_mask is not None:
        pred = pred[cell_mask]
        real = real[cell_mask]

    pred_flat = np.ravel(pred)
    real_flat = np.ravel(real)

    if np.std(pred_flat) == 0 or np.std(real_flat) == 0:
        return 0.0
    return np.corrcoef(pred_flat, real_flat)[0, 1]


def r_squared(pred, real, cell_mask=None):
    if cell_mask is not None:
        pred = pred[cell_mask]
        real = real[cell_mask]
        
    pred_flat = np.ravel(pred)
    real_flat = np.ravel(real)
    
    ss_res = np.sum((real_flat - pred_flat) ** 2)
    ss_tot = np.sum((real_flat - np.mean(real_flat)) ** 2)
    
    if ss_tot == 0:
        return 0.0 if ss_res > 0 else 1.0
        
    return 1.0 - (ss_res / ss_tot)


def iou(pred, real, threshold=0.5):
    pred_mask = pred > threshold
    real_mask = real > threshold
    intersection = np.logical_and(pred_mask, real_mask).sum()
    union = np.logical_or(pred_mask, real_mask).sum()
    return intersection / union if union > 0 else 1.0


def dice(pred, real, threshold=0.5):
    pred_mask = pred > threshold
    real_mask = real > threshold
    intersection = np.logical_and(pred_mask, real_mask).sum()
    total_area = pred_mask.sum() + real_mask.sum()
    return (2 * intersection) / total_area if total_area > 0 else 1.0
  

def morphology_gap(pred, real, threshold=0.5):
    """Calculates absolute error for area and eccentricity."""
    pred_mask = pred > threshold
    real_mask = real > threshold
    
    pred_props = measure.regionprops(measure.label(pred_mask))
    real_props = measure.regionprops(measure.label(real_mask))
    
    pred_region = max(pred_props, key=lambda r: r.area) if pred_props else None
    real_region = max(real_props, key=lambda r: r.area) if real_props else None
    
    if not pred_region and not real_region:
        return 0.0, 0.0  # Both empty, no gap
    
    if not pred_region:
        return float(real_region.area), np.nan  # Missed prediction
    
    if not real_region:
        return float(pred_region.area), np.nan  # False positive prediction
        
    area_err = abs(pred_region.area - real_region.area)
    eccentricity_err = abs(pred_region.eccentricity - real_region.eccentricity)
    
    return float(area_err), float(eccentricity_err)
  

def init_metric_dict(num_channels=3):
    """Initializes the dictionary for standard continuous metrics (all channels)."""
    return {
        channel: {
            "mae": [],
            "mse": [],
            "psnr": [],
            "pearson_corr": [],
            "r_squared": [],
        }
        for channel in range(num_channels)
    }


def init_mask_metric_dict():
    """Initializes the dictionary specifically for Nucleus (1) and Cell Mask (2)."""
    return {
        channel: {
            "iou": [],
            "dice": [],
            "area_error": [],
            "eccentricity_error": []
        }
        for channel in [1, 2]
    }


def summarize_metrics(total_metrics, mask_metrics):
    """Converts raw lists of metrics into mean and standard deviation."""
    summary = {
        "channels": {},
        "masks": {}
    }

    # 1. Summarize standard channel metrics (MAE, MSE, PSNR, Pearson, R2)
    for pred_name in total_metrics:
        summary["channels"][pred_name] = {}
        for channel in total_metrics[pred_name]:
            summary["channels"][pred_name][channel] = {}
            for metric_name, values in total_metrics[pred_name][channel].items():
                values = np.array(values, dtype=np.float32)
                summary["channels"][pred_name][channel][metric_name] = {
                    "mean": float(np.nanmean(values)),
                    "std": float(np.nanstd(values)),
                }

    # 2. Summarize mask-specific metrics (IoU, Dice, Area/Ecc Errors) for Channel 1 & 2
    for pred_name in mask_metrics:
        summary["masks"][pred_name] = {}
        for channel in mask_metrics[pred_name]:
            summary["masks"][pred_name][channel] = {}
            for metric_name, values in mask_metrics[pred_name][channel].items():
                values = np.array(values, dtype=np.float32)
                summary["masks"][pred_name][channel][metric_name] = {
                    "mean": float(np.nanmean(values)),
                    "std": float(np.nanstd(values)),
                }

    return summary


def metrics_to_dataframe(summary):
    """Flattens the summarized dictionary into a structured Pandas DataFrame."""
    rows = []

    for pred_name, pred_data in summary["channels"].items():
        for channel, channel_metrics in pred_data.items():
            row = {
                "prediction": pred_name,
                "channel": channel,
            }
            
            # Append standard metrics
            for metric_name, stats in channel_metrics.items():
                row[f"{metric_name}_mean"] = stats["mean"]
                row[f"{metric_name}_std"] = stats["std"]

            # Append mask metrics ONLY if the current channel is Nucleus (1) or Cell Mask (2)
            base_pred_name = pred_name.split("@")[0]  # Extracts "c_pred" from "c_pred@cell"
            
            if channel in [1, 2] and base_pred_name in summary["masks"]:
                for metric_name, stats in summary["masks"][base_pred_name][channel].items():
                    row[f"{metric_name}_mean"] = stats["mean"]
                    row[f"{metric_name}_std"] = stats["std"]
            else:
                # Fill with NaNs for Protein channel (0) where shape/mask metrics don't apply
                for m in ["iou", "dice", "area_error", "eccentricity_error"]:
                    row[f"{m}_mean"] = np.nan
                    row[f"{m}_std"] = np.nan
                
            rows.append(row)

    return pd.DataFrame(rows)


def compute_file_metrics(c_pred_path, p_pred_path, real_path):
    # Load images and normalize to [0, 1]
    c_pred = np.array(Image.open(c_pred_path).convert("RGB")).astype(np.float32) / 255.0
    p_pred = np.array(Image.open(p_pred_path).convert("RGB")).astype(np.float32) / 255.0
    real = np.array(Image.open(real_path).convert("RGB")).astype(np.float32) / 255.0
    
    # Channel mapping: 0=Protein, 1=Nucleus, 2=Cell Mask
    cell_mask = real[..., 2] > 0  

    file_total_metrics = {
        "c_pred": init_metric_dict(num_channels=3),
        "p_pred": init_metric_dict(num_channels=3),
        "c_pred@cell": init_metric_dict(num_channels=3),
        "p_pred@cell": init_metric_dict(num_channels=3),
    }
    
    file_mask_metrics = {
        "c_pred": init_mask_metric_dict(),
        "p_pred": init_mask_metric_dict(),
    }

    for channel in range(3):
        c_ch = c_pred[..., channel]
        p_ch = p_pred[..., channel]
        r_ch = real[..., channel]

        # 1. Compute continuous metrics (MAE, MSE, PSNR, Pearson, R2) across ALL channels
        file_total_metrics["c_pred"][channel]["mae"].append(mae(c_ch, r_ch))
        file_total_metrics["c_pred"][channel]["mse"].append(mse(c_ch, r_ch))
        file_total_metrics["c_pred"][channel]["psnr"].append(psnr(c_ch, r_ch))
        file_total_metrics["c_pred"][channel]["pearson_corr"].append(pearson_corr(c_ch, r_ch))
        file_total_metrics["c_pred"][channel]["r_squared"].append(r_squared(c_ch, r_ch))

        file_total_metrics["p_pred"][channel]["mae"].append(mae(p_ch, r_ch))
        file_total_metrics["p_pred"][channel]["mse"].append(mse(p_ch, r_ch))
        file_total_metrics["p_pred"][channel]["psnr"].append(psnr(p_ch, r_ch))
        file_total_metrics["p_pred"][channel]["pearson_corr"].append(pearson_corr(p_ch, r_ch))
        file_total_metrics["p_pred"][channel]["r_squared"].append(r_squared(p_ch, r_ch))

        # Metrics strictly inside the true cell mask (@cell)
        file_total_metrics["c_pred@cell"][channel]["mae"].append(mae(c_ch, r_ch, cell_mask))
        file_total_metrics["c_pred@cell"][channel]["mse"].append(mse(c_ch, r_ch, cell_mask))
        file_total_metrics["c_pred@cell"][channel]["psnr"].append(psnr(c_ch, r_ch, cell_mask))
        file_total_metrics["c_pred@cell"][channel]["pearson_corr"].append(pearson_corr(c_ch, r_ch, cell_mask))
        file_total_metrics["c_pred@cell"][channel]["r_squared"].append(r_squared(c_ch, r_ch, cell_mask))

        file_total_metrics["p_pred@cell"][channel]["mae"].append(mae(p_ch, r_ch, cell_mask))
        file_total_metrics["p_pred@cell"][channel]["mse"].append(mse(p_ch, r_ch, cell_mask))
        file_total_metrics["p_pred@cell"][channel]["psnr"].append(psnr(p_ch, r_ch, cell_mask))
        file_total_metrics["p_pred@cell"][channel]["pearson_corr"].append(pearson_corr(p_ch, r_ch, cell_mask))
        file_total_metrics["p_pred@cell"][channel]["r_squared"].append(r_squared(p_ch, r_ch, cell_mask))

        # 2. Compute mask/morphology metrics ONLY for Nucleus (1) and Cell Mask (2)
        if channel in [1, 2]:
            c_area_err, c_ecc_err = morphology_gap(c_ch, r_ch)
            file_mask_metrics["c_pred"][channel]["iou"].append(iou(c_ch, r_ch))
            file_mask_metrics["c_pred"][channel]["dice"].append(dice(c_ch, r_ch))
            file_mask_metrics["c_pred"][channel]["area_error"].append(c_area_err)
            file_mask_metrics["c_pred"][channel]["eccentricity_error"].append(c_ecc_err)

            p_area_err, p_ecc_err = morphology_gap(p_ch, r_ch)
            file_mask_metrics["p_pred"][channel]["iou"].append(iou(p_ch, r_ch))
            file_mask_metrics["p_pred"][channel]["dice"].append(dice(p_ch, r_ch))
            file_mask_metrics["p_pred"][channel]["area_error"].append(p_area_err)
            file_mask_metrics["p_pred"][channel]["eccentricity_error"].append(p_ecc_err)

    return file_total_metrics, file_mask_metrics


def merge_metrics(total_metrics, file_total_metrics, mask_metrics, file_mask_metrics):
    """Aggregates metrics from individual files into the master dictionaries."""
    for pred_name, pred_metrics in file_total_metrics.items():
        for channel, channel_metrics in pred_metrics.items():
            for metric_name, values in channel_metrics.items():
                total_metrics[pred_name][channel][metric_name].extend(values)

    for pred_name, p_metrics in file_mask_metrics.items():
        for channel, channel_metrics in p_metrics.items():
            for metric_name, values in channel_metrics.items():
                mask_metrics[pred_name][channel][metric_name].extend(values)


def main(args):
    c_pred_dir = os.path.join(args.folder, 'c_pred')
    p_pred_dir = os.path.join(args.folder, 'p_pred')
    real_dir = os.path.join(args.folder, 'real')
    
    total_metrics = {
        "c_pred": init_metric_dict(num_channels=3),
        "p_pred": init_metric_dict(num_channels=3),
        "c_pred@cell": init_metric_dict(num_channels=3),
        "p_pred@cell": init_metric_dict(num_channels=3),
    }
    
    mask_metrics = {
        "c_pred": init_mask_metric_dict(),
        "p_pred": init_mask_metric_dict(),
    }

    filenames = sorted(os.listdir(real_dir))
    file_jobs = [
        (
            os.path.join(c_pred_dir, f),
            os.path.join(p_pred_dir, f),
            os.path.join(real_dir, f),
        )
        for f in filenames
    ]

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(compute_file_metrics, c_pred_path, p_pred_path, real_path)
            for c_pred_path, p_pred_path, real_path in file_jobs
        ]

        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Computing metrics"):
            file_total_metrics, file_mask_metrics = future.result()
            merge_metrics(total_metrics, file_total_metrics, mask_metrics, file_mask_metrics)

    summary = summarize_metrics(total_metrics, mask_metrics)
    df = metrics_to_dataframe(summary)
    
    print(df.to_string())
    
    output_csv = os.path.join(args.folder, "channel_metrics_summary.csv")
    df.to_csv(output_csv, index=False)
    print(f"\nResults successfully saved to {output_csv}")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    sys.exit(main(args))