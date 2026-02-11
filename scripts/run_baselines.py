#!/usr/bin/env python3
"""
TrajectoryDiff: Classical Baseline Evaluation

Evaluates non-learning interpolation baselines (IDW, RBF, NearestNeighbor,
DistanceTransform) on the test set and saves results as JSON.

Usage:
    python scripts/run_baselines.py
    python scripts/run_baselines.py --output results/baselines.json
    python scripts/run_baselines.py --max-samples 100 --data-dir data/raw
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data import RadioMapDataModule
from data.floor_plan import png_to_db
from evaluation.metrics import compute_all_metrics
from models.baselines.interpolation import get_all_baselines


def denormalize_to_dbm(x: np.ndarray, min_val: float = -186.0, max_val: float = -47.0) -> np.ndarray:
    """Denormalize from [-1, 1] to dBm scale (RadioMapSeer: [-186, -47])."""
    return ((x + 1) / 2) * (max_val - min_val) + min_val


def evaluate_baselines(
    data_dir: str = "data/raw",
    output_path: str = "experiments/eval_results/baselines.json",
    max_samples: int = 0,
    batch_size: int = 1,
    num_workers: int = 4,
) -> dict:
    """Run all classical baselines on the test set."""

    # Setup data
    print("Setting up test data...")
    datamodule = RadioMapDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        sampling_strategy="trajectory",
        num_trajectories=3,
        points_per_trajectory=100,
    )
    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()

    # Get baselines
    baselines = get_all_baselines()
    print(f"Baselines: {list(baselines.keys())}")

    # Results storage
    results = {name: {
        "rmse": [], "mae": [], "ssim": [], "rmse_observed": [], "rmse_unobserved": [],
        "rmse_dbm": [], "mae_dbm": [], "rmse_observed_dbm": [], "rmse_unobserved_dbm": [],
    } for name in baselines}
    timing = {name: 0.0 for name in baselines}

    total = 0
    for batch in tqdm(test_loader, desc="Evaluating baselines"):
        if max_samples > 0 and total >= max_samples:
            break

        for i in range(batch["radio_map"].shape[0]):
            # Extract numpy arrays (squeeze channel dim)
            gt = batch["radio_map"][i, 0].numpy()
            sparse_rss = batch["sparse_rss"][i, 0].numpy()
            traj_mask = batch["trajectory_mask"][i, 0].numpy()

            for name, baseline in baselines.items():
                t0 = time.time()
                pred = baseline(sparse_rss, traj_mask)
                timing[name] += time.time() - t0

                # Compute metrics in normalized space
                metrics = compute_all_metrics(
                    pred, gt,
                    trajectory_mask=traj_mask,
                )

                results[name]["rmse"].append(metrics["rmse"])
                results[name]["mae"].append(metrics["mae"])
                if "ssim" in metrics:
                    results[name]["ssim"].append(metrics["ssim"])
                if "rmse_observed" in metrics:
                    results[name]["rmse_observed"].append(metrics["rmse_observed"])
                if "rmse_unobserved" in metrics:
                    results[name]["rmse_unobserved"].append(metrics["rmse_unobserved"])

                # dBm-scale metrics
                pred_dbm = denormalize_to_dbm(pred)
                gt_dbm = denormalize_to_dbm(gt)
                diff_dbm = pred_dbm - gt_dbm
                results[name]["rmse_dbm"].append(float(np.sqrt(np.mean(diff_dbm ** 2))))
                results[name]["mae_dbm"].append(float(np.mean(np.abs(diff_dbm))))
                if traj_mask.any():
                    obs_mask = traj_mask > 0.5
                    unobs_mask = traj_mask <= 0.5
                    if obs_mask.any():
                        results[name]["rmse_observed_dbm"].append(
                            float(np.sqrt(np.mean(diff_dbm[obs_mask] ** 2)))
                        )
                    if unobs_mask.any():
                        results[name]["rmse_unobserved_dbm"].append(
                            float(np.sqrt(np.mean(diff_dbm[unobs_mask] ** 2)))
                        )

            total += 1

    # Aggregate
    summary = {}
    for name in baselines:
        r = results[name]
        summary[name] = {
            "rmse_mean": float(np.mean(r["rmse"])),
            "rmse_std": float(np.std(r["rmse"])),
            "mae_mean": float(np.mean(r["mae"])),
            "mae_std": float(np.std(r["mae"])),
            "num_samples": len(r["rmse"]),
            "total_time_s": round(timing[name], 2),
            "avg_time_ms": round(timing[name] / max(len(r["rmse"]), 1) * 1000, 2),
        }
        if r["ssim"]:
            summary[name]["ssim_mean"] = float(np.mean(r["ssim"]))
            summary[name]["ssim_std"] = float(np.std(r["ssim"]))
        if r["rmse_observed"]:
            summary[name]["rmse_observed_mean"] = float(np.nanmean(r["rmse_observed"]))
        if r["rmse_unobserved"]:
            summary[name]["rmse_unobserved_mean"] = float(np.nanmean(r["rmse_unobserved"]))
        # dBm-scale aggregates
        if r["rmse_dbm"]:
            summary[name]["rmse_dbm_mean"] = float(np.mean(r["rmse_dbm"]))
            summary[name]["rmse_dbm_std"] = float(np.std(r["rmse_dbm"]))
        if r["mae_dbm"]:
            summary[name]["mae_dbm_mean"] = float(np.mean(r["mae_dbm"]))
            summary[name]["mae_dbm_std"] = float(np.std(r["mae_dbm"]))
        if r["rmse_observed_dbm"]:
            summary[name]["rmse_observed_dbm_mean"] = float(np.nanmean(r["rmse_observed_dbm"]))
        if r["rmse_unobserved_dbm"]:
            summary[name]["rmse_unobserved_dbm_mean"] = float(np.nanmean(r["rmse_unobserved_dbm"]))

    # Print results (dBm as primary)
    print("\n" + "=" * 80)
    print("BASELINE RESULTS (dBm scale)")
    print("=" * 80)
    print(f"{'Method':<25} {'RMSE(dBm)':>10} {'MAE(dBm)':>10} {'SSIM':>8} {'Time(ms)':>10}")
    print("-" * 80)
    for name, s in summary.items():
        ssim_str = f"{s.get('ssim_mean', float('nan')):.4f}"
        rmse_dbm = s.get('rmse_dbm_mean', float('nan'))
        mae_dbm = s.get('mae_dbm_mean', float('nan'))
        print(f"{name:<25} {rmse_dbm:>10.2f} {mae_dbm:>10.2f} {ssim_str:>8} {s['avg_time_ms']:>10.1f}")

    print(f"\n{'Method':<25} {'RMSE(norm)':>10} {'MAE(norm)':>10}")
    print("-" * 50)
    for name, s in summary.items():
        print(f"{name:<25} {s['rmse_mean']:>10.4f} {s['mae_mean']:>10.4f}")

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate classical baselines")
    parser.add_argument("--data-dir", default="data/raw", help="Path to RadioMapSeer data")
    parser.add_argument("--output", default="experiments/eval_results/baselines.json", help="Output JSON path")
    parser.add_argument("--max-samples", type=int, default=0, help="Max test samples (0=all)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for data loading")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers")
    args = parser.parse_args()

    evaluate_baselines(
        data_dir=args.data_dir,
        output_path=args.output,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
