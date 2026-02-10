#!/usr/bin/env python3
"""
TrajectoryDiff: Uncertainty Calibration Analysis

Loads a trained checkpoint, generates multiple samples per test input,
computes mean/std (uncertainty), and evaluates calibration metrics.

Usage:
    python scripts/analyze_uncertainty.py --checkpoint experiments/trajectory_full/checkpoints/best.ckpt
    python scripts/analyze_uncertainty.py --checkpoint model.ckpt --num-samples 10 --max-test 50
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data import RadioMapDataModule
from evaluation.metrics import (
    uncertainty_calibration,
    uncertainty_error_correlation,
)
from training import DiffusionInference, denormalize_radio_map


@torch.no_grad()
def analyze_uncertainty(
    checkpoint_path: str,
    num_samples: int = 10,
    max_test: int = 0,
    data_dir: str = "data/raw",
    batch_size: int = 4,
    num_workers: int = 4,
    output_dir: str = "experiments/eval_results/uncertainty",
    use_ddim: bool = True,
):
    """Run uncertainty analysis on test set."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"Loading checkpoint: {checkpoint_path}")
    inference = DiffusionInference.from_checkpoint(checkpoint_path, device=device, use_ema=True)

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

    # Collect results
    all_mean_pred = []
    all_std_pred = []
    all_gt = []
    all_traj_mask = []

    total = 0
    for batch in tqdm(test_loader, desc="Generating samples"):
        if max_test > 0 and total >= max_test:
            break

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        condition = {
            "building_map": batch.get("building_map"),
            "sparse_rss": batch.get("sparse_rss"),
            "trajectory_mask": batch.get("trajectory_mask"),
            "coverage_density": batch.get("coverage_density"),
            "tx_position": batch.get("tx_position"),
        }
        condition = {k: v for k, v in condition.items() if v is not None}

        # Generate multiple samples
        samples_list = []
        for s in range(num_samples):
            sample = inference.sample(condition, use_ddim=use_ddim, progress=False)
            samples_list.append(sample.cpu())

        # Stack: [num_samples, B, 1, H, W]
        samples = torch.stack(samples_list, dim=0)
        mean_pred = samples.mean(dim=0)
        std_pred = samples.std(dim=0)

        all_mean_pred.append(mean_pred)
        all_std_pred.append(std_pred)
        all_gt.append(batch["radio_map"].cpu())
        if "trajectory_mask" in batch:
            all_traj_mask.append(batch["trajectory_mask"].cpu())

        total += mean_pred.shape[0]

    # Concatenate
    mean_pred = torch.cat(all_mean_pred, dim=0).numpy()
    std_pred = torch.cat(all_std_pred, dim=0).numpy()
    gt = torch.cat(all_gt, dim=0).numpy()
    traj_mask = torch.cat(all_traj_mask, dim=0).numpy() if all_traj_mask else None

    print(f"\nAnalyzed {mean_pred.shape[0]} test samples with {num_samples} diffusion samples each")

    # ============================================================
    # Compute calibration metrics
    # ============================================================
    # Flatten to 2D for metrics (squeeze channel dim)
    mean_flat = mean_pred[:, 0]  # (N, H, W)
    std_flat = std_pred[:, 0]
    gt_flat = gt[:, 0]

    # Global calibration
    calib = uncertainty_calibration(
        mean_flat.reshape(-1),
        std_flat.reshape(-1),
        gt_flat.reshape(-1),
    )

    errors = np.abs(mean_flat - gt_flat)
    corr = uncertainty_error_correlation(std_flat.reshape(-1), errors.reshape(-1))

    results = {
        "num_test_samples": int(mean_pred.shape[0]),
        "num_diffusion_samples": num_samples,
        "calibration_error": calib["calibration_error"],
        "fraction_within_1std": calib["fraction_within_1std"],
        "fraction_within_2std": calib["fraction_within_2std"],
        "uncertainty_error_correlation": corr,
        "mean_uncertainty": float(std_flat.mean()),
        "mean_error": float(errors.mean()),
    }

    # On-trajectory vs off-trajectory analysis
    if traj_mask is not None:
        mask_flat = traj_mask[:, 0]  # (N, H, W)
        on_traj = mask_flat > 0.5
        off_traj = mask_flat <= 0.5

        if on_traj.sum() > 0:
            results["uncertainty_on_trajectory"] = float(std_flat[on_traj].mean())
            results["error_on_trajectory"] = float(errors[on_traj].mean())
        if off_traj.sum() > 0:
            results["uncertainty_off_trajectory"] = float(std_flat[off_traj].mean())
            results["error_off_trajectory"] = float(errors[off_traj].mean())
        results["uncertainty_ratio_off_on"] = round(
            results.get("uncertainty_off_trajectory", 0) /
            max(results.get("uncertainty_on_trajectory", 1e-8), 1e-8), 3
        )

    # ============================================================
    # Print results
    # ============================================================
    print("\n" + "=" * 60)
    print("UNCERTAINTY CALIBRATION RESULTS")
    print("=" * 60)
    print(f"Calibration Error (ECE):       {results['calibration_error']:.4f}")
    print(f"Fraction within 1 std:         {results['fraction_within_1std']:.4f} (ideal: 0.683)")
    print(f"Fraction within 2 std:         {results['fraction_within_2std']:.4f} (ideal: 0.954)")
    print(f"Uncertainty-Error Correlation: {results['uncertainty_error_correlation']:.4f}")
    print(f"Mean Uncertainty:              {results['mean_uncertainty']:.4f}")
    print(f"Mean Error:                    {results['mean_error']:.4f}")

    if traj_mask is not None:
        print(f"\nSpatial Uncertainty Breakdown:")
        print(f"  On trajectory:   uncertainty={results.get('uncertainty_on_trajectory', 'N/A'):.4f}, "
              f"error={results.get('error_on_trajectory', 'N/A'):.4f}")
        print(f"  Off trajectory:  uncertainty={results.get('uncertainty_off_trajectory', 'N/A'):.4f}, "
              f"error={results.get('error_off_trajectory', 'N/A'):.4f}")
        print(f"  Off/On ratio:    {results.get('uncertainty_ratio_off_on', 'N/A')}")

    # ============================================================
    # Generate calibration plot
    # ============================================================
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot 1: Calibration curve
        expected, observed = calib["calibration_curve"]
        axes[0].plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        axes[0].plot(expected, observed, "b-o", label="Model")
        axes[0].set_xlabel("Expected coverage")
        axes[0].set_ylabel("Observed coverage")
        axes[0].set_title(f"Calibration Curve (ECE={results['calibration_error']:.3f})")
        axes[0].legend()
        axes[0].set_aspect("equal")

        # Plot 2: Uncertainty vs Error scatter (downsampled)
        n_points = min(10000, errors.size)
        idx = np.random.choice(errors.size, n_points, replace=False)
        axes[1].scatter(std_flat.reshape(-1)[idx], errors.reshape(-1)[idx], alpha=0.1, s=1)
        axes[1].set_xlabel("Predicted Uncertainty (std)")
        axes[1].set_ylabel("Actual Error |pred - gt|")
        axes[1].set_title(f"Uncertainty vs Error (r={corr:.3f})")

        # Plot 3: Uncertainty histogram by region
        if traj_mask is not None:
            axes[2].hist(std_flat[on_traj].reshape(-1), bins=50, alpha=0.6, label="On trajectory", density=True)
            axes[2].hist(std_flat[off_traj].reshape(-1), bins=50, alpha=0.6, label="Off trajectory", density=True)
            axes[2].set_xlabel("Uncertainty (std)")
            axes[2].set_ylabel("Density")
            axes[2].set_title("Uncertainty Distribution by Region")
            axes[2].legend()
        else:
            axes[2].hist(std_flat.reshape(-1), bins=50, alpha=0.7)
            axes[2].set_xlabel("Uncertainty (std)")
            axes[2].set_ylabel("Count")
            axes[2].set_title("Uncertainty Distribution")

        plt.tight_layout()
        fig_path = output_dir / "calibration_plots.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nPlots saved to: {fig_path}")
    except ImportError:
        print("\nmatplotlib not available, skipping plots")

    # Save results JSON
    results_path = output_dir / "uncertainty_results.json"
    # Convert numpy types for JSON serialization
    serializable = {k: v for k, v in results.items() if not isinstance(v, np.ndarray)}
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Results saved to: {results_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Uncertainty calibration analysis")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--num-samples", type=int, default=10, help="Diffusion samples per input")
    parser.add_argument("--max-test", type=int, default=0, help="Max test samples (0=all)")
    parser.add_argument("--data-dir", default="data/raw", help="Path to RadioMapSeer data")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--output-dir", default="experiments/eval_results/uncertainty", help="Output directory")
    args = parser.parse_args()

    analyze_uncertainty(
        checkpoint_path=args.checkpoint,
        num_samples=args.num_samples,
        max_test=args.max_test,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
    )
