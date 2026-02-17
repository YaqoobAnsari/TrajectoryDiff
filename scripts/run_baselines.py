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
from scipy import stats
from skimage.metrics import structural_similarity
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data import RadioMapDataModule
from data.floor_plan import png_to_db
from evaluation.metrics import compute_all_metrics
from models.baselines.interpolation import get_all_baselines

try:
    import torch
    from models.baselines.supervised_unet import SupervisedUNetBaseline
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def denormalize_to_dbm(x: np.ndarray, min_val: float = -186.0, max_val: float = -47.0) -> np.ndarray:
    """Denormalize from [-1, 1] to dBm scale (RadioMapSeer: [-186, -47])."""
    return ((x + 1) / 2) * (max_val - min_val) + min_val


def compute_significance(
    metrics_a: list,
    metrics_b: list,
    n_bootstrap: int = 1000,
) -> dict:
    """
    Compute statistical significance between two sets of metrics (C4).

    Uses Wilcoxon signed-rank test for paired samples and bootstrap for 95% CI.
    """
    metrics_a = np.array(metrics_a)
    metrics_b = np.array(metrics_b)

    # Wilcoxon signed-rank test (paired)
    stat, p_value = stats.wilcoxon(metrics_a, metrics_b, alternative='two-sided')

    # Bootstrap 95% CI for mean difference
    diffs = metrics_a - metrics_b
    mean_diff = float(np.mean(diffs))

    bootstrap_means = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        indices = rng.choice(len(diffs), size=len(diffs), replace=True)
        bootstrap_means.append(np.mean(diffs[indices]))

    ci_lower = float(np.percentile(bootstrap_means, 2.5))
    ci_upper = float(np.percentile(bootstrap_means, 97.5))

    return {
        'p_value': float(p_value),
        'mean_diff': mean_diff,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'significant': p_value < 0.05,
    }


def evaluate_baselines(
    data_dir: str = "data/raw",
    output_path: str = "experiments/eval_results/baselines.json",
    max_samples: int = 0,
    batch_size: int = 1,
    num_workers: int = 4,
    reference_results: str = None,
    supervised_checkpoint: str = None,  # C3: Add supervised baseline
) -> dict:
    """Run all classical baselines (and optionally supervised U-Net) on the test set."""

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

    # C3: Add supervised U-Net if checkpoint provided
    supervised_model = None
    if supervised_checkpoint and TORCH_AVAILABLE:
        ckpt_path = Path(supervised_checkpoint)
        if ckpt_path.exists():
            print(f"Loading supervised U-Net from {supervised_checkpoint}")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            supervised_model = SupervisedUNetBaseline.load_from_checkpoint(
                supervised_checkpoint,
                map_location=device,
            )
            supervised_model.eval()
            supervised_model.to(device)
            # Add to baselines dict (but handle separately since it needs batching)
            print("Supervised U-Net loaded successfully")
        else:
            print(f"Warning: Supervised checkpoint not found: {supervised_checkpoint}")
    elif supervised_checkpoint and not TORCH_AVAILABLE:
        print("Warning: PyTorch not available, skipping supervised baseline")

    print(f"Classical baselines: {list(baselines.keys())}")
    if supervised_model:
        print("+ Supervised U-Net")

    # Results storage
    all_baseline_names = list(baselines.keys())
    if supervised_model:
        all_baseline_names.append("supervised_unet")

    results = {name: {
        "rmse": [], "mae": [], "ssim": [], "rmse_observed": [], "rmse_unobserved": [],
        "rmse_dbm": [], "mae_dbm": [], "rmse_observed_dbm": [], "rmse_unobserved_dbm": [],
        # Per-region dBm metrics (fair comparison)
        "rmse_free_space_dbm": [], "rmse_building_dbm": [],
        "mae_free_space_dbm": [], "mae_building_dbm": [],
        "rmse_free_unobs_dbm": [], "rmse_free_obs_dbm": [],
        "ssim_free_space": [],
    } for name in all_baseline_names}
    timing = {name: 0.0 for name in all_baseline_names}

    total = 0
    for batch in tqdm(test_loader, desc="Evaluating baselines"):
        if max_samples > 0 and total >= max_samples:
            break

        # C3: Evaluate supervised U-Net on full batch (GPU-accelerated)
        if supervised_model:
            device = next(supervised_model.parameters()).device
            with torch.no_grad():
                batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                condition = {
                    'building_map': batch_gpu.get('building_map'),
                    'sparse_rss': batch_gpu.get('sparse_rss'),
                    'trajectory_mask': batch_gpu.get('trajectory_mask'),
                    'coverage_density': batch_gpu.get('coverage_density'),
                    'tx_position': batch_gpu.get('tx_position'),
                }
                condition = {k: v for k, v in condition.items() if v is not None}

                t0 = time.time()
                pred_supervised = supervised_model(condition).cpu().numpy()
                timing["supervised_unet"] += time.time() - t0

        for i in range(batch["radio_map"].shape[0]):
            # Extract numpy arrays (squeeze channel dim)
            gt = batch["radio_map"][i, 0].numpy()
            sparse_rss = batch["sparse_rss"][i, 0].numpy()
            traj_mask = batch["trajectory_mask"][i, 0].numpy()
            building_map = batch["building_map"][i, 0].numpy()

            # Region masks (building_map in [-1,1]: >0 = free space, <=0 = building)
            free_space_mask = building_map > 0.0
            building_mask = ~free_space_mask
            obs_mask = traj_mask > 0.5
            free_obs_mask = free_space_mask & obs_mask
            free_unobs_mask = free_space_mask & ~obs_mask

            # Evaluate classical baselines
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

                # Per-region dBm metrics
                if free_space_mask.any():
                    results[name]["rmse_free_space_dbm"].append(
                        float(np.sqrt(np.mean(diff_dbm[free_space_mask] ** 2)))
                    )
                    results[name]["mae_free_space_dbm"].append(
                        float(np.mean(np.abs(diff_dbm[free_space_mask])))
                    )
                    # SSIM on free-space dBm
                    fs_pred = pred_dbm.copy()
                    fs_gt = gt_dbm.copy()
                    # Mask buildings to zero for SSIM (only free space matters)
                    fs_pred[building_mask] = 0.0
                    fs_gt[building_mask] = 0.0
                    ssim_fs = structural_similarity(
                        fs_gt, fs_pred, data_range=139.0,
                    )
                    results[name]["ssim_free_space"].append(float(ssim_fs))
                if building_mask.any():
                    results[name]["rmse_building_dbm"].append(
                        float(np.sqrt(np.mean(diff_dbm[building_mask] ** 2)))
                    )
                    results[name]["mae_building_dbm"].append(
                        float(np.mean(np.abs(diff_dbm[building_mask])))
                    )
                if obs_mask.any():
                    results[name]["rmse_observed_dbm"].append(
                        float(np.sqrt(np.mean(diff_dbm[obs_mask] ** 2)))
                    )
                if (~obs_mask).any():
                    results[name]["rmse_unobserved_dbm"].append(
                        float(np.sqrt(np.mean(diff_dbm[~obs_mask] ** 2)))
                    )
                if free_obs_mask.any():
                    results[name]["rmse_free_obs_dbm"].append(
                        float(np.sqrt(np.mean(diff_dbm[free_obs_mask] ** 2)))
                    )
                if free_unobs_mask.any():
                    results[name]["rmse_free_unobs_dbm"].append(
                        float(np.sqrt(np.mean(diff_dbm[free_unobs_mask] ** 2)))
                    )

            # C3: Process supervised U-Net results
            if supervised_model:
                pred = pred_supervised[i, 0]
                metrics = compute_all_metrics(pred, gt, trajectory_mask=traj_mask)

                results["supervised_unet"]["rmse"].append(metrics["rmse"])
                results["supervised_unet"]["mae"].append(metrics["mae"])
                if "ssim" in metrics:
                    results["supervised_unet"]["ssim"].append(metrics["ssim"])
                if "rmse_observed" in metrics:
                    results["supervised_unet"]["rmse_observed"].append(metrics["rmse_observed"])
                if "rmse_unobserved" in metrics:
                    results["supervised_unet"]["rmse_unobserved"].append(metrics["rmse_unobserved"])

                # dBm-scale metrics (same per-region breakdown)
                pred_dbm = denormalize_to_dbm(pred)
                gt_dbm = denormalize_to_dbm(gt)
                diff_dbm = pred_dbm - gt_dbm
                results["supervised_unet"]["rmse_dbm"].append(float(np.sqrt(np.mean(diff_dbm ** 2))))
                results["supervised_unet"]["mae_dbm"].append(float(np.mean(np.abs(diff_dbm))))
                if free_space_mask.any():
                    results["supervised_unet"]["rmse_free_space_dbm"].append(
                        float(np.sqrt(np.mean(diff_dbm[free_space_mask] ** 2)))
                    )
                    results["supervised_unet"]["mae_free_space_dbm"].append(
                        float(np.mean(np.abs(diff_dbm[free_space_mask])))
                    )
                if building_mask.any():
                    results["supervised_unet"]["rmse_building_dbm"].append(
                        float(np.sqrt(np.mean(diff_dbm[building_mask] ** 2)))
                    )
                    results["supervised_unet"]["mae_building_dbm"].append(
                        float(np.mean(np.abs(diff_dbm[building_mask])))
                    )
                if obs_mask.any():
                    results["supervised_unet"]["rmse_observed_dbm"].append(
                        float(np.sqrt(np.mean(diff_dbm[obs_mask] ** 2)))
                    )
                if (~obs_mask).any():
                    results["supervised_unet"]["rmse_unobserved_dbm"].append(
                        float(np.sqrt(np.mean(diff_dbm[~obs_mask] ** 2)))
                    )
                if free_obs_mask.any():
                    results["supervised_unet"]["rmse_free_obs_dbm"].append(
                        float(np.sqrt(np.mean(diff_dbm[free_obs_mask] ** 2)))
                    )
                if free_unobs_mask.any():
                    results["supervised_unet"]["rmse_free_unobs_dbm"].append(
                        float(np.sqrt(np.mean(diff_dbm[free_unobs_mask] ** 2)))
                    )

            total += 1

    # Aggregate
    summary = {}
    for name in all_baseline_names:
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
            summary[name]["rmse_observed_dbm_std"] = float(np.nanstd(r["rmse_observed_dbm"]))
        if r["rmse_unobserved_dbm"]:
            summary[name]["rmse_unobserved_dbm_mean"] = float(np.nanmean(r["rmse_unobserved_dbm"]))
            summary[name]["rmse_unobserved_dbm_std"] = float(np.nanstd(r["rmse_unobserved_dbm"]))
        # Per-region aggregates
        for key in ["rmse_free_space_dbm", "mae_free_space_dbm", "rmse_building_dbm",
                     "mae_building_dbm", "rmse_free_obs_dbm", "rmse_free_unobs_dbm",
                     "ssim_free_space"]:
            if r[key]:
                summary[name][f"{key}_mean"] = float(np.nanmean(r[key]))
                summary[name][f"{key}_std"] = float(np.nanstd(r[key]))

    # Print results (dBm as primary)
    print("\n" + "=" * 100)
    print("BASELINE RESULTS — ALL PIXELS (dBm scale)")
    print("=" * 100)
    print(f"{'Method':<25} {'RMSE(dBm)':>10} {'MAE(dBm)':>10} {'SSIM':>8} {'Time(ms)':>10}")
    print("-" * 100)
    for name, s in summary.items():
        ssim_str = f"{s.get('ssim_mean', float('nan')):.4f}"
        rmse_dbm = s.get('rmse_dbm_mean', float('nan'))
        mae_dbm = s.get('mae_dbm_mean', float('nan'))
        print(f"{name:<25} {rmse_dbm:>10.2f} {mae_dbm:>10.2f} {ssim_str:>8} {s['avg_time_ms']:>10.1f}")

    print("\n" + "=" * 100)
    print("BASELINE RESULTS — PER-REGION BREAKDOWN (dBm scale)")
    print("=" * 100)
    print(f"{'Method':<25} {'Free RMSE':>10} {'Bldg RMSE':>10} {'Obs RMSE':>10} {'Unobs RMSE':>12} {'FreeUnobs':>10} {'Free SSIM':>10}")
    print("-" * 100)
    for name, s in summary.items():
        fs_rmse = s.get('rmse_free_space_dbm_mean', float('nan'))
        bl_rmse = s.get('rmse_building_dbm_mean', float('nan'))
        ob_rmse = s.get('rmse_observed_dbm_mean', float('nan'))
        uo_rmse = s.get('rmse_unobserved_dbm_mean', float('nan'))
        fu_rmse = s.get('rmse_free_unobs_dbm_mean', float('nan'))
        fs_ssim = s.get('ssim_free_space_mean', float('nan'))
        print(f"{name:<25} {fs_rmse:>10.2f} {bl_rmse:>10.2f} {ob_rmse:>10.2f} {uo_rmse:>12.2f} {fu_rmse:>10.2f} {fs_ssim:>10.4f}")

    print(f"\n{'Method':<25} {'RMSE(norm)':>10} {'MAE(norm)':>10}")
    print("-" * 50)
    for name, s in summary.items():
        print(f"{name:<25} {s['rmse_mean']:>10.4f} {s['mae_mean']:>10.4f}")

    print("\n[Legend]")
    print("  Free RMSE     = Free-space pixels only (streets, walkable areas)")
    print("  Bldg RMSE     = Building pixels only")
    print("  Obs RMSE      = Trajectory-observed pixels")
    print("  Unobs RMSE    = All unobserved pixels")
    print("  FreeUnobs     = Free-space unobserved (FAIR comparison metric)")
    print("  Free SSIM     = SSIM on free-space region (dBm scale, range=139)")

    # C4: Compare against reference (diffusion model) if provided
    if reference_results and Path(reference_results).exists():
        print("\n" + "=" * 80)
        print("STATISTICAL SIGNIFICANCE vs Reference Model")
        print("=" * 80)
        with open(reference_results) as f:
            ref_data = json.load(f)

        if 'per_sample_rmse_dbm' in ref_data:
            ref_rmse = ref_data['per_sample_rmse_dbm']
            print(f"Reference RMSE: {np.mean(ref_rmse):.2f} dBm")
            print(f"\n{'Method':<25} {'p-value':>10} {'Mean Diff':>12} {'95% CI':>20} {'Significant':>12}")
            print("-" * 80)

            for name in baselines:
                if results[name]["rmse_dbm"]:
                    baseline_rmse = results[name]["rmse_dbm"][:len(ref_rmse)]  # Ensure same length
                    sig = compute_significance(baseline_rmse, ref_rmse)
                    ci_str = f"[{sig['ci_lower']:.2f}, {sig['ci_upper']:.2f}]"
                    sig_mark = "**" if sig['significant'] else ""
                    print(f"{name:<25} {sig['p_value']:>10.4f} {sig['mean_diff']:>12.2f} {ci_str:>20} {sig_mark:>12}")

            # Store significance in summary
            for name in all_baseline_names:
                if results[name]["rmse_dbm"]:
                    baseline_rmse = results[name]["rmse_dbm"][:len(ref_rmse)]
                    sig = compute_significance(baseline_rmse, ref_rmse)
                    summary[name]['significance_vs_reference'] = sig

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
    parser.add_argument("--reference-results", default=None, help="Reference model results JSON for significance testing (C4)")
    parser.add_argument("--supervised-checkpoint", default=None, help="Supervised U-Net checkpoint for evaluation (C3)")
    args = parser.parse_args()

    evaluate_baselines(
        data_dir=args.data_dir,
        output_path=args.output,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        reference_results=args.reference_results,
        supervised_checkpoint=args.supervised_checkpoint,
    )
