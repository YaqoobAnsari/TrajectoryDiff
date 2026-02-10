#!/usr/bin/env python3
"""
TrajectoryDiff: Paper Figure Generation

Generates publication-quality figures from experiment results:
1. Qualitative comparison (GT vs predictions vs baselines)
2. Uncertainty maps
3. Calibration plots
4. Ablation bar charts
5. Coverage sweep curves

Usage:
    python scripts/generate_figures.py --results-dir experiments/eval_results --output-dir figures
    python scripts/generate_figures.py --checkpoint experiments/trajectory_full/checkpoints/best.ckpt
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.visualization import set_style, RADIO_CMAP, ERROR_CMAP


def load_json(path: Path) -> dict:
    """Load a JSON file."""
    with open(path) as f:
        return json.load(f)


# ============================================================
# Figure 1: Ablation Bar Chart
# ============================================================
def plot_ablation_chart(results_dir: Path, output_dir: Path):
    """Generate ablation study bar chart from experiment metrics."""
    ablation_experiments = [
        ("trajectory_full", "Full Model"),
        ("ablation_no_physics_loss", "- Physics Loss"),
        ("ablation_no_coverage_attention", "- Coverage Attn"),
        ("ablation_no_trajectory_mask", "- Traj Mask"),
        ("ablation_no_coverage_density", "- Coverage Density"),
        ("ablation_no_tx_position", "- TX Position"),
        ("ablation_small_unet", "Small UNet"),
    ]

    names = []
    rmse_vals = []
    found = False

    for exp_name, label in ablation_experiments:
        metrics_path = results_dir / f"{exp_name}_eval.json"
        if not metrics_path.exists():
            # Try alternative naming
            metrics_path = results_dir / exp_name / "metrics.json"
        if metrics_path.exists():
            metrics = load_json(metrics_path)
            rmse_key = "rmse_dbm" if "rmse_dbm" in metrics else "rmse"
            names.append(label)
            rmse_vals.append(metrics[rmse_key])
            found = True

    if not found:
        print("  [SKIP] No ablation metrics found")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2ecc71"] + ["#e74c3c"] * (len(names) - 1)
    bars = ax.barh(range(len(names)), rmse_vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("RMSE (dBm)")
    ax.set_title("Ablation Study: Component Contributions")
    ax.invert_yaxis()

    # Add value labels
    for bar, val in zip(bars, rmse_vals):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=10)

    plt.tight_layout()
    fig.savefig(output_dir / "ablation_bar_chart.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "ablation_bar_chart.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved ablation_bar_chart.pdf")


# ============================================================
# Figure 2: Coverage Sweep Curve
# ============================================================
def plot_coverage_sweep(results_dir: Path, output_dir: Path):
    """Generate coverage vs RMSE curve."""
    coverage_levels = [1, 5, 10, 20]
    exp_names = [f"coverage_sweep_{c}pct" for c in coverage_levels]

    coverages = []
    rmse_vals = []
    found = False

    for cov, exp_name in zip(coverage_levels, exp_names):
        metrics_path = results_dir / f"{exp_name}_eval.json"
        if not metrics_path.exists():
            metrics_path = results_dir / exp_name / "metrics.json"
        if metrics_path.exists():
            metrics = load_json(metrics_path)
            rmse_key = "rmse_dbm" if "rmse_dbm" in metrics else "rmse"
            coverages.append(cov)
            rmse_vals.append(metrics[rmse_key])
            found = True

    if not found:
        print("  [SKIP] No coverage sweep metrics found")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(coverages, rmse_vals, "b-o", linewidth=2, markersize=8, label="TrajectoryDiff")
    ax.set_xlabel("Coverage (%)")
    ax.set_ylabel("RMSE (dBm)")
    ax.set_title("Effect of Coverage on Reconstruction Quality")
    ax.set_xticks(coverage_levels)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "coverage_sweep.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "coverage_sweep.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved coverage_sweep.pdf")


# ============================================================
# Figure 3: Main Results Comparison Table (as figure)
# ============================================================
def plot_main_results_table(results_dir: Path, output_dir: Path):
    """Generate main results comparison table as a figure."""
    experiments = [
        ("baselines", "Classical Baselines"),
        ("trajectory_full", "TrajectoryDiff (Full)"),
        ("trajectory_baseline", "Trajectory Baseline"),
        ("uniform_baseline", "Uniform Baseline"),
    ]

    rows = []

    # Load baselines
    baselines_path = results_dir / "baselines.json"
    if baselines_path.exists():
        baselines = load_json(baselines_path)
        for name, data in baselines.items():
            rows.append([
                name.replace("_", " ").title(),
                f"{data.get('rmse_mean', 'N/A'):.4f}" if isinstance(data.get('rmse_mean'), (int, float)) else "N/A",
                f"{data.get('mae_mean', 'N/A'):.4f}" if isinstance(data.get('mae_mean'), (int, float)) else "N/A",
                f"{data.get('ssim_mean', 'N/A'):.4f}" if isinstance(data.get('ssim_mean'), (int, float)) else "N/A",
            ])

    # Load model results
    for exp_name, label in experiments[1:]:
        metrics_path = results_dir / f"{exp_name}_eval.json"
        if not metrics_path.exists():
            metrics_path = results_dir / exp_name / "metrics.json"
        if metrics_path.exists():
            m = load_json(metrics_path)
            rows.append([
                label,
                f"{m.get('rmse_dbm', m.get('rmse', 'N/A')):.2f}",
                f"{m.get('mae_dbm', m.get('mae', 'N/A')):.2f}",
                f"{m.get('ssim', 'N/A'):.4f}" if isinstance(m.get('ssim'), (int, float)) else "N/A",
            ])

    if not rows:
        print("  [SKIP] No main results found")
        return

    fig, ax = plt.subplots(figsize=(10, max(3, len(rows) * 0.5 + 1.5)))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Method", "RMSE (dBm)", "MAE (dBm)", "SSIM"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)

    # Style header
    for j in range(4):
        table[0, j].set_facecolor("#3498db")
        table[0, j].set_text_props(color="white", fontweight="bold")

    ax.set_title("Main Results Comparison", fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()
    fig.savefig(output_dir / "main_results_table.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "main_results_table.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved main_results_table.pdf")


# ============================================================
# Figure 4: Qualitative Comparison Grid
# ============================================================
def plot_qualitative_comparison(checkpoint_path: str, output_dir: Path, num_samples: int = 4):
    """Generate qualitative comparison figure from a trained model."""
    import torch

    from data import RadioMapDataModule
    from training import DiffusionInference, denormalize_radio_map
    from models.baselines.interpolation import IDWBaseline, NearestNeighborBaseline

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"  Loading model from {checkpoint_path}")
    inference = DiffusionInference.from_checkpoint(checkpoint_path, device=device, use_ema=True)

    datamodule = RadioMapDataModule(
        data_dir="data/raw",
        batch_size=num_samples,
        num_workers=4,
        sampling_strategy="trajectory",
    )
    datamodule.setup("test")
    batch = next(iter(datamodule.test_dataloader()))
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    condition = {
        "building_map": batch.get("building_map"),
        "sparse_rss": batch.get("sparse_rss"),
        "trajectory_mask": batch.get("trajectory_mask"),
        "coverage_density": batch.get("coverage_density"),
        "tx_position": batch.get("tx_position"),
    }
    condition = {k: v for k, v in condition.items() if v is not None}

    print("  Generating samples...")
    with torch.no_grad():
        pred = inference.sample(condition, use_ddim=True, progress=False).cpu().numpy()

    gt = batch["radio_map"].cpu().numpy()
    building = batch["building_map"].cpu().numpy()
    sparse = batch["sparse_rss"].cpu().numpy()
    traj_mask = batch["trajectory_mask"].cpu().numpy()

    # Run classical baselines
    idw = IDWBaseline()
    nn_baseline = NearestNeighborBaseline()

    n = min(num_samples, pred.shape[0])

    # Column order: Building, Trajectory, GT, Ours, IDW, NN, Error
    fig, axes = plt.subplots(n, 7, figsize=(28, 4 * n))
    if n == 1:
        axes = axes[None, :]

    col_titles = ["Building", "Trajectory", "Ground Truth", "Ours", "IDW", "NN", "|Error| (Ours)"]

    for i in range(n):
        bm = building[i, 0]
        gt_i = gt[i, 0]
        pred_i = pred[i, 0]
        sp = sparse[i, 0]
        mask = traj_mask[i, 0]

        vmin = min(gt_i.min(), pred_i.min())
        vmax = max(gt_i.max(), pred_i.max())

        # Building map
        axes[i, 0].imshow(bm, cmap="gray")

        # Trajectory mask
        axes[i, 1].imshow(bm, cmap="gray", alpha=0.5)
        ys, xs = np.where(mask > 0)
        axes[i, 1].scatter(xs, ys, c="red", s=3, alpha=0.6)

        # Ground truth
        axes[i, 2].imshow(gt_i, cmap=RADIO_CMAP, vmin=vmin, vmax=vmax)

        # Our prediction
        axes[i, 3].imshow(pred_i, cmap=RADIO_CMAP, vmin=vmin, vmax=vmax)

        # IDW
        idw_pred = idw(sp, mask)
        axes[i, 4].imshow(idw_pred, cmap=RADIO_CMAP, vmin=vmin, vmax=vmax)

        # Nearest Neighbor
        nn_pred = nn_baseline(sp, mask)
        axes[i, 5].imshow(nn_pred, cmap=RADIO_CMAP, vmin=vmin, vmax=vmax)

        # Error map
        error = np.abs(gt_i - pred_i)
        axes[i, 6].imshow(error, cmap=ERROR_CMAP)

        for j in range(7):
            axes[i, j].axis("off")
            if i == 0:
                axes[i, j].set_title(col_titles[j], fontsize=12)

    plt.tight_layout()
    fig.savefig(output_dir / "qualitative_comparison.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "qualitative_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved qualitative_comparison.pdf")


# ============================================================
# Figure 5: Uncertainty Visualization
# ============================================================
def plot_uncertainty_figure(checkpoint_path: str, output_dir: Path, num_diffusion_samples: int = 10):
    """Generate uncertainty visualization from a trained model."""
    import torch

    from data import RadioMapDataModule
    from training import DiffusionInference

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inference = DiffusionInference.from_checkpoint(checkpoint_path, device=device, use_ema=True)

    datamodule = RadioMapDataModule(
        data_dir="data/raw",
        batch_size=3,
        num_workers=4,
        sampling_strategy="trajectory",
    )
    datamodule.setup("test")
    batch = next(iter(datamodule.test_dataloader()))
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    condition = {k: v for k, v in {
        "building_map": batch.get("building_map"),
        "sparse_rss": batch.get("sparse_rss"),
        "trajectory_mask": batch.get("trajectory_mask"),
        "coverage_density": batch.get("coverage_density"),
        "tx_position": batch.get("tx_position"),
    }.items() if v is not None}

    print(f"  Generating {num_diffusion_samples} samples for uncertainty...")
    samples_list = []
    with torch.no_grad():
        for _ in range(num_diffusion_samples):
            s = inference.sample(condition, use_ddim=True, progress=False)
            samples_list.append(s.cpu())

    samples = torch.stack(samples_list, dim=0)  # [K, B, 1, H, W]
    mean_pred = samples.mean(dim=0).numpy()
    std_pred = samples.std(dim=0).numpy()
    gt = batch["radio_map"].cpu().numpy()
    traj_mask = batch["trajectory_mask"].cpu().numpy()

    n = mean_pred.shape[0]
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes[None, :]

    col_titles = ["Trajectory Mask", "Mean Prediction", "Uncertainty (Std)", "Actual Error"]

    for i in range(n):
        mask = traj_mask[i, 0]
        mean_i = mean_pred[i, 0]
        std_i = std_pred[i, 0]
        gt_i = gt[i, 0]
        error_i = np.abs(mean_i - gt_i)

        axes[i, 0].imshow(mask, cmap="hot")
        axes[i, 1].imshow(mean_i, cmap=RADIO_CMAP)
        im2 = axes[i, 2].imshow(std_i, cmap="plasma")
        plt.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)
        im3 = axes[i, 3].imshow(error_i, cmap=ERROR_CMAP)
        plt.colorbar(im3, ax=axes[i, 3], fraction=0.046, pad=0.04)

        for j in range(4):
            axes[i, j].axis("off")
            if i == 0:
                axes[i, j].set_title(col_titles[j], fontsize=12)

    fig.suptitle("Uncertainty Estimation via Diffusion Sampling", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / "uncertainty_visualization.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "uncertainty_visualization.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved uncertainty_visualization.pdf")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--results-dir", default="experiments/eval_results",
                        help="Directory with evaluation results JSON files")
    parser.add_argument("--output-dir", default="figures", help="Output directory for figures")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to best checkpoint (for qualitative + uncertainty figures)")
    parser.add_argument("--num-samples", type=int, default=4, help="Samples for qualitative figure")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_style()

    print("=" * 60)
    print("TrajectoryDiff: Paper Figure Generation")
    print("=" * 60)
    print(f"Results dir: {results_dir}")
    print(f"Output dir:  {output_dir}")
    print()

    # Figures from metrics JSON files
    print("Generating metrics-based figures...")
    plot_ablation_chart(results_dir, output_dir)
    plot_coverage_sweep(results_dir, output_dir)
    plot_main_results_table(results_dir, output_dir)

    # Figures requiring model checkpoint
    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"\nGenerating model-based figures (checkpoint: {args.checkpoint})...")
        plot_qualitative_comparison(args.checkpoint, output_dir, num_samples=args.num_samples)
        plot_uncertainty_figure(args.checkpoint, output_dir)
    elif args.checkpoint:
        print(f"\nWARNING: Checkpoint not found: {args.checkpoint}")
        print("  Skipping qualitative comparison and uncertainty figures.")
    else:
        print("\nNo checkpoint provided, skipping model-based figures.")
        print("  Use --checkpoint to generate qualitative and uncertainty figures.")

    print(f"\nAll figures saved to: {output_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
