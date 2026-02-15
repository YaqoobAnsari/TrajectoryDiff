#!/usr/bin/env python3
"""
TrajectoryDiff Evaluation Script

Evaluates trained models on test set and generates visualizations.

Usage:
    python scripts/evaluate.py checkpoint=/path/to/model.ckpt
    python scripts/evaluate.py checkpoint=/path/to/model.ckpt data.sampling.strategy=uniform
    python scripts/evaluate.py checkpoint=/path/to/model.ckpt --num-samples 100

Examples:
    # Evaluate on test set
    python scripts/evaluate.py checkpoint=experiments/baseline/checkpoints/last.ckpt

    # Evaluate with uncertainty estimation
    python scripts/evaluate.py checkpoint=model.ckpt +uncertainty=true +num_uncertainty_samples=10

    # Compare trajectory vs uniform sampling
    python scripts/evaluate.py checkpoint=model.ckpt data.sampling.strategy=uniform
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from scipy import stats
from tqdm import tqdm

from data import RadioMapDataModule
from evaluation.metrics import compute_sample_diversity, compute_masked_ssim, compute_masked_psnr
from training import DiffusionModule, DiffusionInference, denormalize_radio_map


def compute_rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute Root Mean Square Error."""
    return torch.sqrt(((pred - target) ** 2).mean(dim=(1, 2, 3)))


def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute Mean Absolute Error."""
    return (pred - target).abs().mean(dim=(1, 2, 3))


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 2.0) -> torch.Tensor:
    """Compute Peak Signal-to-Noise Ratio."""
    mse = ((pred - target) ** 2).mean(dim=(1, 2, 3))
    return 10 * torch.log10(max_val ** 2 / (mse + 1e-8))


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 2.0,
) -> torch.Tensor:
    """Compute SSIM using skimage (proper sliding-window implementation).

    Args:
        pred: Predicted tensor (B, 1, H, W) in [-1, 1]
        target: Target tensor (B, 1, H, W) in [-1, 1]
        data_range: Value range of the data (2.0 for [-1, 1])

    Returns:
        Per-sample SSIM values (B,)
    """
    from evaluation.metrics import ssim as ssim_metric

    batch_ssim = []
    for i in range(pred.shape[0]):
        p = pred[i].squeeze().cpu().numpy()
        t = target[i].squeeze().cpu().numpy()
        batch_ssim.append(ssim_metric(p, t, data_range=data_range))
    return torch.tensor(batch_ssim)


def compute_significance(
    metrics_a: List[float],
    metrics_b: List[float],
    n_bootstrap: int = 1000,
) -> Dict[str, float]:
    """
    Compute statistical significance between two sets of metrics (C4).

    Uses Wilcoxon signed-rank test for paired samples and bootstrap for 95% CI.

    Args:
        metrics_a: Per-sample metrics from method A
        metrics_b: Per-sample metrics from method B (paired with A)
        n_bootstrap: Number of bootstrap resamples

    Returns:
        Dict with p_value, ci_lower, ci_upper, mean_diff
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


def compute_trajectory_rmse(
    pred: torch.Tensor,
    target: torch.Tensor,
    trajectory_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute RMSE only on trajectory points (observed regions)."""
    mask = trajectory_mask > 0
    if mask.sum() == 0:
        return torch.zeros(pred.shape[0])

    errors = []
    for i in range(pred.shape[0]):
        m = mask[i]
        if m.sum() > 0:
            error = torch.sqrt(((pred[i][m] - target[i][m]) ** 2).mean())
            errors.append(error)
        else:
            errors.append(torch.tensor(0.0))

    return torch.stack(errors)


def compute_blind_spot_rmse(
    pred: torch.Tensor,
    target: torch.Tensor,
    trajectory_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute RMSE on blind spots (unobserved regions)."""
    mask = trajectory_mask == 0
    if mask.sum() == 0:
        return torch.zeros(pred.shape[0])

    errors = []
    for i in range(pred.shape[0]):
        m = mask[i]
        if m.sum() > 0:
            error = torch.sqrt(((pred[i][m] - target[i][m]) ** 2).mean())
            errors.append(error)
        else:
            errors.append(torch.tensor(0.0))

    return torch.stack(errors)


@torch.no_grad()
def evaluate_model(
    inference: DiffusionInference,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_samples: Optional[int] = None,
    use_ddim: bool = True,
    compute_uncertainty: bool = False,
    num_uncertainty_samples: int = 10,
    compute_diversity: bool = False,
    num_diversity_samples: int = 5,
) -> Dict[str, float]:
    """
    Evaluate model on dataloader.

    Args:
        inference: DiffusionInference instance
        dataloader: Test dataloader
        device: Computation device
        max_samples: Maximum samples to evaluate
        use_ddim: Use DDIM sampling
        compute_uncertainty: Whether to compute uncertainty estimates
        num_uncertainty_samples: Number of samples for uncertainty

    Returns:
        Dictionary of metrics
    """
    all_rmse = []
    all_mae = []
    all_psnr = []
    all_ssim = []
    all_traj_rmse = []
    all_blind_rmse = []
    all_uncertainty = []
    all_diversity = []  # C6
    # M10: Per-region metrics
    all_ssim_observed = []
    all_ssim_unobserved = []
    all_psnr_observed = []
    all_psnr_unobserved = []

    total_samples = 0

    for batch in tqdm(dataloader, desc="Evaluating"):
        if max_samples and total_samples >= max_samples:
            break

        # Move to device
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # Extract condition and ground truth
        ground_truth = batch['radio_map']
        condition = {
            'building_map': batch.get('building_map'),
            'sparse_rss': batch.get('sparse_rss'),
            'trajectory_mask': batch.get('trajectory_mask'),
            'coverage_density': batch.get('coverage_density'),
            'tx_position': batch.get('tx_position'),
        }
        condition = {k: v for k, v in condition.items() if v is not None}

        # Generate samples (C6: with diversity option)
        if compute_uncertainty or compute_diversity:
            num_gen_samples = max(num_uncertainty_samples, num_diversity_samples)
            samples_list = []
            for _ in range(num_gen_samples):
                s = inference.sample(condition, use_ddim=use_ddim, progress=False)
                samples_list.append(s)
            samples_tensor = torch.stack(samples_list, dim=0)  # (K, B, 1, H, W)
            samples = samples_tensor.mean(dim=0)  # Mean prediction

            if compute_uncertainty:
                uncertainty = samples_tensor.std(dim=0)
                all_uncertainty.append(uncertainty.mean(dim=(1, 2, 3)).cpu())

            # C6: Compute diversity
            if compute_diversity:
                for b in range(samples_tensor.shape[1]):  # For each batch item
                    diversity = compute_sample_diversity(samples_tensor[:, b])
                    all_diversity.append(diversity['mean_std'])
        else:
            samples = inference.sample(condition, use_ddim=use_ddim, progress=False)

        # Denormalize to dBm for paper-ready metrics
        samples_dbm = denormalize_radio_map(samples)
        gt_dbm = denormalize_radio_map(ground_truth)

        # Compute metrics in dBm scale (C7: both PSNR and SSIM on same scale)
        all_rmse.append(compute_rmse(samples_dbm, gt_dbm).cpu())
        all_mae.append(compute_mae(samples_dbm, gt_dbm).cpu())
        all_psnr.append(compute_psnr(samples_dbm, gt_dbm, max_val=139.0).cpu())  # dBm range [-186, -47]
        all_ssim.append(compute_ssim(samples_dbm, gt_dbm, data_range=139.0).cpu())  # dBm scale

        # Trajectory-aware metrics (dBm scale)
        if 'trajectory_mask' in batch:
            traj_mask = batch['trajectory_mask']
            all_traj_rmse.append(compute_trajectory_rmse(samples_dbm, gt_dbm, traj_mask).cpu())
            all_blind_rmse.append(compute_blind_spot_rmse(samples_dbm, gt_dbm, traj_mask).cpu())

            # M10: Per-region SSIM/PSNR
            for b in range(samples_dbm.shape[0]):
                mask = traj_mask[b, 0].cpu().numpy()
                pred_np = samples_dbm[b, 0].cpu().numpy()
                gt_np = gt_dbm[b, 0].cpu().numpy()

                # Observed region
                observed_mask = mask > 0.5
                if observed_mask.any():
                    all_ssim_observed.append(compute_masked_ssim(pred_np, gt_np, observed_mask, data_range=139.0))
                    all_psnr_observed.append(compute_masked_psnr(pred_np, gt_np, observed_mask, data_range=139.0))

                # Unobserved region
                unobserved_mask = mask <= 0.5
                if unobserved_mask.any():
                    all_ssim_unobserved.append(compute_masked_ssim(pred_np, gt_np, unobserved_mask, data_range=139.0))
                    all_psnr_unobserved.append(compute_masked_psnr(pred_np, gt_np, unobserved_mask, data_range=139.0))

        total_samples += ground_truth.shape[0]

    # Aggregate metrics
    rmse_values = torch.cat(all_rmse).cpu().numpy()
    mae_values = torch.cat(all_mae).cpu().numpy()
    psnr_values = torch.cat(all_psnr).cpu().numpy()
    ssim_values = torch.cat(all_ssim).cpu().numpy()

    metrics = {
        'rmse_dbm': float(rmse_values.mean()),
        'rmse_dbm_std': float(rmse_values.std()),
        'mae_dbm': float(mae_values.mean()),
        'mae_dbm_std': float(mae_values.std()),
        'psnr': float(psnr_values.mean()),
        'psnr_std': float(psnr_values.std()),
        'ssim': float(ssim_values.mean()),
        'ssim_std': float(ssim_values.std()),
        'num_samples': total_samples,
        # C4: Store per-sample metrics for significance testing
        'per_sample_rmse_dbm': rmse_values.tolist(),
        'per_sample_mae_dbm': mae_values.tolist(),
        'per_sample_psnr': psnr_values.tolist(),
        'per_sample_ssim': ssim_values.tolist(),
    }

    if all_traj_rmse:
        traj_rmse_values = torch.cat(all_traj_rmse).cpu().numpy()
        blind_rmse_values = torch.cat(all_blind_rmse).cpu().numpy()
        metrics['trajectory_rmse_dbm'] = float(traj_rmse_values.mean())
        metrics['blind_spot_rmse_dbm'] = float(blind_rmse_values.mean())
        metrics['per_sample_trajectory_rmse_dbm'] = traj_rmse_values.tolist()
        metrics['per_sample_blind_spot_rmse_dbm'] = blind_rmse_values.tolist()

    if all_uncertainty:
        uncertainty_values = torch.cat(all_uncertainty).cpu().numpy()
        metrics['mean_uncertainty'] = float(uncertainty_values.mean())
        metrics['per_sample_uncertainty'] = uncertainty_values.tolist()

    # C6: Diversity metrics
    if all_diversity:
        metrics['mean_diversity'] = float(np.mean(all_diversity))
        metrics['std_diversity'] = float(np.std(all_diversity))

    # M10: Per-region metrics
    if all_ssim_observed:
        metrics['ssim_observed'] = float(np.nanmean(all_ssim_observed))
        metrics['psnr_observed'] = float(np.nanmean(all_psnr_observed))
    if all_ssim_unobserved:
        metrics['ssim_unobserved'] = float(np.nanmean(all_ssim_unobserved))
        metrics['psnr_unobserved'] = float(np.nanmean(all_psnr_unobserved))

    return metrics


def save_visualizations(
    inference: DiffusionInference,
    dataloader: torch.utils.data.DataLoader,
    output_dir: Path,
    device: torch.device,
    num_samples: int = 8,
):
    """Generate and save visualization samples."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get a batch
    batch = next(iter(dataloader))
    batch = {
        k: v.to(device)[:num_samples] if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }

    condition = {
        'building_map': batch.get('building_map'),
        'sparse_rss': batch.get('sparse_rss'),
        'trajectory_mask': batch.get('trajectory_mask'),
        'coverage_density': batch.get('coverage_density'),
        'tx_position': batch.get('tx_position'),
    }
    condition = {k: v for k, v in condition.items() if v is not None}

    # Generate samples
    samples = inference.sample(condition, use_ddim=True, progress=True)

    # Create visualization
    n = min(num_samples, samples.shape[0])
    fig, axes = plt.subplots(n, 5, figsize=(20, 4 * n))

    if n == 1:
        axes = axes[None, :]

    for i in range(n):
        # Building map
        if 'building_map' in batch:
            axes[i, 0].imshow(batch['building_map'][i, 0].cpu().numpy(), cmap='gray')
            axes[i, 0].set_title('Building Map')

        # Sparse RSS
        if 'sparse_rss' in batch:
            im = axes[i, 1].imshow(batch['sparse_rss'][i, 0].cpu().numpy(), cmap='viridis')
            axes[i, 1].set_title('Sparse RSS')
            plt.colorbar(im, ax=axes[i, 1])

        # Trajectory mask
        if 'trajectory_mask' in batch:
            axes[i, 2].imshow(batch['trajectory_mask'][i, 0].cpu().numpy(), cmap='hot')
            axes[i, 2].set_title('Trajectory Mask')

        # Ground truth
        im = axes[i, 3].imshow(batch['radio_map'][i, 0].cpu().numpy(), cmap='viridis')
        axes[i, 3].set_title('Ground Truth')
        plt.colorbar(im, ax=axes[i, 3])

        # Generated
        im = axes[i, 4].imshow(samples[i, 0].cpu().numpy(), cmap='viridis')
        axes[i, 4].set_title('Generated')
        plt.colorbar(im, ax=axes[i, 4])

        for ax in axes[i]:
            ax.axis('off')

    plt.tight_layout()
    fig.savefig(output_dir / 'samples.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Create error map visualization
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))

    if n == 1:
        axes = axes[None, :]

    for i in range(n):
        gt = batch['radio_map'][i, 0].cpu().numpy()
        gen = samples[i, 0].cpu().numpy()
        error = np.abs(gt - gen)

        axes[i, 0].imshow(gt, cmap='viridis')
        axes[i, 0].set_title('Ground Truth')

        axes[i, 1].imshow(gen, cmap='viridis')
        axes[i, 1].set_title('Generated')

        im = axes[i, 2].imshow(error, cmap='hot')
        axes[i, 2].set_title(f'|Error| (RMSE: {np.sqrt((error**2).mean()):.4f})')
        plt.colorbar(im, ax=axes[i, 2])

        for ax in axes[i]:
            ax.axis('off')

    plt.tight_layout()
    fig.savefig(output_dir / 'error_maps.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Visualizations saved to {output_dir}")


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main evaluation function."""

    # Check for checkpoint
    if not cfg.get('checkpoint'):
        raise ValueError("Must provide checkpoint path: python evaluate.py checkpoint=/path/to/model.ckpt")

    checkpoint_path = Path(cfg.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print("=" * 60)
    print("TrajectoryDiff Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    print("\nLoading model...")
    inference = DiffusionInference.from_checkpoint(
        checkpoint_path,
        device=device,
        use_ema=True,
    )

    # Setup data
    print("Setting up data...")
    datamodule = RadioMapDataModule(
        data_dir=cfg.data.dataset.root,
        batch_size=cfg.data.loader.batch_size,
        num_workers=cfg.data.loader.num_workers,
        train_ratio=cfg.data.splits.train,
        val_ratio=cfg.data.splits.val,
        sampling_strategy=cfg.data.sampling.strategy,
        num_trajectories=cfg.data.sampling.trajectory.num_trajectories,
        points_per_trajectory=cfg.data.sampling.trajectory.points_per_trajectory,
        trajectory_method=cfg.data.sampling.trajectory.method,
        rss_noise_std=cfg.data.sampling.trajectory.rss_noise_std,
        position_noise_std=cfg.data.sampling.trajectory.get('position_noise_std', 0.5),
    )
    datamodule.setup('test')
    test_loader = datamodule.test_dataloader()

    # Evaluation settings
    max_samples = cfg.get('max_samples', None)
    compute_unc = cfg.get('uncertainty', False)
    num_unc_samples = cfg.get('num_uncertainty_samples', 10)
    compute_div = cfg.get('diversity', False)  # C6
    num_div_samples = cfg.get('num_diversity_samples', 5)  # C6

    # Evaluate
    print("\nEvaluating...")
    metrics = evaluate_model(
        inference=inference,
        dataloader=test_loader,
        device=device,
        max_samples=max_samples,
        use_ddim=True,
        compute_uncertainty=compute_unc,
        num_uncertainty_samples=num_unc_samples,
        compute_diversity=compute_div,
        num_diversity_samples=num_div_samples,
    )

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Samples evaluated: {metrics['num_samples']}")
    print(f"\nReconstruction Metrics (dBm scale):")
    print(f"  RMSE:  {metrics['rmse_dbm']:.2f} +/- {metrics['rmse_dbm_std']:.2f} dBm")
    print(f"  MAE:   {metrics['mae_dbm']:.2f} +/- {metrics['mae_dbm_std']:.2f} dBm")
    print(f"  PSNR:  {metrics['psnr']:.2f} +/- {metrics['psnr_std']:.2f} dB (dBm scale, range=139)")
    print(f"  SSIM:  {metrics['ssim']:.4f} +/- {metrics['ssim_std']:.4f}")

    if 'trajectory_rmse_dbm' in metrics:
        print(f"\nTrajectory-Aware Metrics (dBm scale):")
        print(f"  Trajectory RMSE:  {metrics['trajectory_rmse_dbm']:.2f} dBm (on observed points)")
        print(f"  Blind Spot RMSE:  {metrics['blind_spot_rmse_dbm']:.2f} dBm (on unobserved points)")

    # M10: Per-region metrics
    if 'ssim_observed' in metrics:
        print(f"\nPer-Region Metrics (M10):")
        print(f"  SSIM (observed):    {metrics['ssim_observed']:.4f}")
        print(f"  SSIM (unobserved):  {metrics['ssim_unobserved']:.4f}")
        print(f"  PSNR (observed):    {metrics['psnr_observed']:.2f} dB")
        print(f"  PSNR (unobserved):  {metrics['psnr_unobserved']:.2f} dB")

    if 'mean_uncertainty' in metrics:
        print(f"\nUncertainty Estimation:")
        print(f"  Mean Uncertainty: {metrics['mean_uncertainty']:.4f}")

    # C6: Diversity metrics
    if 'mean_diversity' in metrics:
        print(f"\nSample Diversity (C6):")
        print(f"  Mean Diversity: {metrics['mean_diversity']:.4f}")
        print(f"  Std Diversity:  {metrics['std_diversity']:.4f}")

    # Save results
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    results_path = output_dir / 'metrics.json'
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {results_path}")

    # Generate visualizations
    if cfg.get('visualize', True):
        print("\nGenerating visualizations...")
        save_visualizations(
            inference=inference,
            dataloader=test_loader,
            output_dir=output_dir / 'visualizations',
            device=device,
            num_samples=cfg.get('num_vis_samples', 8),
        )

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
