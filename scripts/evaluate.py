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
from tqdm import tqdm

from data import RadioMapDataModule
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
    window_size: int = 11,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> torch.Tensor:
    """Compute Structural Similarity Index (simplified version)."""
    # Simple SSIM computation
    mu_pred = pred.mean(dim=(2, 3), keepdim=True)
    mu_target = target.mean(dim=(2, 3), keepdim=True)

    sigma_pred = ((pred - mu_pred) ** 2).mean(dim=(2, 3), keepdim=True)
    sigma_target = ((target - mu_target) ** 2).mean(dim=(2, 3), keepdim=True)
    sigma_pred_target = ((pred - mu_pred) * (target - mu_target)).mean(dim=(2, 3), keepdim=True)

    numerator = (2 * mu_pred * mu_target + C1) * (2 * sigma_pred_target + C2)
    denominator = (mu_pred ** 2 + mu_target ** 2 + C1) * (sigma_pred + sigma_target + C2)

    ssim = numerator / denominator
    return ssim.mean(dim=(1, 2, 3))


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

        # Generate samples
        if compute_uncertainty:
            samples_list = []
            for _ in range(num_uncertainty_samples):
                s = inference.sample(condition, use_ddim=use_ddim, progress=False)
                samples_list.append(s)
            samples = torch.stack(samples_list, dim=0).mean(dim=0)
            uncertainty = torch.stack(samples_list, dim=0).std(dim=0)
            all_uncertainty.append(uncertainty.mean(dim=(1, 2, 3)).cpu())
        else:
            samples = inference.sample(condition, use_ddim=use_ddim, progress=False)

        # Compute metrics
        all_rmse.append(compute_rmse(samples, ground_truth).cpu())
        all_mae.append(compute_mae(samples, ground_truth).cpu())
        all_psnr.append(compute_psnr(samples, ground_truth).cpu())
        all_ssim.append(compute_ssim(samples, ground_truth).cpu())

        # Trajectory-aware metrics
        if 'trajectory_mask' in batch:
            traj_mask = batch['trajectory_mask']
            all_traj_rmse.append(compute_trajectory_rmse(samples, ground_truth, traj_mask).cpu())
            all_blind_rmse.append(compute_blind_spot_rmse(samples, ground_truth, traj_mask).cpu())

        total_samples += ground_truth.shape[0]

    # Aggregate metrics
    metrics = {
        'rmse': torch.cat(all_rmse).mean().item(),
        'rmse_std': torch.cat(all_rmse).std().item(),
        'mae': torch.cat(all_mae).mean().item(),
        'mae_std': torch.cat(all_mae).std().item(),
        'psnr': torch.cat(all_psnr).mean().item(),
        'psnr_std': torch.cat(all_psnr).std().item(),
        'ssim': torch.cat(all_ssim).mean().item(),
        'ssim_std': torch.cat(all_ssim).std().item(),
        'num_samples': total_samples,
    }

    if all_traj_rmse:
        metrics['trajectory_rmse'] = torch.cat(all_traj_rmse).mean().item()
        metrics['blind_spot_rmse'] = torch.cat(all_blind_rmse).mean().item()

    if all_uncertainty:
        metrics['mean_uncertainty'] = torch.cat(all_uncertainty).mean().item()

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
        data_root=cfg.data.dataset.root,
        batch_size=cfg.data.loader.batch_size,
        num_workers=cfg.data.loader.num_workers,
        image_size=cfg.data.image.height,
        train_split=cfg.data.splits.train,
        val_split=cfg.data.splits.val,
        augment=False,  # No augmentation for evaluation
        sampling_strategy=cfg.data.sampling.strategy,
        num_trajectories=cfg.data.sampling.trajectory.num_trajectories,
        points_per_trajectory=cfg.data.sampling.trajectory.points_per_trajectory,
        rss_noise_std=cfg.data.sampling.trajectory.rss_noise_std,
    )
    datamodule.setup('test')
    test_loader = datamodule.test_dataloader()

    # Evaluation settings
    max_samples = cfg.get('max_samples', None)
    compute_unc = cfg.get('uncertainty', False)
    num_unc_samples = cfg.get('num_uncertainty_samples', 10)

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
    )

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Samples evaluated: {metrics['num_samples']}")
    print(f"\nReconstruction Metrics:")
    print(f"  RMSE:  {metrics['rmse']:.4f} +/- {metrics['rmse_std']:.4f}")
    print(f"  MAE:   {metrics['mae']:.4f} +/- {metrics['mae_std']:.4f}")
    print(f"  PSNR:  {metrics['psnr']:.2f} +/- {metrics['psnr_std']:.2f} dB")
    print(f"  SSIM:  {metrics['ssim']:.4f} +/- {metrics['ssim_std']:.4f}")

    if 'trajectory_rmse' in metrics:
        print(f"\nTrajectory-Aware Metrics:")
        print(f"  Trajectory RMSE:  {metrics['trajectory_rmse']:.4f} (on observed points)")
        print(f"  Blind Spot RMSE:  {metrics['blind_spot_rmse']:.4f} (on unobserved points)")

    if 'mean_uncertainty' in metrics:
        print(f"\nUncertainty Estimation:")
        print(f"  Mean Uncertainty: {metrics['mean_uncertainty']:.4f}")

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
