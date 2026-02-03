"""
Training callbacks for TrajectoryDiff.

Contains callbacks for logging, visualization, and monitoring during training.
"""

from typing import Any, Dict, Optional
from pathlib import Path

import torch
import lightning as L
from lightning.pytorch.callbacks import Callback


class WandBSampleLogger(Callback):
    """
    Weights & Biases callback for logging samples during training.

    Logs generated samples, ground truth comparisons, and conditioning
    inputs to W&B for visualization.
    """

    def __init__(
        self,
        every_n_epochs: int = 5,
        num_samples: int = 4,
        use_ddim: bool = True,
        log_conditioning: bool = True,
    ):
        """
        Args:
            every_n_epochs: Log samples every N epochs
            num_samples: Number of samples to generate and log
            use_ddim: Use DDIM for faster sampling
            log_conditioning: Also log conditioning inputs
        """
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.num_samples = num_samples
        self.use_ddim = use_ddim
        self.log_conditioning = log_conditioning

    def on_validation_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: 'DiffusionModule',
    ):
        """Generate and log samples at end of validation epoch."""
        if trainer.sanity_checking:
            return

        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return

        # Check if wandb is available
        if trainer.logger is None or not hasattr(trainer.logger, 'experiment'):
            return

        try:
            import wandb
        except ImportError:
            return

        # Get validation batch
        val_dataloader = trainer.val_dataloaders
        if val_dataloader is None:
            return

        batch = next(iter(val_dataloader))

        # Move to device and limit samples
        device = pl_module.device
        batch = {
            k: v.to(device)[:self.num_samples] if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # Extract condition
        condition = pl_module._extract_condition(batch)

        # Generate samples
        with torch.no_grad():
            samples = pl_module.sample(condition, use_ddim=self.use_ddim, progress=False)

        # Log to wandb
        self._log_samples(trainer, batch, samples)

    def _log_samples(
        self,
        trainer: L.Trainer,
        batch: Dict[str, torch.Tensor],
        samples: torch.Tensor,
    ):
        """Log samples to W&B."""
        import wandb
        import numpy as np

        log_dict = {}
        n = samples.shape[0]

        # Log individual samples
        for i in range(n):
            # Generated sample
            gen_img = samples[i, 0].cpu().numpy()
            log_dict[f'samples/generated_{i}'] = wandb.Image(
                gen_img,
                caption=f'Generated sample {i}',
            )

            # Ground truth
            if 'radio_map' in batch:
                gt_img = batch['radio_map'][i, 0].cpu().numpy()
                log_dict[f'samples/ground_truth_{i}'] = wandb.Image(
                    gt_img,
                    caption=f'Ground truth {i}',
                )

            # Conditioning inputs
            if self.log_conditioning:
                if 'building_map' in batch:
                    bm_img = batch['building_map'][i, 0].cpu().numpy()
                    log_dict[f'conditioning/building_map_{i}'] = wandb.Image(
                        bm_img,
                        caption=f'Building map {i}',
                    )

                if 'sparse_rss' in batch:
                    rss_img = batch['sparse_rss'][i, 0].cpu().numpy()
                    log_dict[f'conditioning/sparse_rss_{i}'] = wandb.Image(
                        rss_img,
                        caption=f'Sparse RSS {i}',
                    )

                if 'trajectory_mask' in batch:
                    mask_img = batch['trajectory_mask'][i, 0].cpu().numpy()
                    log_dict[f'conditioning/trajectory_mask_{i}'] = wandb.Image(
                        mask_img,
                        caption=f'Trajectory mask {i}',
                    )

        # Create comparison grid
        try:
            comparison = self._create_comparison_grid(batch, samples)
            if comparison is not None:
                log_dict['samples/comparison_grid'] = wandb.Image(
                    comparison,
                    caption='Left to right: Building, Sparse RSS, Ground Truth, Generated',
                )
        except Exception:
            pass

        # Log
        trainer.logger.experiment.log(
            log_dict,
            step=trainer.global_step,
        )

    def _create_comparison_grid(
        self,
        batch: Dict[str, torch.Tensor],
        samples: torch.Tensor,
    ):
        """Create a comparison grid of inputs and outputs."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            from io import BytesIO
            from PIL import Image

            n = min(4, samples.shape[0])
            fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n))

            if n == 1:
                axes = axes[None, :]

            for i in range(n):
                # Building map
                if 'building_map' in batch:
                    axes[i, 0].imshow(batch['building_map'][i, 0].cpu().numpy(), cmap='gray')
                    axes[i, 0].set_title('Building')
                else:
                    axes[i, 0].axis('off')

                # Sparse RSS
                if 'sparse_rss' in batch:
                    axes[i, 1].imshow(batch['sparse_rss'][i, 0].cpu().numpy(), cmap='viridis')
                    axes[i, 1].set_title('Sparse RSS')
                else:
                    axes[i, 1].axis('off')

                # Ground truth
                if 'radio_map' in batch:
                    axes[i, 2].imshow(batch['radio_map'][i, 0].cpu().numpy(), cmap='viridis')
                    axes[i, 2].set_title('Ground Truth')
                else:
                    axes[i, 2].axis('off')

                # Generated
                axes[i, 3].imshow(samples[i, 0].cpu().numpy(), cmap='viridis')
                axes[i, 3].set_title('Generated')

                for ax in axes[i]:
                    ax.axis('off')

            plt.tight_layout()

            # Convert to image
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img = Image.open(buf)
            plt.close(fig)

            return img

        except Exception:
            return None


class MetricsLogger(Callback):
    """
    Callback for computing and logging additional metrics.

    Computes RMSE, SSIM, and other radio map-specific metrics
    during validation.
    """

    def __init__(
        self,
        compute_every_n_epochs: int = 1,
        num_eval_samples: int = 100,
    ):
        """
        Args:
            compute_every_n_epochs: Compute full metrics every N epochs
            num_eval_samples: Number of samples to use for metric computation
        """
        super().__init__()
        self.compute_every_n_epochs = compute_every_n_epochs
        self.num_eval_samples = num_eval_samples

    def on_validation_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: 'DiffusionModule',
    ):
        """Compute and log metrics at end of validation."""
        if trainer.sanity_checking:
            return

        if (trainer.current_epoch + 1) % self.compute_every_n_epochs != 0:
            return

        val_dataloader = trainer.val_dataloaders
        if val_dataloader is None:
            return

        # Compute metrics over multiple batches
        all_rmse = []
        all_mae = []
        total_samples = 0

        with torch.no_grad():
            for batch in val_dataloader:
                if total_samples >= self.num_eval_samples:
                    break

                # Move to device
                device = pl_module.device
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                # Extract condition and ground truth
                condition = pl_module._extract_condition(batch)
                ground_truth = batch['radio_map']

                # Generate samples
                samples = pl_module.sample(condition, use_ddim=True, progress=False)

                # Compute metrics
                rmse = torch.sqrt(((samples - ground_truth) ** 2).mean(dim=(1, 2, 3)))
                mae = (samples - ground_truth).abs().mean(dim=(1, 2, 3))

                all_rmse.append(rmse.cpu())
                all_mae.append(mae.cpu())

                total_samples += ground_truth.shape[0]

        # Aggregate metrics
        if len(all_rmse) > 0:
            rmse = torch.cat(all_rmse).mean().item()
            mae = torch.cat(all_mae).mean().item()

            # Log metrics
            pl_module.log('val/rmse', rmse, sync_dist=True)
            pl_module.log('val/mae', mae, sync_dist=True)


class CheckpointEveryNSteps(Callback):
    """
    Save checkpoints every N training steps.

    Useful for long training runs where you want intermediate checkpoints.
    """

    def __init__(
        self,
        save_step_frequency: int = 5000,
        prefix: str = 'step',
        save_path: Optional[str] = None,
    ):
        """
        Args:
            save_step_frequency: Save every N steps
            prefix: Filename prefix
            save_path: Directory to save checkpoints (uses trainer default if None)
        """
        super().__init__()
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.save_path = Path(save_path) if save_path else None

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ):
        """Check if we should save a checkpoint."""
        if trainer.global_step > 0 and trainer.global_step % self.save_step_frequency == 0:
            # Determine save path
            if self.save_path is not None:
                save_dir = self.save_path
            elif trainer.log_dir is not None:
                save_dir = Path(trainer.log_dir) / 'checkpoints'
            else:
                save_dir = Path('checkpoints')

            save_dir.mkdir(parents=True, exist_ok=True)

            # Save checkpoint
            ckpt_path = save_dir / f'{self.prefix}_{trainer.global_step}.ckpt'
            trainer.save_checkpoint(str(ckpt_path))


class GradientMonitor(Callback):
    """
    Monitor gradient statistics during training.

    Logs gradient norms and detects potential training issues.
    """

    def __init__(self, log_every_n_steps: int = 100):
        """
        Args:
            log_every_n_steps: Log gradient stats every N steps
        """
        super().__init__()
        self.log_every_n_steps = log_every_n_steps

    def on_after_backward(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """Compute and log gradient statistics after backward pass."""
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        # Compute gradient norm
        total_norm = 0.0
        for p in pl_module.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2

        total_norm = total_norm ** 0.5

        # Log
        pl_module.log('train/grad_norm', total_norm)

        # Warn if gradient is suspiciously large
        if total_norm > 100:
            print(f"Warning: Large gradient norm detected: {total_norm:.2f}")


__all__ = [
    'WandBSampleLogger',
    'MetricsLogger',
    'CheckpointEveryNSteps',
    'GradientMonitor',
]
