"""
PyTorch Lightning Module for Trajectory-Conditioned Diffusion.

Handles training, validation, and inference for the diffusion model.
Includes optional physics-informed losses and coverage-aware attention.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

import sys
sys.path.insert(0, 'src')

from models.diffusion import GaussianDiffusion, DDIMSampler
from models.encoders import TrajectoryConditionedUNet


class DiffusionModule(L.LightningModule):
    """
    Lightning module for training trajectory-conditioned diffusion models.

    Handles:
    - Training with random timestep sampling
    - Validation with fixed timesteps for consistent metrics
    - Sample generation during validation
    - Metric computation (MSE, trajectory-aware metrics)
    - Optional physics-informed losses (TrajectoryDiffLoss)
    - Optional coverage-aware attention (CoverageAwareUNet)
    - Optimizer and scheduler configuration
    """

    def __init__(
        self,
        # Model configuration
        unet_size: str = 'medium',
        image_size: int = 256,
        condition_channels: int = 64,
        # Diffusion configuration
        num_timesteps: int = 1000,
        beta_schedule: str = 'cosine',
        prediction_type: str = 'epsilon',
        loss_type: str = 'mse',
        # Conditioning configuration
        use_building_map: bool = True,
        use_sparse_rss: bool = True,
        use_trajectory_mask: bool = True,
        use_coverage_density: bool = True,
        use_tx_position: bool = True,
        # Physics loss configuration
        use_physics_losses: bool = False,
        trajectory_consistency_weight: float = 0.1,
        coverage_weighted: bool = True,
        distance_decay_weight: float = 0.01,
        # Coverage attention configuration
        use_coverage_attention: bool = False,
        coverage_temperature: float = 1.0,
        # Training configuration
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_steps: int = 100000,
        ema_decay: float = 0.9999,
        use_ema: bool = True,
        # Sampling configuration
        sample_every_n_epochs: int = 5,
        num_samples: int = 4,
        ddim_steps: int = 50,
        # Memory optimization
        use_gradient_checkpointing: bool = False,
    ):
        """
        Initialize diffusion training module.

        Args:
            unet_size: Size of U-Net ('small', 'medium', 'large')
            image_size: Input image size
            condition_channels: Conditioning tensor channels
            num_timesteps: Number of diffusion steps
            beta_schedule: Noise schedule type
            prediction_type: What model predicts ('epsilon', 'x0', 'v')
            loss_type: Loss function ('mse', 'l1', 'huber')
            use_building_map: Include building map in conditioning
            use_sparse_rss: Include sparse RSS measurements
            use_trajectory_mask: Include trajectory mask
            use_coverage_density: Include coverage density
            use_tx_position: Include TX position encoding
            use_physics_losses: Enable physics-informed losses
            trajectory_consistency_weight: Weight for trajectory consistency loss
            coverage_weighted: Use coverage-weighted diffusion loss
            distance_decay_weight: Weight for distance decay regularization
            use_coverage_attention: Use CoverageAwareUNet
            coverage_temperature: Temperature for coverage attention modulation
            learning_rate: Initial learning rate
            weight_decay: AdamW weight decay
            warmup_steps: Linear warmup steps
            max_steps: Maximum training steps (for scheduler)
            ema_decay: EMA decay rate
            use_ema: Whether to use EMA
            sample_every_n_epochs: Generate samples every N epochs
            num_samples: Number of samples to generate
            ddim_steps: DDIM sampling steps
        """
        super().__init__()
        self.save_hyperparameters()

        # Create model (with optional coverage-aware attention)
        self.model = TrajectoryConditionedUNet(
            unet_size=unet_size,
            image_size=image_size,
            condition_channels=condition_channels,
            use_building_map=use_building_map,
            use_sparse_rss=use_sparse_rss,
            use_trajectory_mask=use_trajectory_mask,
            use_coverage_density=use_coverage_density,
            use_tx_position=use_tx_position,
            use_coverage_attention=use_coverage_attention,
            coverage_temperature=coverage_temperature,
        )

        # Enable gradient checkpointing for memory savings (trades compute for memory)
        if use_gradient_checkpointing:
            if hasattr(self.model, 'unet'):
                self.model.unet.gradient_checkpointing = True
            self.model.gradient_checkpointing_enable = True

        # Create diffusion process
        self.diffusion = GaussianDiffusion(
            num_timesteps=num_timesteps,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
            loss_type=loss_type,
        )

        # Physics-informed losses (optional)
        self.use_physics_losses = use_physics_losses
        if use_physics_losses:
            from .losses import TrajectoryDiffLoss
            self.physics_loss = TrajectoryDiffLoss(
                diffusion_weight=1.0,
                trajectory_consistency_weight=trajectory_consistency_weight,
                coverage_weighted=coverage_weighted,
                distance_decay_weight=distance_decay_weight,
            )

        # EMA model (optional)
        self.use_ema = use_ema
        if use_ema:
            self.ema_model = self._create_ema_model()
            self.ema_decay = ema_decay

        # DDIM sampler for fast inference
        self.ddim_sampler = DDIMSampler(self.diffusion, ddim_num_steps=ddim_steps)

        # Sampling configuration
        self.sample_every_n_epochs = sample_every_n_epochs
        self.num_samples = num_samples

        # For tracking best validation loss
        self.best_val_loss = float('inf')

    def _create_ema_model(self) -> nn.Module:
        """Create EMA copy of model."""
        import copy
        ema_model = copy.deepcopy(self.model)
        for param in ema_model.parameters():
            param.requires_grad = False
        return ema_model

    @torch.no_grad()
    def _update_ema(self):
        """Update EMA model parameters."""
        if not self.use_ema:
            return

        for ema_param, model_param in zip(
            self.ema_model.parameters(),
            self.model.parameters()
        ):
            ema_param.data.mul_(self.ema_decay).add_(
                model_param.data, alpha=1 - self.ema_decay
            )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass through model."""
        return self.model(
            x, t,
            building_map=condition.get('building_map'),
            sparse_rss=condition.get('sparse_rss'),
            trajectory_mask=condition.get('trajectory_mask'),
            coverage_density=condition.get('coverage_density'),
            tx_position=condition.get('tx_position'),
        )

    def _extract_condition(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Extract conditioning inputs from batch."""
        condition = {}

        if self.hparams.use_building_map and 'building_map' in batch:
            condition['building_map'] = batch['building_map']

        if self.hparams.use_sparse_rss and 'sparse_rss' in batch:
            condition['sparse_rss'] = batch['sparse_rss']

        if self.hparams.use_trajectory_mask and 'trajectory_mask' in batch:
            condition['trajectory_mask'] = batch['trajectory_mask']

        if self.hparams.use_coverage_density and 'coverage_density' in batch:
            condition['coverage_density'] = batch['coverage_density']

        if self.hparams.use_tx_position and 'tx_position' in batch:
            condition['tx_position'] = batch['tx_position']

        return condition

    def _compute_target(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Compute prediction target based on prediction type."""
        if self.hparams.prediction_type == 'epsilon':
            return noise
        elif self.hparams.prediction_type == 'x0':
            return x_0
        else:  # 'v'
            sqrt_alpha = self.diffusion._extract(
                self.diffusion.sqrt_alphas_cumprod, t, x_0.shape
            )
            sqrt_one_minus_alpha = self.diffusion._extract(
                self.diffusion.sqrt_one_minus_alphas_cumprod, t, x_0.shape
            )
            return sqrt_alpha * noise - sqrt_one_minus_alpha * x_0

    def _predict_x0(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        pred: torch.Tensor,
    ) -> torch.Tensor:
        """Recover predicted x0 from model output, clamped to [-1, 1]."""
        if self.hparams.prediction_type == 'epsilon':
            pred_x0 = self.diffusion.predict_x0_from_epsilon(x_t, t, pred)
        elif self.hparams.prediction_type == 'x0':
            pred_x0 = pred
        else:  # 'v'
            sqrt_alpha = self.diffusion._extract(
                self.diffusion.sqrt_alphas_cumprod, t, x_t.shape
            )
            sqrt_one_minus = self.diffusion._extract(
                self.diffusion.sqrt_one_minus_alphas_cumprod, t, x_t.shape
            )
            pred_x0 = sqrt_alpha * x_t - sqrt_one_minus * pred
        return pred_x0.clamp(-1, 1)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Optional[torch.Tensor]:
        """
        Training step with OOM recovery.

        Args:
            batch: Dictionary containing radio_map, building_map, sparse_rss, etc.
            batch_idx: Batch index

        Returns:
            Loss tensor, or None if OOM occurred (Lightning skips the step)
        """
        try:
            return self._training_step_inner(batch, batch_idx)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            self.print(f"WARNING: CUDA OOM at step {self.global_step}, skipping batch")
            return None

    def _training_step_inner(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Inner training step logic."""
        # Get ground truth radio map
        x_0 = batch['radio_map']
        B = x_0.shape[0]

        # Sample random timesteps
        t = self.diffusion.sample_timesteps(B, x_0.device)

        # Extract conditioning
        condition = self._extract_condition(batch)

        # Forward diffusion
        noise = torch.randn_like(x_0)
        x_t = self.diffusion.q_sample(x_0, t, noise=noise)

        # Model prediction
        pred = self.forward(x_t, t, condition)

        # Compute target
        target = self._compute_target(x_0, t, noise)

        if self.use_physics_losses:
            # Recover pred_x0 for physics losses
            pred_x0 = self._predict_x0(x_t, t, pred)

            # Combined loss with physics components
            losses = self.physics_loss(
                noise_pred=pred,
                noise_target=target,
                pred_x0=pred_x0,
                batch=batch,
            )
            loss = losses['total']

            # Log individual loss components
            for key, val in losses.items():
                self.log(f'train/{key}', val, prog_bar=(key == 'total'))
        else:
            # Standard diffusion loss
            if self.hparams.loss_type == 'mse':
                loss = F.mse_loss(pred, target)
            elif self.hparams.loss_type == 'l1':
                loss = F.l1_loss(pred, target)
            else:  # 'huber'
                loss = F.smooth_l1_loss(pred, target)
            self.log('train/loss', loss, prog_bar=True)

        # Update EMA
        if self.use_ema:
            self._update_ema()

        self.log('train/timestep_mean', t.float().mean())

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Validation step.

        Args:
            batch: Dictionary containing radio_map, building_map, sparse_rss, etc.
            batch_idx: Batch index

        Returns:
            Dictionary with loss and metrics
        """
        x_0 = batch['radio_map']
        B = x_0.shape[0]

        # Sample timesteps for validation
        t = self.diffusion.sample_timesteps(B, x_0.device)

        # Extract conditioning
        condition = self._extract_condition(batch)

        # Forward diffusion
        noise = torch.randn_like(x_0)
        x_t = self.diffusion.q_sample(x_0, t, noise=noise)

        # Use EMA model for validation if available
        model = self.ema_model if self.use_ema else self.model
        model.eval()  # Ensure EMA model is in eval mode (Lightning only manages self.model)

        with torch.no_grad():
            pred = model(
                x_t, t,
                building_map=condition.get('building_map'),
                sparse_rss=condition.get('sparse_rss'),
                trajectory_mask=condition.get('trajectory_mask'),
                coverage_density=condition.get('coverage_density'),
                tx_position=condition.get('tx_position'),
            )

        # Compute target and loss
        target = self._compute_target(x_0, t, noise)
        loss = F.mse_loss(pred, target)

        # Log metrics
        self.log('val/loss', loss, prog_bar=True, sync_dist=True)

        return {'val_loss': loss}

    def on_validation_epoch_end(self):
        """Generate samples at end of validation epoch."""
        if (self.current_epoch + 1) % self.sample_every_n_epochs != 0:
            return

        # Generate samples (would need a validation batch for conditioning)
        # This is typically done in a callback for cleaner code
        pass

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # AdamW optimizer
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999),
        )

        # Learning rate scheduler with warmup
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=self.hparams.warmup_steps,
        )

        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_steps - self.hparams.warmup_steps,
            eta_min=self.hparams.learning_rate * 0.01,
        )

        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.hparams.warmup_steps],
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        }

    @torch.no_grad()
    def sample(
        self,
        condition: Dict[str, torch.Tensor],
        num_samples: Optional[int] = None,
        use_ddim: bool = True,
        progress: bool = False,
    ) -> torch.Tensor:
        """
        Generate samples from the model.

        Args:
            condition: Conditioning inputs
            num_samples: Number of samples (inferred from condition if None)
            use_ddim: Use DDIM for faster sampling
            progress: Show progress bar

        Returns:
            Generated radio maps (B, 1, H, W)
        """
        # Determine batch size from condition
        for key in condition:
            if condition[key] is not None:
                B = condition[key].shape[0]
                H = condition[key].shape[2] if condition[key].dim() > 2 else self.hparams.image_size
                W = condition[key].shape[3] if condition[key].dim() > 3 else self.hparams.image_size
                device = condition[key].device
                break
        else:
            raise ValueError("At least one conditioning input required")

        if num_samples is not None:
            B = num_samples

        # Use EMA model for sampling
        model = self.ema_model if self.use_ema else self.model
        model.eval()

        # Create wrapper that passes condition
        class ConditionedModel(nn.Module):
            def __init__(self, model, condition):
                super().__init__()
                self.model = model
                self.condition = condition

            def forward(self, x, t, **kwargs):
                return self.model(
                    x, t,
                    building_map=self.condition.get('building_map'),
                    sparse_rss=self.condition.get('sparse_rss'),
                    trajectory_mask=self.condition.get('trajectory_mask'),
                    coverage_density=self.condition.get('coverage_density'),
                    tx_position=self.condition.get('tx_position'),
                )

            def parameters(self):
                return self.model.parameters()

        conditioned_model = ConditionedModel(model, condition)

        shape = (B, 1, H, W)

        if use_ddim:
            samples = self.ddim_sampler.sample(
                conditioned_model,
                shape=shape,
                progress=progress,
            )
        else:
            samples = self.diffusion.p_sample_loop(
                conditioned_model,
                shape=shape,
                progress=progress,
            )

        return samples

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        """Save EMA model in checkpoint."""
        if self.use_ema:
            checkpoint['ema_state_dict'] = self.ema_model.state_dict()

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]):
        """Load EMA model from checkpoint."""
        if self.use_ema and 'ema_state_dict' in checkpoint:
            self.ema_model.load_state_dict(checkpoint['ema_state_dict'])


class SampleCallback(L.Callback):
    """Callback for generating and logging samples during training."""

    def __init__(
        self,
        every_n_epochs: int = 5,
        num_samples: int = 4,
        use_ddim: bool = True,
    ):
        """
        Args:
            every_n_epochs: Generate samples every N epochs
            num_samples: Number of samples to generate
            use_ddim: Use DDIM for faster sampling
        """
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.num_samples = num_samples
        self.use_ddim = use_ddim

    def on_validation_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: DiffusionModule,
    ):
        """Generate samples at end of validation."""
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return

        # Get a validation batch for conditioning
        val_dataloader = trainer.val_dataloaders
        if val_dataloader is None:
            return

        batch = next(iter(val_dataloader))

        # Move to device
        device = pl_module.device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Limit to num_samples
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key][:self.num_samples]

        # Extract condition
        condition = pl_module._extract_condition(batch)

        # Generate samples
        samples = pl_module.sample(condition, use_ddim=self.use_ddim)

        # Log to tensorboard/wandb if available
        if trainer.logger is not None:
            self._log_samples(trainer, pl_module, batch, samples)

    def _log_samples(
        self,
        trainer: L.Trainer,
        pl_module: DiffusionModule,
        batch: Dict[str, torch.Tensor],
        samples: torch.Tensor,
    ):
        """Log samples to logger."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            # Create comparison figure
            n = min(self.num_samples, samples.shape[0])
            fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))

            if n == 1:
                axes = axes[None, :]

            for i in range(n):
                # Building map
                if 'building_map' in batch:
                    axes[i, 0].imshow(batch['building_map'][i, 0].cpu().numpy(), cmap='gray')
                    axes[i, 0].set_title('Building Map')

                # Sparse RSS
                if 'sparse_rss' in batch:
                    axes[i, 1].imshow(batch['sparse_rss'][i, 0].cpu().numpy(), cmap='viridis')
                    axes[i, 1].set_title('Sparse RSS')

                # Ground truth
                if 'radio_map' in batch:
                    axes[i, 2].imshow(batch['radio_map'][i, 0].cpu().numpy(), cmap='viridis')
                    axes[i, 2].set_title('Ground Truth')

                # Generated
                axes[i, 3].imshow(samples[i, 0].cpu().numpy(), cmap='viridis')
                axes[i, 3].set_title('Generated')

                for ax in axes[i]:
                    ax.axis('off')

            plt.tight_layout()

            # Log figure
            if hasattr(trainer.logger, 'experiment'):
                if hasattr(trainer.logger.experiment, 'add_figure'):
                    trainer.logger.experiment.add_figure(
                        'samples',
                        fig,
                        global_step=trainer.global_step
                    )
                elif hasattr(trainer.logger.experiment, 'log'):
                    # wandb
                    import wandb
                    trainer.logger.experiment.log({
                        'samples': wandb.Image(fig),
                        'epoch': trainer.current_epoch,
                    })

            plt.close(fig)

        except Exception as e:
            print(f"Failed to log samples: {e}")


def get_diffusion_module(
    preset: str = 'default',
    **kwargs,
) -> DiffusionModule:
    """
    Factory function to create diffusion module with presets.

    Args:
        preset: Configuration preset ('default', 'fast', 'quality')
        **kwargs: Override preset parameters

    Returns:
        DiffusionModule instance
    """
    presets = {
        'default': dict(
            unet_size='medium',
            num_timesteps=1000,
            beta_schedule='cosine',
            learning_rate=1e-4,
            warmup_steps=1000,
        ),
        'fast': dict(
            unet_size='small',
            num_timesteps=500,
            beta_schedule='linear',
            learning_rate=2e-4,
            warmup_steps=500,
        ),
        'quality': dict(
            unet_size='large',
            num_timesteps=1000,
            beta_schedule='cosine',
            learning_rate=5e-5,
            warmup_steps=2000,
        ),
    }

    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Choose from {list(presets.keys())}")

    config = presets[preset]
    config.update(kwargs)

    return DiffusionModule(**config)
