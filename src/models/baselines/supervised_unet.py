"""
Supervised U-Net Baseline (C3).

Directly predicts radio maps from sparse observations using the same
architecture as the diffusion model, but trained with simple MSE loss.

This provides a fair comparison to show the value of diffusion modeling.
"""

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from models.encoders import TrajectoryConditionedUNet


class SupervisedUNetBaseline(L.LightningModule):
    """
    Supervised baseline using the same UNet architecture as the diffusion model.

    Trains to directly predict the radio map from sparse observations using MSE loss.
    This is a deterministic baseline without uncertainty estimation.
    """

    def __init__(
        self,
        # Model configuration
        unet_size: str = 'medium',
        image_size: int = 256,
        condition_channels: int = 64,
        # Conditioning configuration
        use_building_map: bool = True,
        use_sparse_rss: bool = True,
        use_trajectory_mask: bool = True,
        use_coverage_density: bool = True,
        use_tx_position: bool = True,
        # Training configuration
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_epochs: int = 5,
        # Loss configuration
        loss_type: str = 'mse',  # 'mse', 'l1', or 'huber'
        mask_unobserved_weight: float = 1.0,  # Weight for unobserved regions
    ):
        """
        Initialize supervised U-Net baseline.

        Args:
            unet_size: Size of U-Net ('small', 'medium', 'large')
            image_size: Input image size
            condition_channels: Conditioning tensor channels
            use_building_map: Include building map in conditioning
            use_sparse_rss: Include sparse RSS measurements
            use_trajectory_mask: Include trajectory mask
            use_coverage_density: Include coverage density
            use_tx_position: Include TX position encoding
            learning_rate: Initial learning rate
            weight_decay: AdamW weight decay
            warmup_epochs: Number of warmup epochs
            loss_type: Loss function type
            mask_unobserved_weight: Relative weight for unobserved regions
        """
        super().__init__()
        self.save_hyperparameters()

        # Create model (same architecture as diffusion, but no coverage attention)
        self.model = TrajectoryConditionedUNet(
            unet_size=unet_size,
            image_size=image_size,
            condition_channels=condition_channels,
            use_building_map=use_building_map,
            use_sparse_rss=use_sparse_rss,
            use_trajectory_mask=use_trajectory_mask,
            use_coverage_density=use_coverage_density,
            use_tx_position=use_tx_position,
            use_coverage_attention=False,  # Not needed for supervised baseline
        )

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.loss_type = loss_type
        self.mask_unobserved_weight = mask_unobserved_weight

    def forward(
        self,
        condition: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass: directly predict radio map.

        Args:
            condition: Dictionary with conditioning inputs

        Returns:
            Predicted radio map (B, 1, H, W)
        """
        # Model takes (x_t, t, condition) - for supervised, we pass zeros
        # since there's no diffusion timestep
        B = condition['sparse_rss'].shape[0] if 'sparse_rss' in condition else \
            condition['building_map'].shape[0]
        device = condition['sparse_rss'].device if 'sparse_rss' in condition else \
            condition['building_map'].device

        # Use dummy x_t and t (all zeros/zero timestep)
        x_t = torch.zeros(B, 1, self.hparams.image_size, self.hparams.image_size, device=device)
        t = torch.zeros(B, dtype=torch.long, device=device)

        # Model forward returns predicted noise/x0 - we interpret it as predicted radio map
        pred = self.model(x_t, t, **condition)
        return pred

    def compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        trajectory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute reconstruction loss.

        Args:
            pred: Predicted radio map (B, 1, H, W)
            target: Ground truth radio map (B, 1, H, W)
            trajectory_mask: Optional mask of observed regions (B, 1, H, W)

        Returns:
            Scalar loss
        """
        if self.loss_type == 'mse':
            loss = F.mse_loss(pred, target, reduction='none')
        elif self.loss_type == 'l1':
            loss = F.l1_loss(pred, target, reduction='none')
        elif self.loss_type == 'huber':
            loss = F.huber_loss(pred, target, reduction='none', delta=1.0)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Optional weighting: emphasize unobserved regions
        if trajectory_mask is not None and self.mask_unobserved_weight != 1.0:
            observed_mask = trajectory_mask > 0.5
            unobserved_mask = trajectory_mask <= 0.5

            weights = torch.ones_like(loss)
            weights[unobserved_mask] = self.mask_unobserved_weight

            loss = loss * weights

        return loss.mean()

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
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

        # Forward pass
        pred = self(condition)

        # Compute loss
        loss = self.compute_loss(
            pred,
            ground_truth,
            trajectory_mask=batch.get('trajectory_mask'),
        )

        # Log
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step."""
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

        # Forward pass
        pred = self(condition)

        # Compute loss
        loss = self.compute_loss(
            pred,
            ground_truth,
            trajectory_mask=batch.get('trajectory_mask'),
        )

        # Additional metrics
        mse = F.mse_loss(pred, ground_truth)
        mae = F.l1_loss(pred, ground_truth)

        # Log
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val/mse', mse, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/mae', mae, on_step=False, on_epoch=True, sync_dist=True)

        return {'val_loss': loss}

    def predict_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Prediction step for inference."""
        condition = {
            'building_map': batch.get('building_map'),
            'sparse_rss': batch.get('sparse_rss'),
            'trajectory_mask': batch.get('trajectory_mask'),
            'coverage_density': batch.get('coverage_density'),
            'tx_position': batch.get('tx_position'),
        }
        condition = {k: v for k, v in condition.items() if v is not None}

        return self(condition)

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
        )

        # Compute warmup and total steps from trainer
        # Note: self.trainer raises RuntimeError when not attached (Lightning 2.x)
        try:
            _trainer = self.trainer
        except RuntimeError:
            _trainer = None

        if _trainer is not None and hasattr(_trainer, 'estimated_stepping_batches'):
            total_steps = _trainer.estimated_stepping_batches
            max_epochs = _trainer.max_epochs or 200
            if not math.isfinite(total_steps) or max_epochs < 1:
                total_steps = 100000
                steps_per_epoch = 500
            else:
                steps_per_epoch = total_steps // max(1, max_epochs)
            warmup_steps = self.warmup_epochs * steps_per_epoch
        else:
            warmup_steps = 1000
            total_steps = 100000

        warmup_steps = max(1, warmup_steps) if self.warmup_epochs > 0 else 1

        # Linear warmup + cosine decay (same as diffusion model)
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01 if self.warmup_epochs > 0 else 1.0,
            end_factor=1.0,
            total_iters=warmup_steps,
        )

        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(total_steps) - warmup_steps),
            eta_min=self.learning_rate * 0.01,
        )

        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            },
        }
