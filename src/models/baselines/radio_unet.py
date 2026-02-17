"""
RadioUNet baseline (Levie et al., 2021) adapted to sparse trajectory setting.

A non-diffusion UNet that directly predicts radio maps from building map + TX
position + sparse observations via MSE regression. Uses raw channel concatenation
instead of a learned condition encoder, isolating the value of our novel components.

Key differences from our TrajectoryConditionedUNet:
- No condition encoder — raw channel concatenation
- No time embedding — not a diffusion model
- No coverage attention — standard self-attention only
- No diffusion — single forward pass, MSE loss
"""

import math
import os
import sys
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from models.diffusion.unet import (
    ResidualBlock,
    AttentionBlock,
    Downsample,
    Upsample,
    Swish,
    get_norm,
)


def _build_tx_distance_map(
    tx_position: torch.Tensor, H: int, W: int
) -> torch.Tensor:
    """Build normalized inverse-distance map from TX position.

    Args:
        tx_position: (B, 2) in [0, 1] normalized coordinates.
        H, W: Spatial dimensions.

    Returns:
        (B, 1, H, W) inverse-distance map in ~[0, 1].
    """
    device = tx_position.device
    y = torch.linspace(0, 1, H, device=device)
    x = torch.linspace(0, 1, W, device=device)
    yy, xx = torch.meshgrid(y, x, indexing='ij')

    tx_x = tx_position[:, 0].view(-1, 1, 1)
    tx_y = tx_position[:, 1].view(-1, 1, 1)

    dist = torch.sqrt((xx[None] - tx_x) ** 2 + (yy[None] - tx_y) ** 2 + 1e-6)
    inv_dist = 1.0 / (1.0 + dist * 10.0)
    return inv_dist.unsqueeze(1)


class RadioUNetBaseline(L.LightningModule):
    """RadioUNet-style baseline adapted to sparse trajectory setting.

    Direct regression UNet: concat(building_map, sparse_rss, trajectory_mask,
    tx_distance_map) → predicted radio map. No diffusion, no learned encoder.
    """

    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 64,
        channel_mult: tuple = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attention_resolutions: tuple = (32,),
        dropout: float = 0.1,
        num_heads: int = 4,
        image_size: int = 256,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_epochs: int = 5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.image_size = image_size

        num_groups = 32
        ch = base_channels
        current_res = image_size

        # Input conv
        self.input_conv = nn.Conv2d(in_channels, ch, 3, padding=1)

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        skip_channels = [ch]

        for level, mult in enumerate(channel_mult):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                layers = [ResidualBlock(ch, out_ch, time_emb_dim=None,
                                        dropout=dropout, num_groups=num_groups)]
                if current_res in attention_resolutions:
                    layers.append(AttentionBlock(out_ch, num_heads,
                                                 num_groups=num_groups))
                self.encoder_blocks.append(nn.ModuleList(layers))
                skip_channels.append(out_ch)
                ch = out_ch

            if level < len(channel_mult) - 1:
                self.encoder_blocks.append(nn.ModuleList([Downsample(ch)]))
                skip_channels.append(ch)
                current_res //= 2

        # Middle
        self.middle = nn.ModuleList([
            ResidualBlock(ch, ch, time_emb_dim=None, dropout=dropout,
                          num_groups=num_groups),
            AttentionBlock(ch, num_heads, num_groups=num_groups),
            ResidualBlock(ch, ch, time_emb_dim=None, dropout=dropout,
                          num_groups=num_groups),
        ])

        # Decoder
        self.decoder_blocks = nn.ModuleList()
        for level, mult in list(enumerate(channel_mult))[::-1]:
            out_ch = base_channels * mult
            for _ in range(num_res_blocks + 1):
                skip_ch = skip_channels.pop()
                layers = [ResidualBlock(ch + skip_ch, out_ch, time_emb_dim=None,
                                        dropout=dropout, num_groups=num_groups)]
                if current_res in attention_resolutions:
                    layers.append(AttentionBlock(out_ch, num_heads,
                                                 num_groups=num_groups))
                self.decoder_blocks.append(nn.ModuleList(layers))
                ch = out_ch

            if level > 0:
                self.decoder_blocks.append(nn.ModuleList([Upsample(ch)]))
                current_res *= 2

        # Output
        self.output_conv = nn.Sequential(
            get_norm('group', ch, num_groups),
            Swish(),
            nn.Conv2d(ch, 1, 3, padding=1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _unet_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run through encoder → middle → decoder."""
        h = self.input_conv(x)
        skips = [h]

        for block in self.encoder_blocks:
            for layer in block:
                if isinstance(layer, ResidualBlock):
                    h = layer(h)
                else:
                    h = layer(h)
            skips.append(h)

        for layer in self.middle:
            h = layer(h) if isinstance(layer, AttentionBlock) else layer(h)

        for block in self.decoder_blocks:
            for layer in block:
                if isinstance(layer, ResidualBlock):
                    h = torch.cat([h, skips.pop()], dim=1)
                    h = layer(h)
                else:
                    h = layer(h)

        return self.output_conv(h)

    def forward(self, condition: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict radio map from condition dict.

        Args:
            condition: Dict with building_map, sparse_rss, trajectory_mask, tx_position.

        Returns:
            Predicted radio map (B, 1, H, W).
        """
        bm = condition['building_map']
        B, _, H, W = bm.shape
        tx_dist = _build_tx_distance_map(condition['tx_position'], H, W)

        x = torch.cat([bm, condition['sparse_rss'],
                        condition['trajectory_mask'], tx_dist], dim=1)
        return self._unet_forward(x)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        condition = {k: batch[k] for k in
                     ('building_map', 'sparse_rss', 'trajectory_mask', 'tx_position')}
        pred = self(condition)
        loss = F.mse_loss(pred, batch['radio_map'])
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict:
        condition = {k: batch[k] for k in
                     ('building_map', 'sparse_rss', 'trajectory_mask', 'tx_position')}
        pred = self(condition)
        gt = batch['radio_map']
        loss = F.mse_loss(pred, gt)
        mae = F.l1_loss(pred, gt)
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True,
                 sync_dist=True)
        self.log('val/mae', mae, on_step=False, on_epoch=True, sync_dist=True)
        return {'val_loss': loss}

    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        condition = {k: batch[k] for k in
                     ('building_map', 'sparse_rss', 'trajectory_mask', 'tx_position')}
        return self(condition)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate,
                          weight_decay=self.weight_decay, betas=(0.9, 0.999))

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
            total_steps = 100000
            warmup_steps = 1000

        warmup_steps = max(1, warmup_steps) if self.warmup_epochs > 0 else 1

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01 if self.warmup_epochs > 0 else 1.0,
            end_factor=1.0, total_iters=warmup_steps,
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(total_steps) - warmup_steps),
            eta_min=self.learning_rate * 0.01,
        )
        scheduler = SequentialLR(optimizer,
                                  schedulers=[warmup_scheduler, cosine_scheduler],
                                  milestones=[warmup_steps])
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'step',
                             'frequency': 1},
        }
