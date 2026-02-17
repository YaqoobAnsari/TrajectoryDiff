"""
RMDM baseline (Xu et al., 2025) adapted to sparse trajectory setting.

Dual-UNet diffusion model with physics-conductor anchor fusion.
Current SOTA on RadioMapSeer (with dense observations).

Architecture:
- PhysicsConductor: small UNet that produces physics-based anchor from
  building_map + tx_distance (no diffusion, no sparse observations).
- DetailSculptor: diffusion UNet with multiplicative anchor fusion at each
  resolution level. Conditioned on sparse_rss + trajectory_mask.

Key adaptations for sparse trajectory setting:
- Sculptor conditioning uses sparse_rss + trajectory_mask (not dense)
- No Helmholtz PDE loss (requires dense ground truth for Laplacian)
- Reuses our GaussianDiffusion and DDIMSampler for diffusion process
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
    UNet,
    ResidualBlock,
    AttentionBlock,
    Downsample,
    Upsample,
    Swish,
    get_norm,
)
from models.diffusion.ddpm import (
    GaussianDiffusion,
    DDIMSampler,
    SinusoidalPositionEmbedding,
)


def _build_tx_distance_map(
    tx_position: torch.Tensor, H: int, W: int
) -> torch.Tensor:
    """Build normalized inverse-distance map from TX position."""
    device = tx_position.device
    y = torch.linspace(0, 1, H, device=device)
    x = torch.linspace(0, 1, W, device=device)
    yy, xx = torch.meshgrid(y, x, indexing='ij')

    tx_x = tx_position[:, 0].view(-1, 1, 1)
    tx_y = tx_position[:, 1].view(-1, 1, 1)

    dist = torch.sqrt((xx[None] - tx_x) ** 2 + (yy[None] - tx_y) ** 2 + 1e-6)
    inv_dist = 1.0 / (1.0 + dist * 10.0)
    return inv_dist.unsqueeze(1)


class _AnchorFusionUNet(UNet):
    """UNet with multiplicative anchor fusion at resolution transitions.

    After each downsample in the encoder and after the bottleneck, features
    are modulated: ``h = h * (1 + anchor_at_resolution)``.
    This lets the physics-based anchor guide feature extraction at all scales.
    """

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        anchor: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        time_emb = self.time_embed(t)

        if cond is not None:
            x = torch.cat([x, cond], dim=1)

        h = self.input_conv(x)
        skips = [h]
        prev_res = h.shape[2]

        for block in self.encoder_blocks:
            for layer in block:
                if isinstance(layer, ResidualBlock):
                    h = layer(h, time_emb)
                elif isinstance(layer, AttentionBlock):
                    h = layer(h)
                else:
                    h = layer(h)

            # Anchor fusion when resolution changes (after downsample)
            if anchor is not None and h.shape[2] != prev_res:
                a = F.interpolate(anchor, size=h.shape[2:],
                                  mode='bilinear', align_corners=False)
                h = h * (1.0 + a)
                prev_res = h.shape[2]

            skips.append(h)

        # Middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock):
                h = layer(h, time_emb)
            else:
                h = layer(h)

        # Anchor fusion after bottleneck
        if anchor is not None:
            a = F.interpolate(anchor, size=h.shape[2:],
                              mode='bilinear', align_corners=False)
            h = h * (1.0 + a)

        # Decoder
        for block in self.decoder_blocks:
            for layer in block:
                if isinstance(layer, ResidualBlock):
                    skip = skips.pop()
                    h = torch.cat([h, skip], dim=1)
                    h = layer(h, time_emb)
                elif isinstance(layer, AttentionBlock):
                    h = layer(h)
                else:
                    h = layer(h)

        return self.output_conv(h)


class RMDMBaseline(L.LightningModule):
    """RMDM-style baseline: dual-UNet diffusion with anchor fusion.

    Training:
        1. Conductor: building_map + tx_distance → anchor (MSE vs ground truth)
        2. Sculptor: DDPM training on x_t with anchor fusion (noise prediction MSE)
        3. Combined loss = diffusion_loss + conductor_weight * conductor_loss

    Inference:
        1. Conductor produces anchor from building_map + tx_distance
        2. DDIM sampling with sculptor + anchor fusion
    """

    def __init__(
        self,
        # Architecture
        image_size: int = 256,
        sculptor_channels: int = 64,
        sculptor_channel_mult: tuple = (1, 2, 4, 8),
        sculptor_num_res_blocks: int = 2,
        sculptor_attention_resolutions: tuple = (32,),
        sculptor_dropout: float = 0.1,
        sculptor_num_heads: int = 4,
        conductor_channels: int = 32,
        conductor_channel_mult: tuple = (1, 2, 4),
        # Diffusion
        num_timesteps: int = 1000,
        beta_schedule: str = 'cosine',
        prediction_type: str = 'epsilon',
        ddim_steps: int = 50,
        # Training
        conductor_weight: float = 0.1,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_epochs: int = 5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.conductor_weight = conductor_weight
        self.image_size = image_size

        # Physics-Conductor: building_map (1ch) + tx_distance (1ch) → anchor (1ch)
        # Small UNet — uses time embedding internally but we always pass t=0.
        self.conductor = UNet(
            in_channels=2,
            out_channels=1,
            model_channels=conductor_channels,
            channel_mult=conductor_channel_mult,
            num_res_blocks=1,
            attention_resolutions=(),
            dropout=0.0,
            num_heads=4,
            cond_channels=0,
            image_size=image_size,
        )

        # Detail-Sculptor: diffusion UNet with anchor fusion
        # Condition: sparse_rss (1ch) + trajectory_mask (1ch) = 2ch
        self.sculptor = _AnchorFusionUNet(
            in_channels=1,
            out_channels=1,
            model_channels=sculptor_channels,
            channel_mult=sculptor_channel_mult,
            num_res_blocks=sculptor_num_res_blocks,
            attention_resolutions=sculptor_attention_resolutions,
            dropout=sculptor_dropout,
            num_heads=sculptor_num_heads,
            cond_channels=2,
            image_size=image_size,
        )

        # Diffusion process (reuse ours)
        self.diffusion = GaussianDiffusion(
            num_timesteps=num_timesteps,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
        )

        # DDIM sampler
        self.ddim_sampler = DDIMSampler(self.diffusion, ddim_num_steps=ddim_steps)

    def _conductor_forward(
        self,
        building_map: torch.Tensor,
        tx_position: torch.Tensor,
    ) -> torch.Tensor:
        """Run conductor to produce physics-based anchor."""
        B, _, H, W = building_map.shape
        tx_dist = _build_tx_distance_map(tx_position, H, W)
        conductor_input = torch.cat([building_map, tx_dist], dim=1)
        t_zero = torch.zeros(B, dtype=torch.long, device=building_map.device)
        anchor = self.conductor(conductor_input, t_zero)
        return anchor.clamp(-1, 1)

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x_0 = batch['radio_map']
        B = x_0.shape[0]

        # Conductor → anchor
        anchor = self._conductor_forward(batch['building_map'], batch['tx_position'])
        conductor_loss = F.mse_loss(anchor, x_0)

        # Diffusion training on sculptor
        t = self.diffusion.sample_timesteps(B, x_0.device)
        noise = torch.randn_like(x_0)
        x_t = self.diffusion.q_sample(x_0, t, noise)

        cond = torch.cat([batch['sparse_rss'], batch['trajectory_mask']], dim=1)
        pred = self.sculptor(x_t, t, cond=cond, anchor=anchor.detach())

        # Noise prediction loss
        if self.hparams.prediction_type == 'epsilon':
            target = noise
        elif self.hparams.prediction_type == 'x0':
            target = x_0
        else:
            raise ValueError(f"Unsupported prediction_type: {self.hparams.prediction_type}")

        diffusion_loss = F.mse_loss(pred, target)

        loss = diffusion_loss + self.conductor_weight * conductor_loss

        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/diffusion_loss', diffusion_loss)
        self.log('train/conductor_loss', conductor_loss)

        return loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        x_0 = batch['radio_map']
        B = x_0.shape[0]

        anchor = self._conductor_forward(batch['building_map'], batch['tx_position'])

        # Deterministic timesteps/noise for consistent val/loss
        val_rng = torch.Generator(device='cpu')
        val_rng.manual_seed(42 + batch_idx)
        t = torch.randint(0, self.diffusion.num_timesteps, (B,),
                          generator=val_rng).to(x_0.device)
        noise = torch.randn(x_0.shape, generator=val_rng).to(
            device=x_0.device, dtype=x_0.dtype)

        x_t = self.diffusion.q_sample(x_0, t, noise)
        cond = torch.cat([batch['sparse_rss'], batch['trajectory_mask']], dim=1)
        pred = self.sculptor(x_t, t, cond=cond, anchor=anchor.detach())

        loss = F.mse_loss(pred, noise)
        conductor_loss = F.mse_loss(anchor, x_0)

        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True,
                 sync_dist=True)
        self.log('val/conductor_loss', conductor_loss, on_step=False, on_epoch=True,
                 sync_dist=True)

        return {'val_loss': loss}

    @torch.no_grad()
    def sample(
        self,
        condition: Dict[str, torch.Tensor],
        use_ddim: bool = True,
        progress: bool = False,
    ) -> torch.Tensor:
        """Generate radio map samples.

        Args:
            condition: Dict with building_map, sparse_rss, trajectory_mask, tx_position.
            use_ddim: Use DDIM sampling (faster).
            progress: Show progress bar.

        Returns:
            Generated radio maps (B, 1, H, W).
        """
        building_map = condition['building_map']
        B, _, H, W = building_map.shape

        # Conductor → anchor
        anchor = self._conductor_forward(building_map, condition['tx_position'])

        # Build sculptor condition
        cond = torch.cat([condition['sparse_rss'],
                          condition['trajectory_mask']], dim=1)

        # Wrap sculptor with fixed condition + anchor for DDIM sampler
        class _Wrapped(nn.Module):
            def __init__(self_, sculptor, cond, anchor):
                super().__init__()
                self_.sculptor = sculptor
                self_._cond = cond
                self_._anchor = anchor

            def forward(self_, x_t, t, **kwargs):
                return self_.sculptor(x_t, t, cond=self_._cond,
                                      anchor=self_._anchor)

        wrapped = _Wrapped(self.sculptor, cond, anchor)
        shape = (B, 1, H, W)

        if use_ddim:
            samples = self.ddim_sampler.sample(wrapped, shape, progress=progress)
        else:
            samples = self.diffusion.p_sample_loop(wrapped, shape, progress=progress)

        return samples.clamp(-1, 1)

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
