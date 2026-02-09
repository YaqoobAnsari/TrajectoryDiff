"""
Coverage-Aware U-Net for TrajectoryDiff.

Extends the standard UNet by replacing AttentionBlocks with
CoverageAwareAttentionBlocks. This is a key novel contribution:
attention weights are modulated by coverage density so the model
trusts high-coverage regions and explores low-coverage blind spots.

Usage:
    unet = get_coverage_aware_unet('medium', coverage_temperature=1.0,
                                    in_channels=1, out_channels=1,
                                    cond_channels=64, image_size=256)
    out = unet(x, t, cond=cond, coverage=coverage_density)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .unet import (
    UNet,
    ResidualBlock,
    AttentionBlock,
    Downsample,
    Upsample,
)
from .attention import CoverageAwareAttentionBlock, downsample_coverage


class CoverageAwareUNet(UNet):
    """
    UNet with CoverageAwareAttention replacing standard self-attention.

    After constructing a standard UNet, replaces all AttentionBlock
    instances with CoverageAwareAttentionBlock. The forward pass
    threads coverage density through the network, downsampling it
    to match feature resolutions at each attention layer.

    When coverage=None is passed to forward(), the attention blocks
    fall back to standard self-attention (no coverage modulation),
    making this a strict superset of the standard UNet.
    """

    def __init__(self, coverage_temperature: float = 1.0, **kwargs):
        """
        Args:
            coverage_temperature: Temperature for coverage modulation.
                Higher = more uniform attention, lower = more coverage-dependent.
            **kwargs: All arguments passed to UNet.__init__()
        """
        super().__init__(**kwargs)
        self.coverage_temperature = coverage_temperature
        self._replace_attention_blocks()

    def _replace_attention_blocks(self):
        """Replace all standard AttentionBlocks with CoverageAwareAttentionBlocks."""
        def replace_in_modulelist(module_list: nn.ModuleList):
            for i, block in enumerate(module_list):
                if isinstance(block, nn.ModuleList):
                    for j, layer in enumerate(block):
                        if isinstance(layer, AttentionBlock):
                            block[j] = CoverageAwareAttentionBlock(
                                channels=layer.channels,
                                coverage_temperature=self.coverage_temperature,
                            )
                elif isinstance(block, AttentionBlock):
                    module_list[i] = CoverageAwareAttentionBlock(
                        channels=block.channels,
                        coverage_temperature=self.coverage_temperature,
                    )

        replace_in_modulelist(self.encoder_blocks)
        replace_in_modulelist(self.middle)
        replace_in_modulelist(self.decoder_blocks)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        coverage: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass with coverage-aware attention.

        Args:
            x: Noisy input (B, in_channels, H, W)
            t: Timesteps (B,)
            cond: Conditioning input (B, cond_channels, H, W)
            coverage: Coverage density map (B, 1, H, W) in [0, 1].
                      When None, attention blocks use standard self-attention.

        Returns:
            Predicted noise or x_0 (B, out_channels, H, W)
        """
        # Time embedding
        time_emb = self.time_embed(t)

        # Concatenate conditioning
        if cond is not None:
            x = torch.cat([x, cond], dim=1)

        # Input conv
        h = self.input_conv(x)

        # Encoder with skip connections
        skips = [h]

        for block in self.encoder_blocks:
            for layer in block:
                if isinstance(layer, ResidualBlock):
                    h = layer(h, time_emb)
                elif isinstance(layer, CoverageAwareAttentionBlock):
                    cov = self._get_coverage(coverage, h.shape[2], h.shape[3])
                    h = layer(h, coverage=cov)
                elif isinstance(layer, (Downsample, Upsample)):
                    h = layer(h)
                else:
                    h = layer(h)
            skips.append(h)

        # Middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock):
                h = layer(h, time_emb)
            elif isinstance(layer, CoverageAwareAttentionBlock):
                cov = self._get_coverage(coverage, h.shape[2], h.shape[3])
                h = layer(h, coverage=cov)
            else:
                h = layer(h)

        # Decoder with skip connections
        for block in self.decoder_blocks:
            for layer in block:
                if isinstance(layer, ResidualBlock):
                    skip = skips.pop()
                    h = torch.cat([h, skip], dim=1)
                    h = layer(h, time_emb)
                elif isinstance(layer, CoverageAwareAttentionBlock):
                    cov = self._get_coverage(coverage, h.shape[2], h.shape[3])
                    h = layer(h, coverage=cov)
                elif isinstance(layer, (Downsample, Upsample)):
                    h = layer(h)
                else:
                    h = layer(h)

        return self.output_conv(h)

    @staticmethod
    def _get_coverage(
        coverage: Optional[torch.Tensor],
        target_h: int,
        target_w: int,
    ) -> Optional[torch.Tensor]:
        """Downsample coverage to match current feature resolution."""
        if coverage is None:
            return None
        return downsample_coverage(coverage, (target_h, target_w))


def get_coverage_aware_unet(
    size: str = 'medium',
    in_channels: int = 1,
    out_channels: int = 1,
    cond_channels: int = 0,
    image_size: int = 256,
    coverage_temperature: float = 1.0,
) -> CoverageAwareUNet:
    """
    Factory function to get CoverageAwareUNet by size.

    Args:
        size: 'small', 'medium', or 'large'
        in_channels: Input channels (radio map = 1)
        out_channels: Output channels
        cond_channels: Conditioning channels from encoder
        image_size: Input image size
        coverage_temperature: Coverage modulation temperature

    Returns:
        CoverageAwareUNet instance
    """
    configs = {
        'small': dict(
            model_channels=32,
            channel_mult=(1, 2, 4),
            num_res_blocks=1,
            attention_resolutions=(32, 16),
            dropout=0.0,
            num_heads=4,
        ),
        'medium': dict(
            model_channels=64,
            channel_mult=(1, 2, 4, 8),
            num_res_blocks=2,
            attention_resolutions=(32, 16, 8),
            dropout=0.1,
            num_heads=4,
        ),
        'large': dict(
            model_channels=128,
            channel_mult=(1, 2, 4, 8),
            num_res_blocks=3,
            attention_resolutions=(32, 16, 8),
            dropout=0.1,
            num_heads=8,
        ),
    }

    if size not in configs:
        raise ValueError(f"Unknown size: {size}. Choose from {list(configs.keys())}")

    config = configs[size]
    return CoverageAwareUNet(
        coverage_temperature=coverage_temperature,
        in_channels=in_channels,
        out_channels=out_channels,
        cond_channels=cond_channels,
        image_size=image_size,
        **config,
    )


__all__ = [
    'CoverageAwareUNet',
    'get_coverage_aware_unet',
]
