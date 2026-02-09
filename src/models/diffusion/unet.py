"""
U-Net Architecture for Diffusion Models.

Implements a U-Net with:
- Residual blocks with GroupNorm
- Self-attention at specified resolutions
- Time embedding injection
- Conditioning input support (building map, sparse RSS, etc.)

Based on:
- Ronneberger et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation"
- Ho et al. "Denoising Diffusion Probabilistic Models"
- Dhariwal & Nichol "Diffusion Models Beat GANs on Image Synthesis"
"""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ddpm import SinusoidalPositionEmbedding


def get_norm(norm_type: str, num_channels: int, num_groups: int = 32) -> nn.Module:
    """Get normalization layer."""
    if norm_type == 'group':
        return nn.GroupNorm(min(num_groups, num_channels), num_channels)
    elif norm_type == 'batch':
        return nn.BatchNorm2d(num_channels)
    elif norm_type == 'instance':
        return nn.InstanceNorm2d(num_channels)
    elif norm_type == 'layer':
        return nn.GroupNorm(1, num_channels)
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")


class Swish(nn.Module):
    """Swish activation function: x * sigmoid(x)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class ResidualBlock(nn.Module):
    """
    Residual block with time embedding injection.

    Architecture:
        x -> GroupNorm -> Swish -> Conv -> GroupNorm -> Swish -> Conv -> + x
                                      ^
                                      |
                                  time_emb
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: Optional[int] = None,
        dropout: float = 0.0,
        norm_type: str = 'group',
        num_groups: int = 32,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # First convolution path
        self.norm1 = get_norm(norm_type, in_channels, num_groups)
        self.act1 = Swish()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Time embedding projection
        self.time_emb_proj = None
        if time_emb_dim is not None:
            self.time_emb_proj = nn.Sequential(
                Swish(),
                nn.Linear(time_emb_dim, out_channels),
            )

        # Second convolution path
        self.norm2 = get_norm(norm_type, out_channels, num_groups)
        self.act2 = Swish()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Skip connection (if channels change)
        self.skip = nn.Identity()
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        time_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)

        # Add time embedding
        if self.time_emb_proj is not None and time_emb is not None:
            time_emb = self.time_emb_proj(time_emb)[:, :, None, None]
            h = h + time_emb

        h = self.norm2(h)
        h = self.act2(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Self-attention block for spatial attention."""

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        head_dim: Optional[int] = None,
        norm_type: str = 'group',
        num_groups: int = 32,
    ):
        super().__init__()

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = head_dim or (channels // num_heads)
        self.scale = self.head_dim ** -0.5

        self.norm = get_norm(norm_type, channels, num_groups)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        residual = x

        x = self.norm(x)

        qkv = self.qkv(x)
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Use F.scaled_dot_product_attention for Flash Attention on supported GPUs
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)

        out = self.proj(out)
        return out + residual


class Downsample(nn.Module):
    """Downsampling layer using strided convolution."""

    def __init__(self, channels: int, use_conv: bool = True):
        super().__init__()

        if use_conv:
            self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        else:
            self.op = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    """Upsampling layer using interpolation + convolution."""

    def __init__(self, channels: int, use_conv: bool = True):
        super().__init__()

        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.use_conv:
            x = self.conv(x)
        return x


class UNet(nn.Module):
    """
    U-Net for diffusion models.

    Architecture:
        Input -> Encoder -> Middle -> Decoder -> Output

    Features:
        - Configurable depth and channel multipliers
        - Residual blocks with time embedding
        - Self-attention at specified resolutions
        - Skip connections between encoder and decoder
        - Conditioning input support
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        model_channels: int = 64,
        channel_mult: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (16, 8),
        dropout: float = 0.0,
        num_heads: int = 4,
        num_groups: int = 32,
        time_emb_dim: Optional[int] = None,
        cond_channels: int = 0,
        image_size: int = 256,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.channel_mult = channel_mult
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.num_heads = num_heads
        self.image_size = image_size

        # Time embedding
        time_emb_dim = time_emb_dim or model_channels * 4
        self.time_emb_dim = time_emb_dim

        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(model_channels),
            nn.Linear(model_channels, time_emb_dim),
            Swish(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Input convolution
        total_in_channels = in_channels + cond_channels
        self.input_conv = nn.Conv2d(total_in_channels, model_channels, kernel_size=3, padding=1)

        # Build encoder and decoder with explicit skip channel tracking
        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        # Track skip connection channels: [(channels, resolution), ...]
        skip_channels = []

        ch = model_channels
        current_res = image_size

        # Initial skip from input_conv
        skip_channels.append(ch)

        # Encoder
        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult

            # Residual blocks at this level
            for _ in range(num_res_blocks):
                block = ResidualBlock(ch, out_ch, time_emb_dim, dropout, num_groups=num_groups)

                if current_res in attention_resolutions:
                    block = nn.ModuleList([
                        block,
                        AttentionBlock(out_ch, num_heads, num_groups=num_groups)
                    ])
                else:
                    block = nn.ModuleList([block])

                self.encoder_blocks.append(block)
                skip_channels.append(out_ch)
                ch = out_ch

            # Downsample (except last level)
            if level < len(channel_mult) - 1:
                self.encoder_blocks.append(nn.ModuleList([Downsample(ch)]))
                skip_channels.append(ch)
                current_res //= 2

        # Middle
        self.middle = nn.ModuleList([
            ResidualBlock(ch, ch, time_emb_dim, dropout, num_groups=num_groups),
            AttentionBlock(ch, num_heads, num_groups=num_groups),
            ResidualBlock(ch, ch, time_emb_dim, dropout, num_groups=num_groups),
        ])

        # Decoder (reverse order)
        for level, mult in list(enumerate(channel_mult))[::-1]:
            out_ch = model_channels * mult

            # Extra block for upsampled skip + num_res_blocks for regular skips
            for i in range(num_res_blocks + 1):
                skip_ch = skip_channels.pop()
                block = ResidualBlock(ch + skip_ch, out_ch, time_emb_dim, dropout, num_groups=num_groups)

                if current_res in attention_resolutions:
                    block = nn.ModuleList([
                        block,
                        AttentionBlock(out_ch, num_heads, num_groups=num_groups)
                    ])
                else:
                    block = nn.ModuleList([block])

                self.decoder_blocks.append(block)
                ch = out_ch

            # Upsample (except last level)
            if level > 0:
                self.decoder_blocks.append(nn.ModuleList([Upsample(ch)]))
                current_res *= 2

        # Output
        self.output_conv = nn.Sequential(
            get_norm('group', ch, num_groups),
            Swish(),
            nn.Conv2d(ch, out_channels, kernel_size=3, padding=1),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Zero initialize final output conv
        nn.init.zeros_(self.output_conv[-1].weight)
        if self.output_conv[-1].bias is not None:
            nn.init.zeros_(self.output_conv[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Noisy input (B, in_channels, H, W)
            t: Timesteps (B,)
            cond: Conditioning input (B, cond_channels, H, W)
            **kwargs: Additional conditioning (ignored)

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
                elif isinstance(layer, AttentionBlock):
                    h = layer(h)
                else:
                    h = layer(h)
            skips.append(h)

        # Middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock):
                h = layer(h, time_emb)
            else:
                h = layer(h)

        # Decoder with skip connections
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


class UNetSmall(UNet):
    """Small U-Net for fast experimentation."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        cond_channels: int = 0,
        image_size: int = 256,
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            model_channels=32,
            channel_mult=(1, 2, 4),
            num_res_blocks=1,
            attention_resolutions=(32, 16),
            dropout=0.0,
            num_heads=4,
            cond_channels=cond_channels,
            image_size=image_size,
            **kwargs,
        )


class UNetMedium(UNet):
    """Medium U-Net - good balance."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        cond_channels: int = 0,
        image_size: int = 256,
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            model_channels=64,
            channel_mult=(1, 2, 4, 8),
            num_res_blocks=2,
            attention_resolutions=(32, 16, 8),
            dropout=0.1,
            num_heads=4,
            cond_channels=cond_channels,
            image_size=image_size,
            **kwargs,
        )


class UNetLarge(UNet):
    """Large U-Net for best quality."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        cond_channels: int = 0,
        image_size: int = 256,
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            model_channels=128,
            channel_mult=(1, 2, 4, 8),
            num_res_blocks=3,
            attention_resolutions=(32, 16, 8),
            dropout=0.1,
            num_heads=8,
            cond_channels=cond_channels,
            image_size=image_size,
            **kwargs,
        )


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_unet(
    size: str = 'medium',
    in_channels: int = 1,
    out_channels: int = 1,
    cond_channels: int = 0,
    image_size: int = 256,
) -> UNet:
    """Factory function to get U-Net by size."""
    models = {
        'small': UNetSmall,
        'medium': UNetMedium,
        'large': UNetLarge,
    }

    if size not in models:
        raise ValueError(f"Unknown size: {size}. Choose from {list(models.keys())}")

    return models[size](
        in_channels=in_channels,
        out_channels=out_channels,
        cond_channels=cond_channels,
        image_size=image_size,
    )
