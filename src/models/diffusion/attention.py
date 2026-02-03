"""
Coverage-Aware Attention for TrajectoryDiff.

Novel attention mechanism that modulates attention weights by coverage density.
This is the key architectural contribution for ECCV/CVPR.

Key insight: When denoising, we should:
- Trust features from high-coverage regions (we have data there)
- Be more exploratory in low-coverage regions (blind spots)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CoverageAwareAttention(nn.Module):
    """
    Novel attention mechanism that modulates attention by coverage density.

    Key insight: When denoising, we should:
    - Trust features from high-coverage regions (we have data there)
    - Be more exploratory in low-coverage regions (blind spots)

    This is implemented by scaling attention weights based on
    the coverage density of key positions.

    Args:
        dim: Feature dimension
        num_heads: Number of attention heads
        coverage_temperature: Temperature for coverage modulation.
                              Higher = more uniform, lower = more coverage-dependent.
        qkv_bias: Whether to use bias in QKV projection
        dropout: Attention dropout rate
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        coverage_temperature: float = 1.0,
        qkv_bias: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.coverage_temperature = coverage_temperature

        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"

        self.to_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

        # Learnable coverage modulation
        # Maps scalar coverage to per-head scaling factors
        self.coverage_gate = nn.Sequential(
            nn.Linear(1, dim // 4),
            nn.SiLU(),
            nn.Linear(dim // 4, num_heads),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,                 # (B, N, D) spatial features flattened
        coverage: Optional[torch.Tensor] = None,  # (B, N, 1) coverage density per position
    ) -> torch.Tensor:
        """
        Apply coverage-aware attention.

        Args:
            x: Input features of shape (B, N, D)
            coverage: Optional coverage density of shape (B, N, 1) where 1 = on trajectory

        Returns:
            Output features of shape (B, N, D)
        """
        B, N, D = x.shape

        # Standard QKV projection
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: t.view(B, N, self.num_heads, self.head_dim).transpose(1, 2),
            qkv
        )  # Each: (B, heads, N, head_dim)

        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)

        # NOVEL: Modulate attention by coverage density
        if coverage is not None:
            # coverage_gate produces per-head scaling factors
            coverage_scale = self.coverage_gate(coverage)  # (B, N, heads)
            coverage_scale = coverage_scale.permute(0, 2, 1).unsqueeze(-1)  # (B, heads, N, 1)

            # Scale attention to keys based on their coverage
            # High coverage keys get higher attention
            key_coverage = coverage_scale.transpose(-2, -1)  # (B, heads, 1, N)

            # Temperature-scaled modulation
            # Higher temperature = more uniform attention
            # Lower temperature = more coverage-dependent
            coverage_weight = key_coverage ** (1.0 / self.coverage_temperature)

            # Multiplicative attention modulation
            attn = attn * coverage_weight

        # Softmax normalization
        attn = attn.softmax(dim=-1)

        # Apply attention to values
        out = attn @ v  # (B, heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, D)

        return self.to_out(out)


class CoverageAwareTransformerBlock(nn.Module):
    """
    Transformer block with coverage-aware attention.

    Standard pre-norm transformer block but with coverage-aware attention
    instead of standard self-attention.

    Args:
        dim: Feature dimension
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dim multiplier
        dropout: Dropout rate
        coverage_temperature: Temperature for coverage modulation
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        coverage_temperature: float = 1.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = CoverageAwareAttention(
            dim,
            num_heads,
            coverage_temperature=coverage_temperature,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(
        self,
        x: torch.Tensor,
        coverage: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional coverage modulation.

        Args:
            x: Input features (B, N, D)
            coverage: Optional coverage density (B, N, 1)

        Returns:
            Output features (B, N, D)
        """
        x = x + self.attn(self.norm1(x), coverage)
        x = x + self.ffn(self.norm2(x))
        return x


class CoverageAwareAttentionBlock(nn.Module):
    """
    Spatial attention block compatible with U-Net architecture.

    Takes 4D spatial features (B, C, H, W), flattens to sequence,
    applies coverage-aware attention, and reshapes back.

    This is designed to be a drop-in replacement for standard
    attention blocks in diffusion U-Nets.

    Args:
        channels: Number of input/output channels
        num_heads: Number of attention heads (default: channels // 32)
        coverage_temperature: Temperature for coverage modulation
    """

    def __init__(
        self,
        channels: int,
        num_heads: Optional[int] = None,
        coverage_temperature: float = 1.0,
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads or max(1, channels // 32)

        # Group normalization before attention
        self.norm = nn.GroupNorm(32, channels)

        # Coverage-aware attention
        self.attention = CoverageAwareAttention(
            dim=channels,
            num_heads=self.num_heads,
            coverage_temperature=coverage_temperature,
        )

    def forward(
        self,
        x: torch.Tensor,                  # (B, C, H, W)
        coverage: Optional[torch.Tensor] = None,  # (B, 1, H, W)
    ) -> torch.Tensor:
        """
        Apply coverage-aware attention to spatial features.

        Args:
            x: Input features (B, C, H, W)
            coverage: Optional coverage density map (B, 1, H, W)

        Returns:
            Output features (B, C, H, W)
        """
        B, C, H, W = x.shape

        # Normalize
        h = self.norm(x)

        # Reshape to sequence: (B, C, H, W) -> (B, H*W, C)
        h = h.view(B, C, H * W).transpose(1, 2)

        # Reshape coverage if provided: (B, 1, H, W) -> (B, H*W, 1)
        if coverage is not None:
            coverage = coverage.view(B, 1, H * W).transpose(1, 2)

        # Apply attention
        h = self.attention(h, coverage)

        # Reshape back: (B, H*W, C) -> (B, C, H, W)
        h = h.transpose(1, 2).view(B, C, H, W)

        # Residual connection
        return x + h


class AdaptiveCoverageAttention(nn.Module):
    """
    Attention with adaptive coverage fusion.

    Learns to combine standard attention and coverage-weighted attention
    based on the input features. This provides flexibility during training
    to balance between the two modalities.

    Args:
        dim: Feature dimension
        num_heads: Number of attention heads
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

        # Coverage embedding
        self.coverage_embed = nn.Sequential(
            nn.Linear(1, dim // 4),
            nn.SiLU(),
            nn.Linear(dim // 4, num_heads),
        )

        # Learnable fusion weight (how much to trust coverage)
        self.fusion_gate = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.SiLU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,                 # (B, N, D)
        coverage: Optional[torch.Tensor] = None,  # (B, N, 1)
    ) -> torch.Tensor:
        """
        Apply adaptive coverage-aware attention.

        Args:
            x: Input features (B, N, D)
            coverage: Optional coverage density (B, N, 1)

        Returns:
            Output features (B, N, D)
        """
        B, N, D = x.shape

        # QKV projection
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: t.view(B, N, self.num_heads, self.head_dim).transpose(1, 2),
            qkv
        )

        # Standard attention scores
        attn_standard = (q @ k.transpose(-2, -1)) * self.scale

        if coverage is not None:
            # Coverage-weighted attention
            coverage_bias = self.coverage_embed(coverage)  # (B, N, heads)
            coverage_bias = coverage_bias.permute(0, 2, 1).unsqueeze(-2)  # (B, heads, 1, N)

            # Add coverage as bias to attention (keys with high coverage get boosted)
            attn_coverage = attn_standard + coverage_bias

            # Compute fusion weight from input features
            # Pool over sequence dimension
            x_pooled = x.mean(dim=1)  # (B, D)
            fusion_weight = self.fusion_gate(x_pooled)  # (B, 1)
            fusion_weight = fusion_weight.view(B, 1, 1, 1)  # (B, 1, 1, 1)

            # Fuse standard and coverage-weighted attention
            attn = fusion_weight * attn_coverage + (1 - fusion_weight) * attn_standard
        else:
            attn = attn_standard

        # Softmax and apply to values
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, D)

        return self.to_out(out)


def downsample_coverage(
    coverage: torch.Tensor,
    target_size: tuple,
) -> torch.Tensor:
    """
    Downsample coverage map to match feature resolution.

    Uses average pooling to downsample, preserving the average
    coverage density in each region.

    Args:
        coverage: Coverage map (B, 1, H, W)
        target_size: Target spatial size (H', W')

    Returns:
        Downsampled coverage (B, 1, H', W')
    """
    return F.adaptive_avg_pool2d(coverage, target_size)


def upsample_coverage(
    coverage: torch.Tensor,
    target_size: tuple,
) -> torch.Tensor:
    """
    Upsample coverage map to match feature resolution.

    Uses bilinear interpolation to upsample.

    Args:
        coverage: Coverage map (B, 1, H, W)
        target_size: Target spatial size (H', W')

    Returns:
        Upsampled coverage (B, 1, H', W')
    """
    return F.interpolate(
        coverage,
        size=target_size,
        mode='bilinear',
        align_corners=False,
    )
