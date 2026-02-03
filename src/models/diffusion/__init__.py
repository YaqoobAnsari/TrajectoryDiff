"""
Diffusion model components for TrajectoryDiff.

Implements DDPM (Denoising Diffusion Probabilistic Models) and related
components for trajectory-conditioned radio map generation.
"""

from .ddpm import (
    GaussianDiffusion,
    DDIMSampler,
    SinusoidalPositionEmbedding,
    linear_beta_schedule,
    cosine_beta_schedule,
    sigmoid_beta_schedule,
)

from .unet import (
    UNet,
    UNetSmall,
    UNetMedium,
    UNetLarge,
    ResidualBlock,
    AttentionBlock,
    get_unet,
    count_parameters,
)

from .attention import (
    CoverageAwareAttention,
    CoverageAwareTransformerBlock,
    CoverageAwareAttentionBlock,
    AdaptiveCoverageAttention,
    downsample_coverage,
    upsample_coverage,
)

__all__ = [
    # DDPM
    'GaussianDiffusion',
    'DDIMSampler',
    'SinusoidalPositionEmbedding',
    'linear_beta_schedule',
    'cosine_beta_schedule',
    'sigmoid_beta_schedule',
    # U-Net
    'UNet',
    'UNetSmall',
    'UNetMedium',
    'UNetLarge',
    'ResidualBlock',
    'AttentionBlock',
    'get_unet',
    'count_parameters',
    # Coverage-Aware Attention
    'CoverageAwareAttention',
    'CoverageAwareTransformerBlock',
    'CoverageAwareAttentionBlock',
    'AdaptiveCoverageAttention',
    'downsample_coverage',
    'upsample_coverage',
]
