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
    CoverageAwareAttentionBlock,
    downsample_coverage,
    upsample_coverage,
)

from .coverage_unet import (
    CoverageAwareUNet,
    get_coverage_aware_unet,
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
    # Coverage-Aware U-Net
    'CoverageAwareUNet',
    'get_coverage_aware_unet',
    # Coverage-Aware Attention
    'CoverageAwareAttention',
    'CoverageAwareAttentionBlock',
    'downsample_coverage',
    'upsample_coverage',
]
