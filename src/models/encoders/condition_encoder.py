"""
Condition Encoder for Trajectory-Conditioned Diffusion.

Encodes trajectory-related inputs (building map, sparse RSS, coverage density,
TX position) into conditioning features for the diffusion model.

Multiple fusion strategies supported:
- Concatenation: Simple channel-wise concatenation
- FiLM: Feature-wise Linear Modulation
- Cross-attention: Attention-based conditioning (future)
"""

import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding2D(nn.Module):
    """
    2D positional encoding for spatial features.

    Encodes (x, y) position as a spatial feature map.
    """

    def __init__(self, channels: int, max_resolution: int = 256):
        """
        Args:
            channels: Number of output channels (must be divisible by 4)
            max_resolution: Maximum spatial resolution
        """
        super().__init__()
        assert channels % 4 == 0, "channels must be divisible by 4"

        self.channels = channels
        self.max_resolution = max_resolution

        # Create position encoding lookup
        pe_channels = channels // 4
        div_term = torch.exp(
            torch.arange(0, pe_channels, dtype=torch.float) *
            (-math.log(10000.0) / pe_channels)
        )
        self.register_buffer('div_term', div_term)

    def forward(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """
        Generate 2D positional encoding.

        Args:
            H: Height
            W: Width
            device: Device to create tensor on

        Returns:
            Positional encoding (1, channels, H, W)
        """
        pe_channels = self.channels // 4

        # Create position grids
        y_pos = torch.arange(H, device=device).float() / self.max_resolution
        x_pos = torch.arange(W, device=device).float() / self.max_resolution

        # Apply sinusoidal encoding
        y_enc_sin = torch.sin(y_pos[:, None] * self.div_term[None, :])  # (H, C/4)
        y_enc_cos = torch.cos(y_pos[:, None] * self.div_term[None, :])  # (H, C/4)
        x_enc_sin = torch.sin(x_pos[:, None] * self.div_term[None, :])  # (W, C/4)
        x_enc_cos = torch.cos(x_pos[:, None] * self.div_term[None, :])  # (W, C/4)

        # Expand to 2D
        y_enc_sin = y_enc_sin[:, None, :].expand(H, W, pe_channels)  # (H, W, C/4)
        y_enc_cos = y_enc_cos[:, None, :].expand(H, W, pe_channels)
        x_enc_sin = x_enc_sin[None, :, :].expand(H, W, pe_channels)
        x_enc_cos = x_enc_cos[None, :, :].expand(H, W, pe_channels)

        # Concatenate
        pe = torch.cat([y_enc_sin, y_enc_cos, x_enc_sin, x_enc_cos], dim=-1)
        pe = pe.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)

        return pe


class TxPositionEncoder(nn.Module):
    """
    Encodes transmitter position as a spatial feature map.

    Creates a distance-based encoding centered on the TX position.
    Expects tx_position in [0, 1] normalized coordinates.
    """

    def __init__(
        self,
        channels: int,
        encoding_type: str = 'gaussian',
        sigma: float = 0.04,
    ):
        """
        Args:
            channels: Number of output channels
            encoding_type: 'gaussian', 'distance', or 'sinusoidal'
            sigma: Spread for gaussian encoding (in [0,1] normalized space;
                   0.04 ≈ 10 pixels at 256x256 resolution)
        """
        super().__init__()

        self.channels = channels
        self.encoding_type = encoding_type
        self.sigma = sigma

        if encoding_type == 'sinusoidal':
            self.mlp = nn.Sequential(
                nn.Linear(2, channels),
                nn.SiLU(),
                nn.Linear(channels, channels),
            )
        elif encoding_type == 'gaussian':
            # Multiple Gaussian scales (from very local to map-wide)
            self.sigmas = nn.Parameter(
                torch.tensor([sigma * (2 ** i) for i in range(channels)]),
                requires_grad=False
            )

    def forward(
        self,
        tx_position: torch.Tensor,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """
        Encode transmitter position.

        Args:
            tx_position: TX coordinates (B, 2) in [0, 1] normalized range
            H: Output height
            W: Output width

        Returns:
            Encoded position (B, channels, H, W)
        """
        B = tx_position.shape[0]
        device = tx_position.device

        # Create coordinate grids in [0, 1] to match normalized tx_position
        y_coords = torch.linspace(0, 1, H, device=device)
        x_coords = torch.linspace(0, 1, W, device=device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

        # Compute distance from TX (all in [0, 1] space)
        tx_x = tx_position[:, 0][:, None, None]  # (B, 1, 1)
        tx_y = tx_position[:, 1][:, None, None]  # (B, 1, 1)

        dist_sq = (xx[None] - tx_x) ** 2 + (yy[None] - tx_y) ** 2  # (B, H, W)
        dist = torch.sqrt(dist_sq + 1e-6)

        if self.encoding_type == 'gaussian':
            # Multi-scale Gaussian encoding
            encoding = torch.exp(-dist_sq[:, None] / (2 * self.sigmas[None, :, None, None] ** 2))

        elif self.encoding_type == 'distance':
            # Normalized distance encoding
            encoding = 1.0 / (1.0 + dist / self.sigma)
            encoding = encoding.unsqueeze(1).expand(-1, self.channels, -1, -1)

        elif self.encoding_type == 'sinusoidal':
            # Sinusoidal encoding (tx_position already in [0, 1])
            pos_embed = self.mlp(tx_position)  # (B, channels)
            encoding = pos_embed[:, :, None, None].expand(-1, -1, H, W)

        return encoding


class ConvBlock(nn.Module):
    """Simple convolutional block with normalization and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        norm: bool = True,
        activation: bool = True,
    ):
        super().__init__()

        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]
        if norm:
            layers.append(nn.GroupNorm(min(32, out_channels), out_channels))
        if activation:
            layers.append(nn.SiLU())

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConditionEncoder(nn.Module):
    """
    Encodes trajectory conditioning inputs for the diffusion model.

    Input channels:
        - building_map: Building layout (1 channel)
        - sparse_rss: Sparse RSS measurements (1 channel)
        - trajectory_mask: Binary mask of sampled locations (1 channel)
        - coverage_density: Smoothed coverage density (1 channel)
        - tx_position: Transmitter location (2D coordinate)

    Output:
        Conditioning tensor (B, out_channels, H, W) for U-Net input
    """

    def __init__(
        self,
        out_channels: int = 64,
        hidden_channels: int = 32,
        use_building_map: bool = True,
        use_sparse_rss: bool = True,
        use_trajectory_mask: bool = True,
        use_coverage_density: bool = True,
        use_tx_position: bool = True,
        tx_encoding_type: str = 'gaussian',
        tx_channels: int = 8,
        use_positional_encoding: bool = False,
        positional_channels: int = 16,
        fusion_type: str = 'concat',  # 'concat', 'film', or 'add'
    ):
        """
        Initialize condition encoder.

        Args:
            out_channels: Output conditioning channels
            hidden_channels: Hidden layer channels
            use_building_map: Include building map
            use_sparse_rss: Include sparse RSS measurements
            use_trajectory_mask: Include trajectory/sampling mask
            use_coverage_density: Include coverage density map
            use_tx_position: Include transmitter position
            tx_encoding_type: Type of TX position encoding
            tx_channels: Channels for TX encoding
            use_positional_encoding: Add 2D positional encoding
            positional_channels: Channels for positional encoding
            fusion_type: How to fuse different inputs
        """
        super().__init__()

        self.use_building_map = use_building_map
        self.use_sparse_rss = use_sparse_rss
        self.use_trajectory_mask = use_trajectory_mask
        self.use_coverage_density = use_coverage_density
        self.use_tx_position = use_tx_position
        self.use_positional_encoding = use_positional_encoding
        self.fusion_type = fusion_type

        # Count input channels
        in_channels = 0
        if use_building_map:
            in_channels += 1
        if use_sparse_rss:
            in_channels += 1
        if use_trajectory_mask:
            in_channels += 1
        if use_coverage_density:
            in_channels += 1
        if use_tx_position:
            in_channels += tx_channels
            self.tx_encoder = TxPositionEncoder(
                channels=tx_channels,
                encoding_type=tx_encoding_type,
            )
        if use_positional_encoding:
            in_channels += positional_channels
            self.pos_encoder = PositionalEncoding2D(channels=positional_channels)

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Encoder network
        if fusion_type == 'concat':
            self.encoder = nn.Sequential(
                ConvBlock(in_channels, hidden_channels),
                ConvBlock(hidden_channels, hidden_channels),
                ConvBlock(hidden_channels, out_channels, norm=True, activation=False),
            )
        elif fusion_type == 'add':
            # Separate encoders for each input type
            self.encoders = nn.ModuleDict()
            if use_building_map:
                self.encoders['building'] = ConvBlock(1, out_channels)
            if use_sparse_rss:
                self.encoders['sparse_rss'] = ConvBlock(1, out_channels)
            if use_trajectory_mask:
                self.encoders['mask'] = ConvBlock(1, out_channels)
            if use_coverage_density:
                self.encoders['coverage'] = ConvBlock(1, out_channels)
            if use_tx_position:
                self.encoders['tx'] = ConvBlock(tx_channels, out_channels)

            # Fusion layer
            self.fusion = ConvBlock(out_channels, out_channels, norm=True, activation=False)

    def forward(
        self,
        building_map: Optional[torch.Tensor] = None,
        sparse_rss: Optional[torch.Tensor] = None,
        trajectory_mask: Optional[torch.Tensor] = None,
        coverage_density: Optional[torch.Tensor] = None,
        tx_position: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode conditioning inputs.

        Args:
            building_map: Building layout (B, 1, H, W)
            sparse_rss: Sparse RSS values (B, 1, H, W)
            trajectory_mask: Sampling mask (B, 1, H, W)
            coverage_density: Coverage density (B, 1, H, W)
            tx_position: TX location (B, 2)

        Returns:
            Conditioning tensor (B, out_channels, H, W)
        """
        # Determine spatial dimensions from first available input
        for tensor in [building_map, sparse_rss, trajectory_mask, coverage_density]:
            if tensor is not None:
                B, _, H, W = tensor.shape
                device = tensor.device
                break
        else:
            raise ValueError("At least one spatial input must be provided")

        if self.fusion_type == 'concat':
            # Concatenate all inputs
            inputs = []

            if self.use_building_map and building_map is not None:
                inputs.append(building_map)

            if self.use_sparse_rss and sparse_rss is not None:
                inputs.append(sparse_rss)

            if self.use_trajectory_mask and trajectory_mask is not None:
                inputs.append(trajectory_mask)

            if self.use_coverage_density and coverage_density is not None:
                inputs.append(coverage_density)

            if self.use_tx_position and tx_position is not None:
                tx_encoding = self.tx_encoder(tx_position, H, W)
                inputs.append(tx_encoding)

            if self.use_positional_encoding:
                pos_encoding = self.pos_encoder(H, W, device).expand(B, -1, -1, -1)
                inputs.append(pos_encoding)

            x = torch.cat(inputs, dim=1)
            return self.encoder(x)

        elif self.fusion_type == 'add':
            # Encode separately and add
            features = torch.zeros(B, self.out_channels, H, W, device=device)

            if self.use_building_map and building_map is not None:
                features = features + self.encoders['building'](building_map)

            if self.use_sparse_rss and sparse_rss is not None:
                features = features + self.encoders['sparse_rss'](sparse_rss)

            if self.use_trajectory_mask and trajectory_mask is not None:
                features = features + self.encoders['mask'](trajectory_mask)

            if self.use_coverage_density and coverage_density is not None:
                features = features + self.encoders['coverage'](coverage_density)

            if self.use_tx_position and tx_position is not None:
                tx_encoding = self.tx_encoder(tx_position, H, W)
                features = features + self.encoders['tx'](tx_encoding)

            return self.fusion(features)


class TrajectoryConditionedUNet(nn.Module):
    """
    Complete trajectory-conditioned diffusion model.

    Combines the condition encoder with the U-Net denoiser.
    Optionally uses CoverageAwareUNet for coverage-modulated attention.
    """

    def __init__(
        self,
        # U-Net parameters
        unet_size: str = 'medium',
        image_size: int = 256,
        # Condition encoder parameters
        condition_channels: int = 64,
        use_building_map: bool = True,
        use_sparse_rss: bool = True,
        use_trajectory_mask: bool = True,
        use_coverage_density: bool = True,
        use_tx_position: bool = True,
        tx_encoding_type: str = 'gaussian',
        # Coverage-aware attention
        use_coverage_attention: bool = False,
        coverage_temperature: float = 1.0,
    ):
        """
        Initialize trajectory-conditioned model.

        Args:
            unet_size: Size of U-Net ('small', 'medium', 'large')
            image_size: Input image size
            condition_channels: Conditioning tensor channels
            use_building_map: Include building map in conditioning
            use_sparse_rss: Include sparse RSS measurements
            use_trajectory_mask: Include trajectory mask
            use_coverage_density: Include coverage density
            use_tx_position: Include TX position encoding
            tx_encoding_type: Type of TX position encoding
            use_coverage_attention: Use CoverageAwareUNet instead of standard UNet
            coverage_temperature: Temperature for coverage attention modulation
        """
        super().__init__()

        self.use_coverage_attention = use_coverage_attention

        # Condition encoder
        self.condition_encoder = ConditionEncoder(
            out_channels=condition_channels,
            use_building_map=use_building_map,
            use_sparse_rss=use_sparse_rss,
            use_trajectory_mask=use_trajectory_mask,
            use_coverage_density=use_coverage_density,
            use_tx_position=use_tx_position,
            tx_encoding_type=tx_encoding_type,
        )

        # U-Net (radio map has 1 channel, conditioning adds condition_channels)
        if use_coverage_attention:
            from ..diffusion.coverage_unet import get_coverage_aware_unet
            self.unet = get_coverage_aware_unet(
                size=unet_size,
                in_channels=1,
                out_channels=1,
                cond_channels=condition_channels,
                image_size=image_size,
                coverage_temperature=coverage_temperature,
            )
        else:
            from ..diffusion import get_unet
            self.unet = get_unet(
                size=unet_size,
                in_channels=1,
                out_channels=1,
                cond_channels=condition_channels,
                image_size=image_size,
            )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        building_map: Optional[torch.Tensor] = None,
        sparse_rss: Optional[torch.Tensor] = None,
        trajectory_mask: Optional[torch.Tensor] = None,
        coverage_density: Optional[torch.Tensor] = None,
        tx_position: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Noisy radio map (B, 1, H, W)
            t: Timesteps (B,)
            building_map: Building layout (B, 1, H, W)
            sparse_rss: Sparse RSS values (B, 1, H, W)
            trajectory_mask: Sampling mask (B, 1, H, W)
            coverage_density: Coverage density (B, 1, H, W)
            tx_position: TX location (B, 2)

        Returns:
            Predicted noise or x_0 (B, 1, H, W)
        """
        # Encode conditioning
        cond = self.condition_encoder(
            building_map=building_map,
            sparse_rss=sparse_rss,
            trajectory_mask=trajectory_mask,
            coverage_density=coverage_density,
            tx_position=tx_position,
        )

        # U-Net forward (coverage passed through for CoverageAwareUNet;
        # standard UNet ignores it via **kwargs)
        return self.unet(x, t, cond=cond, coverage=coverage_density)


def get_condition_encoder(
    preset: str = 'full',
    out_channels: int = 64,
) -> ConditionEncoder:
    """
    Factory function to get condition encoder presets.

    Args:
        preset: 'full', 'minimal', or 'building_only'
        out_channels: Output channels

    Returns:
        ConditionEncoder instance
    """
    presets = {
        'full': dict(
            use_building_map=True,
            use_sparse_rss=True,
            use_trajectory_mask=True,
            use_coverage_density=True,
            use_tx_position=True,
        ),
        'minimal': dict(
            use_building_map=True,
            use_sparse_rss=True,
            use_trajectory_mask=False,
            use_coverage_density=False,
            use_tx_position=True,
        ),
        'building_only': dict(
            use_building_map=True,
            use_sparse_rss=False,
            use_trajectory_mask=False,
            use_coverage_density=False,
            use_tx_position=True,
        ),
    }

    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Choose from {list(presets.keys())}")

    return ConditionEncoder(out_channels=out_channels, **presets[preset])
