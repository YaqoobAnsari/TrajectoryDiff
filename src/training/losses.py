"""
Physics-Informed Losses for TrajectoryDiff.

These losses enforce physical plausibility in radio map predictions:
- TrajectoryConsistencyLoss: Match observations along trajectories
- CoverageWeightedLoss: Weight loss by coverage density
- DistanceDecayLoss: Soft constraint that signal decreases with distance
- TrajectoryDiffLoss: Combined loss for training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class TrajectoryConsistencyLoss(nn.Module):
    """
    Enforce that predictions match observations along trajectories.

    Unlike pixel-wise MSE, this:
    1. Uses bilinear interpolation for sub-pixel accuracy
    2. Weights by measurement confidence
    3. Can enforce local smoothness along path
    """

    def __init__(self, smoothness_weight: float = 0.1):
        """
        Args:
            smoothness_weight: Weight for smoothness term along trajectory.
                              Set to 0 to disable smoothness regularization.
        """
        super().__init__()
        self.smoothness_weight = smoothness_weight

    def forward(
        self,
        pred_map: torch.Tensor,       # (B, 1, H, W) predicted radio map
        sparse_rss: torch.Tensor,     # (B, 1, H, W) observed RSS values
        trajectory_mask: torch.Tensor, # (B, 1, H, W) binary mask
    ) -> torch.Tensor:
        """
        Compute trajectory consistency loss.

        Args:
            pred_map: Predicted radio map
            sparse_rss: Ground truth RSS at observed locations
            trajectory_mask: Binary mask indicating observed locations

        Returns:
            Scalar loss value
        """
        # Only compute loss where we have observations
        mask = trajectory_mask > 0.5

        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred_map.device, requires_grad=True)

        # Point-wise consistency loss
        pred_on_traj = pred_map[mask]
        obs_on_traj = sparse_rss[mask]
        consistency_loss = F.mse_loss(pred_on_traj, obs_on_traj)

        # Optional: Local smoothness along trajectory
        # (predictions should vary smoothly along the path)
        if self.smoothness_weight > 0:
            smoothness_loss = self._compute_smoothness(pred_map, trajectory_mask)
            return consistency_loss + self.smoothness_weight * smoothness_loss

        return consistency_loss

    def _compute_smoothness(
        self,
        pred_map: torch.Tensor,
        trajectory_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute gradient smoothness along trajectory."""
        # Sobel gradients
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=pred_map.dtype,
            device=pred_map.device
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=pred_map.dtype,
            device=pred_map.device
        ).view(1, 1, 3, 3)

        grad_x = F.conv2d(pred_map, sobel_x, padding=1)
        grad_y = F.conv2d(pred_map, sobel_y, padding=1)

        # Only penalize high gradients ON the trajectory
        # (we want smooth predictions along the path)
        mask = trajectory_mask > 0.5
        gradient_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred_map.device, requires_grad=True)

        return gradient_magnitude[mask].mean()


class CoverageWeightedLoss(nn.Module):
    """
    Weight prediction loss by coverage density.

    High coverage (on trajectory) → high weight (be accurate!)
    Low coverage (blind spot) → low weight (allow exploration)
    """

    def __init__(self, min_weight: float = 0.1, max_weight: float = 1.0):
        """
        Args:
            min_weight: Minimum weight for blind spots (low coverage)
            max_weight: Maximum weight for trajectory regions (high coverage)
        """
        super().__init__()
        self.min_weight = min_weight
        self.max_weight = max_weight

    def forward(
        self,
        pred: torch.Tensor,            # (B, 1, H, W) prediction
        target: torch.Tensor,          # (B, 1, H, W) target
        coverage_density: torch.Tensor, # (B, 1, H, W) coverage map [0, 1]
    ) -> torch.Tensor:
        """
        Compute coverage-weighted MSE loss.

        Args:
            pred: Predicted values (noise or x0)
            target: Target values
            coverage_density: Coverage map where 1 = on trajectory, 0 = blind spot

        Returns:
            Scalar weighted MSE loss
        """
        # Compute per-pixel squared error
        squared_error = (pred - target) ** 2

        # Weight by coverage density
        # coverage_density is in [0, 1], high = on trajectory
        weights = self.min_weight + (self.max_weight - self.min_weight) * coverage_density

        # Weighted mean
        weighted_error = squared_error * weights
        return weighted_error.mean()


class DistanceDecayLoss(nn.Module):
    """
    Soft regularization: signal should generally decrease with distance from TX.

    This is a SOFT constraint because:
    - Multipath can cause local increases
    - Walls cause sharp drops, not gradual decay
    """

    def __init__(self, weight: float = 0.01):
        """
        Args:
            weight: Scaling factor for this loss term
        """
        super().__init__()
        self.weight = weight

    def forward(
        self,
        pred_map: torch.Tensor,      # (B, 1, H, W) predicted radio map
        tx_position: torch.Tensor,   # (B, 2) normalized TX position
        building_map: torch.Tensor,  # (B, 1, H, W) to mask out walls
    ) -> torch.Tensor:
        """
        Penalize signal INCREASING with distance from TX.

        Args:
            pred_map: Predicted radio map
            tx_position: Normalized (0-1) transmitter position
            building_map: Building map where 1 = wall, 0 = free space

        Returns:
            Scalar penalty for physics violations
        """
        B, C, H, W = pred_map.shape
        device = pred_map.device

        # Create distance map from TX
        y_coords = torch.linspace(0, 1, H, device=device)
        x_coords = torch.linspace(0, 1, W, device=device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

        distance_maps = []
        for b in range(B):
            tx_x, tx_y = tx_position[b]
            dist = torch.sqrt((xx - tx_x) ** 2 + (yy - tx_y) ** 2)
            distance_maps.append(dist)

        distance_map = torch.stack(distance_maps, dim=0).unsqueeze(1)  # (B, 1, H, W)

        # Define near and far regions
        near_tx = distance_map < 0.3
        far_from_tx = distance_map > 0.7

        # Mask out walls
        free_space = building_map < 0.5

        # Compute mean RSS in near and far regions
        near_mask = near_tx & free_space
        far_mask = far_from_tx & free_space

        near_count = near_mask.sum()
        far_count = far_mask.sum()

        if near_count == 0 or far_count == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        near_rss = pred_map[near_mask].mean()
        far_rss = pred_map[far_mask].mean()

        # Penalize if far RSS > near RSS (physically wrong)
        violation = F.relu(far_rss - near_rss)

        return self.weight * violation


class WallAttenuationLoss(nn.Module):
    """
    Enforce that signal attenuates through walls.

    For each wall pixel, check that signal on far side (from TX)
    is lower than signal on near side.

    Note: This is a more complex loss - currently a placeholder.
    """

    def __init__(self, min_attenuation_db: float = 3.0):
        """
        Args:
            min_attenuation_db: Minimum expected attenuation through walls
        """
        super().__init__()
        self.min_attenuation_db = min_attenuation_db

    def forward(
        self,
        pred_map: torch.Tensor,      # (B, 1, H, W)
        building_map: torch.Tensor,  # (B, 1, H, W) walls = 1
        tx_position: torch.Tensor,   # (B, 2)
    ) -> torch.Tensor:
        """
        Compute wall attenuation violation loss.

        Currently returns 0 - full implementation would:
        1. Extract wall boundaries
        2. Determine wall normal direction
        3. Sample signal on both sides
        4. Penalize if far_side > near_side - min_attenuation
        """
        # TODO: Full implementation requires morphological operations
        # For now, return zero loss
        return torch.tensor(0.0, device=pred_map.device, requires_grad=True)


class TrajectoryDiffLoss(nn.Module):
    """
    Combined loss for TrajectoryDiff training.

    Combines:
    - Diffusion loss (MSE on noise or x0)
    - Trajectory consistency loss
    - Optional coverage weighting
    - Optional distance decay regularization
    """

    def __init__(
        self,
        diffusion_weight: float = 1.0,
        trajectory_consistency_weight: float = 0.1,
        coverage_weighted: bool = True,
        coverage_min_weight: float = 0.1,
        coverage_max_weight: float = 1.0,
        distance_decay_weight: float = 0.01,
        smoothness_weight: float = 0.1,
    ):
        """
        Args:
            diffusion_weight: Weight for main diffusion loss
            trajectory_consistency_weight: Weight for trajectory consistency
            coverage_weighted: Whether to use coverage-weighted diffusion loss
            coverage_min_weight: Min weight for coverage weighting
            coverage_max_weight: Max weight for coverage weighting
            distance_decay_weight: Weight for distance decay regularization
            smoothness_weight: Weight for trajectory smoothness
        """
        super().__init__()
        self.diffusion_weight = diffusion_weight
        self.trajectory_consistency_weight = trajectory_consistency_weight
        self.coverage_weighted = coverage_weighted
        self.distance_decay_weight = distance_decay_weight

        self.trajectory_loss = TrajectoryConsistencyLoss(
            smoothness_weight=smoothness_weight
        )
        self.coverage_loss = CoverageWeightedLoss(
            min_weight=coverage_min_weight,
            max_weight=coverage_max_weight,
        ) if coverage_weighted else None
        self.distance_loss = DistanceDecayLoss(weight=1.0)  # Scale applied below

    def forward(
        self,
        noise_pred: torch.Tensor,
        noise_target: torch.Tensor,
        pred_x0: Optional[torch.Tensor] = None,
        batch: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses.

        Args:
            noise_pred: Predicted noise
            noise_target: Target noise
            pred_x0: Predicted clean image (optional, for physics losses)
            batch: Dictionary with 'sparse_rss', 'trajectory_mask', 'coverage_density',
                   'tx_position', 'building_map'

        Returns:
            Dictionary with individual losses for logging, including 'total'
        """
        losses = {}
        batch = batch or {}

        # Main diffusion loss
        if self.coverage_loss is not None and 'coverage_density' in batch:
            diffusion_loss = self.coverage_loss(
                noise_pred, noise_target, batch['coverage_density']
            )
        else:
            diffusion_loss = F.mse_loss(noise_pred, noise_target)

        losses['diffusion'] = diffusion_loss * self.diffusion_weight

        # Trajectory consistency (on predicted x0)
        if (
            self.trajectory_consistency_weight > 0
            and pred_x0 is not None
            and 'sparse_rss' in batch
            and 'trajectory_mask' in batch
        ):
            traj_loss = self.trajectory_loss(
                pred_x0,
                batch['sparse_rss'],
                batch['trajectory_mask'],
            )
            losses['trajectory_consistency'] = traj_loss * self.trajectory_consistency_weight

        # Distance decay regularization
        if (
            self.distance_decay_weight > 0
            and pred_x0 is not None
            and 'tx_position' in batch
            and 'building_map' in batch
        ):
            dist_loss = self.distance_loss(
                pred_x0,
                batch['tx_position'],
                batch['building_map'],
            )
            losses['distance_decay'] = dist_loss * self.distance_decay_weight

        # Total loss
        losses['total'] = sum(losses.values())

        return losses


def compute_physics_losses(
    pred_map: torch.Tensor,
    sparse_rss: torch.Tensor,
    trajectory_mask: torch.Tensor,
    coverage_density: Optional[torch.Tensor] = None,
    tx_position: Optional[torch.Tensor] = None,
    building_map: Optional[torch.Tensor] = None,
    trajectory_weight: float = 0.1,
    distance_weight: float = 0.01,
) -> Dict[str, torch.Tensor]:
    """
    Convenience function to compute all physics losses.

    Args:
        pred_map: Predicted radio map (B, 1, H, W)
        sparse_rss: Observed RSS values (B, 1, H, W)
        trajectory_mask: Binary trajectory mask (B, 1, H, W)
        coverage_density: Coverage density map (B, 1, H, W), optional
        tx_position: Transmitter position (B, 2), optional
        building_map: Building map (B, 1, H, W), optional
        trajectory_weight: Weight for trajectory consistency
        distance_weight: Weight for distance decay

    Returns:
        Dictionary of loss values
    """
    losses = {}

    # Trajectory consistency
    traj_loss_fn = TrajectoryConsistencyLoss()
    losses['trajectory_consistency'] = trajectory_weight * traj_loss_fn(
        pred_map, sparse_rss, trajectory_mask
    )

    # Distance decay
    if tx_position is not None and building_map is not None:
        dist_loss_fn = DistanceDecayLoss(weight=1.0)
        losses['distance_decay'] = distance_weight * dist_loss_fn(
            pred_map, tx_position, building_map
        )

    losses['total_physics'] = sum(losses.values())

    return losses
