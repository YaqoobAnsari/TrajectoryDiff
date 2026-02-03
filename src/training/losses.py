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

        # Pre-register Sobel kernels as buffers (avoids re-creation every forward pass)
        if smoothness_weight > 0:
            self.register_buffer('sobel_x', torch.tensor(
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
            ).view(1, 1, 3, 3))
            self.register_buffer('sobel_y', torch.tensor(
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
            ).view(1, 1, 3, 3))

    def forward(
        self,
        pred_map: torch.Tensor,       # (B, 1, H, W) predicted radio map
        sparse_rss: torch.Tensor,     # (B, 1, H, W) observed RSS values
        trajectory_mask: torch.Tensor, # (B, 1, H, W) binary mask
        reduction: str = 'mean',      # 'mean' or 'none'
    ) -> torch.Tensor:
        """
        Compute trajectory consistency loss.

        Args:
            pred_map: Predicted radio map
            sparse_rss: Ground truth RSS at observed locations
            trajectory_mask: Binary mask indicating observed locations
            reduction: 'mean' returns scalar; 'none' returns (B,) per-sample losses

        Returns:
            Scalar loss value (reduction='mean') or (B,) per-sample losses (reduction='none')
        """
        B = pred_map.shape[0]
        mask = trajectory_mask > 0.5

        if reduction == 'none':
            # Per-sample losses for SNR weighting
            per_sample = []
            for i in range(B):
                sample_mask = mask[i]
                if sample_mask.sum() == 0:
                    per_sample.append(torch.tensor(0.0, device=pred_map.device))
                else:
                    sample_loss = F.mse_loss(
                        pred_map[i][sample_mask], sparse_rss[i][sample_mask]
                    )
                    if self.smoothness_weight > 0:
                        smooth = self._compute_smoothness(
                            pred_map[i:i+1], trajectory_mask[i:i+1]
                        )
                        sample_loss = sample_loss + self.smoothness_weight * smooth
                    per_sample.append(sample_loss)
            return torch.stack(per_sample)

        # Original scalar reduction
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred_map.device, requires_grad=True)

        # Point-wise consistency loss
        pred_on_traj = pred_map[mask]
        obs_on_traj = sparse_rss[mask]
        consistency_loss = F.mse_loss(pred_on_traj, obs_on_traj)

        # Optional: Local smoothness along trajectory
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
        # Use pre-registered Sobel buffers (cast to input dtype for mixed precision)
        grad_x = F.conv2d(pred_map, self.sobel_x.to(pred_map.dtype), padding=1)
        grad_y = F.conv2d(pred_map, self.sobel_y.to(pred_map.dtype), padding=1)

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

    NOTE: Spatial weighting violates the standard ELBO bound because the
    uniform-weight diffusion loss is a (scaled) variational lower bound on
    log p(x), and non-uniform weighting breaks this correspondence. This is
    intentional: our objective is task-specific reconstruction quality (minimizing
    dBm RMSE in observed regions), NOT generative modeling of the radio-map
    distribution. The coverage weighting trades theoretically-grounded generation
    for better practical reconstruction fidelity where measurements exist.
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

    def __init__(self, weight: float = 0.01, image_size: int = 256):
        """
        Args:
            weight: Scaling factor for this loss term
            image_size: Expected spatial dimension for pre-computed meshgrid
        """
        super().__init__()
        self.weight = weight
        self._cached_size = image_size

        # Pre-register coordinate grids as buffers (avoids re-creation every forward)
        # Note: torch.meshgrid returns views sharing memory; must .clone() before
        # registering as buffers, otherwise load_state_dict() fails with
        # "more than one element of the written-to tensor refers to a single memory location"
        y_coords = torch.linspace(0, 1, image_size)
        x_coords = torch.linspace(0, 1, image_size)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        self.register_buffer('yy', yy.clone())
        self.register_buffer('xx', xx.clone())

    def _get_grid(self, H: int, W: int, device: torch.device) -> tuple:
        """Get coordinate grids, using cached buffers when sizes match."""
        if H == self._cached_size and W == self._cached_size:
            return self.yy, self.xx
        # Fallback for non-standard sizes (e.g., tests with small images)
        y_coords = torch.linspace(0, 1, H, device=device)
        x_coords = torch.linspace(0, 1, W, device=device)
        return torch.meshgrid(y_coords, x_coords, indexing='ij')

    def forward(
        self,
        pred_map: torch.Tensor,      # (B, 1, H, W) predicted radio map
        tx_position: torch.Tensor,   # (B, 2) normalized TX position
        building_map: torch.Tensor,  # (B, 1, H, W) to mask out walls
        reduction: str = 'mean',     # 'mean' or 'none'
    ) -> torch.Tensor:
        """
        Penalize signal INCREASING with distance from TX.

        Computes per-sample near/far means (not mixing pixels across samples)
        then averages across the batch.

        Args:
            pred_map: Predicted radio map
            tx_position: Normalized (0-1) transmitter position
            building_map: Building map in [-1,1] where -1 = wall, +1 = walkable
            reduction: 'mean' returns scalar; 'none' returns (B,) per-sample losses

        Returns:
            Scalar penalty (reduction='mean') or (B,) per-sample penalties (reduction='none')
        """
        B, C, H, W = pred_map.shape

        # Vectorized distance computation
        tx_x = tx_position[:, 0].view(B, 1, 1)
        tx_y = tx_position[:, 1].view(B, 1, 1)

        yy, xx = self._get_grid(H, W, pred_map.device)
        distance_map = torch.sqrt(
            (xx.unsqueeze(0) - tx_x) ** 2 +
            (yy.unsqueeze(0) - tx_y) ** 2
        ).unsqueeze(1)  # (B, 1, H, W)

        near_tx = distance_map < 0.3
        far_from_tx = distance_map > 0.7
        free_space = building_map > 0.0

        near_mask = near_tx & free_space  # (B, 1, H, W)
        far_mask = far_from_tx & free_space  # (B, 1, H, W)

        # Per-sample near/far means to avoid mixing pixels across batch
        per_sample_violations = []
        for i in range(B):
            nm = near_mask[i]  # (1, H, W)
            fm = far_mask[i]
            if nm.sum() == 0 or fm.sum() == 0:
                per_sample_violations.append(
                    torch.tensor(0.0, device=pred_map.device)
                )
            else:
                near_rss = pred_map[i][nm].mean()
                far_rss = pred_map[i][fm].mean()
                per_sample_violations.append(F.relu(far_rss - near_rss))

        per_sample = torch.stack(per_sample_violations)  # (B,)

        if reduction == 'none':
            return self.weight * per_sample
        return self.weight * per_sample.mean()


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
