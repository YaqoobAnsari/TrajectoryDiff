"""
Tests for TrajectoryDiff physics-informed losses.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from training.losses import (
    TrajectoryConsistencyLoss,
    CoverageWeightedLoss,
    DistanceDecayLoss,
    WallAttenuationLoss,
    TrajectoryDiffLoss,
    compute_physics_losses,
)


class TestTrajectoryConsistencyLoss:
    """Tests for TrajectoryConsistencyLoss."""

    def test_basic_computation(self):
        """Test basic loss computation."""
        loss_fn = TrajectoryConsistencyLoss(smoothness_weight=0.0)

        B, H, W = 2, 64, 64
        pred_map = torch.randn(B, 1, H, W, requires_grad=True)
        sparse_rss = torch.randn(B, 1, H, W)
        trajectory_mask = torch.zeros(B, 1, H, W)
        trajectory_mask[:, :, 20:30, 20:30] = 1.0  # Some trajectory region

        loss = loss_fn(pred_map, sparse_rss, trajectory_mask)

        assert loss.shape == ()
        assert loss.item() >= 0
        assert loss.requires_grad

    def test_perfect_match(self):
        """Test loss is zero when prediction matches observation."""
        loss_fn = TrajectoryConsistencyLoss(smoothness_weight=0.0)

        B, H, W = 2, 64, 64
        pred_map = torch.ones(B, 1, H, W)
        sparse_rss = torch.ones(B, 1, H, W)
        trajectory_mask = torch.zeros(B, 1, H, W)
        trajectory_mask[:, :, 20:30, 20:30] = 1.0

        loss = loss_fn(pred_map, sparse_rss, trajectory_mask)

        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_empty_mask(self):
        """Test handling of empty trajectory mask."""
        loss_fn = TrajectoryConsistencyLoss()

        B, H, W = 2, 64, 64
        pred_map = torch.randn(B, 1, H, W)
        sparse_rss = torch.randn(B, 1, H, W)
        trajectory_mask = torch.zeros(B, 1, H, W)  # Empty mask

        loss = loss_fn(pred_map, sparse_rss, trajectory_mask)

        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_with_smoothness(self):
        """Test loss with smoothness regularization enabled."""
        loss_fn = TrajectoryConsistencyLoss(smoothness_weight=0.1)

        B, H, W = 2, 64, 64
        pred_map = torch.randn(B, 1, H, W)
        sparse_rss = pred_map.clone()  # Perfect match for consistency
        trajectory_mask = torch.zeros(B, 1, H, W)
        trajectory_mask[:, :, 20:30, 20:30] = 1.0

        loss = loss_fn(pred_map, sparse_rss, trajectory_mask)

        # Loss should be positive due to smoothness term (unless pred is constant)
        assert loss.item() >= 0

    def test_gradient_flow(self):
        """Test that gradients flow through the loss."""
        loss_fn = TrajectoryConsistencyLoss()

        pred_map = torch.randn(2, 1, 32, 32, requires_grad=True)
        sparse_rss = torch.randn(2, 1, 32, 32)
        trajectory_mask = torch.zeros(2, 1, 32, 32)
        trajectory_mask[:, :, 10:20, 10:20] = 1.0

        loss = loss_fn(pred_map, sparse_rss, trajectory_mask)
        loss.backward()

        assert pred_map.grad is not None
        assert not torch.all(pred_map.grad == 0)


class TestCoverageWeightedLoss:
    """Tests for CoverageWeightedLoss."""

    def test_basic_computation(self):
        """Test basic loss computation."""
        loss_fn = CoverageWeightedLoss(min_weight=0.1, max_weight=1.0)

        B, H, W = 2, 64, 64
        pred = torch.randn(B, 1, H, W, requires_grad=True)
        target = torch.randn(B, 1, H, W)
        coverage = torch.rand(B, 1, H, W)

        loss = loss_fn(pred, target, coverage)

        assert loss.shape == ()
        assert loss.item() >= 0
        assert loss.requires_grad

    def test_perfect_prediction(self):
        """Test loss is zero when prediction matches target."""
        loss_fn = CoverageWeightedLoss()

        pred = torch.ones(2, 1, 32, 32)
        target = torch.ones(2, 1, 32, 32)
        coverage = torch.rand(2, 1, 32, 32)

        loss = loss_fn(pred, target, coverage)

        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_coverage_weighting(self):
        """Test that high coverage regions contribute more to loss."""
        loss_fn = CoverageWeightedLoss(min_weight=0.0, max_weight=1.0)

        # Create scenario with error only in specific region
        pred = torch.zeros(1, 1, 32, 32)
        target = torch.zeros(1, 1, 32, 32)
        target[:, :, 10:20, 10:20] = 1.0  # Error region

        # High coverage on error region
        coverage_high = torch.zeros(1, 1, 32, 32)
        coverage_high[:, :, 10:20, 10:20] = 1.0

        # Low coverage on error region
        coverage_low = torch.zeros(1, 1, 32, 32)
        coverage_low[:, :, 10:20, 10:20] = 0.0

        loss_high = loss_fn(pred, target, coverage_high)
        loss_low = loss_fn(pred, target, coverage_low)

        # High coverage should give higher loss
        assert loss_high.item() > loss_low.item()

    def test_gradient_flow(self):
        """Test gradient flow through coverage-weighted loss."""
        loss_fn = CoverageWeightedLoss()

        pred = torch.randn(2, 1, 32, 32, requires_grad=True)
        target = torch.randn(2, 1, 32, 32)
        coverage = torch.rand(2, 1, 32, 32)

        loss = loss_fn(pred, target, coverage)
        loss.backward()

        assert pred.grad is not None


class TestDistanceDecayLoss:
    """Tests for DistanceDecayLoss."""

    def test_basic_computation(self):
        """Test basic loss computation."""
        loss_fn = DistanceDecayLoss(weight=0.01)

        B, H, W = 2, 64, 64
        pred_map = torch.randn(B, 1, H, W)
        tx_position = torch.rand(B, 2)  # Random TX positions
        building_map = torch.ones(B, 1, H, W)  # All walkable (+1 in [-1,1])

        loss = loss_fn(pred_map, tx_position, building_map)

        assert loss.shape == ()
        assert loss.item() >= 0

    def test_physically_correct_signal(self):
        """Test loss is low for physically plausible signal decay."""
        loss_fn = DistanceDecayLoss(weight=1.0)

        B, H, W = 1, 64, 64

        # TX at center
        tx_position = torch.tensor([[0.5, 0.5]])

        # Create distance-decaying signal (strong near TX, weak far)
        y_coords = torch.linspace(0, 1, H)
        x_coords = torch.linspace(0, 1, W)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        distance = torch.sqrt((xx - 0.5) ** 2 + (yy - 0.5) ** 2)
        pred_map = (1.0 - distance).unsqueeze(0).unsqueeze(0)  # Decay with distance

        building_map = torch.ones(B, 1, H, W)  # All walkable (+1 in [-1,1])

        loss = loss_fn(pred_map, tx_position, building_map)

        # Should be low (no violation) - signal decreases with distance
        assert loss.item() == pytest.approx(0.0, abs=0.1)

    def test_physically_wrong_signal(self):
        """Test loss penalizes signal increasing with distance."""
        loss_fn = DistanceDecayLoss(weight=1.0)

        B, H, W = 1, 64, 64

        # TX at center
        tx_position = torch.tensor([[0.5, 0.5]])

        # Create WRONG signal (weak near TX, strong far) - physically impossible
        y_coords = torch.linspace(0, 1, H)
        x_coords = torch.linspace(0, 1, W)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        distance = torch.sqrt((xx - 0.5) ** 2 + (yy - 0.5) ** 2)
        pred_map = distance.unsqueeze(0).unsqueeze(0)  # INCREASE with distance

        building_map = torch.ones(B, 1, H, W)  # All walkable (+1 in [-1,1])

        loss = loss_fn(pred_map, tx_position, building_map)

        # Should be positive (violation detected)
        assert loss.item() > 0

    def test_wall_masking(self):
        """Test that walls are properly masked out."""
        loss_fn = DistanceDecayLoss(weight=1.0)

        B, H, W = 1, 64, 64
        pred_map = torch.randn(B, 1, H, W)
        tx_position = torch.tensor([[0.5, 0.5]])

        # Full wall coverage - should not be able to compute loss
        building_map = -torch.ones(B, 1, H, W)  # All walls (-1 in [-1,1])

        loss = loss_fn(pred_map, tx_position, building_map)

        # With all walls, should return zero (no free space to evaluate)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)


class TestWallAttenuationLoss:
    """Tests for WallAttenuationLoss (placeholder implementation)."""

    def test_returns_zero(self):
        """Test that placeholder returns zero loss."""
        loss_fn = WallAttenuationLoss()

        pred_map = torch.randn(2, 1, 32, 32)
        building_map = torch.zeros(2, 1, 32, 32)
        tx_position = torch.rand(2, 2)

        loss = loss_fn(pred_map, building_map, tx_position)

        # Placeholder should return 0
        assert loss.item() == pytest.approx(0.0, abs=1e-6)


class TestTrajectoryDiffLoss:
    """Tests for combined TrajectoryDiffLoss."""

    def test_basic_computation(self):
        """Test basic combined loss computation."""
        loss_fn = TrajectoryDiffLoss(
            diffusion_weight=1.0,
            trajectory_consistency_weight=0.1,
            coverage_weighted=False,
            distance_decay_weight=0.0,
        )

        B, H, W = 2, 32, 32
        noise_pred = torch.randn(B, 1, H, W, requires_grad=True)
        noise_target = torch.randn(B, 1, H, W)
        pred_x0 = torch.randn(B, 1, H, W, requires_grad=True)

        batch = {
            'sparse_rss': torch.randn(B, 1, H, W),
            'trajectory_mask': torch.zeros(B, 1, H, W),
        }
        batch['trajectory_mask'][:, :, 10:20, 10:20] = 1.0

        losses = loss_fn(noise_pred, noise_target, pred_x0, batch)

        assert 'diffusion' in losses
        assert 'trajectory_consistency' in losses
        assert 'total' in losses
        assert losses['total'].requires_grad

    def test_with_coverage_weighting(self):
        """Test combined loss with coverage weighting enabled."""
        loss_fn = TrajectoryDiffLoss(
            diffusion_weight=1.0,
            trajectory_consistency_weight=0.1,
            coverage_weighted=True,
        )

        B, H, W = 2, 32, 32
        noise_pred = torch.randn(B, 1, H, W)
        noise_target = torch.randn(B, 1, H, W)

        batch = {
            'coverage_density': torch.rand(B, 1, H, W),
            'sparse_rss': torch.randn(B, 1, H, W),
            'trajectory_mask': torch.zeros(B, 1, H, W),
        }

        losses = loss_fn(noise_pred, noise_target, None, batch)

        assert 'diffusion' in losses
        assert 'total' in losses

    def test_with_all_components(self):
        """Test combined loss with all physics components."""
        loss_fn = TrajectoryDiffLoss(
            diffusion_weight=1.0,
            trajectory_consistency_weight=0.1,
            coverage_weighted=True,
            distance_decay_weight=0.01,
        )

        B, H, W = 2, 32, 32
        noise_pred = torch.randn(B, 1, H, W)
        noise_target = torch.randn(B, 1, H, W)
        pred_x0 = torch.randn(B, 1, H, W)

        batch = {
            'coverage_density': torch.rand(B, 1, H, W),
            'sparse_rss': torch.randn(B, 1, H, W),
            'trajectory_mask': torch.zeros(B, 1, H, W),
            'tx_position': torch.rand(B, 2),
            'building_map': torch.ones(B, 1, H, W),  # All walkable (+1 in [-1,1])
        }
        batch['trajectory_mask'][:, :, 10:20, 10:20] = 1.0

        losses = loss_fn(noise_pred, noise_target, pred_x0, batch)

        assert 'diffusion' in losses
        assert 'trajectory_consistency' in losses
        assert 'distance_decay' in losses
        assert 'total' in losses

    def test_gradient_flow(self):
        """Test gradient flows through combined loss."""
        loss_fn = TrajectoryDiffLoss(
            diffusion_weight=1.0,
            trajectory_consistency_weight=0.1,
        )

        noise_pred = torch.randn(2, 1, 32, 32, requires_grad=True)
        noise_target = torch.randn(2, 1, 32, 32)
        pred_x0 = torch.randn(2, 1, 32, 32, requires_grad=True)

        batch = {
            'sparse_rss': torch.randn(2, 1, 32, 32),
            'trajectory_mask': torch.zeros(2, 1, 32, 32),
        }
        batch['trajectory_mask'][:, :, 10:20, 10:20] = 1.0

        losses = loss_fn(noise_pred, noise_target, pred_x0, batch)
        losses['total'].backward()

        assert noise_pred.grad is not None


class TestComputePhysicsLosses:
    """Tests for compute_physics_losses convenience function."""

    def test_basic_usage(self):
        """Test basic usage of convenience function."""
        B, H, W = 2, 32, 32
        pred_map = torch.randn(B, 1, H, W)
        sparse_rss = torch.randn(B, 1, H, W)
        trajectory_mask = torch.zeros(B, 1, H, W)
        trajectory_mask[:, :, 10:20, 10:20] = 1.0

        losses = compute_physics_losses(
            pred_map=pred_map,
            sparse_rss=sparse_rss,
            trajectory_mask=trajectory_mask,
            trajectory_weight=0.1,
        )

        assert 'trajectory_consistency' in losses
        assert 'total_physics' in losses

    def test_with_optional_components(self):
        """Test with all optional components."""
        B, H, W = 2, 32, 32
        pred_map = torch.randn(B, 1, H, W)
        sparse_rss = torch.randn(B, 1, H, W)
        trajectory_mask = torch.zeros(B, 1, H, W)
        trajectory_mask[:, :, 10:20, 10:20] = 1.0

        losses = compute_physics_losses(
            pred_map=pred_map,
            sparse_rss=sparse_rss,
            trajectory_mask=trajectory_mask,
            tx_position=torch.rand(B, 2),
            building_map=torch.ones(B, 1, H, W),  # All walkable (+1 in [-1,1])
            trajectory_weight=0.1,
            distance_weight=0.01,
        )

        assert 'trajectory_consistency' in losses
        assert 'distance_decay' in losses
        assert 'total_physics' in losses


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
