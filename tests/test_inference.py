"""
Tests for inference module.

Run with: pytest tests/test_inference.py -v
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class TestDiffusionInference:
    """Test DiffusionInference wrapper."""

    @pytest.fixture
    def module(self):
        """Create a diffusion module for testing."""
        from training import DiffusionModule

        return DiffusionModule(
            unet_size='small',
            image_size=64,
            condition_channels=32,
            num_timesteps=50,
            use_ema=False,
        )

    @pytest.fixture
    def inference(self, module):
        """Create inference wrapper."""
        from training import DiffusionInference

        return DiffusionInference(module, device=torch.device('cpu'), use_ema=False)

    @pytest.fixture
    def condition(self):
        """Create sample condition."""
        return {
            'building_map': torch.randn(2, 1, 64, 64),
            'sparse_rss': torch.randn(2, 1, 64, 64),
            'trajectory_mask': torch.randn(2, 1, 64, 64),
            'coverage_density': torch.randn(2, 1, 64, 64),
            'tx_position': torch.tensor([[32.0, 32.0], [48.0, 16.0]]),
        }

    def test_inference_init(self, inference):
        """Inference wrapper should initialize correctly."""
        assert inference.module is not None
        assert inference.device == torch.device('cpu')

    def test_sample_basic(self, inference, condition):
        """Should generate samples from condition."""
        samples = inference.sample(
            condition,
            use_ddim=True,
            progress=False,
        )

        assert samples.shape == (2, 1, 64, 64)
        assert not torch.isnan(samples).any()

    def test_sample_batched(self, inference, condition):
        """Should handle batched sampling."""
        # Create larger batch
        large_condition = {
            k: v.repeat(4, 1, 1, 1) if v.dim() == 4 else v.repeat(4, 1)
            for k, v in condition.items()
        }

        samples = inference.sample(
            large_condition,
            use_ddim=True,
            batch_size=2,  # Process in batches of 2
            progress=False,
        )

        assert samples.shape == (8, 1, 64, 64)


class TestDenormalize:
    """Test denormalization functions."""

    def test_denormalize_radio_map(self):
        """Should denormalize from [-1, 1] to dBm."""
        from training import denormalize_radio_map

        # Test edge cases
        x = torch.tensor([-1.0, 0.0, 1.0])
        result = denormalize_radio_map(x, min_val=-120.0, max_val=0.0)

        assert torch.isclose(result[0], torch.tensor(-120.0))
        assert torch.isclose(result[1], torch.tensor(-60.0))
        assert torch.isclose(result[2], torch.tensor(0.0))

    def test_denormalize_preserves_shape(self):
        """Should preserve tensor shape."""
        from training import denormalize_radio_map

        x = torch.randn(2, 1, 64, 64)
        result = denormalize_radio_map(x)

        assert result.shape == x.shape


class TestUncertainty:
    """Test uncertainty estimation."""

    @pytest.fixture
    def inference(self):
        """Create inference wrapper for uncertainty tests."""
        from training import DiffusionModule, DiffusionInference

        module = DiffusionModule(
            unet_size='small',
            image_size=64,
            condition_channels=32,
            num_timesteps=50,
            use_ema=False,
        )
        return DiffusionInference(module, device=torch.device('cpu'))

    @pytest.fixture
    def condition(self):
        """Create sample condition."""
        return {
            'building_map': torch.randn(1, 1, 64, 64),
            'sparse_rss': torch.randn(1, 1, 64, 64),
            'trajectory_mask': torch.randn(1, 1, 64, 64),
            'coverage_density': torch.randn(1, 1, 64, 64),
            'tx_position': torch.tensor([[32.0, 32.0]]),
        }

    def test_compute_uncertainty(self, inference, condition):
        """Should compute mean and std from multiple samples."""
        from training import compute_uncertainty

        mean, std = compute_uncertainty(
            inference,
            condition,
            num_samples=3,  # Small number for speed
            use_ddim=True,
        )

        assert mean.shape == (1, 1, 64, 64)
        assert std.shape == (1, 1, 64, 64)
        assert (std >= 0).all()  # Std should be non-negative


class TestInterpolation:
    """Test condition interpolation."""

    @pytest.fixture
    def inference(self):
        """Create inference wrapper."""
        from training import DiffusionModule, DiffusionInference

        module = DiffusionModule(
            unet_size='small',
            image_size=64,
            condition_channels=32,
            num_timesteps=50,
            use_ema=False,
        )
        return DiffusionInference(module, device=torch.device('cpu'))

    def test_sample_interpolation(self, inference):
        """Should generate interpolated samples."""
        from training import sample_interpolation

        condition1 = {
            'building_map': torch.zeros(1, 1, 64, 64),
            'sparse_rss': torch.zeros(1, 1, 64, 64),
            'trajectory_mask': torch.zeros(1, 1, 64, 64),
            'coverage_density': torch.zeros(1, 1, 64, 64),
            'tx_position': torch.tensor([[32.0, 32.0]]),
        }

        condition2 = {
            'building_map': torch.ones(1, 1, 64, 64),
            'sparse_rss': torch.ones(1, 1, 64, 64),
            'trajectory_mask': torch.ones(1, 1, 64, 64),
            'coverage_density': torch.ones(1, 1, 64, 64),
            'tx_position': torch.tensor([[48.0, 48.0]]),
        }

        samples = sample_interpolation(
            inference,
            condition1,
            condition2,
            num_steps=3,  # Few steps for speed
            use_ddim=True,
        )

        assert samples.shape == (3, 1, 64, 64)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
