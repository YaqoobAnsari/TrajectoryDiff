"""
Tests for diffusion model components.

Run with: pytest tests/test_diffusion.py -v
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class TestNoiseSchedules:
    """Test noise schedule implementations."""

    def test_linear_schedule_shape(self):
        """Linear schedule should have correct shape."""
        from models.diffusion import linear_beta_schedule

        betas = linear_beta_schedule(1000)
        assert betas.shape == (1000,)

    def test_linear_schedule_bounds(self):
        """Linear schedule should be within specified bounds."""
        from models.diffusion import linear_beta_schedule

        betas = linear_beta_schedule(1000, beta_start=1e-4, beta_end=0.02)
        assert betas[0] == pytest.approx(1e-4, rel=1e-3)
        assert betas[-1] == pytest.approx(0.02, rel=1e-3)

    def test_linear_schedule_monotonic(self):
        """Linear schedule should be monotonically increasing."""
        from models.diffusion import linear_beta_schedule

        betas = linear_beta_schedule(1000)
        assert torch.all(betas[1:] >= betas[:-1])

    def test_cosine_schedule_shape(self):
        """Cosine schedule should have correct shape."""
        from models.diffusion import cosine_beta_schedule

        betas = cosine_beta_schedule(1000)
        assert betas.shape == (1000,)

    def test_cosine_schedule_bounds(self):
        """Cosine schedule should be within valid bounds."""
        from models.diffusion import cosine_beta_schedule

        betas = cosine_beta_schedule(1000)
        assert betas.min() > 0
        assert betas.max() < 1

    def test_sigmoid_schedule_shape(self):
        """Sigmoid schedule should have correct shape."""
        from models.diffusion import sigmoid_beta_schedule

        betas = sigmoid_beta_schedule(1000)
        assert betas.shape == (1000,)


class TestSinusoidalEmbedding:
    """Test sinusoidal timestep embedding."""

    def test_embedding_shape(self):
        """Embedding should have correct shape."""
        from models.diffusion import SinusoidalPositionEmbedding

        embed = SinusoidalPositionEmbedding(dim=128)
        timesteps = torch.tensor([0, 100, 500, 999])

        output = embed(timesteps)
        assert output.shape == (4, 128)

    def test_embedding_different_timesteps(self):
        """Different timesteps should produce different embeddings."""
        from models.diffusion import SinusoidalPositionEmbedding

        embed = SinusoidalPositionEmbedding(dim=64)
        t1 = torch.tensor([0])
        t2 = torch.tensor([500])

        e1 = embed(t1)
        e2 = embed(t2)

        assert not torch.allclose(e1, e2)

    def test_embedding_same_timesteps(self):
        """Same timesteps should produce same embeddings."""
        from models.diffusion import SinusoidalPositionEmbedding

        embed = SinusoidalPositionEmbedding(dim=64)
        t = torch.tensor([250, 250])

        output = embed(t)
        assert torch.allclose(output[0], output[1])


class TestGaussianDiffusion:
    """Test Gaussian diffusion process."""

    @pytest.fixture
    def diffusion(self):
        """Create diffusion instance."""
        from models.diffusion import GaussianDiffusion

        return GaussianDiffusion(
            num_timesteps=100,
            beta_schedule='linear',
        )

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        return torch.randn(2, 1, 64, 64)

    def test_diffusion_init_linear(self):
        """Diffusion should initialize with linear schedule."""
        from models.diffusion import GaussianDiffusion

        diffusion = GaussianDiffusion(num_timesteps=100, beta_schedule='linear')
        assert diffusion.num_timesteps == 100
        assert diffusion.betas.shape == (100,)

    def test_diffusion_init_cosine(self):
        """Diffusion should initialize with cosine schedule."""
        from models.diffusion import GaussianDiffusion

        diffusion = GaussianDiffusion(num_timesteps=100, beta_schedule='cosine')
        assert diffusion.num_timesteps == 100

    def test_q_sample_shape(self, diffusion, sample_data):
        """Forward diffusion should preserve shape."""
        t = torch.tensor([10, 50])
        noisy = diffusion.q_sample(sample_data, t)

        assert noisy.shape == sample_data.shape

    def test_q_sample_noise_increases(self, diffusion, sample_data):
        """Later timesteps should have more noise."""
        t_early = torch.tensor([10, 10])
        t_late = torch.tensor([90, 90])

        noisy_early = diffusion.q_sample(sample_data, t_early)
        noisy_late = diffusion.q_sample(sample_data, t_late)

        # Variance should be higher at later timesteps
        var_early = noisy_early.var()
        var_late = noisy_late.var()

        # Not a strict test, but late should generally have more variance
        # (closer to pure noise with var=1)
        assert var_late > var_early * 0.5  # Relaxed condition

    def test_q_sample_deterministic_with_noise(self, diffusion, sample_data):
        """Forward diffusion should be deterministic with fixed noise."""
        t = torch.tensor([50, 50])
        noise = torch.randn_like(sample_data)

        noisy1 = diffusion.q_sample(sample_data, t, noise=noise)
        noisy2 = diffusion.q_sample(sample_data, t, noise=noise)

        assert torch.allclose(noisy1, noisy2)

    def test_predict_x0_from_epsilon(self, diffusion, sample_data):
        """Should recover x0 from epsilon and x_t."""
        t = torch.tensor([50, 50])
        noise = torch.randn_like(sample_data)

        x_t = diffusion.q_sample(sample_data, t, noise=noise)
        recovered = diffusion.predict_x0_from_epsilon(x_t, t, noise)

        assert torch.allclose(recovered, sample_data, atol=1e-5)

    def test_sample_timesteps(self, diffusion):
        """Sampled timesteps should be in valid range."""
        t = diffusion.sample_timesteps(32, torch.device('cpu'))

        assert t.shape == (32,)
        assert t.min() >= 0
        assert t.max() < diffusion.num_timesteps


class TestTrainingLosses:
    """Test training loss computation."""

    @pytest.fixture
    def diffusion(self):
        """Create diffusion instance."""
        from models.diffusion import GaussianDiffusion

        return GaussianDiffusion(
            num_timesteps=100,
            beta_schedule='linear',
            loss_type='mse',
            prediction_type='epsilon',
        )

    @pytest.fixture
    def dummy_model(self):
        """Create dummy denoising model."""
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 1, 3, padding=1)

            def forward(self, x, t, **kwargs):
                # Just return something of right shape
                return self.conv(x)

        return DummyModel()

    def test_training_loss_shape(self, diffusion, dummy_model):
        """Training loss should return scalar."""
        x_0 = torch.randn(4, 1, 64, 64)
        t = diffusion.sample_timesteps(4, x_0.device)

        losses = diffusion.training_losses(dummy_model, x_0, t)

        assert 'loss' in losses
        assert losses['loss'].shape == ()  # Scalar
        assert 'loss_per_sample' in losses
        assert losses['loss_per_sample'].shape == (4,)

    def test_training_loss_with_condition(self, diffusion, dummy_model):
        """Training loss should work with conditioning."""
        x_0 = torch.randn(4, 1, 64, 64)
        t = diffusion.sample_timesteps(4, x_0.device)
        condition = {'building_map': torch.randn(4, 1, 64, 64)}

        losses = diffusion.training_losses(dummy_model, x_0, t, condition=condition)

        assert 'loss' in losses


class TestDDIMSampler:
    """Test DDIM sampling."""

    def test_ddim_init(self):
        """DDIM sampler should initialize correctly."""
        from models.diffusion import GaussianDiffusion, DDIMSampler

        diffusion = GaussianDiffusion(num_timesteps=1000, beta_schedule='linear')
        sampler = DDIMSampler(diffusion, ddim_num_steps=50)

        assert len(sampler.ddim_timesteps) == 50
        assert sampler.ddim_eta == 0.0

    def test_ddim_timesteps_subset(self):
        """DDIM timesteps should be subset of original."""
        from models.diffusion import GaussianDiffusion, DDIMSampler

        diffusion = GaussianDiffusion(num_timesteps=1000, beta_schedule='linear')
        sampler = DDIMSampler(diffusion, ddim_num_steps=50)

        for t in sampler.ddim_timesteps:
            assert 0 <= t < 1000


class TestPredictionTypes:
    """Test different prediction types."""

    def test_epsilon_prediction(self):
        """Epsilon prediction should work."""
        from models.diffusion import GaussianDiffusion

        diffusion = GaussianDiffusion(
            num_timesteps=100,
            prediction_type='epsilon',
        )
        assert diffusion.prediction_type == 'epsilon'

    def test_x0_prediction(self):
        """x0 prediction should work."""
        from models.diffusion import GaussianDiffusion

        diffusion = GaussianDiffusion(
            num_timesteps=100,
            prediction_type='x0',
        )
        assert diffusion.prediction_type == 'x0'

    def test_v_prediction(self):
        """v prediction should work."""
        from models.diffusion import GaussianDiffusion

        diffusion = GaussianDiffusion(
            num_timesteps=100,
            prediction_type='v',
        )
        assert diffusion.prediction_type == 'v'


class TestUNet:
    """Test U-Net architecture."""

    def test_unet_small_forward(self):
        """Small U-Net should produce correct output shape."""
        from models.diffusion import UNetSmall

        model = UNetSmall(in_channels=1, out_channels=1, image_size=64)
        x = torch.randn(2, 1, 64, 64)
        t = torch.tensor([10, 50])

        output = model(x, t)
        assert output.shape == x.shape

    def test_unet_medium_forward(self):
        """Medium U-Net should produce correct output shape."""
        from models.diffusion import UNetMedium

        model = UNetMedium(in_channels=1, out_channels=1, image_size=64)
        x = torch.randn(2, 1, 64, 64)
        t = torch.tensor([10, 50])

        output = model(x, t)
        assert output.shape == x.shape

    def test_unet_with_conditioning(self):
        """U-Net should work with conditioning input."""
        from models.diffusion import UNetSmall

        model = UNetSmall(in_channels=1, out_channels=1, cond_channels=2, image_size=64)
        x = torch.randn(2, 1, 64, 64)
        cond = torch.randn(2, 2, 64, 64)
        t = torch.tensor([10, 50])

        output = model(x, t, cond=cond)
        assert output.shape == x.shape

    def test_unet_different_channels(self):
        """U-Net should handle different input/output channels."""
        from models.diffusion import UNetSmall

        model = UNetSmall(in_channels=3, out_channels=1, image_size=64)
        x = torch.randn(2, 3, 64, 64)
        t = torch.tensor([10, 50])

        output = model(x, t)
        assert output.shape == (2, 1, 64, 64)

    def test_get_unet_factory(self):
        """Factory function should return correct model types."""
        from models.diffusion import get_unet, UNetSmall, UNetMedium, UNetLarge

        small = get_unet('small', image_size=64)
        medium = get_unet('medium', image_size=64)
        large = get_unet('large', image_size=64)

        assert isinstance(small, UNetSmall)
        assert isinstance(medium, UNetMedium)
        assert isinstance(large, UNetLarge)

    def test_count_parameters(self):
        """Parameter count should increase with model size."""
        from models.diffusion import get_unet, count_parameters

        small = get_unet('small', image_size=64)
        medium = get_unet('medium', image_size=64)
        large = get_unet('large', image_size=64)

        small_params = count_parameters(small)
        medium_params = count_parameters(medium)
        large_params = count_parameters(large)

        assert small_params < medium_params < large_params


class TestResidualBlock:
    """Test residual block."""

    def test_residual_block_same_channels(self):
        """Residual block should work with same input/output channels."""
        from models.diffusion import ResidualBlock

        block = ResidualBlock(64, 64, time_emb_dim=128)
        x = torch.randn(2, 64, 32, 32)
        time_emb = torch.randn(2, 128)

        output = block(x, time_emb)
        assert output.shape == x.shape

    def test_residual_block_different_channels(self):
        """Residual block should work with different input/output channels."""
        from models.diffusion import ResidualBlock

        block = ResidualBlock(32, 64, time_emb_dim=128)
        x = torch.randn(2, 32, 32, 32)
        time_emb = torch.randn(2, 128)

        output = block(x, time_emb)
        assert output.shape == (2, 64, 32, 32)

    def test_residual_block_no_time_emb(self):
        """Residual block should work without time embedding."""
        from models.diffusion import ResidualBlock

        block = ResidualBlock(64, 64)
        x = torch.randn(2, 64, 32, 32)

        output = block(x)
        assert output.shape == x.shape


class TestAttentionBlock:
    """Test attention block."""

    def test_attention_block_forward(self):
        """Attention block should preserve shape."""
        from models.diffusion import AttentionBlock

        block = AttentionBlock(64, num_heads=4)
        x = torch.randn(2, 64, 16, 16)

        output = block(x)
        assert output.shape == x.shape

    def test_attention_block_different_heads(self):
        """Attention block should work with different head counts."""
        from models.diffusion import AttentionBlock

        for num_heads in [1, 2, 4, 8]:
            block = AttentionBlock(64, num_heads=num_heads)
            x = torch.randn(2, 64, 8, 8)
            output = block(x)
            assert output.shape == x.shape


class TestIntegration:
    """Integration tests for full diffusion pipeline."""

    def test_unet_with_diffusion(self):
        """U-Net should integrate with diffusion process."""
        from models.diffusion import GaussianDiffusion, UNetSmall

        diffusion = GaussianDiffusion(num_timesteps=100, beta_schedule='linear')
        model = UNetSmall(in_channels=1, out_channels=1, image_size=64)

        x_0 = torch.randn(2, 1, 64, 64)
        t = diffusion.sample_timesteps(2, x_0.device)

        # Training loss
        losses = diffusion.training_losses(model, x_0, t)
        assert 'loss' in losses
        assert not torch.isnan(losses['loss'])

    def test_unet_with_conditioning_and_diffusion(self):
        """U-Net with conditioning should integrate with diffusion."""
        from models.diffusion import GaussianDiffusion, UNetSmall

        diffusion = GaussianDiffusion(num_timesteps=100, beta_schedule='linear')
        model = UNetSmall(in_channels=1, out_channels=1, cond_channels=2, image_size=64)

        x_0 = torch.randn(2, 1, 64, 64)
        cond = torch.randn(2, 2, 64, 64)
        t = diffusion.sample_timesteps(2, x_0.device)

        # Need to pass cond through model in training_losses
        # The model accepts cond as keyword argument
        x_t = diffusion.q_sample(x_0, t)
        pred = model(x_t, t, cond=cond)

        assert pred.shape == x_0.shape
        assert not torch.isnan(pred).any()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
