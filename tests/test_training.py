"""
Tests for training module components.

Run with: pytest tests/test_training.py -v
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class TestDiffusionModule:
    """Test DiffusionModule Lightning module."""

    @pytest.fixture
    def module(self):
        """Create a small diffusion module for testing."""
        from training import DiffusionModule

        return DiffusionModule(
            unet_size='small',
            image_size=64,
            condition_channels=32,
            num_timesteps=100,
            beta_schedule='linear',
            learning_rate=1e-4,
            warmup_steps=10,
            max_steps=100,
            use_ema=False,  # Disable EMA for faster tests
        )

    @pytest.fixture
    def batch(self):
        """Create a sample batch."""
        return {
            'radio_map': torch.randn(2, 1, 64, 64),
            'building_map': torch.randn(2, 1, 64, 64),
            'sparse_rss': torch.randn(2, 1, 64, 64),
            'trajectory_mask': torch.randn(2, 1, 64, 64),
            'coverage_density': torch.randn(2, 1, 64, 64),
            'tx_position': torch.tensor([[32.0, 32.0], [48.0, 16.0]]),
        }

    def test_module_init(self, module):
        """Module should initialize correctly."""
        assert module.model is not None
        assert module.diffusion is not None
        assert module.hparams.unet_size == 'small'

    def test_training_step(self, module, batch):
        """Training step should return loss."""
        loss = module.training_step(batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()
        assert not torch.isnan(loss)
        assert loss > 0

    def test_validation_step(self, module, batch):
        """Validation step should return metrics dict."""
        result = module.validation_step(batch, batch_idx=0)

        assert isinstance(result, dict)
        assert 'val_loss' in result
        assert not torch.isnan(result['val_loss'])

    def test_configure_optimizers(self, module):
        """Should configure optimizer and scheduler."""
        config = module.configure_optimizers()

        assert 'optimizer' in config
        assert 'lr_scheduler' in config
        assert config['lr_scheduler']['interval'] == 'step'

    def test_extract_condition(self, module, batch):
        """Should extract conditioning inputs from batch."""
        condition = module._extract_condition(batch)

        assert 'building_map' in condition
        assert 'sparse_rss' in condition
        assert 'tx_position' in condition

    def test_forward(self, module, batch):
        """Forward pass should produce correct shape."""
        x = batch['radio_map']
        t = torch.tensor([10, 50])
        condition = module._extract_condition(batch)

        output = module(x, t, condition)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()


class TestDiffusionModuleWithEMA:
    """Test DiffusionModule with EMA."""

    @pytest.fixture
    def module_with_ema(self):
        """Create module with EMA enabled."""
        from training import DiffusionModule

        return DiffusionModule(
            unet_size='small',
            image_size=64,
            condition_channels=32,
            num_timesteps=100,
            use_ema=True,
            ema_decay=0.999,
        )

    def test_ema_model_created(self, module_with_ema):
        """EMA model should be created."""
        assert module_with_ema.ema_model is not None

    def test_ema_params_frozen(self, module_with_ema):
        """EMA model parameters should be frozen."""
        for param in module_with_ema.ema_model.parameters():
            assert not param.requires_grad

    def test_ema_update(self, module_with_ema):
        """EMA update should modify EMA parameters."""
        # Get initial EMA parameters
        initial_params = [p.clone() for p in module_with_ema.ema_model.parameters()]

        # Modify model parameters
        for param in module_with_ema.model.parameters():
            param.data.add_(torch.randn_like(param) * 0.1)

        # Update EMA
        module_with_ema._update_ema()

        # Check EMA parameters changed
        changed = False
        for initial, current in zip(initial_params, module_with_ema.ema_model.parameters()):
            if not torch.allclose(initial, current):
                changed = True
                break

        assert changed, "EMA parameters should have changed"


class TestSampling:
    """Test sampling functionality."""

    @pytest.fixture
    def module(self):
        """Create module for sampling tests."""
        from training import DiffusionModule

        return DiffusionModule(
            unet_size='small',
            image_size=64,
            condition_channels=32,
            num_timesteps=50,  # Fewer steps for faster tests
            use_ema=False,
        )

    def test_sample_ddim(self, module):
        """DDIM sampling should produce correct shape."""
        condition = {
            'building_map': torch.randn(2, 1, 64, 64),
            'sparse_rss': torch.randn(2, 1, 64, 64),
            'trajectory_mask': torch.randn(2, 1, 64, 64),
            'coverage_density': torch.randn(2, 1, 64, 64),
            'tx_position': torch.tensor([[32.0, 32.0], [48.0, 16.0]]),
        }

        samples = module.sample(condition, use_ddim=True, progress=False)

        assert samples.shape == (2, 1, 64, 64)
        assert not torch.isnan(samples).any()


class TestFactoryFunction:
    """Test factory function."""

    def test_get_diffusion_module_default(self):
        """Default preset should work."""
        from training import get_diffusion_module

        module = get_diffusion_module('default', image_size=64, use_ema=False)

        assert module.hparams.unet_size == 'medium'
        assert module.hparams.num_timesteps == 1000

    def test_get_diffusion_module_fast(self):
        """Fast preset should work."""
        from training import get_diffusion_module

        module = get_diffusion_module('fast', image_size=64, use_ema=False)

        assert module.hparams.unet_size == 'small'
        assert module.hparams.num_timesteps == 500

    def test_get_diffusion_module_quality(self):
        """Quality preset should work."""
        from training import get_diffusion_module

        module = get_diffusion_module('quality', image_size=64, use_ema=False)

        assert module.hparams.unet_size == 'large'

    def test_get_diffusion_module_override(self):
        """Should allow overriding preset parameters."""
        from training import get_diffusion_module

        module = get_diffusion_module(
            'default',
            image_size=64,
            learning_rate=5e-5,
            use_ema=False,
        )

        assert module.hparams.learning_rate == 5e-5


class TestCheckpointing:
    """Test checkpoint save/load."""

    @pytest.fixture
    def module_with_ema(self):
        """Create module with EMA for checkpoint tests."""
        from training import DiffusionModule

        return DiffusionModule(
            unet_size='small',
            image_size=64,
            condition_channels=32,
            num_timesteps=50,
            use_ema=True,
        )

    def test_save_checkpoint_includes_ema(self, module_with_ema):
        """Checkpoint should include EMA state."""
        checkpoint = {}
        module_with_ema.on_save_checkpoint(checkpoint)

        assert 'ema_state_dict' in checkpoint
        assert len(checkpoint['ema_state_dict']) > 0

    def test_load_checkpoint_structure(self, module_with_ema):
        """Loading checkpoint should not raise errors."""
        # Save checkpoint
        checkpoint = {}
        module_with_ema.on_save_checkpoint(checkpoint)

        # Create a new module
        from training import DiffusionModule
        new_module = DiffusionModule(
            unet_size='small',
            image_size=64,
            condition_channels=32,
            num_timesteps=50,
            use_ema=True,
        )

        # Load checkpoint (should not raise)
        new_module.on_load_checkpoint(checkpoint)

        # Verify state dict keys match
        orig_keys = set(module_with_ema.ema_model.state_dict().keys())
        new_keys = set(new_module.ema_model.state_dict().keys())
        assert orig_keys == new_keys


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
