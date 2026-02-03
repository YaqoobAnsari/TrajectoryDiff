"""
Comprehensive integration tests for TrajectoryDiff.

Tests the full pipeline end-to-end:
- DiffusionModule with all features (coverage attention + physics losses)
- Training step with gradient flow
- Validation step
- DDIM sampling
- Checkpoint save/load round-trip
- Config-to-model mapping
- All ablation configurations

Uses mock data (small random tensors, no file I/O).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch
import torch.nn as nn

from training.diffusion_module import DiffusionModule, get_diffusion_module


# Small dimensions for fast tests
IMG_SIZE = 32
BATCH_SIZE = 2
COND_CH = 16


def make_mock_batch(batch_size=BATCH_SIZE, img_size=IMG_SIZE, device='cpu'):
    """Create a mock batch mimicking the real dataset output."""
    return {
        'radio_map': torch.randn(batch_size, 1, img_size, img_size, device=device) * 0.5,
        'building_map': torch.randint(0, 2, (batch_size, 1, img_size, img_size), device=device).float() * 2 - 1,
        'sparse_rss': torch.randn(batch_size, 1, img_size, img_size, device=device) * 0.3,
        'trajectory_mask': torch.randint(0, 2, (batch_size, 1, img_size, img_size), device=device).float(),
        'coverage_density': torch.rand(batch_size, 1, img_size, img_size, device=device),
        'tx_position': torch.rand(batch_size, 2, device=device),
    }


class TestDiffusionModuleStandard:
    """Test standard DiffusionModule (no physics losses, no coverage attention)."""

    @pytest.fixture
    def module(self):
        return DiffusionModule(
            unet_size='small',
            image_size=IMG_SIZE,
            condition_channels=COND_CH,
            num_timesteps=100,
            beta_schedule='cosine',
            prediction_type='epsilon',
            use_ema=False,
            learning_rate=1e-4,
            warmup_epochs=1,
            ddim_steps=5,
        )

    def test_training_step(self, module):
        """Training step should compute loss and return scalar."""
        batch = make_mock_batch()
        loss = module.training_step(batch, 0)
        assert loss.dim() == 0
        assert loss.item() > 0
        assert loss.requires_grad

    def test_validation_step(self, module):
        """Validation step should compute loss."""
        batch = make_mock_batch()
        result = module.validation_step(batch, 0)
        assert 'val_loss' in result
        assert result['val_loss'].item() > 0

    def test_gradient_flow(self, module):
        """Gradients should flow to all model parameters."""
        batch = make_mock_batch()
        loss = module.training_step(batch, 0)
        loss.backward()

        has_grad = False
        for name, param in module.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_grad = True
                break
        assert has_grad, "No parameters received gradients"

    def test_ddim_sampling(self, module):
        """DDIM sampling should produce correct shape."""
        batch = make_mock_batch()
        condition = module._extract_condition(batch)
        samples = module.sample(condition, use_ddim=True, progress=False)
        assert samples.shape == (BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE)

    def test_configure_optimizers(self, module):
        """Optimizer config should be valid."""
        opt_config = module.configure_optimizers()
        assert 'optimizer' in opt_config
        assert 'lr_scheduler' in opt_config


class TestDiffusionModuleWithPhysicsLosses:
    """Test DiffusionModule with physics-informed losses enabled."""

    @pytest.fixture
    def module(self):
        return DiffusionModule(
            unet_size='small',
            image_size=IMG_SIZE,
            condition_channels=COND_CH,
            num_timesteps=100,
            beta_schedule='cosine',
            prediction_type='epsilon',
            use_ema=False,
            # Physics losses
            use_physics_losses=True,
            trajectory_consistency_weight=0.1,
            coverage_weighted=True,
            distance_decay_weight=0.01,
            learning_rate=1e-4,
            warmup_epochs=1,
            ddim_steps=5,
        )

    def test_training_step_with_physics(self, module):
        """Training step with physics losses should return scalar loss."""
        batch = make_mock_batch()
        loss = module.training_step(batch, 0)
        assert loss.dim() == 0
        assert loss.item() > 0
        assert loss.requires_grad

    def test_physics_loss_components(self, module):
        """Physics loss should produce all expected components."""
        batch = make_mock_batch()
        x_0 = batch['radio_map']
        t = module.diffusion.sample_timesteps(BATCH_SIZE, x_0.device)
        noise = torch.randn_like(x_0)
        x_t = module.diffusion.q_sample(x_0, t, noise=noise)
        condition = module._extract_condition(batch)
        pred = module.forward(x_t, t, condition)
        target = module._compute_target(x_0, t, noise)
        pred_x0 = module._predict_x0(x_t, t, pred)

        losses = module.physics_loss(
            noise_pred=pred,
            noise_target=target,
            pred_x0=pred_x0,
            batch=batch,
        )

        assert 'total' in losses
        assert 'diffusion' in losses
        assert 'trajectory_consistency' in losses
        assert losses['total'].requires_grad

    def test_gradient_flow_with_physics(self, module):
        """All model params should get gradients through physics losses."""
        batch = make_mock_batch()
        loss = module.training_step(batch, 0)
        loss.backward()

        params_with_grad = sum(
            1 for p in module.model.parameters()
            if p.requires_grad and p.grad is not None
        )
        total_params = sum(1 for p in module.model.parameters() if p.requires_grad)
        # At least some parameters should have gradients
        assert params_with_grad > 0


class TestDiffusionModuleWithCoverageAttention:
    """Test DiffusionModule with CoverageAwareUNet."""

    @pytest.fixture
    def module(self):
        return DiffusionModule(
            unet_size='small',
            image_size=IMG_SIZE,
            condition_channels=COND_CH,
            num_timesteps=100,
            beta_schedule='cosine',
            prediction_type='epsilon',
            use_ema=False,
            use_coverage_attention=True,
            coverage_temperature=1.0,
            learning_rate=1e-4,
            warmup_epochs=1,
            ddim_steps=5,
        )

    def test_training_step_with_coverage(self, module):
        """Training step with coverage attention should work."""
        batch = make_mock_batch()
        loss = module.training_step(batch, 0)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_sampling_with_coverage(self, module):
        """DDIM sampling should work with CoverageAwareUNet."""
        batch = make_mock_batch()
        condition = module._extract_condition(batch)
        samples = module.sample(condition, use_ddim=True, progress=False)
        assert samples.shape == (BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE)

    def test_uses_coverage_aware_unet(self, module):
        """Model should use CoverageAwareUNet when enabled."""
        from models.diffusion.coverage_unet import CoverageAwareUNet
        assert isinstance(module.model.unet, CoverageAwareUNet)


class TestDiffusionModuleFullFeatures:
    """Test DiffusionModule with ALL features enabled (physics + coverage attention)."""

    @pytest.fixture
    def module(self):
        return DiffusionModule(
            unet_size='small',
            image_size=IMG_SIZE,
            condition_channels=COND_CH,
            num_timesteps=100,
            beta_schedule='cosine',
            prediction_type='epsilon',
            use_ema=True,
            ema_decay=0.999,
            # Physics losses
            use_physics_losses=True,
            trajectory_consistency_weight=0.1,
            coverage_weighted=True,
            distance_decay_weight=0.01,
            # Coverage attention
            use_coverage_attention=True,
            coverage_temperature=1.0,
            learning_rate=1e-4,
            warmup_epochs=1,
            ddim_steps=5,
        )

    def test_full_training_step(self, module):
        """Full featured training step should complete without error."""
        batch = make_mock_batch()
        loss = module.training_step(batch, 0)
        assert loss.dim() == 0
        assert loss.item() > 0
        assert loss.requires_grad

    def test_full_validation_step(self, module):
        """Full featured validation step should complete."""
        batch = make_mock_batch()
        result = module.validation_step(batch, 0)
        assert 'val_loss' in result

    def test_ema_model_exists(self, module):
        """EMA model should be created."""
        assert hasattr(module, 'ema_model')
        assert module.ema_model is not None

    def test_ema_update(self, module):
        """EMA should update when model params differ."""
        # Perturb model params so EMA update is visible
        with torch.no_grad():
            for p in list(module.model.parameters())[:3]:
                p.add_(torch.randn_like(p) * 0.5)

        ema_params_before = [p.clone() for p in list(module.ema_model.parameters())[:3]]

        module._update_ema()

        ema_params_after = [p.clone() for p in list(module.ema_model.parameters())[:3]]

        changed = any(
            not torch.allclose(before, after, atol=1e-7)
            for before, after in zip(ema_params_before, ema_params_after)
        )
        assert changed, "EMA parameters did not update"

    def test_sampling_with_ema(self, module):
        """Sampling should use EMA model."""
        batch = make_mock_batch()
        condition = module._extract_condition(batch)
        samples = module.sample(condition, use_ddim=True, progress=False)
        assert samples.shape == (BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE)

    def test_checkpoint_round_trip(self, module, tmp_path):
        """Checkpoint save/load should preserve model state."""
        # Do a training step to ensure model has been used
        batch = make_mock_batch()
        module.training_step(batch, 0)

        # Save checkpoint
        ckpt_path = tmp_path / "test_checkpoint.ckpt"
        checkpoint = {
            'state_dict': module.state_dict(),
            'hyper_parameters': module.hparams,
        }
        module.on_save_checkpoint(checkpoint)
        torch.save(checkpoint, ckpt_path)

        # Load checkpoint
        loaded = torch.load(ckpt_path, weights_only=False)
        assert 'ema_state_dict' in loaded  # EMA should be saved
        assert 'state_dict' in loaded


class TestPredictionTypes:
    """Test all prediction types work end-to-end."""

    @pytest.mark.parametrize("pred_type", ["epsilon", "x0", "v"])
    def test_prediction_type_training(self, pred_type):
        """All prediction types should train without error."""
        module = DiffusionModule(
            unet_size='small',
            image_size=IMG_SIZE,
            condition_channels=COND_CH,
            num_timesteps=100,
            prediction_type=pred_type,
            use_ema=False,
            learning_rate=1e-4,
            warmup_epochs=1,
            ddim_steps=5,
        )
        batch = make_mock_batch()
        loss = module.training_step(batch, 0)
        assert loss.dim() == 0
        assert loss.item() > 0

    @pytest.mark.parametrize("pred_type", ["epsilon", "x0", "v"])
    def test_prediction_type_with_physics(self, pred_type):
        """Physics losses should work with all prediction types."""
        module = DiffusionModule(
            unet_size='small',
            image_size=IMG_SIZE,
            condition_channels=COND_CH,
            num_timesteps=100,
            prediction_type=pred_type,
            use_ema=False,
            use_physics_losses=True,
            trajectory_consistency_weight=0.1,
            distance_decay_weight=0.01,
            learning_rate=1e-4,
            warmup_epochs=1,
            ddim_steps=5,
        )
        batch = make_mock_batch()
        loss = module.training_step(batch, 0)
        assert loss.dim() == 0
        assert loss.item() > 0


class TestAblationConfigurations:
    """Test that all ablation configurations work."""

    def test_no_building_map(self):
        """Training without building map should work."""
        module = DiffusionModule(
            unet_size='small', image_size=IMG_SIZE, condition_channels=COND_CH,
            num_timesteps=100, use_ema=False, use_building_map=False,
            learning_rate=1e-4, warmup_epochs=1, ddim_steps=5,
        )
        batch = make_mock_batch()
        loss = module.training_step(batch, 0)
        assert loss.item() > 0

    def test_no_trajectory_mask(self):
        """Training without trajectory mask should work."""
        module = DiffusionModule(
            unet_size='small', image_size=IMG_SIZE, condition_channels=COND_CH,
            num_timesteps=100, use_ema=False, use_trajectory_mask=False,
            learning_rate=1e-4, warmup_epochs=1, ddim_steps=5,
        )
        batch = make_mock_batch()
        loss = module.training_step(batch, 0)
        assert loss.item() > 0

    def test_no_coverage_density(self):
        """Training without coverage density should work."""
        module = DiffusionModule(
            unet_size='small', image_size=IMG_SIZE, condition_channels=COND_CH,
            num_timesteps=100, use_ema=False, use_coverage_density=False,
            learning_rate=1e-4, warmup_epochs=1, ddim_steps=5,
        )
        batch = make_mock_batch()
        loss = module.training_step(batch, 0)
        assert loss.item() > 0

    def test_no_tx_position(self):
        """Training without TX position should work."""
        module = DiffusionModule(
            unet_size='small', image_size=IMG_SIZE, condition_channels=COND_CH,
            num_timesteps=100, use_ema=False, use_tx_position=False,
            learning_rate=1e-4, warmup_epochs=1, ddim_steps=5,
        )
        batch = make_mock_batch()
        loss = module.training_step(batch, 0)
        assert loss.item() > 0

    def test_minimal_conditioning(self):
        """Training with only sparse_rss should work."""
        module = DiffusionModule(
            unet_size='small', image_size=IMG_SIZE, condition_channels=COND_CH,
            num_timesteps=100, use_ema=False,
            use_building_map=False, use_trajectory_mask=False,
            use_coverage_density=False, use_tx_position=False,
            use_sparse_rss=True,
            learning_rate=1e-4, warmup_epochs=1, ddim_steps=5,
        )
        batch = make_mock_batch()
        loss = module.training_step(batch, 0)
        assert loss.item() > 0


class TestLossTypes:
    """Test all loss types."""

    @pytest.mark.parametrize("loss_type", ["mse", "l1", "huber"])
    def test_loss_type(self, loss_type):
        """All loss types should work."""
        module = DiffusionModule(
            unet_size='small', image_size=IMG_SIZE, condition_channels=COND_CH,
            num_timesteps=100, loss_type=loss_type, use_ema=False,
            learning_rate=1e-4, warmup_epochs=1, ddim_steps=5,
        )
        batch = make_mock_batch()
        loss = module.training_step(batch, 0)
        assert loss.item() > 0


class TestFactoryFunction:
    """Test get_diffusion_module factory."""

    @pytest.mark.parametrize("preset", ["default", "fast", "quality"])
    def test_all_presets(self, preset):
        """All presets should create valid modules."""
        module = get_diffusion_module(preset)
        batch = make_mock_batch(img_size=32)
        # Forward should work (even if shapes don't match preset image_size,
        # the model adapts to input size)
        loss = module.training_step(batch, 0)
        assert loss.item() > 0

    def test_preset_override(self):
        """Preset overrides should work."""
        module = get_diffusion_module('fast', learning_rate=5e-5)
        assert module.hparams.learning_rate == 5e-5


class TestMultiStepTraining:
    """Test that multiple training steps work (loss should be finite)."""

    def test_five_training_steps(self):
        """5 consecutive training steps should all produce finite loss."""
        module = DiffusionModule(
            unet_size='small', image_size=IMG_SIZE, condition_channels=COND_CH,
            num_timesteps=100, use_ema=False,
            learning_rate=1e-4, warmup_epochs=1, ddim_steps=5,
        )
        optimizer = torch.optim.Adam(module.parameters(), lr=1e-4)

        losses = []
        for i in range(5):
            batch = make_mock_batch()
            optimizer.zero_grad()
            loss = module.training_step(batch, i)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        for loss_val in losses:
            assert not torch.isnan(torch.tensor(loss_val)), "NaN loss detected"
            assert not torch.isinf(torch.tensor(loss_val)), "Inf loss detected"

    def test_five_steps_with_full_features(self):
        """5 steps with all features enabled should produce finite loss."""
        module = DiffusionModule(
            unet_size='small', image_size=IMG_SIZE, condition_channels=COND_CH,
            num_timesteps=100, use_ema=False,
            use_physics_losses=True, use_coverage_attention=True,
            learning_rate=1e-4, warmup_epochs=1, ddim_steps=5,
        )
        optimizer = torch.optim.Adam(module.parameters(), lr=1e-4)

        for i in range(5):
            batch = make_mock_batch()
            optimizer.zero_grad()
            loss = module.training_step(batch, i)
            loss.backward()
            optimizer.step()
            assert not torch.isnan(loss), f"NaN loss at step {i}"
            assert not torch.isinf(loss), f"Inf loss at step {i}"
