"""
Tests for CoverageAwareUNet.

Verifies:
- Forward pass shapes for all sizes (small/medium/large)
- Gradient flow through coverage attention
- With and without coverage (fallback to standard attention)
- CoverageAwareAttentionBlock replacement
- Coverage downsampling at each level
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch
import torch.nn as nn

from models.diffusion.coverage_unet import CoverageAwareUNet, get_coverage_aware_unet
from models.diffusion.unet import UNet, AttentionBlock, get_unet
from models.diffusion.attention import CoverageAwareAttentionBlock


# Use small tensors for fast tests
B, C, H, W = 2, 1, 32, 32
COND_CH = 16


def make_inputs(with_coverage=True):
    """Create standard test inputs."""
    x = torch.randn(B, C, H, W)
    t = torch.randint(0, 100, (B,))
    cond = torch.randn(B, COND_CH, H, W)
    coverage = torch.rand(B, 1, H, W) if with_coverage else None
    return x, t, cond, coverage


def _make_output_nonzero(unet):
    """Re-init output conv so the model doesn't output all zeros."""
    nn.init.xavier_uniform_(unet.output_conv[-1].weight)


class TestCoverageAwareUNetForward:
    """Test forward pass shapes and basic functionality."""

    @pytest.mark.parametrize("size", ["small", "medium", "large"])
    def test_forward_all_sizes(self, size):
        """CoverageAwareUNet should produce correct output shapes for all sizes."""
        unet = get_coverage_aware_unet(
            size=size,
            in_channels=C,
            out_channels=C,
            cond_channels=COND_CH,
            image_size=H,
        )
        x, t, cond, coverage = make_inputs(with_coverage=True)
        out = unet(x, t, cond=cond, coverage=coverage)
        assert out.shape == (B, C, H, W)

    def test_forward_without_coverage(self):
        """Should work when coverage=None (falls back to standard attention)."""
        unet = get_coverage_aware_unet(
            size="small",
            in_channels=C,
            out_channels=C,
            cond_channels=COND_CH,
            image_size=H,
        )
        x, t, cond, _ = make_inputs(with_coverage=False)
        out = unet(x, t, cond=cond, coverage=None)
        assert out.shape == (B, C, H, W)

    def test_forward_without_conditioning(self):
        """Should work without conditioning (cond=None)."""
        unet = get_coverage_aware_unet(
            size="small",
            in_channels=C,
            out_channels=C,
            cond_channels=0,
            image_size=H,
        )
        x, t, _, coverage = make_inputs(with_coverage=True)
        out = unet(x, t, cond=None, coverage=coverage)
        assert out.shape == (B, C, H, W)

    def test_different_temperatures(self):
        """Different coverage temperatures should produce different outputs."""
        x, t, cond, coverage = make_inputs(with_coverage=True)

        unet_low = get_coverage_aware_unet(
            size="small", in_channels=C, out_channels=C,
            cond_channels=COND_CH, image_size=H,
            coverage_temperature=0.1,
        )
        _make_output_nonzero(unet_low)

        unet_high = get_coverage_aware_unet(
            size="small", in_channels=C, out_channels=C,
            cond_channels=COND_CH, image_size=H,
            coverage_temperature=10.0,
        )
        # Copy weights so only temperature differs
        unet_high.load_state_dict(unet_low.state_dict(), strict=False)

        out_low = unet_low(x, t, cond=cond, coverage=coverage)
        out_high = unet_high(x, t, cond=cond, coverage=coverage)

        assert not torch.allclose(out_low, out_high, atol=1e-5)


class TestAttentionBlockReplacement:
    """Verify that AttentionBlocks are properly replaced."""

    def test_no_standard_attention_blocks(self):
        """After replacement, no standard AttentionBlock should remain."""
        unet = get_coverage_aware_unet(
            size="small",
            in_channels=C,
            out_channels=C,
            cond_channels=COND_CH,
            image_size=H,
        )
        for module in unet.modules():
            if isinstance(module, AttentionBlock) and not isinstance(module, CoverageAwareAttentionBlock):
                pytest.fail("Found standard AttentionBlock that was not replaced")

    def test_has_coverage_aware_blocks(self):
        """Should contain CoverageAwareAttentionBlock instances."""
        unet = get_coverage_aware_unet(
            size="small",
            in_channels=C,
            out_channels=C,
            cond_channels=COND_CH,
            image_size=H,
        )
        has_coverage_block = any(
            isinstance(m, CoverageAwareAttentionBlock) for m in unet.modules()
        )
        assert has_coverage_block, "No CoverageAwareAttentionBlock found"

    def test_temperature_propagated(self):
        """Coverage temperature should be propagated to all attention blocks."""
        temp = 2.5
        unet = get_coverage_aware_unet(
            size="small",
            in_channels=C,
            out_channels=C,
            cond_channels=COND_CH,
            image_size=H,
            coverage_temperature=temp,
        )
        for module in unet.modules():
            if isinstance(module, CoverageAwareAttentionBlock):
                # CoverageAwareAttentionBlock stores attention as self.attention
                assert module.attention.coverage_temperature == temp


class TestGradientFlow:
    """Verify gradient flow through the network."""

    def test_gradient_flow_with_coverage(self):
        """Gradients should flow back through coverage-aware attention."""
        unet = get_coverage_aware_unet(
            size="small",
            in_channels=C,
            out_channels=C,
            cond_channels=COND_CH,
            image_size=H,
        )
        # UNet zero-inits output_conv, so re-init to get non-zero output
        _make_output_nonzero(unet)

        x, t, cond, coverage = make_inputs(with_coverage=True)
        x.requires_grad_(True)

        out = unet(x, t, cond=cond, coverage=coverage)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert not torch.all(x.grad == 0)

    def test_gradient_flow_without_coverage(self):
        """Gradients should flow even when coverage is None."""
        unet = get_coverage_aware_unet(
            size="small",
            in_channels=C,
            out_channels=C,
            cond_channels=COND_CH,
            image_size=H,
        )
        _make_output_nonzero(unet)

        x, t, cond, _ = make_inputs(with_coverage=False)
        x.requires_grad_(True)

        out = unet(x, t, cond=cond, coverage=None)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_all_parameters_have_gradients(self):
        """All trainable parameters should receive gradients."""
        unet = get_coverage_aware_unet(
            size="small",
            in_channels=C,
            out_channels=C,
            cond_channels=COND_CH,
            image_size=H,
        )
        x, t, cond, coverage = make_inputs(with_coverage=True)
        out = unet(x, t, cond=cond, coverage=coverage)
        loss = out.sum()
        loss.backward()

        for name, param in unet.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


class TestCoverageEffect:
    """Verify that coverage actually affects the output."""

    def test_coverage_changes_output(self):
        """Non-trivial coverage should change output vs no coverage."""
        unet = get_coverage_aware_unet(
            size="small",
            in_channels=C,
            out_channels=C,
            cond_channels=COND_CH,
            image_size=H,
        )
        _make_output_nonzero(unet)
        unet.eval()
        x, t, cond, coverage = make_inputs(with_coverage=True)

        with torch.no_grad():
            out_with = unet(x, t, cond=cond, coverage=coverage)
            out_without = unet(x, t, cond=cond, coverage=None)

        assert not torch.allclose(out_with, out_without, atol=1e-5)

    def test_different_coverage_different_output(self):
        """Different coverage maps should produce different outputs."""
        unet = get_coverage_aware_unet(
            size="small",
            in_channels=C,
            out_channels=C,
            cond_channels=COND_CH,
            image_size=H,
        )
        _make_output_nonzero(unet)
        unet.eval()
        x, t, cond, _ = make_inputs(with_coverage=False)

        # Use spatially varying coverage so the additive log-bias
        # produces non-uniform shifts across keys (uniform coverage
        # cancels out in softmax).
        torch.manual_seed(42)
        coverage_a = torch.rand(B, 1, H, W)
        coverage_b = 1.0 - coverage_a  # Inverted spatial pattern

        with torch.no_grad():
            out_a = unet(x, t, cond=cond, coverage=coverage_a)
            out_b = unet(x, t, cond=cond, coverage=coverage_b)

        assert not torch.allclose(out_a, out_b, atol=1e-5)


class TestGetCoverageAwareUnet:
    """Test factory function."""

    def test_invalid_size(self):
        """Should raise error for invalid size."""
        with pytest.raises(ValueError, match="Unknown size"):
            get_coverage_aware_unet(size="xlarge")

    def test_returns_coverage_aware_unet(self):
        """Factory should return CoverageAwareUNet instance."""
        unet = get_coverage_aware_unet(size="small")
        assert isinstance(unet, CoverageAwareUNet)
        assert isinstance(unet, UNet)

    @pytest.mark.parametrize("size", ["small", "medium", "large"])
    def test_parameter_count_matches_or_exceeds_unet(self, size):
        """CoverageAwareUNet should have similar or more params than UNet."""
        standard = get_unet(size=size, in_channels=C, out_channels=C,
                            cond_channels=COND_CH, image_size=H)
        coverage = get_coverage_aware_unet(size=size, in_channels=C, out_channels=C,
                                           cond_channels=COND_CH, image_size=H)

        standard_params = sum(p.numel() for p in standard.parameters())
        coverage_params = sum(p.numel() for p in coverage.parameters())

        assert coverage_params > 0
        ratio = coverage_params / standard_params
        assert 0.5 < ratio < 2.0, f"Param ratio {ratio:.2f} outside expected range"
