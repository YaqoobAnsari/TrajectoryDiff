"""
Tests for CoverageAwareAttention module.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.diffusion.attention import (
    CoverageAwareAttention,
    CoverageAwareAttentionBlock,
    downsample_coverage,
    upsample_coverage,
)


class TestCoverageAwareAttention:
    """Tests for CoverageAwareAttention."""

    def test_basic_forward(self):
        """Test basic forward pass without coverage."""
        attn = CoverageAwareAttention(dim=64, num_heads=4)

        x = torch.randn(2, 16, 64)  # (B, N, D)
        out = attn(x)

        assert out.shape == x.shape

    def test_forward_with_coverage(self):
        """Test forward pass with coverage input."""
        attn = CoverageAwareAttention(dim=64, num_heads=4)

        x = torch.randn(2, 16, 64)  # (B, N, D)
        coverage = torch.rand(2, 16, 1)  # (B, N, 1)

        out = attn(x, coverage)

        assert out.shape == x.shape

    def test_gradient_flow(self):
        """Test gradient flows through attention."""
        attn = CoverageAwareAttention(dim=64, num_heads=4)

        x = torch.randn(2, 16, 64, requires_grad=True)
        coverage = torch.rand(2, 16, 1)

        out = attn(x, coverage)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_different_heads(self):
        """Test with different number of heads."""
        for num_heads in [1, 2, 4, 8]:
            attn = CoverageAwareAttention(dim=64, num_heads=num_heads)
            x = torch.randn(2, 16, 64)
            out = attn(x)
            assert out.shape == x.shape

    def test_coverage_affects_output(self):
        """Test that coverage actually affects the output."""
        attn = CoverageAwareAttention(dim=64, num_heads=4)

        x = torch.randn(2, 16, 64)

        # High coverage everywhere
        coverage_high = torch.ones(2, 16, 1)

        # Low coverage everywhere
        coverage_low = torch.zeros(2, 16, 1)

        out_high = attn(x, coverage_high)
        out_low = attn(x, coverage_low)

        # Outputs should be different
        assert not torch.allclose(out_high, out_low)

    def test_temperature_effect(self):
        """Test that temperature affects coverage modulation."""
        x = torch.randn(2, 16, 64)
        coverage = torch.rand(2, 16, 1)

        # Low temperature = more coverage-dependent
        attn_low_temp = CoverageAwareAttention(dim=64, num_heads=4, coverage_temperature=0.5)
        # High temperature = more uniform
        attn_high_temp = CoverageAwareAttention(dim=64, num_heads=4, coverage_temperature=2.0)

        out_low = attn_low_temp(x, coverage)
        out_high = attn_high_temp(x, coverage)

        # Outputs should be different
        assert not torch.allclose(out_low, out_high)


class TestCoverageAwareAttentionBlock:
    """Tests for CoverageAwareAttentionBlock (spatial version)."""

    def test_basic_forward(self):
        """Test basic forward pass with spatial input."""
        block = CoverageAwareAttentionBlock(channels=64)

        x = torch.randn(2, 64, 16, 16)  # (B, C, H, W)
        out = block(x)

        assert out.shape == x.shape

    def test_forward_with_coverage(self):
        """Test forward with spatial coverage map."""
        block = CoverageAwareAttentionBlock(channels=64)

        x = torch.randn(2, 64, 16, 16)  # (B, C, H, W)
        coverage = torch.rand(2, 1, 16, 16)  # (B, 1, H, W)

        out = block(x, coverage)

        assert out.shape == x.shape

    def test_gradient_flow(self):
        """Test gradient flow through spatial attention."""
        block = CoverageAwareAttentionBlock(channels=64)

        x = torch.randn(2, 64, 8, 8, requires_grad=True)
        coverage = torch.rand(2, 1, 8, 8)

        out = block(x, coverage)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None

    def test_different_spatial_sizes(self):
        """Test with different spatial sizes."""
        block = CoverageAwareAttentionBlock(channels=64)

        for size in [8, 16, 32]:
            x = torch.randn(2, 64, size, size)
            out = block(x)
            assert out.shape == x.shape


class TestCoverageResampling:
    """Tests for coverage resampling functions."""

    def test_downsample_coverage(self):
        """Test coverage downsampling."""
        coverage = torch.rand(2, 1, 64, 64)
        target_size = (16, 16)

        downsampled = downsample_coverage(coverage, target_size)

        assert downsampled.shape == (2, 1, 16, 16)

    def test_upsample_coverage(self):
        """Test coverage upsampling."""
        coverage = torch.rand(2, 1, 16, 16)
        target_size = (64, 64)

        upsampled = upsample_coverage(coverage, target_size)

        assert upsampled.shape == (2, 1, 64, 64)

    def test_downsample_preserves_average(self):
        """Test that downsampling preserves approximate average."""
        coverage = torch.rand(2, 1, 64, 64)
        target_size = (16, 16)

        downsampled = downsample_coverage(coverage, target_size)

        # Average should be approximately preserved
        orig_mean = coverage.mean()
        down_mean = downsampled.mean()

        assert abs(orig_mean.item() - down_mean.item()) < 0.1

    def test_round_trip(self):
        """Test down then up sampling."""
        coverage = torch.rand(2, 1, 64, 64)

        downsampled = downsample_coverage(coverage, (16, 16))
        upsampled = upsample_coverage(downsampled, (64, 64))

        # Shape should be preserved
        assert upsampled.shape == coverage.shape


class TestIntegration:
    """Integration tests for coverage-aware attention in U-Net context."""

    def test_multiple_resolutions(self):
        """Test attention at multiple resolutions (as in U-Net)."""
        block_64 = CoverageAwareAttentionBlock(channels=64)
        block_128 = CoverageAwareAttentionBlock(channels=128)
        block_256 = CoverageAwareAttentionBlock(channels=256)

        # Simulate U-Net encoder path
        x1 = torch.randn(2, 64, 64, 64)   # High res
        x2 = torch.randn(2, 128, 32, 32)  # Mid res
        x3 = torch.randn(2, 256, 16, 16)  # Low res

        coverage = torch.rand(2, 1, 64, 64)

        # Downsample coverage for each resolution
        cov1 = coverage
        cov2 = downsample_coverage(coverage, (32, 32))
        cov3 = downsample_coverage(coverage, (16, 16))

        # Apply attention at each level
        out1 = block_64(x1, cov1)
        out2 = block_128(x2, cov2)
        out3 = block_256(x3, cov3)

        assert out1.shape == x1.shape
        assert out2.shape == x2.shape
        assert out3.shape == x3.shape

    def test_training_mode(self):
        """Test attention behaves correctly in training vs eval mode."""
        block = CoverageAwareAttentionBlock(channels=64)

        x = torch.randn(2, 64, 16, 16)
        coverage = torch.rand(2, 1, 16, 16)

        # Training mode
        block.train()
        out_train = block(x, coverage)

        # Eval mode
        block.eval()
        out_eval = block(x, coverage)

        assert out_train.shape == out_eval.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
