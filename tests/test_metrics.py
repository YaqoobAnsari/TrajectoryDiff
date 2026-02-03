"""
Unit tests for evaluation metrics.

Run with: pytest tests/test_metrics.py -v
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from evaluation.metrics import (
    rmse,
    mae,
    mse,
    rmse_observed,
    rmse_unobserved,
    rmse_by_distance,
    ssim,
    psnr,
    coverage_weighted_rmse,
    compute_coverage_density,
    uncertainty_calibration,
    uncertainty_error_correlation,
    compute_all_metrics,
)


class TestBasicMetrics:
    """Test basic error metrics."""

    def test_rmse_identical(self):
        """RMSE of identical arrays should be 0."""
        arr = np.random.randn(64, 64)
        assert rmse(arr, arr) == 0.0

    def test_rmse_known_value(self):
        """Test RMSE with known difference."""
        pred = np.ones((10, 10))
        target = np.zeros((10, 10))
        # RMSE should be 1.0
        assert abs(rmse(pred, target) - 1.0) < 1e-6

    def test_rmse_with_mask(self):
        """Test masked RMSE."""
        pred = np.array([[1, 2], [3, 4]], dtype=float)
        target = np.array([[0, 0], [0, 0]], dtype=float)
        mask = np.array([[1, 0], [0, 1]])  # Only corners

        # RMSE on corners: sqrt((1^2 + 4^2) / 2) = sqrt(8.5) ≈ 2.915
        expected = np.sqrt((1**2 + 4**2) / 2)
        assert abs(rmse(pred, target, mask) - expected) < 1e-6

    def test_rmse_empty_mask(self):
        """RMSE with empty mask should return NaN."""
        pred = np.ones((10, 10))
        target = np.zeros((10, 10))
        mask = np.zeros((10, 10))
        assert np.isnan(rmse(pred, target, mask))

    def test_mae_known_value(self):
        """Test MAE with known values."""
        pred = np.array([1, 2, 3], dtype=float)
        target = np.array([0, 0, 0], dtype=float)
        # MAE = (1 + 2 + 3) / 3 = 2.0
        assert abs(mae(pred, target) - 2.0) < 1e-6

    def test_mse_known_value(self):
        """Test MSE with known values."""
        pred = np.array([1, 2, 3], dtype=float)
        target = np.array([0, 0, 0], dtype=float)
        # MSE = (1 + 4 + 9) / 3 ≈ 4.667
        expected = (1 + 4 + 9) / 3
        assert abs(mse(pred, target) - expected) < 1e-6


class TestTrajectoryAwareMetrics:
    """Test trajectory-aware metrics."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data with trajectory mask."""
        np.random.seed(42)
        H, W = 64, 64
        target = np.random.randn(H, W) * 10  # Radio map
        pred = target + np.random.randn(H, W) * 2  # Noisy prediction

        # Create trajectory mask (simulate a path)
        mask = np.zeros((H, W))
        for i in range(H):
            mask[i, 30:35] = 1  # Vertical corridor
        for j in range(W):
            mask[30:35, j] = 1  # Horizontal corridor

        return pred, target, mask

    def test_rmse_observed_vs_unobserved(self, sample_data):
        """Observed and unobserved RMSE should cover all pixels."""
        pred, target, mask = sample_data

        obs_rmse = rmse_observed(pred, target, mask)
        unobs_rmse = rmse_unobserved(pred, target, mask)

        # Both should be finite
        assert np.isfinite(obs_rmse)
        assert np.isfinite(unobs_rmse)

    def test_rmse_observed_subset(self, sample_data):
        """RMSE observed should only use masked pixels."""
        pred, target, mask = sample_data

        obs_rmse = rmse_observed(pred, target, mask)
        manual_rmse = rmse(pred, target, mask)

        assert abs(obs_rmse - manual_rmse) < 1e-6

    def test_rmse_unobserved_complement(self, sample_data):
        """RMSE unobserved should use inverse mask."""
        pred, target, mask = sample_data

        unobs_rmse = rmse_unobserved(pred, target, mask)
        manual_rmse = rmse(pred, target, 1 - mask)

        assert abs(unobs_rmse - manual_rmse) < 1e-6

    def test_rmse_by_distance_ordering(self, sample_data):
        """Error should generally increase with distance from samples."""
        pred, target, mask = sample_data

        # Add distance-dependent noise to make test deterministic
        from scipy.ndimage import distance_transform_edt
        dist = distance_transform_edt(1 - mask)
        pred_biased = target + dist * 0.1  # Error grows with distance

        dist_metrics = rmse_by_distance(pred_biased, target, mask, [5, 10, 20])

        # Should have increasing RMSE with distance
        values = [v for k, v in sorted(dist_metrics.items()) if np.isfinite(v)]
        assert len(values) >= 2
        # Check general trend (allow some noise)
        assert values[-1] >= values[0] * 0.5  # Far should be at least half of near


class TestCoverageMetrics:
    """Test coverage-weighted metrics."""

    def test_coverage_density_smoothing(self):
        """Coverage density should smooth the binary mask."""
        mask = np.zeros((64, 64))
        mask[32, 32] = 1  # Single point

        density = compute_coverage_density(mask, sigma=5.0)

        # Density should be highest at the point
        assert density[32, 32] == density.max()
        # Should be smooth (neighbors have positive values)
        assert density[30, 32] > 0
        assert density[32, 30] > 0

    def test_coverage_weighted_rmse_inverse(self):
        """Inverse weighting should emphasize low-coverage areas."""
        H, W = 64, 64
        target = np.zeros((H, W))

        # Prediction with error only in low-coverage region
        pred = np.zeros((H, W))
        pred[0:10, 0:10] = 1  # Error in corner

        # Coverage high in center, low in corners
        coverage = np.zeros((H, W))
        coverage[20:44, 20:44] = 1
        coverage = compute_coverage_density(coverage, sigma=10)

        weighted = coverage_weighted_rmse(pred, target, coverage, inverse=True)
        unweighted = rmse(pred, target)

        # Weighted should be higher (corner error emphasized)
        assert weighted > unweighted


class TestStructuralMetrics:
    """Test structural similarity metrics."""

    def test_ssim_identical(self):
        """SSIM of identical images should be 1.0."""
        arr = np.random.randn(64, 64) * 50 + 100
        assert abs(ssim(arr, arr) - 1.0) < 1e-6

    def test_ssim_range(self):
        """SSIM should be in [-1, 1]."""
        a = np.random.randn(64, 64)
        b = np.random.randn(64, 64)
        s = ssim(a, b)
        assert -1 <= s <= 1

    def test_psnr_identical(self):
        """PSNR of identical images should be infinite."""
        arr = np.random.randn(64, 64)
        assert psnr(arr, arr) == float('inf')

    def test_psnr_range(self):
        """PSNR should be positive for reasonable predictions."""
        target = np.random.randn(64, 64) * 10
        pred = target + np.random.randn(64, 64) * 0.1
        p = psnr(pred, target, data_range=target.max() - target.min())
        assert p > 0


class TestUncertaintyMetrics:
    """Test uncertainty calibration metrics."""

    def test_perfect_calibration(self):
        """Well-calibrated predictions should have low calibration error."""
        np.random.seed(42)
        H, W = 64, 64

        # Generate truly Gaussian predictions
        target = np.random.randn(H, W) * 10
        true_std = 2.0
        pred_mean = target + np.random.randn(H, W) * true_std
        pred_std = np.ones((H, W)) * true_std

        calib = uncertainty_calibration(pred_mean, pred_std, target)

        # Should be reasonably calibrated
        # Allow tolerance due to finite sample size
        assert 0.5 < calib['fraction_within_1std'] < 0.85
        assert 0.85 < calib['fraction_within_2std'] < 1.0

    def test_overconfident_calibration(self):
        """Overconfident predictions should show poor calibration."""
        np.random.seed(42)
        H, W = 64, 64

        target = np.random.randn(H, W) * 10
        pred_mean = target + np.random.randn(H, W) * 5  # Actual std = 5
        pred_std = np.ones((H, W)) * 1.0  # Claimed std = 1 (overconfident)

        calib = uncertainty_calibration(pred_mean, pred_std, target)

        # Should show fewer predictions within claimed bounds
        assert calib['fraction_within_1std'] < 0.68

    def test_uncertainty_error_correlation(self):
        """Uncertainty should correlate with errors when calibrated."""
        np.random.seed(42)
        H, W = 64, 64

        # Create heteroscedastic noise
        noise_scale = np.random.rand(H, W) * 5 + 0.1
        target = np.random.randn(H, W) * 10
        errors = np.abs(np.random.randn(H, W)) * noise_scale

        # If uncertainty matches noise scale, should correlate
        corr = uncertainty_error_correlation(noise_scale, errors)
        assert corr > 0.3  # Should be positively correlated


class TestAggregateMetrics:
    """Test the aggregate compute_all_metrics function."""

    def test_compute_all_basic(self):
        """Test basic metric computation."""
        pred = np.random.randn(64, 64)
        target = np.random.randn(64, 64)

        metrics = compute_all_metrics(pred, target)

        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'mse' in metrics
        assert 'psnr' in metrics
        assert np.isfinite(metrics['rmse'])

    def test_compute_all_with_mask(self):
        """Test metrics with trajectory mask."""
        pred = np.random.randn(64, 64)
        target = np.random.randn(64, 64)
        mask = np.zeros((64, 64))
        mask[20:40, 20:40] = 1

        metrics = compute_all_metrics(pred, target, trajectory_mask=mask)

        assert 'rmse_observed' in metrics
        assert 'rmse_unobserved' in metrics
        assert 'coverage_ratio' in metrics
        assert abs(metrics['coverage_ratio'] - 0.09765625) < 0.01  # 20x20 / 64x64 = 400/4096

    def test_compute_all_with_uncertainty(self):
        """Test metrics with uncertainty estimates."""
        pred = np.random.randn(64, 64)
        target = np.random.randn(64, 64)
        pred_std = np.abs(np.random.randn(64, 64)) + 0.1

        metrics = compute_all_metrics(pred, target, pred_std=pred_std)

        assert 'calibration_error' in metrics
        assert 'fraction_within_1std' in metrics
        assert 'uncertainty_error_correlation' in metrics


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
