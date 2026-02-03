"""
Evaluation metrics for radio map prediction.

This module provides metrics for evaluating radio map generation quality,
with special focus on trajectory-based sampling scenarios where we care about:
1. Overall reconstruction quality (RMSE, SSIM)
2. Interpolation quality (observed regions)
3. Extrapolation quality (unobserved regions / blind spots)
4. Uncertainty calibration

Key insight: For trajectory-conditioned models, RMSE_unobserved is the critical
metric - it measures how well we extrapolate into regions without samples.
"""

from typing import Optional, Tuple, Dict, Union
import numpy as np

try:
    from skimage.metrics import structural_similarity as ssim_func
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# =============================================================================
# Core Metrics
# =============================================================================

def rmse(
    pred: np.ndarray,
    target: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> float:
    """
    Compute Root Mean Squared Error.

    Args:
        pred: Predicted radio map, shape (H, W) or (B, H, W)
        target: Ground truth radio map, same shape as pred
        mask: Optional binary mask, 1 = include, 0 = exclude

    Returns:
        RMSE value (lower is better)

    Example:
        >>> pred = np.random.randn(256, 256)
        >>> target = pred + np.random.randn(256, 256) * 0.1
        >>> rmse(pred, target)  # Should be ~0.1
    """
    diff = pred - target

    if mask is not None:
        # Only compute on masked pixels
        mask = mask.astype(bool)
        if not np.any(mask):
            return float('nan')
        diff = diff[mask]

    return float(np.sqrt(np.mean(diff ** 2)))


def mae(
    pred: np.ndarray,
    target: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> float:
    """
    Compute Mean Absolute Error.

    Args:
        pred: Predicted radio map
        target: Ground truth radio map
        mask: Optional binary mask

    Returns:
        MAE value (lower is better)
    """
    diff = np.abs(pred - target)

    if mask is not None:
        mask = mask.astype(bool)
        if not np.any(mask):
            return float('nan')
        diff = diff[mask]

    return float(np.mean(diff))


def mse(
    pred: np.ndarray,
    target: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> float:
    """
    Compute Mean Squared Error.

    Args:
        pred: Predicted radio map
        target: Ground truth radio map
        mask: Optional binary mask

    Returns:
        MSE value (lower is better)
    """
    diff = pred - target

    if mask is not None:
        mask = mask.astype(bool)
        if not np.any(mask):
            return float('nan')
        diff = diff[mask]

    return float(np.mean(diff ** 2))


# =============================================================================
# Trajectory-Aware Metrics (Key for our method)
# =============================================================================

def rmse_observed(
    pred: np.ndarray,
    target: np.ndarray,
    trajectory_mask: np.ndarray
) -> float:
    """
    RMSE on pixels WHERE we have trajectory samples (interpolation quality).

    This measures how well the model reconstructs regions with observations.
    Should be low for any reasonable method.

    Args:
        pred: Predicted radio map, shape (H, W)
        target: Ground truth radio map, shape (H, W)
        trajectory_mask: Binary mask, 1 = sampled location, 0 = unsampled

    Returns:
        RMSE on observed pixels
    """
    return rmse(pred, target, mask=trajectory_mask)


def rmse_unobserved(
    pred: np.ndarray,
    target: np.ndarray,
    trajectory_mask: np.ndarray
) -> float:
    """
    RMSE on pixels WITHOUT trajectory samples (extrapolation quality).

    THIS IS THE KEY METRIC for trajectory-conditioned models.
    It measures how well we predict blind spots and unvisited regions.

    Args:
        pred: Predicted radio map, shape (H, W)
        target: Ground truth radio map, shape (H, W)
        trajectory_mask: Binary mask, 1 = sampled, 0 = unsampled

    Returns:
        RMSE on unobserved pixels (our key metric)
    """
    unobserved_mask = 1 - trajectory_mask
    return rmse(pred, target, mask=unobserved_mask)


def rmse_by_distance(
    pred: np.ndarray,
    target: np.ndarray,
    trajectory_mask: np.ndarray,
    distance_bins: Optional[list] = None
) -> Dict[str, float]:
    """
    Compute RMSE binned by distance from nearest trajectory sample.

    This shows how error grows as we move further from observations.
    Trajectory-aware models should show slower error growth.

    Args:
        pred: Predicted radio map, shape (H, W)
        target: Ground truth radio map, shape (H, W)
        trajectory_mask: Binary mask of sampled locations
        distance_bins: Distance thresholds in pixels, default [5, 10, 20, 50]

    Returns:
        Dict with RMSE for each distance bin
    """
    from scipy.ndimage import distance_transform_edt

    if distance_bins is None:
        distance_bins = [5, 10, 20, 50]

    # Compute distance from each pixel to nearest sample
    dist_to_sample = distance_transform_edt(1 - trajectory_mask)

    results = {}
    prev_threshold = 0

    for threshold in distance_bins:
        bin_mask = (dist_to_sample > prev_threshold) & (dist_to_sample <= threshold)
        if np.any(bin_mask):
            results[f'rmse_dist_{prev_threshold}_{threshold}'] = rmse(pred, target, mask=bin_mask)
        else:
            results[f'rmse_dist_{prev_threshold}_{threshold}'] = float('nan')
        prev_threshold = threshold

    # Beyond last threshold
    beyond_mask = dist_to_sample > distance_bins[-1]
    if np.any(beyond_mask):
        results[f'rmse_dist_{distance_bins[-1]}_inf'] = rmse(pred, target, mask=beyond_mask)
    else:
        results[f'rmse_dist_{distance_bins[-1]}_inf'] = float('nan')

    return results


# =============================================================================
# Structural Metrics
# =============================================================================

def ssim(
    pred: np.ndarray,
    target: np.ndarray,
    data_range: Optional[float] = None,
    win_size: int = 7
) -> float:
    """
    Compute Structural Similarity Index (SSIM).

    SSIM measures perceptual similarity, capturing structure beyond pixel-wise error.
    Useful for assessing whether predicted radio maps have realistic spatial patterns.

    Args:
        pred: Predicted radio map, shape (H, W)
        target: Ground truth radio map, shape (H, W)
        data_range: Range of data values. If None, computed from target.
        win_size: Window size for SSIM computation

    Returns:
        SSIM value in [-1, 1], higher is better

    Raises:
        ImportError: If scikit-image is not installed
    """
    if not SKIMAGE_AVAILABLE:
        raise ImportError("scikit-image required for SSIM. Install with: pip install scikit-image")

    if data_range is None:
        data_range = target.max() - target.min()
        if data_range == 0:
            data_range = 1.0

    # Ensure 2D
    if pred.ndim > 2:
        pred = pred.squeeze()
    if target.ndim > 2:
        target = target.squeeze()

    return float(ssim_func(
        pred,
        target,
        data_range=data_range,
        win_size=win_size,
        channel_axis=None
    ))


def psnr(
    pred: np.ndarray,
    target: np.ndarray,
    data_range: Optional[float] = None
) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR).

    Args:
        pred: Predicted radio map
        target: Ground truth radio map
        data_range: Range of data values

    Returns:
        PSNR in dB (higher is better)
    """
    if data_range is None:
        data_range = target.max() - target.min()
        if data_range == 0:
            return float('inf')

    mse_val = mse(pred, target)
    if mse_val == 0:
        return float('inf')

    return float(10 * np.log10((data_range ** 2) / mse_val))


def compute_masked_ssim(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    data_range: Optional[float] = None,
    win_size: int = 7
) -> float:
    """
    Compute SSIM only within a masked region (M10).

    Args:
        pred: Predicted radio map, shape (H, W)
        target: Ground truth radio map, shape (H, W)
        mask: Binary mask (1=include, 0=exclude), shape (H, W)
        data_range: Range of data values
        win_size: Window size for SSIM

    Returns:
        SSIM value computed only on masked region
    """
    if not SKIMAGE_AVAILABLE:
        raise ImportError("scikit-image required for SSIM")

    # Apply mask
    mask_bool = mask.astype(bool)
    if not np.any(mask_bool):
        return float('nan')

    # For masked SSIM, we compute on the full image but weight by mask
    # Note: This is an approximation - true masked SSIM would require
    # sliding window only over masked regions
    pred_masked = pred.copy()
    target_masked = target.copy()

    # Set unmasked regions to mean to minimize their influence
    if np.any(~mask_bool):
        pred_masked[~mask_bool] = np.mean(pred[mask_bool])
        target_masked[~mask_bool] = np.mean(target[mask_bool])

    if data_range is None:
        data_range = target.max() - target.min()
        if data_range == 0:
            data_range = 1.0

    return ssim(pred_masked, target_masked, data_range=data_range, win_size=win_size)


def compute_masked_psnr(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    data_range: Optional[float] = None
) -> float:
    """
    Compute PSNR only within a masked region (M10).

    Args:
        pred: Predicted radio map
        target: Ground truth radio map
        mask: Binary mask (1=include, 0=exclude)
        data_range: Range of data values

    Returns:
        PSNR computed only on masked region
    """
    mask_bool = mask.astype(bool)
    if not np.any(mask_bool):
        return float('nan')

    mse_val = mse(pred, target, mask=mask)
    if mse_val == 0:
        return float('inf')

    if data_range is None:
        data_range = target.max() - target.min()
        if data_range == 0:
            return float('inf')

    return float(10 * np.log10((data_range ** 2) / mse_val))


# =============================================================================
# Sample Diversity Metrics (C6)
# =============================================================================

def compute_sample_diversity(samples: Union[np.ndarray, 'torch.Tensor']) -> Dict[str, float]:
    """
    Compute diversity metrics across multiple samples (C6).

    Measures the variability across N samples of the same input,
    quantifying epistemic uncertainty and model diversity.

    Args:
        samples: Multiple samples, shape (N, C, H, W) or (N, H, W)
            where N is number of samples

    Returns:
        Dict with diversity metrics:
        - mean_std: Mean per-pixel standard deviation
        - median_std: Median per-pixel standard deviation
        - max_std: Maximum per-pixel standard deviation
        - mean_range: Mean per-pixel range (max - min)
    """
    if TORCH_AVAILABLE and isinstance(samples, torch.Tensor):
        samples = samples.cpu().numpy()

    # Ensure shape (N, H, W)
    if samples.ndim == 4:
        samples = samples.squeeze(1)  # Remove channel dim if present

    # Compute per-pixel std across samples
    per_pixel_std = np.std(samples, axis=0)  # (H, W)
    per_pixel_range = np.max(samples, axis=0) - np.min(samples, axis=0)  # (H, W)

    return {
        'mean_std': float(np.mean(per_pixel_std)),
        'median_std': float(np.median(per_pixel_std)),
        'max_std': float(np.max(per_pixel_std)),
        'mean_range': float(np.mean(per_pixel_range)),
        'std_90th_percentile': float(np.percentile(per_pixel_std, 90)),
    }


# =============================================================================
# Coverage-Weighted Metrics
# =============================================================================

def coverage_weighted_rmse(
    pred: np.ndarray,
    target: np.ndarray,
    coverage_density: np.ndarray,
    inverse: bool = True
) -> float:
    """
    RMSE weighted by coverage density.

    By default (inverse=True), gives HIGHER weight to LOW coverage regions.
    This emphasizes extrapolation quality over interpolation.

    Args:
        pred: Predicted radio map, shape (H, W)
        target: Ground truth radio map, shape (H, W)
        coverage_density: Coverage density map (e.g., Gaussian-smoothed trajectory mask)
        inverse: If True, weight inversely by coverage (emphasize blind spots)

    Returns:
        Weighted RMSE value
    """
    diff_sq = (pred - target) ** 2

    if inverse:
        # Higher weight where coverage is low
        # Add small epsilon to avoid division by zero
        weights = 1.0 / (coverage_density + 1e-6)
    else:
        weights = coverage_density

    # Normalize weights
    weights = weights / weights.sum()

    return float(np.sqrt(np.sum(weights * diff_sq)))


def compute_coverage_density(
    trajectory_mask: np.ndarray,
    sigma: float = 5.0
) -> np.ndarray:
    """
    Compute Gaussian-smoothed coverage density from trajectory mask.

    Args:
        trajectory_mask: Binary mask of sampled locations
        sigma: Gaussian smoothing sigma in pixels

    Returns:
        Coverage density map, same shape as input
    """
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(trajectory_mask.astype(float), sigma=sigma)


# =============================================================================
# Uncertainty Metrics
# =============================================================================

def uncertainty_calibration(
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    target: np.ndarray,
    n_bins: int = 10
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Evaluate uncertainty calibration.

    A well-calibrated model should have:
    - Errors within 1 std for ~68% of predictions
    - Errors within 2 std for ~95% of predictions

    Args:
        pred_mean: Mean prediction, shape (H, W)
        pred_std: Standard deviation (uncertainty), shape (H, W)
        target: Ground truth, shape (H, W)
        n_bins: Number of bins for calibration curve

    Returns:
        Dict containing:
        - 'calibration_error': Expected Calibration Error (ECE)
        - 'fraction_within_1std': Should be ~0.68
        - 'fraction_within_2std': Should be ~0.95
        - 'calibration_curve': (expected, observed) arrays for plotting
    """
    # Compute z-scores
    z_scores = np.abs(pred_mean - target) / (pred_std + 1e-8)

    # Fractions within confidence intervals
    within_1std = float(np.mean(z_scores <= 1))
    within_2std = float(np.mean(z_scores <= 2))

    # Calibration curve
    # For each confidence level, what fraction of predictions are within that confidence?
    confidence_levels = np.linspace(0, 3, n_bins + 1)[1:]  # z-scores from 0 to 3
    expected_coverage = []
    observed_coverage = []

    for z in confidence_levels:
        # Expected: CDF of standard normal at z (both tails)
        from scipy.stats import norm
        expected = 2 * norm.cdf(z) - 1  # Probability within ±z
        observed = float(np.mean(z_scores <= z))
        expected_coverage.append(expected)
        observed_coverage.append(observed)

    expected_coverage = np.array(expected_coverage)
    observed_coverage = np.array(observed_coverage)

    # Expected Calibration Error (area between curves)
    ece = float(np.mean(np.abs(expected_coverage - observed_coverage)))

    return {
        'calibration_error': ece,
        'fraction_within_1std': within_1std,
        'fraction_within_2std': within_2std,
        'calibration_curve': (expected_coverage, observed_coverage)
    }


def uncertainty_error_correlation(
    pred_std: np.ndarray,
    errors: np.ndarray
) -> float:
    """
    Compute correlation between predicted uncertainty and actual errors.

    A good uncertainty estimate should correlate with actual errors:
    high uncertainty where errors are high, low uncertainty where errors are low.

    Args:
        pred_std: Predicted standard deviation (uncertainty)
        errors: Actual absolute errors |pred - target|

    Returns:
        Pearson correlation coefficient, should be positive and high
    """
    # Flatten
    std_flat = pred_std.flatten()
    err_flat = errors.flatten()

    # Remove any NaN or inf
    valid = np.isfinite(std_flat) & np.isfinite(err_flat)
    std_flat = std_flat[valid]
    err_flat = err_flat[valid]

    if len(std_flat) < 2:
        return float('nan')

    correlation = np.corrcoef(std_flat, err_flat)[0, 1]
    return float(correlation)


# =============================================================================
# Aggregate Metrics
# =============================================================================

def compute_all_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    trajectory_mask: Optional[np.ndarray] = None,
    pred_std: Optional[np.ndarray] = None,
    data_range: Optional[float] = None
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.

    This is the main function to call for comprehensive evaluation.

    Args:
        pred: Predicted radio map, shape (H, W)
        target: Ground truth radio map, shape (H, W)
        trajectory_mask: Binary mask of trajectory samples (optional)
        pred_std: Predicted uncertainty/std (optional, for uncertainty metrics)
        data_range: Data range for SSIM/PSNR (optional)

    Returns:
        Dict with all computed metrics

    Example:
        >>> metrics = compute_all_metrics(pred, target, trajectory_mask)
        >>> print(f"Overall RMSE: {metrics['rmse']:.4f}")
        >>> print(f"Unobserved RMSE: {metrics['rmse_unobserved']:.4f}")
    """
    results = {}

    # Basic metrics
    results['rmse'] = rmse(pred, target)
    results['mae'] = mae(pred, target)
    results['mse'] = mse(pred, target)

    # Structural metrics
    if SKIMAGE_AVAILABLE:
        try:
            results['ssim'] = ssim(pred, target, data_range=data_range)
        except Exception:
            results['ssim'] = float('nan')

    results['psnr'] = psnr(pred, target, data_range=data_range)

    # Trajectory-aware metrics
    if trajectory_mask is not None:
        results['rmse_observed'] = rmse_observed(pred, target, trajectory_mask)
        results['rmse_unobserved'] = rmse_unobserved(pred, target, trajectory_mask)

        # Coverage metrics
        coverage_density = compute_coverage_density(trajectory_mask)
        results['rmse_coverage_weighted'] = coverage_weighted_rmse(
            pred, target, coverage_density, inverse=True
        )

        # Coverage statistics
        results['coverage_ratio'] = float(np.mean(trajectory_mask))

        # Distance-based metrics
        dist_metrics = rmse_by_distance(pred, target, trajectory_mask)
        results.update(dist_metrics)

    # Uncertainty metrics
    if pred_std is not None:
        errors = np.abs(pred - target)
        calib = uncertainty_calibration(pred, pred_std, target)
        results['calibration_error'] = calib['calibration_error']
        results['fraction_within_1std'] = calib['fraction_within_1std']
        results['fraction_within_2std'] = calib['fraction_within_2std']
        results['uncertainty_error_correlation'] = uncertainty_error_correlation(pred_std, errors)

    return results


# =============================================================================
# PyTorch Compatibility
# =============================================================================

if TORCH_AVAILABLE:
    def torch_rmse(
        pred: 'torch.Tensor',
        target: 'torch.Tensor',
        mask: Optional['torch.Tensor'] = None
    ) -> 'torch.Tensor':
        """
        PyTorch-compatible RMSE for use in training.

        Args:
            pred: Predicted tensor
            target: Target tensor
            mask: Optional mask tensor

        Returns:
            RMSE as a scalar tensor (differentiable)
        """
        diff = pred - target

        if mask is not None:
            mask = mask.bool()
            diff = diff[mask]

        return torch.sqrt(torch.mean(diff ** 2))

    def torch_masked_mse_loss(
        pred: 'torch.Tensor',
        target: 'torch.Tensor',
        mask: Optional['torch.Tensor'] = None,
        reduction: str = 'mean'
    ) -> 'torch.Tensor':
        """
        Masked MSE loss for training.

        Args:
            pred: Predicted tensor
            target: Target tensor
            mask: Optional mask (1 = include, 0 = exclude)
            reduction: 'mean', 'sum', or 'none'

        Returns:
            Loss tensor
        """
        diff_sq = (pred - target) ** 2

        if mask is not None:
            diff_sq = diff_sq * mask
            if reduction == 'mean':
                return diff_sq.sum() / (mask.sum() + 1e-8)
            elif reduction == 'sum':
                return diff_sq.sum()
            else:
                return diff_sq
        else:
            if reduction == 'mean':
                return diff_sq.mean()
            elif reduction == 'sum':
                return diff_sq.sum()
            else:
                return diff_sq
