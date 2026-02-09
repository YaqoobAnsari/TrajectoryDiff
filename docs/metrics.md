# Evaluation Metrics

This document describes the evaluation metrics used in TrajectoryDiff.

## Key Insight

For trajectory-conditioned radio map generation, the **most important metric is RMSE on unobserved regions**. This measures how well the model extrapolates into blind spots — areas without trajectory samples.

## Metrics Overview

### Basic Metrics

| Metric | Formula | Range | Better |
|--------|---------|-------|--------|
| **RMSE** | √(mean((pred - target)²)) | [0, ∞) | Lower |
| **MAE** | mean(\|pred - target\|) | [0, ∞) | Lower |
| **MSE** | mean((pred - target)²) | [0, ∞) | Lower |
| **PSNR** | 10·log₁₀(range²/MSE) | [0, ∞) dB | Higher |
| **SSIM** | Structural similarity | [-1, 1] | Higher |

### Trajectory-Aware Metrics (Our Focus)

| Metric | Description | Why It Matters |
|--------|-------------|----------------|
| **RMSE_observed** | RMSE on pixels with trajectory samples | Interpolation quality |
| **RMSE_unobserved** | RMSE on pixels without samples | **Extrapolation quality (KEY)** |
| **RMSE_by_distance** | RMSE binned by distance to nearest sample | Error growth with distance |
| **Coverage_weighted_RMSE** | Higher weight on low-coverage regions | Emphasizes blind spots |

### Uncertainty Metrics

| Metric | Expected Value | Description |
|--------|----------------|-------------|
| **fraction_within_1std** | ~0.68 | Predictions within ±1σ |
| **fraction_within_2std** | ~0.95 | Predictions within ±2σ |
| **calibration_error** | ~0 | Deviation from ideal calibration |
| **uncertainty_correlation** | >0.5 | Correlation between σ and \|error\| |

## Usage

```python
from src.evaluation.metrics import compute_all_metrics

# Basic usage
metrics = compute_all_metrics(pred, target)
print(f"RMSE: {metrics['rmse']:.4f}")

# With trajectory mask (recommended)
metrics = compute_all_metrics(pred, target, trajectory_mask=mask)
print(f"RMSE (observed): {metrics['rmse_observed']:.4f}")
print(f"RMSE (unobserved): {metrics['rmse_unobserved']:.4f}")  # KEY METRIC

# With uncertainty
metrics = compute_all_metrics(pred, target, pred_std=std)
print(f"Calibration error: {metrics['calibration_error']:.4f}")
```

### dBm-Scale Metrics (for papers)

The evaluation script (`scripts/evaluate.py`) denormalizes predictions from [-1, 1] back to dBm before computing RMSE and MAE. This produces paper-ready metrics:

```python
from src.training.inference import denormalize_radio_map

# Denormalize: [-1, 1] -> [0, 255] -> dBm
samples_dbm = denormalize_radio_map(samples)
gt_dbm = denormalize_radio_map(ground_truth)

# RMSE/MAE in dBm scale (meaningful for radio propagation)
rmse_dbm = compute_rmse(samples_dbm, gt_dbm)

# PSNR/SSIM stay in normalized space (standard practice)
psnr = compute_psnr(samples, ground_truth)
```

## Interpretation Guide

### RMSE Ranges (for pathloss in dB)

| RMSE (dB) | Quality |
|-----------|---------|
| < 2 | Excellent |
| 2-5 | Good |
| 5-10 | Moderate |
| > 10 | Poor |

### Key Comparisons

1. **RMSE_unobserved vs RMSE_observed**:
   - If unobserved >> observed: Model struggles with extrapolation
   - If unobserved ≈ observed: Good generalization

2. **TrajectoryDiff vs Baselines**:
   - Our hypothesis: TrajectoryDiff should excel on RMSE_unobserved
   - Uniform-sampling methods should struggle on trajectory-sampled data

3. **Distance-based RMSE**:
   - Error should increase with distance from samples
   - Trajectory-aware models should show slower increase

## Implementation Details

### Distance-Based RMSE

Uses scipy's `distance_transform_edt` to compute distance from each pixel to the nearest trajectory sample. Default bins: [0-5, 5-10, 10-20, 20-50, 50+] pixels.

### Coverage Density

Gaussian smoothing of trajectory mask:
```python
coverage = gaussian_filter(trajectory_mask, sigma=5.0)
```

Higher sigma → more smoothing → broader "observed" regions.

### Uncertainty Calibration

Uses z-scores: `z = |pred - target| / pred_std`

A well-calibrated model should have:
- P(z ≤ 1) ≈ 0.68 (1 standard deviation)
- P(z ≤ 2) ≈ 0.95 (2 standard deviations)

---

*See `src/evaluation/metrics.py` for implementation details.*
