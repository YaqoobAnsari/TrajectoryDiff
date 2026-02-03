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
| **NMSE** | MSE / var(target) | [0, ∞) | Lower |

**Note:** Published SOTA on RadioMapSeer (RadioUNet, RadioDiff, RMDM) reports NMSE and normalized RMSE. Our evaluation computes both normalized-space and dBm-scale metrics. SSIM is computed on dBm scale with skimage sliding window (fixed in CVPR audit).

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

2. **TrajectoryDiff vs Classical Baselines**:
   - Our hypothesis: TrajectoryDiff should excel on RMSE_unobserved
   - Uniform-sampling methods should struggle on trajectory-sampled data
   - Fair comparison: free-space RMSE only (classical baselines lack building map)

3. **TrajectoryDiff vs DL Baselines**:
   - vs Supervised UNet: Shows value of diffusion process (same arch, same conditioning)
   - vs RadioUNet: Shows value of learned ConditionEncoder + coverage density + trajectory mask
   - vs RMDM: Shows value of coverage-aware attention vs anchor fusion approach
   - All DL baselines receive building map — all-pixel RMSE is fair between DL methods

4. **Distance-based RMSE**:
   - Error should increase with distance from samples
   - Trajectory-aware models should show slower increase

## Fair Evaluation Methodology (Paper-Critical)

### Information Available to Each Method

| Method | Building Map | Sparse RSS | Trajectory Mask | Coverage Density | TX Position |
|--------|:-----------:|:----------:|:--------------:|:----------------:|:-----------:|
| Classical (IDW, RBF, NN) | No | Yes | Yes | No | No |
| RadioUNet | Yes (as input channel) | Yes | Yes | No | Yes (distance map) |
| Supervised UNet | Yes | Yes | Yes | Yes | Yes |
| RMDM | Yes (via conductor) | Yes | Yes | Yes | Yes |
| TrajectoryDiff (ours) | Yes | Yes | Yes | Yes | Yes |

### The Building Pixel Problem

Classical baselines (IDW, RBF, NN) do NOT receive the building map as input — they only get sparse trajectory observations. The diffusion model and DL baselines receive the building map as a conditioning input. This creates an unfair advantage when evaluating over all pixels:

- **~70% of pixels are buildings** where baselines have zero information
- **Ground truth in buildings**: ~-144 dBm (moderate signal from ray-tracing penetration)
- **Baseline predictions in buildings**: ~-184 dBm (extrapolated from weak free-space observations)
- **Result**: ~40 dBm error in buildings dominates the all-pixel RMSE

### Per-Region Metrics (required for fair comparison)

| Metric | What It Measures | Fair? |
|--------|-----------------|-------|
| **RMSE (all pixels)** | Full image reconstruction quality | Unfair — building map advantage |
| **RMSE (free-space only)** | Interpolation/extrapolation in walkable areas | **Fair** — same info available |
| **RMSE (building only)** | Building interior estimation | Shows building map value |
| **RMSE (observed)** | Accuracy at trajectory points | Baseline sanity check |
| **RMSE (unobserved)** | Extrapolation quality | Key research metric |
| **RMSE (free-space unobserved)** | Extrapolation in walkable blind spots | **Fairest comparison** |

### For the Paper

1. **Primary comparison metric**: Free-space RMSE (both methods have same information)
2. **Key research metric**: Free-space unobserved RMSE (extrapolation into blind spots)
3. **Report all-pixel RMSE** too, but note the building map advantage in text
4. **Building RMSE** shows the value of incorporating building maps (our advantage)
5. **Do NOT compare** with published SOTA (RMDM, RadioDiff) — they use dense observation, completely different protocol

### Coverage Context

- Trajectory coverage: ~300 points / 65,536 pixels = **~0.5% of total map**
- Free-space coverage: ~300 / ~19,000 free pixels = **~1.6% of walkable area**
- This is far sparser than typical radio map prediction papers (which use 1-10% uniform)

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
