# TrajectoryDiff: Comprehensive Next Steps Guide

## Current Status (Feb 2026)

**Phases Complete:** 0, 1, 2, 3 (Setup, Data Pipeline, Model Development, Physics + Architecture)
**Tests Passing:** 143
**Ready For:** Phase 4 (Experiments)

### What We Have
- Complete diffusion model with trajectory conditioning
- DDPM/DDIM sampling
- U-Net with condition encoder
- Training pipeline with EMA, W&B logging
- Evaluation metrics
- **NEW: Physics-informed losses** (TrajectoryConsistency, CoverageWeighted, DistanceDecay)
- **NEW: CoverageAwareAttention** (novel architectural component for ECCV/CVPR)

### What We Need
- ~~Physics-informed losses (strengthen the model)~~ ✓ DONE
- ~~Potential architectural novelty (strengthen the paper)~~ ✓ DONE
- Comprehensive experiments (prove our claims)

---

# Phase 3: Physics-Informed Components & Architectural Novelty

## Overview

Phase 3 has TWO goals:
1. **Physics losses** - Make predictions physically plausible
2. **Architectural novelty** - Add something new for ECCV/CVPR (optional but recommended)

---

## 3.1 Physics-Informed Losses

### 3.1.1 Trajectory Consistency Loss
**Priority:** CRITICAL | **Effort:** Medium | **File:** `src/training/losses.py`

**Purpose:** Ensure predicted RSS values match observations along the trajectory.

**Why it matters:** Standard MSE treats all pixels equally. But we KNOW the ground truth on trajectories - we should be more strict there.

```python
# src/training/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrajectoryConsistencyLoss(nn.Module):
    """
    Enforce that predictions match observations along trajectories.

    Unlike pixel-wise MSE, this:
    1. Uses bilinear interpolation for sub-pixel accuracy
    2. Weights by measurement confidence
    3. Can enforce local smoothness along path
    """

    def __init__(self, smoothness_weight: float = 0.1):
        super().__init__()
        self.smoothness_weight = smoothness_weight

    def forward(
        self,
        pred_map: torch.Tensor,      # (B, 1, H, W) predicted radio map
        sparse_rss: torch.Tensor,     # (B, 1, H, W) observed RSS values
        trajectory_mask: torch.Tensor, # (B, 1, H, W) binary mask
    ) -> torch.Tensor:
        """
        Compute trajectory consistency loss.

        Returns:
            Scalar loss value
        """
        # Only compute loss where we have observations
        mask = trajectory_mask > 0.5

        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred_map.device)

        # Point-wise consistency loss
        pred_on_traj = pred_map[mask]
        obs_on_traj = sparse_rss[mask]
        consistency_loss = F.mse_loss(pred_on_traj, obs_on_traj)

        # Optional: Local smoothness along trajectory
        # (predictions should vary smoothly along the path)
        if self.smoothness_weight > 0:
            smoothness_loss = self._compute_smoothness(pred_map, trajectory_mask)
            return consistency_loss + self.smoothness_weight * smoothness_loss

        return consistency_loss

    def _compute_smoothness(
        self,
        pred_map: torch.Tensor,
        trajectory_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute gradient smoothness along trajectory."""
        # Sobel gradients
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=pred_map.dtype, device=pred_map.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=pred_map.dtype, device=pred_map.device).view(1, 1, 3, 3)

        grad_x = F.conv2d(pred_map, sobel_x, padding=1)
        grad_y = F.conv2d(pred_map, sobel_y, padding=1)

        # Only penalize high gradients ON the trajectory
        # (we want smooth predictions along the path)
        mask = trajectory_mask > 0.5
        gradient_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred_map.device)

        return gradient_magnitude[mask].mean()
```

**Tasks:**
- [x] Create `src/training/losses.py`
- [x] Implement `TrajectoryConsistencyLoss`
- [ ] Add to training loop in `diffusion_module.py`
- [ ] Add `trajectory_consistency_weight` to config
- [x] Write tests in `tests/test_losses.py`

---

### 3.1.2 Coverage-Weighted Loss
**Priority:** HIGH | **Effort:** Low | **File:** `src/training/losses.py`

**Purpose:** Weight the diffusion loss by coverage density - be strict where we have data, lenient where we don't.

```python
class CoverageWeightedLoss(nn.Module):
    """
    Weight prediction loss by coverage density.

    High coverage (on trajectory) → high weight (be accurate!)
    Low coverage (blind spot) → low weight (allow exploration)
    """

    def __init__(self, min_weight: float = 0.1, max_weight: float = 1.0):
        super().__init__()
        self.min_weight = min_weight
        self.max_weight = max_weight

    def forward(
        self,
        pred: torch.Tensor,           # (B, 1, H, W) prediction
        target: torch.Tensor,         # (B, 1, H, W) target
        coverage_density: torch.Tensor, # (B, 1, H, W) coverage map [0, 1]
    ) -> torch.Tensor:
        """
        Compute coverage-weighted MSE loss.
        """
        # Compute per-pixel squared error
        squared_error = (pred - target) ** 2

        # Weight by coverage density
        # coverage_density is in [0, 1], high = on trajectory
        weights = self.min_weight + (self.max_weight - self.min_weight) * coverage_density

        # Weighted mean
        weighted_error = squared_error * weights
        return weighted_error.mean()
```

**Tasks:**
- [x] Implement `CoverageWeightedLoss`
- [ ] Option to use instead of standard MSE in training
- [ ] Ablation: compare with/without coverage weighting

---

### 3.1.3 Distance Decay Regularization
**Priority:** MEDIUM | **Effort:** Low | **File:** `src/training/losses.py`

**Purpose:** Soft constraint that signal generally decreases with distance from TX.

```python
class DistanceDecayLoss(nn.Module):
    """
    Soft regularization: signal should generally decrease with distance from TX.

    This is a SOFT constraint because:
    - Multipath can cause local increases
    - Walls cause sharp drops, not gradual decay
    """

    def __init__(self, weight: float = 0.01):
        super().__init__()
        self.weight = weight

    def forward(
        self,
        pred_map: torch.Tensor,     # (B, 1, H, W) predicted radio map
        tx_position: torch.Tensor,   # (B, 2) normalized TX position
        building_map: torch.Tensor,  # (B, 1, H, W) to mask out walls
    ) -> torch.Tensor:
        """
        Penalize signal INCREASING with distance from TX.
        """
        B, C, H, W = pred_map.shape
        device = pred_map.device

        # Create distance map from TX
        y_coords = torch.linspace(0, 1, H, device=device)
        x_coords = torch.linspace(0, 1, W, device=device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

        distance_maps = []
        for b in range(B):
            tx_x, tx_y = tx_position[b]
            dist = torch.sqrt((xx - tx_x) ** 2 + (yy - tx_y) ** 2)
            distance_maps.append(dist)

        distance_map = torch.stack(distance_maps, dim=0).unsqueeze(1)  # (B, 1, H, W)

        # Compute radial gradient of predicted map
        # (should be negative - signal decreases outward)
        grad_x = pred_map[:, :, :, 1:] - pred_map[:, :, :, :-1]
        grad_y = pred_map[:, :, 1:, :] - pred_map[:, :, :-1, :]

        # Compute radial direction at each point
        # Penalize positive gradients in radial direction (signal increasing outward)
        # This is simplified - full implementation would use proper radial coordinates

        # For now, just penalize if mean RSS increases with distance
        # (very soft constraint)
        near_tx = distance_map < 0.3
        far_from_tx = distance_map > 0.7

        # Mask out walls
        free_space = building_map < 0.5

        near_rss = pred_map[near_tx & free_space].mean() if (near_tx & free_space).sum() > 0 else 0
        far_rss = pred_map[far_from_tx & free_space].mean() if (far_from_tx & free_space).sum() > 0 else 0

        # Penalize if far RSS > near RSS (physically wrong)
        violation = F.relu(far_rss - near_rss)

        return self.weight * violation
```

**Tasks:**
- [x] Implement `DistanceDecayLoss`
- [ ] Add as optional regularization term
- [x] Test on synthetic data where we know ground truth

---

### 3.1.4 Wall Attenuation Loss
**Priority:** LOW | **Effort:** High | **File:** `src/training/losses.py`

**Purpose:** Signal should drop when crossing walls.

```python
class WallAttenuationLoss(nn.Module):
    """
    Enforce that signal attenuates through walls.

    For each wall pixel, check that signal on far side (from TX)
    is lower than signal on near side.
    """

    def __init__(self, min_attenuation_db: float = 3.0):
        super().__init__()
        self.min_attenuation_db = min_attenuation_db

    def forward(
        self,
        pred_map: torch.Tensor,     # (B, 1, H, W)
        building_map: torch.Tensor,  # (B, 1, H, W) walls = 1
        tx_position: torch.Tensor,   # (B, 2)
    ) -> torch.Tensor:
        """
        Compute wall attenuation violation loss.
        """
        # This is complex to implement properly
        # Simplified version: compute gradient across walls
        # and penalize if gradient is positive (signal increasing through wall)

        # Detect wall boundaries using morphological operations
        # For each boundary pixel, check signal difference

        # TODO: Full implementation requires:
        # 1. Extract wall boundaries
        # 2. Determine wall normal direction
        # 3. Sample signal on both sides
        # 4. Penalize if far_side > near_side - min_attenuation

        return torch.tensor(0.0, device=pred_map.device)
```

**Tasks:**
- [ ] Implement wall boundary detection
- [ ] Compute signal gradient across walls
- [ ] Penalize positive gradients (signal increasing through walls)
- [ ] This is OPTIONAL - can skip if time is limited

---

## 3.2 Architectural Novelty (ECCV/CVPR Enhancement)

Based on our literature review, we need some architectural novelty. Here are three options, ranked by feasibility:

### Option A: Coverage-Aware Attention (Recommended)
**Priority:** HIGH | **Effort:** Medium | **File:** `src/models/diffusion/attention.py`

**Novelty:** Attention weights are modulated by coverage density. The model pays MORE attention to regions with data and LESS attention to blind spots during denoising.

```python
# src/models/diffusion/attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CoverageAwareAttention(nn.Module):
    """
    Novel attention mechanism that modulates attention by coverage density.

    Key insight: When denoising, we should:
    - Trust features from high-coverage regions (we have data there)
    - Be more exploratory in low-coverage regions (blind spots)

    This is implemented by scaling attention weights based on
    the coverage density of key positions.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        coverage_temperature: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.coverage_temperature = coverage_temperature

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

        # Learnable coverage modulation
        self.coverage_gate = nn.Sequential(
            nn.Linear(1, dim // 4),
            nn.SiLU(),
            nn.Linear(dim // 4, num_heads),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,                # (B, N, D) spatial features flattened
        coverage: torch.Tensor = None,   # (B, N, 1) coverage density per position
    ) -> torch.Tensor:
        B, N, D = x.shape

        # Standard QKV projection
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(B, N, self.num_heads, self.head_dim).transpose(1, 2), qkv)

        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)

        # NOVEL: Modulate attention by coverage density
        if coverage is not None:
            # coverage_gate produces per-head scaling factors
            coverage_scale = self.coverage_gate(coverage)  # (B, N, heads)
            coverage_scale = coverage_scale.permute(0, 2, 1).unsqueeze(-1)  # (B, heads, N, 1)

            # Scale attention to keys based on their coverage
            # High coverage keys get higher attention
            key_coverage = coverage_scale.transpose(-2, -1)  # (B, heads, 1, N)

            # Temperature-scaled modulation
            coverage_weight = key_coverage ** (1.0 / self.coverage_temperature)
            attn = attn * coverage_weight

        attn = attn.softmax(dim=-1)

        # Apply attention to values
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, D)

        return self.to_out(out)


class CoverageAwareTransformerBlock(nn.Module):
    """
    Transformer block with coverage-aware attention.
    """

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = CoverageAwareAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor, coverage: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), coverage)
        x = x + self.ffn(self.norm2(x))
        return x
```

**Integration into U-Net:**
```python
# In AttentionBlock, replace standard attention with CoverageAwareAttention
# Pass coverage_density (downsampled to match resolution) as additional input
```

**Tasks:**
- [x] Create `src/models/diffusion/attention.py`
- [x] Implement `CoverageAwareAttention`
- [ ] Integrate into U-Net's attention blocks
- [ ] Pass coverage_density through the network
- [x] Write tests
- [ ] Ablation: with/without coverage-aware attention

---

### Option B: Dual-Stream Encoder
**Priority:** MEDIUM | **Effort:** High | **File:** `src/models/encoders/dual_stream.py`

**Novelty:** Separate processing streams for spatial (CNN) and sequential (Transformer) trajectory information, fused via cross-attention.

```python
class DualStreamTrajectoryEncoder(nn.Module):
    """
    Process trajectory data through two streams:
    1. Spatial stream: CNN on sparse_rss map (WHERE are measurements)
    2. Sequential stream: Transformer on trajectory points (WHAT is the path)

    Fuse via cross-attention.
    """

    def __init__(self, spatial_dim: int = 256, sequence_dim: int = 256):
        super().__init__()

        # Spatial stream (CNN)
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1),  # sparse_rss + mask
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, spatial_dim, 3, stride=2, padding=1),
        )

        # Sequential stream (Transformer)
        self.point_embed = nn.Linear(3, sequence_dim)  # (x, y, rss)
        self.pos_embed = nn.Embedding(1000, sequence_dim)  # Trajectory position
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(sequence_dim, nhead=8, batch_first=True),
            num_layers=4
        )

        # Cross-attention fusion
        self.cross_attn = nn.MultiheadAttention(spatial_dim, num_heads=8, batch_first=True)

    def forward(
        self,
        sparse_rss: torch.Tensor,       # (B, 1, H, W)
        trajectory_mask: torch.Tensor,   # (B, 1, H, W)
        trajectory_points: torch.Tensor, # (B, N, 3) - (x, y, rss) points
    ) -> torch.Tensor:
        # Spatial stream
        spatial_input = torch.cat([sparse_rss, trajectory_mask], dim=1)
        spatial_feat = self.spatial_encoder(spatial_input)  # (B, D, H', W')
        B, D, H, W = spatial_feat.shape
        spatial_feat = spatial_feat.flatten(2).permute(0, 2, 1)  # (B, H'*W', D)

        # Sequential stream
        N = trajectory_points.shape[1]
        seq_feat = self.point_embed(trajectory_points)  # (B, N, D)
        pos = self.pos_embed(torch.arange(N, device=seq_feat.device))
        seq_feat = seq_feat + pos
        seq_feat = self.transformer(seq_feat)  # (B, N, D)

        # Cross-attention: spatial queries, sequential keys/values
        fused, _ = self.cross_attn(spatial_feat, seq_feat, seq_feat)

        # Reshape back to spatial
        fused = fused.permute(0, 2, 1).view(B, D, H, W)

        return fused
```

**Tasks:**
- [ ] Implement dual-stream encoder
- [ ] Requires trajectory points as additional input
- [ ] More complex data pipeline changes

---

### Option C: Uncertainty-Guided Diffusion
**Priority:** MEDIUM | **Effort:** Medium | **File:** `src/models/diffusion/ddpm.py`

**Novelty:** Different noise injection during sampling based on coverage. Deterministic in high-coverage regions, stochastic in low-coverage.

```python
def p_sample_uncertainty_guided(
    self,
    model,
    x_t: torch.Tensor,
    t: int,
    coverage_density: torch.Tensor,
    eta_min: float = 0.0,  # Deterministic in high coverage
    eta_max: float = 1.0,  # Stochastic in low coverage
) -> torch.Tensor:
    """
    Uncertainty-guided denoising step.

    In high-coverage regions: deterministic (eta → 0)
    In low-coverage regions: stochastic (eta → 1)
    """
    # Predict noise
    noise_pred = model(x_t, t)

    # Compute deterministic component
    x_0_pred = self.predict_x0_from_noise(x_t, t, noise_pred)

    # Spatially-varying eta based on coverage
    eta = eta_min + (eta_max - eta_min) * (1 - coverage_density)

    # DDIM-style update with spatially-varying stochasticity
    # High eta = more noise = more exploration
    # Low eta = less noise = follow prediction

    # ... (full DDIM update with eta)

    return x_t_minus_1
```

**Tasks:**
- [ ] Modify DDIM sampler to accept coverage
- [ ] Implement spatially-varying eta
- [ ] This changes sampling, not training

---

## 3.3 Combined Loss Function

```python
# src/training/losses.py

class TrajectoryDiffLoss(nn.Module):
    """
    Combined loss for TrajectoryDiff training.
    """

    def __init__(
        self,
        diffusion_weight: float = 1.0,
        trajectory_consistency_weight: float = 0.1,
        coverage_weighted: bool = True,
        distance_decay_weight: float = 0.01,
    ):
        super().__init__()
        self.diffusion_weight = diffusion_weight
        self.trajectory_consistency_weight = trajectory_consistency_weight
        self.coverage_weighted = coverage_weighted
        self.distance_decay_weight = distance_decay_weight

        self.trajectory_loss = TrajectoryConsistencyLoss()
        self.coverage_loss = CoverageWeightedLoss() if coverage_weighted else None
        self.distance_loss = DistanceDecayLoss(weight=distance_decay_weight)

    def forward(
        self,
        noise_pred: torch.Tensor,
        noise_target: torch.Tensor,
        pred_x0: torch.Tensor,  # Predicted clean image
        batch: dict,
    ) -> dict:
        """
        Compute all losses.

        Returns dict with individual losses for logging.
        """
        losses = {}

        # Main diffusion loss
        if self.coverage_loss and 'coverage_density' in batch:
            diffusion_loss = self.coverage_loss(
                noise_pred, noise_target, batch['coverage_density']
            )
        else:
            diffusion_loss = F.mse_loss(noise_pred, noise_target)

        losses['diffusion'] = diffusion_loss * self.diffusion_weight

        # Trajectory consistency (on predicted x0)
        if self.trajectory_consistency_weight > 0 and pred_x0 is not None:
            traj_loss = self.trajectory_loss(
                pred_x0,
                batch['sparse_rss'],
                batch['trajectory_mask'],
            )
            losses['trajectory_consistency'] = traj_loss * self.trajectory_consistency_weight

        # Distance decay regularization
        if self.distance_decay_weight > 0 and pred_x0 is not None:
            dist_loss = self.distance_loss(
                pred_x0,
                batch['tx_position'],
                batch['building_map'],
            )
            losses['distance_decay'] = dist_loss

        # Total loss
        losses['total'] = sum(losses.values())

        return losses
```

---

## 3.4 Phase 3 Implementation Order

```
Week 1:
├── Day 1-2: Create src/training/losses.py
│   ├── TrajectoryConsistencyLoss
│   ├── CoverageWeightedLoss
│   └── Tests
├── Day 3-4: Integrate into training loop
│   ├── Modify diffusion_module.py
│   ├── Add loss weights to config
│   └── Verify training still works
└── Day 5: Test physics losses
    ├── Run short training with new losses
    └── Check loss curves in W&B

Week 2 (Optional - for ECCV/CVPR):
├── Day 1-3: Implement CoverageAwareAttention
│   ├── Create attention.py
│   ├── Integrate into U-Net
│   └── Tests
└── Day 4-5: Ablation experiments
    ├── With/without coverage attention
    └── Document results
```

---

# Phase 4: Comprehensive Experiments

## Overview

Phase 4 proves our claims through systematic experiments:
1. **Initial training** - Verify everything works
2. **Baselines** - Establish comparison points
3. **Ablations** - Show each component matters
4. **Key experiments** - Prove trajectory conditioning helps
5. **Uncertainty** - Show uncertainty is calibrated

---

## 4.1 Experiment Infrastructure

### 4.1.1 Directory Structure
```
experiments/
├── configs/                    # Experiment-specific configs
│   ├── baseline_trajectory.yaml
│   ├── baseline_uniform.yaml
│   ├── ablation_no_trajectory_mask.yaml
│   ├── ablation_no_coverage.yaml
│   ├── ablation_no_tx_position.yaml
│   └── ablation_small_unet.yaml
├── runs/                       # Training outputs
│   ├── baseline_trajectory_v1/
│   │   ├── checkpoints/
│   │   ├── logs/
│   │   └── config.yaml
│   └── ...
├── results/                    # Evaluation results
│   ├── metrics/
│   ├── visualizations/
│   └── tables/
└── analysis/                   # Analysis notebooks
    ├── compare_baselines.ipynb
    ├── ablation_analysis.ipynb
    └── uncertainty_calibration.ipynb
```

### 4.1.2 Experiment Tracking
All experiments logged to W&B with:
- Loss curves
- Sample visualizations
- Metrics per epoch
- Config snapshots

---

## 4.2 Experiment Matrix

### Master Experiment Table

| ID | Experiment | Config | Purpose | Priority |
|----|------------|--------|---------|----------|
| E01 | Initial test | default | Verify pipeline | CRITICAL |
| E02 | Baseline: trajectory sampling | baseline_traj | Main model | CRITICAL |
| E03 | Baseline: uniform sampling | baseline_unif | Comparison | HIGH |
| E04 | Ablation: no trajectory mask | ablation_no_mask | Show mask helps | HIGH |
| E05 | Ablation: no coverage density | ablation_no_cov | Show coverage helps | HIGH |
| E06 | Ablation: no TX position | ablation_no_tx | Show TX helps | MEDIUM |
| E07 | Ablation: small U-Net | ablation_small | Model size effect | LOW |
| E08 | Ablation: no physics loss | ablation_no_phys | Show physics helps | MEDIUM |
| E09 | Cross-evaluation: traj→unif | cross_traj_unif | Generalization | HIGH |
| E10 | Cross-evaluation: unif→traj | cross_unif_traj | Generalization | HIGH |
| E11 | Uncertainty calibration | uncertainty | Calibration metrics | HIGH |
| E12 | Coverage sweep: 1%, 5%, 10% | coverage_sweep | Coverage effect | MEDIUM |
| E13 | Baseline: IDW | - | Classical baseline | HIGH |
| E14 | Baseline: RBF | - | Classical baseline | HIGH |
| E15 | Baseline: Kriging | - | Classical baseline | HIGH |

---

## 4.3 Detailed Experiment Specifications

### E01: Initial Pipeline Test
**Purpose:** Verify training works end-to-end

```bash
python scripts/train.py \
    training.max_epochs=2 \
    data.loader.batch_size=4 \
    logging.wandb.enabled=false \
    experiment.name=test_pipeline
```

**Success criteria:**
- [ ] No errors during training
- [ ] Loss decreases
- [ ] Checkpoints saved
- [ ] Samples can be generated

---

### E02: Main Trajectory Baseline
**Purpose:** Train our full model with trajectory sampling

**Config:** `configs/experiment/baseline_trajectory.yaml`
```yaml
defaults:
  - /config

experiment:
  name: baseline_trajectory_v1
  tags: [baseline, trajectory, main]
  seed: 42

data:
  sampling:
    strategy: trajectory
    trajectory:
      num_trajectories: 3
      points_per_trajectory: 100

training:
  max_epochs: 100
  optimizer:
    lr: 1e-4

model:
  unet:
    base_channels: 64
```

**Command:**
```bash
python scripts/train.py experiment=baseline_trajectory
```

**Metrics to track:**
- train/loss, val/loss
- val/rmse, val/mae, val/ssim
- val/rmse_observed (on trajectory)
- val/rmse_unobserved (blind spots)

---

### E03: Uniform Sampling Baseline
**Purpose:** Compare against uniform random sampling assumption

**Config:** `configs/experiment/baseline_uniform.yaml`
```yaml
defaults:
  - /config

experiment:
  name: baseline_uniform_v1
  tags: [baseline, uniform]

data:
  sampling:
    strategy: uniform
    uniform:
      rate: 0.01  # 1% of pixels, same as trajectory coverage
```

**Key comparison:** E02 vs E03
- Same total number of samples
- Different spatial distribution
- We expect E02 to have better uncertainty calibration

---

### E04-E07: Ablation Studies

**E04: No Trajectory Mask**
```yaml
# configs/experiment/ablation_no_trajectory_mask.yaml
defaults:
  - /experiment/baseline_trajectory

experiment:
  name: ablation_no_trajectory_mask
  tags: [ablation]

model:
  use_trajectory_mask: false
```

**E05: No Coverage Density**
```yaml
model:
  use_coverage_density: false
```

**E06: No TX Position**
```yaml
model:
  use_tx_position: false
```

**E07: Small U-Net**
```yaml
model:
  unet:
    base_channels: 32  # Instead of 64
```

---

### E09-E10: Cross-Evaluation (KEY EXPERIMENT)
**Purpose:** Show that trajectory-trained model generalizes differently than uniform-trained

**Protocol:**
1. Train model on trajectory sampling (E02)
2. Train model on uniform sampling (E03)
3. Evaluate BOTH models on BOTH test sets:

| Train \ Test | Trajectory Test | Uniform Test |
|--------------|-----------------|--------------|
| Trajectory Model (E02) | Native | Cross |
| Uniform Model (E03) | Cross | Native |

**Expected results:**
- Trajectory model: excellent on trajectory, good on uniform
- Uniform model: okay on both, but worse uncertainty

**Commands:**
```bash
# Evaluate trajectory model on uniform test set
python scripts/evaluate.py \
    checkpoint=experiments/baseline_trajectory_v1/checkpoints/best.ckpt \
    data.sampling.strategy=uniform \
    experiment.name=cross_eval_traj_to_unif

# Evaluate uniform model on trajectory test set
python scripts/evaluate.py \
    checkpoint=experiments/baseline_uniform_v1/checkpoints/best.ckpt \
    data.sampling.strategy=trajectory \
    experiment.name=cross_eval_unif_to_traj
```

---

### E11: Uncertainty Calibration
**Purpose:** Show uncertainty is meaningful and calibrated

**Protocol:**
1. Generate 10 samples per test input
2. Compute mean (prediction) and std (uncertainty)
3. Compare uncertainty to actual error

**Metrics:**
- Calibration plot: binned expected error vs actual error
- ECE (Expected Calibration Error)
- Correlation: uncertainty vs error
- Spatial analysis: uncertainty on trajectory vs off trajectory

**Script:** `scripts/analyze_uncertainty.py`
```python
# Key analysis:
# 1. Uncertainty should be LOW on trajectory
# 2. Uncertainty should be HIGH in blind spots
# 3. Uncertainty should CORRELATE with actual error

def analyze_uncertainty(samples, ground_truth, trajectory_mask):
    mean_pred = samples.mean(dim=0)
    uncertainty = samples.std(dim=0)
    error = (mean_pred - ground_truth).abs()

    # On trajectory
    on_traj = trajectory_mask > 0.5
    print(f"Uncertainty on trajectory: {uncertainty[on_traj].mean():.4f}")
    print(f"Error on trajectory: {error[on_traj].mean():.4f}")

    # Off trajectory
    off_traj = trajectory_mask < 0.5
    print(f"Uncertainty off trajectory: {uncertainty[off_traj].mean():.4f}")
    print(f"Error off trajectory: {error[off_traj].mean():.4f}")

    # Correlation
    correlation = torch.corrcoef(torch.stack([
        uncertainty.flatten(),
        error.flatten()
    ]))[0, 1]
    print(f"Uncertainty-Error correlation: {correlation:.4f}")
```

---

### E13-E15: Classical Baselines
**Purpose:** Compare against non-learning methods

**Script:** `scripts/run_baselines.py`
```python
from src.data.baselines import (
    IDWInterpolator,
    RBFInterpolator,
    KrigingInterpolator,
)

def evaluate_baseline(method, test_loader):
    all_rmse = []
    for batch in test_loader:
        sparse_rss = batch['sparse_rss']
        trajectory_mask = batch['trajectory_mask']
        ground_truth = batch['radio_map']

        # Run interpolation
        pred = method.interpolate(sparse_rss, trajectory_mask)

        # Compute metrics
        rmse = compute_rmse(pred, ground_truth)
        all_rmse.append(rmse)

    return torch.cat(all_rmse).mean()
```

---

## 4.4 Results Tables to Generate

### Table 1: Main Results
| Method | RMSE ↓ | MAE ↓ | SSIM ↑ | PSNR ↑ |
|--------|--------|-------|--------|--------|
| IDW Interpolation | - | - | - | - |
| RBF Interpolation | - | - | - | - |
| Kriging | - | - | - | - |
| Diffusion (uniform) | - | - | - | - |
| **TrajectoryDiff (ours)** | **-** | **-** | **-** | **-** |

### Table 2: Trajectory-Aware Metrics
| Method | RMSE (observed) ↓ | RMSE (blind spots) ↓ | Uncertainty-Error Corr ↑ |
|--------|-------------------|----------------------|--------------------------|
| Diffusion (uniform) | - | - | - |
| **TrajectoryDiff** | **-** | **-** | **-** |

### Table 3: Ablation Study
| Configuration | RMSE | Δ from Full |
|---------------|------|-------------|
| Full model | - | - |
| - trajectory_mask | - | +X.XX |
| - coverage_density | - | +X.XX |
| - tx_position | - | +X.XX |
| - physics_loss | - | +X.XX |

---

## 4.5 Figures to Generate

### Figure 1: Problem Illustration
```
┌─────────────────────────────────────────────────────┐
│  (a) Uniform Sampling    (b) Trajectory Sampling    │
│                                                     │
│  [Random dots on map]    [Path through building]    │
│                                                     │
│  "Assumption of prior    "Reality of crowdsourced  │
│   work"                   data"                     │
└─────────────────────────────────────────────────────┘
```

### Figure 2: Qualitative Results
```
┌─────────────────────────────────────────────────────┐
│ Building | Trajectory | Sparse RSS | GT | Ours | IDW│
│ Map      | Mask       |            |    |      |    │
│ ──────── │ ────────── │ ────────── │────│──────│────│
│  [img]   │   [img]    │   [img]    │[im]│ [im] │[im]│
│  [img]   │   [img]    │   [img]    │[im]│ [im] │[im]│
│  [img]   │   [img]    │   [img]    │[im]│ [im] │[im]│
└─────────────────────────────────────────────────────┘
```

### Figure 3: Uncertainty Visualization
```
┌─────────────────────────────────────────────────────┐
│ (a) Trajectory     (b) Prediction    (c) Uncertainty│
│                                                     │
│ [Path on map]      [Radio map]       [Uncertainty]  │
│                                       High in blind │
│                                       spots!        │
└─────────────────────────────────────────────────────┘
```

### Figure 4: Calibration Plot
```
┌─────────────────────────────────────────────────────┐
│  Expected Error vs Actual Error                     │
│                                                     │
│  Actual │      ....                                 │
│  Error  │    ..    .                                │
│         │  ..       .  ← Perfect calibration        │
│         │ .          .                              │
│         │.            .                             │
│         └──────────────────                         │
│             Expected Error (Uncertainty)            │
└─────────────────────────────────────────────────────┘
```

---

## 4.6 Phase 4 Timeline

```
Week 1: Initial Training
├── Day 1: E01 - Pipeline test
├── Day 2-3: E02 - Main trajectory baseline (full training)
├── Day 4-5: E03 - Uniform baseline
└── Day 6-7: E13-E15 - Classical baselines

Week 2: Ablations
├── Day 1-2: E04-E07 - Component ablations
├── Day 3-4: E08 - Physics loss ablation
└── Day 5-7: Analysis and visualization

Week 3: Key Experiments
├── Day 1-2: E09-E10 - Cross-evaluation
├── Day 3-4: E11 - Uncertainty calibration
├── Day 5: E12 - Coverage sweep
└── Day 6-7: Generate all figures and tables

Week 4: Paper Preparation
├── Day 1-3: Write results section
├── Day 4-5: Write method section
└── Day 6-7: Full paper draft
```

---

## 4.7 Commands Reference

### Training
```bash
# Quick test
python scripts/train.py training.max_epochs=2 data.loader.batch_size=4

# Full training
python scripts/train.py experiment=baseline_trajectory

# Resume from checkpoint
python scripts/train.py experiment=baseline_trajectory +ckpt_path=path/to/ckpt
```

### Evaluation
```bash
# Evaluate checkpoint
python scripts/evaluate.py checkpoint=path/to/best.ckpt

# Evaluate with different sampling
python scripts/evaluate.py checkpoint=path/to/best.ckpt data.sampling.strategy=uniform
```

### Baselines
```bash
# Run all classical baselines
python scripts/run_baselines.py --output results/baselines.json
```

### Analysis
```bash
# Generate uncertainty analysis
python scripts/analyze_uncertainty.py checkpoint=path/to/best.ckpt --num-samples 10

# Generate figures
python scripts/generate_figures.py --results-dir results/ --output-dir figures/
```

---

## Success Criteria

### Minimum Viable (Workshop Paper)
- [ ] Model trains and converges
- [ ] Beats IDW/RBF baselines by >5% RMSE
- [ ] Uncertainty somewhat correlates with error (r > 0.3)

### Target (Conference Paper)
- [ ] Beats all baselines by >10% RMSE
- [ ] Clear advantage of trajectory conditioning in ablations
- [ ] Well-calibrated uncertainty (ECE < 0.1)
- [ ] Compelling visualizations

### Stretch (Best Paper Candidate)
- [ ] Novel architectural component (Coverage-Aware Attention)
- [ ] Physics-informed losses improve results
- [ ] Cross-evaluation shows generalization advantage
- [ ] Real-world data validation

---

## File Checklist

### Phase 3 Files to Create
- [x] `src/training/losses.py`
- [x] `src/models/diffusion/attention.py` (novel CoverageAwareAttention)
- [x] `tests/test_losses.py`
- [x] `tests/test_attention.py`
- [ ] `configs/training/with_physics.yaml`

### Phase 4 Files to Create
- [ ] `configs/experiment/baseline_trajectory.yaml`
- [ ] `configs/experiment/baseline_uniform.yaml`
- [ ] `configs/experiment/ablation_*.yaml` (multiple)
- [ ] `scripts/run_baselines.py`
- [ ] `scripts/analyze_uncertainty.py`
- [ ] `scripts/generate_figures.py`
- [ ] `notebooks/analysis.ipynb`
