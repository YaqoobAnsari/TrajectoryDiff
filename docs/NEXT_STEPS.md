# TrajectoryDiff: Next Steps Guide

## Current Status (Feb 2026)

**Phases Complete:** 0, 1, 2 (Setup, Data Pipeline, Model Development)
**Tests Passing:** 101
**Ready For:** Initial training experiments

---

## Phase 3: Physics-Informed Components

### 3.1 Trajectory Consistency Loss
**Priority:** HIGH | **Effort:** Medium

The model should produce RSS values along the trajectory that match observations.

**Implementation:**
```python
# Location: src/training/losses.py

def trajectory_consistency_loss(pred_map, trajectory_points, trajectory_mask):
    """
    Penalize deviation between predicted and observed RSS along trajectory.

    This is STRONGER than pixel-wise MSE because:
    1. We interpolate between trajectory points
    2. We enforce smoothness along the path
    3. We weight by measurement confidence
    """
    # Sample predicted RSS at trajectory locations
    # Compare with observed RSS
    # Return weighted loss
```

**Tasks:**
- [ ] Create `src/training/losses.py` module
- [ ] Implement bilinear sampling for sub-pixel trajectory points
- [ ] Add loss weight hyperparameter to config
- [ ] Test with synthetic trajectories
- [ ] Ablation: compare with/without this loss

### 3.2 Wall Attenuation Prior
**Priority:** MEDIUM | **Effort:** High

Signal should attenuate when crossing walls.

**Implementation:**
```python
def wall_attenuation_loss(pred_map, building_map, tx_position):
    """
    Enforce physical constraint: RSS drops across walls.

    For each wall pixel:
    - Compute gradient of pred_map perpendicular to wall
    - Gradient should be negative (signal decreases through wall)
    """
```

**Tasks:**
- [ ] Extract wall boundaries from building_map
- [ ] Compute signal gradients across walls
- [ ] Implement soft penalty for wrong gradient direction
- [ ] Tune attenuation magnitude based on wall material (if available)

### 3.3 Distance Decay Regularization
**Priority:** LOW | **Effort:** Low

Signal generally decreases with distance from transmitter.

**Tasks:**
- [ ] Compute distance map from TX position
- [ ] Add soft regularization penalizing RSS increase with distance
- [ ] Make it weak (multipath can cause local increases)

---

## Phase 4: Training & Experiments

### 4.1 Initial Training Run
**Priority:** CRITICAL | **Effort:** Low

**Command:**
```bash
cd TrajectoryDiff
python scripts/train.py \
    training.max_epochs=50 \
    data.loader.batch_size=8 \
    logging.wandb.enabled=true \
    experiment.name=initial_test
```

**Tasks:**
- [ ] Verify data loading works with full RadioMapSeer
- [ ] Run 1 epoch to check for errors
- [ ] Monitor GPU memory usage
- [ ] Check W&B logging works
- [ ] Run full 50-epoch training
- [ ] Save best checkpoint

### 4.2 Baseline Comparisons
**Priority:** HIGH | **Effort:** Medium

Compare TrajectoryDiff against:

| Baseline | Implementation | Status |
|----------|---------------|--------|
| IDW Interpolation | `src/data/baselines.py` | ✅ Done |
| RBF Interpolation | `src/data/baselines.py` | ✅ Done |
| Kriging | `src/data/baselines.py` | ✅ Done |
| U-Net Regression | Need to implement | ⬜ TODO |
| Standard Diffusion (no trajectory) | Ablation of our model | ⬜ TODO |

**Tasks:**
- [ ] Run all baselines on test set
- [ ] Create comparison table (RMSE, MAE, SSIM)
- [ ] Generate visualization comparing all methods
- [ ] Analyze where each method fails

### 4.3 Ablation Studies
**Priority:** HIGH | **Effort:** Medium

| Ablation | What to Compare | Expected Result |
|----------|-----------------|-----------------|
| Trajectory Encoding | With vs without trajectory mask | Trajectory helps blind spots |
| TX Position | With vs without TX encoding | TX helps distance-based patterns |
| Coverage Density | With vs without coverage map | Helps uncertainty estimation |
| EMA | With vs without EMA | EMA stabilizes sampling |
| DDIM Steps | 10, 25, 50, 100 | Trade-off speed vs quality |
| U-Net Size | Small, Medium, Large | Larger = better but slower |

**Tasks:**
- [ ] Create experiment configs for each ablation
- [ ] Run all ablations
- [ ] Create ablation table for paper
- [ ] Identify which components matter most

### 4.4 Sampling Strategy Comparison
**Priority:** HIGH | **Effort:** Medium

This is our KEY CLAIM: trajectory sampling is different from uniform.

| Experiment | Sampling | Expected Result |
|------------|----------|-----------------|
| Uniform baseline | Random 1% pixels | Good everywhere, no bias |
| Trajectory (ours) | Walking paths | Good on paths, uncertain elsewhere |
| Mixed | 50% trajectory, 50% uniform | Balanced performance |

**Tasks:**
- [ ] Generate test sets with each sampling strategy
- [ ] Evaluate model trained on trajectory, tested on uniform (and vice versa)
- [ ] Show that trajectory-trained model has calibrated uncertainty
- [ ] Key figure: RMSE vs distance from trajectory

### 4.5 Uncertainty Quantification
**Priority:** HIGH | **Effort:** Medium

Verify uncertainty is meaningful.

**Tasks:**
- [ ] Generate 10 samples per test input
- [ ] Compute mean and std (uncertainty)
- [ ] Create calibration plot: predicted uncertainty vs actual error
- [ ] Show uncertainty is high in blind spots, low on trajectory
- [ ] Compute Expected Calibration Error (ECE)

---

## Phase 5: Paper Preparation

### 5.1 Figures to Create
**Priority:** HIGH | **Effort:** High

| Figure | Description | Tool |
|--------|-------------|------|
| Fig 1 | Problem illustration (uniform vs trajectory) | Matplotlib |
| Fig 2 | Architecture diagram | Draw.io/TikZ |
| Fig 3 | Qualitative results grid | Matplotlib |
| Fig 4 | Uncertainty visualization | Matplotlib |
| Fig 5 | Ablation bar charts | Matplotlib |
| Fig 6 | Coverage vs RMSE curves | Matplotlib |

### 5.2 Tables to Create

| Table | Content |
|-------|---------|
| Table 1 | Main results: all methods comparison |
| Table 2 | Ablation study results |
| Table 3 | Computational cost comparison |

### 5.3 Writing Tasks

- [ ] Abstract (150 words)
- [ ] Introduction (1 page)
- [ ] Related Work (1 page)
- [ ] Method (2-3 pages)
- [ ] Experiments (2-3 pages)
- [ ] Conclusion (0.5 page)
- [ ] Supplementary material

---

## Immediate Action Items (This Week)

### Day 1-2: Verify Training Pipeline
1. [ ] Run `python scripts/train.py training.max_epochs=2` to test
2. [ ] Fix any data loading issues
3. [ ] Verify W&B logging works
4. [ ] Check GPU memory with batch_size=8

### Day 3-4: First Real Training
1. [ ] Run 50-epoch training
2. [ ] Monitor loss curves in W&B
3. [ ] Generate sample visualizations
4. [ ] Evaluate on test set

### Day 5-7: Baseline Comparison
1. [ ] Run all interpolation baselines
2. [ ] Create comparison table
3. [ ] Identify performance gaps
4. [ ] Start ablation experiments

---

## Configuration Files Needed

### Experiment Configs to Create

```yaml
# configs/experiment/trajectory_baseline.yaml
defaults:
  - /model: trajectory_diffusion
  - /data: radiomapseer
  - /training: default

experiment:
  name: trajectory_baseline
  tags: [baseline, trajectory]

data:
  sampling:
    strategy: trajectory
```

```yaml
# configs/experiment/uniform_baseline.yaml
defaults:
  - /model: trajectory_diffusion
  - /data: radiomapseer
  - /training: default

experiment:
  name: uniform_baseline
  tags: [baseline, uniform]

data:
  sampling:
    strategy: uniform
```

```yaml
# configs/experiment/ablation_no_trajectory.yaml
# Ablation: remove trajectory mask conditioning

model:
  use_trajectory_mask: false
  use_coverage_density: false
```

---

## Known Issues to Address

### 1. Trajectory Visualization Quality
The generated trajectories may look unrealistic because:
- A* paths are too straight (shortest path = direct)
- Random walks can be too jagged
- No momentum/smoothing applied yet

**Fix:** Implement trajectory smoothing with Bezier curves or spline interpolation.

### 2. Building Map Quality
RadioMapSeer building maps may need preprocessing:
- Convert to binary (wall/no-wall)
- Handle different image formats
- Normalize to [0, 1] range

### 3. Scale Mismatch
Radio maps are in dBm (-120 to 0), but diffusion works best with [-1, 1]:
- Current normalization: `(x - min) / (max - min) * 2 - 1`
- Need to verify denormalization for evaluation

---

## Success Metrics

### Minimum Viable:
- [ ] Model trains without errors
- [ ] RMSE < baseline interpolation
- [ ] Uncertainty correlates with error (r > 0.5)

### Target:
- [ ] 10%+ RMSE improvement over baselines
- [ ] Well-calibrated uncertainty (ECE < 0.1)
- [ ] Clear trajectory vs uniform advantage

### Stretch:
- [ ] Real-world trajectory data collected
- [ ] Paper submitted to ECCV/CVPR
- [ ] Open-source release

---

## File Structure Reference

```
TrajectoryDiff/
├── configs/
│   ├── config.yaml              # Main entry point
│   ├── data/radiomapseer.yaml   # Data config
│   ├── model/trajectory_diffusion.yaml
│   ├── training/default.yaml
│   └── experiment/              # Create ablation configs here
├── src/
│   ├── data/                    # ✅ Complete
│   │   ├── dataset.py
│   │   ├── datamodule.py
│   │   ├── trajectory_sampler.py
│   │   ├── transforms.py
│   │   └── baselines.py
│   ├── models/                  # ✅ Complete
│   │   ├── diffusion/
│   │   │   ├── ddpm.py
│   │   │   └── unet.py
│   │   └── encoders/
│   │       └── condition_encoder.py
│   ├── training/                # ✅ Complete
│   │   ├── diffusion_module.py
│   │   ├── inference.py
│   │   └── callbacks.py
│   └── evaluation/              # ✅ Complete
│       └── metrics.py
├── scripts/
│   ├── train.py                 # ✅ Complete
│   └── evaluate.py              # ✅ Complete
├── tests/                       # 101 tests passing
└── docs/
    ├── PLAN.md                  # Research plan
    └── NEXT_STEPS.md            # This file
```

---

## Contact & Resources

- **RadioMapSeer Paper**: https://ieeexplore.ieee.org/document/9893908
- **DDPM Paper**: https://arxiv.org/abs/2006.11239
- **DDIM Paper**: https://arxiv.org/abs/2010.02502
- **W&B Dashboard**: https://wandb.ai/your-username/trajectorydiff
