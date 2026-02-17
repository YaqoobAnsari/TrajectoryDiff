# TrajectoryDiff: Next Steps

## Current Status (Feb 17, 2026)

**Version:** v0.4.0-experiment-ready
**Tests:** 199 passing (9 test files)
**Code:** All CVPR audit fixes applied (24 fixes across 14 files)

### First DDIM Evaluation Results — trajectory_full @ Epoch 133

Evaluated via job 2729 on 7g.141gb (50-step DDIM, 8480 test samples). Full results in `experiments/eval_results/trajectory_full_epoch133/`.

| Metric | Value | Notes |
|--------|-------|-------|
| **RMSE (all pixels)** | 37.25 +/- 16.9 dBm | Includes buildings (~70% of pixels) |
| **MAE (all pixels)** | 30.83 +/- 17.1 dBm | |
| **SSIM (all pixels)** | 0.635 +/- 0.166 | |
| **Trajectory RMSE (observed)** | **11.06 dBm** | At trajectory sample points |
| **Blind Spot RMSE (unobserved)** | 37.31 dBm | All non-trajectory pixels (incl. buildings) |
| **SSIM (observed)** | 0.994 | Near-perfect at observed points |

**Assessment**: Model IS learning structure (building outlines, corridor patterns visible). 11 dBm trajectory RMSE is promising. However, 37 dBm all-pixel RMSE is still high — only ~6 dBm better than dataset-mean baseline (~43 dBm). Error maps show signal overestimation near TX (bright blobs instead of fine corridor propagation). High per-sample variance (std=16.9, range 4-105 dBm) indicates under-training. Model needs to reach 200 epochs.

**Missing**: Free-space vs building RMSE breakdown (evaluate.py TODO). The 37 dBm blind spot RMSE is dominated by building pixels.

### Active Jobs

| Job | Experiment | Resource | Progress | Status |
|-----|-----------|----------|----------|--------|
| 2729 | trajectory_full eval + baselines | 7g.141gb GPU | Model eval done, baselines running | Step 2 in progress |
| 2730 | classical baselines (standalone) | CPU (mcore-n01) | ~3500/8480 (41%) | ~3h remaining |
| 2725 | trajectory_baseline train | 2g.35gb GPU | Epoch 37/200 | ~24h to timeout |
| 2726 | uniform_baseline train | 2g.35gb GPU | Epoch 46/200 | ~24h to timeout |

### val/loss Gap Explanation

trajectory_full's higher val/loss (0.00474 vs 0.00335 for uniform_baseline) is **expected and by design**:
- val/loss = plain MSE on noise prediction (uniform over all pixels)
- Training uses coverage-weighted loss (90% of pixels get 0.1x weight) + physics losses (37% of gradient budget)
- The model optimizes a different objective than what val/loss measures
- **Real quality assessment requires DDIM evaluation** (`evaluate.py`) to get dBm-scale RMSE

---

## Immediate TODO

### 1. Resume trajectory_full Training (epochs 133 → 200)
Job 2707 timed out at epoch 135 (48h). Best checkpoint: `epoch=133-val_loss=0.0047.ckpt`. val/loss still decreasing — needs more training.
```bash
# Resume from last.ckpt (no FRESH flag)
bash scripts/submit_experiment.sh trajectory_full 7g.141gb
```
Submit once 7g.141gb slot frees up (after job 2729 finishes).

### 2. Classical Baselines (jobs 2729 step 2 + 2730)
- Job 2730 standalone baselines: ~3h remaining, results → `experiments/eval_results/baselines.json`
- Job 2729 also runs baselines after model eval (redundant but ensures completion)
- Includes **per-region dBm metrics**: free-space, building, observed, unobserved, free-unobserved
- **Fair comparison metric**: Free-space RMSE (baselines don't get building map)
- See `docs/metrics.md` "Fair Evaluation Methodology" for details

### 3. Add Free-Space/Building RMSE to evaluate.py
**Critical TODO before final eval.** Current evaluate.py only computes all-pixel and observed/unobserved metrics. Need to add:
- `rmse_free_space` — RMSE on non-building pixels only
- `rmse_building` — RMSE on building pixels only
- `rmse_free_space_unobserved` — key research metric (extrapolation in walkable blind spots)
This is needed to match the per-region metrics in `run_baselines.py`.

### 4. After Core Training Completes (200 epochs)
```bash
# Evaluate best checkpoints (with per-region metrics once added)
python scripts/evaluate.py checkpoint=experiments/trajectory_full/<date>/checkpoints/best.ckpt
python scripts/evaluate.py checkpoint=experiments/trajectory_baseline/<date>/checkpoints/best.ckpt
python scripts/evaluate.py checkpoint=experiments/uniform_baseline/<date>/checkpoints/best.ckpt
```

---

## Experiment Queue (after core 3 finish)

### Wave 3 — Ablations (5 experiments, 2g.35gb each)
Submit 2 at a time (max 2x 2g.35gb concurrent):
```bash
FRESH=1 bash scripts/submit_experiment.sh ablation_no_physics_loss 2g.35gb
FRESH=1 bash scripts/submit_experiment.sh ablation_no_coverage_attention 2g.35gb
# Then when slots open:
FRESH=1 bash scripts/submit_experiment.sh ablation_no_trajectory_mask 2g.35gb
FRESH=1 bash scripts/submit_experiment.sh ablation_no_coverage_density 2g.35gb
FRESH=1 bash scripts/submit_experiment.sh ablation_no_tx_position 2g.35gb
```

### Wave 4 — Coverage Sweeps (4 experiments)
```bash
FRESH=1 bash scripts/submit_experiment.sh coverage_sweep_1pct 2g.35gb
FRESH=1 bash scripts/submit_experiment.sh coverage_sweep_5pct 2g.35gb
FRESH=1 bash scripts/submit_experiment.sh coverage_sweep_20pct 2g.35gb
# coverage_sweep_10pct is identical to trajectory_full
```

### Wave 5 — Cross-eval + Extras
```bash
FRESH=1 bash scripts/submit_experiment.sh cross_eval_traj_to_uniform 2g.35gb
FRESH=1 bash scripts/submit_experiment.sh cross_eval_uniform_to_traj 2g.35gb
FRESH=1 bash scripts/submit_experiment.sh ablation_small_unet 2g.35gb
```

### DL Baselines (3 experiments, 2g.35gb each)
All three DL baselines are implemented and wired into `train.py` via `model_type` factory:
```bash
# Supervised UNet — same architecture, no diffusion (ablation)
FRESH=1 bash scripts/submit_experiment.sh supervised_unet 2g.35gb

# RadioUNet — Levie et al. 2021, standalone UNet, direct MSE
FRESH=1 bash scripts/submit_experiment.sh radio_unet 2g.35gb

# RMDM — Xu et al. 2025, dual-UNet diffusion with anchor fusion
FRESH=1 bash scripts/submit_experiment.sh rmdm_baseline 7g.141gb
```
- Supervised UNet and RadioUNet: single forward pass, ~2x faster than diffusion
- RMDM: diffusion-based, needs 7g.141gb for dual-UNet memory

---

## Post-Training Pipeline

```bash
# 1. Aggregate all eval results
python scripts/aggregate_results.py --eval-dir experiments/eval_results --verbose

# 2. Generate figures
python scripts/generate_figures.py --results-dir experiments/eval_results --output-dir figures/

# 3. Uncertainty analysis (requires trajectory_full checkpoint)
python scripts/analyze_uncertainty.py --checkpoint experiments/trajectory_full/<date>/checkpoints/best.ckpt

# 4. Statistical significance testing
python scripts/run_baselines.py --reference-results experiments/eval_results/trajectory_full.json
```

**Outputs:**
- `experiments/eval_results/summary.json` — all metrics
- `experiments/eval_results/summary.csv` — spreadsheet format
- `experiments/eval_results/tables.tex` — LaTeX tables for paper
- `figures/` — all paper figures

---

## Paper Outline

### Target Venue
CVPR 2026 or ECCV 2026

### Structure
1. Introduction — trajectory vs uniform sampling gap
2. Related Work — diffusion for radio maps, trajectory-based methods
3. Method — TrajectoryDiff architecture, coverage-aware attention, physics losses
4. Experiments — baselines, ablations, coverage sweeps, uncertainty
5. Conclusion

### Key Figures
- Problem illustration (uniform vs trajectory sampling)
- Architecture diagram
- Qualitative results (6-panel: building, trajectory, sparse, GT, ours, IDW)
- Uncertainty visualization
- Ablation bar chart
- Coverage sweep curve

### Key Tables
- **Main results table** — must use per-region metrics for fair comparison:
  - Primary: Free-space RMSE (dBm) — fair across all methods
  - Research: Free-space unobserved RMSE — extrapolation quality
  - Context: All-pixel RMSE + Building RMSE (with footnote about building map advantage)
  - SSIM on free-space region
- Ablation study (each component's contribution)
- Cross-evaluation (train on trajectory, test on uniform and vice versa)

### Evaluation Methodology Note for Paper
When comparing diffusion models vs classical baselines, clearly state:
1. Diffusion models receive building map as conditioning; classical baselines do not
2. Free-space RMSE is the primary fair comparison metric
3. All-pixel RMSE favors methods with building map access (~35 dBm advantage)
4. Coverage is ~0.5% of total map / ~1.6% of free space (far sparser than published work)
5. Published SOTA uses dense observation — not in the same table
