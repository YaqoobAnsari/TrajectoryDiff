# TrajectoryDiff: Next Steps

## Current Status (Feb 17, 2026)

**Version:** v0.4.0-experiment-ready
**Tests:** 199 passing (9 test files)
**Code:** All CVPR audit fixes applied (24 fixes across 14 files)

### Active Jobs

| Job | Experiment | Resource | Epoch | val/loss | ETA |
|-----|-----------|----------|-------|----------|-----|
| 2707 | trajectory_full | 7g.141gb GPU | 127/200 | 0.00474 | ~3h to timeout, needs resume |
| 2725 | trajectory_baseline | 2g.35gb GPU | 29/200 | 0.00376 | ~30h to timeout, needs resume |
| 2726 | uniform_baseline | 2g.35gb GPU | 36/200 | 0.00335 | ~30h to timeout, needs resume |
| 2728 | classical baselines (re-run) | CPU (mcore-n01) | - | - | ~5h, with per-region metrics |

All 4 SLURM job slots occupied. Job 2727 (original baselines) completed evaluation but crashed before saving JSON due to a Path import bug (now fixed). Job 2728 re-runs with per-region dBm metrics.

### val/loss Gap Explanation

trajectory_full's higher val/loss (0.00474 vs 0.00335 for uniform_baseline) is **expected and by design**:
- val/loss = plain MSE on noise prediction (uniform over all pixels)
- Training uses coverage-weighted loss (90% of pixels get 0.1x weight) + physics losses (37% of gradient budget)
- The model optimizes a different objective than what val/loss measures
- **Real quality assessment requires DDIM evaluation** (`evaluate.py`) to get dBm-scale RMSE

---

## Immediate TODO (when jobs finish)

### 1. Classical Baselines Complete (~5h) — Job 2728
- Results will be in `experiments/eval_results/baselines.json`
- Now includes **per-region dBm metrics**: free-space, building, observed, unobserved, free-unobserved
- **Fair comparison metric**: Free-space RMSE (baselines don't get building map)
- **Key research metric**: Free-space unobserved RMSE (extrapolation quality)
- Do NOT compare all-pixel RMSE directly — 70% is buildings where baselines are structurally blind
- See `docs/metrics.md` "Fair Evaluation Methodology" section for details

### 2. GPU Jobs Timeout (~48h)
All three GPU jobs will hit the 48h wall time before reaching 200 epochs.

**Resume procedure:**
```bash
# Remove FRESH flag — let it find last.ckpt automatically
bash scripts/submit_experiment.sh trajectory_full 7g.141gb
bash scripts/submit_experiment.sh trajectory_baseline 2g.35gb
bash scripts/submit_experiment.sh uniform_baseline 2g.35gb
```

### 2.5. Early Evaluation (before 200 epochs)
**val/loss is NOT the quality metric.** To know if the model needs more training, run DDIM evaluation on the current best checkpoint:
```bash
# Evaluate trajectory_full at current best (epoch 124)
python scripts/evaluate.py \
    checkpoint=experiments/trajectory_full/2026-02-15_13-00-26/checkpoints/epoch=124-val_loss=0.0048.ckpt \
    max_samples=500
```
This runs 50-step DDIM sampling and computes actual dBm RMSE. If RMSE is still high, continue training. If it's plateaued, 200 epochs may be enough.

### 3. After Core Training Completes (200 epochs)
```bash
# Evaluate best checkpoints (all per-region metrics)
python scripts/evaluate.py checkpoint=experiments/trajectory_full/<date>/checkpoints/best.ckpt
python scripts/evaluate.py checkpoint=experiments/trajectory_baseline/<date>/checkpoints/best.ckpt
python scripts/evaluate.py checkpoint=experiments/uniform_baseline/<date>/checkpoints/best.ckpt

# Compare against classical baselines (with significance testing)
python scripts/run_baselines.py --reference-results experiments/eval_results/trajectory_full.json
```

**NOTE**: `evaluate.py` also needs per-region (free-space/building) dBm metrics added to match the baselines script. TODO: add building mask evaluation to evaluate.py before final eval.

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
