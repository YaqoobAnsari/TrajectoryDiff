# TrajectoryDiff Research Plan

## Executive Summary

**Problem**: Existing radio map diffusion methods assume uniformly random samples. Real-world crowdsourced data follows trajectory patterns with systematic spatial bias.

**Solution**: TrajectoryDiff — a trajectory-conditioned diffusion model that explicitly models WHERE samples come from, enabling proper extrapolation into unobserved regions with calibrated uncertainty.

**Novelty Claims**:
1. First diffusion model for trajectory-to-map generation (vs. random-sparse-to-dense)
2. First to explicitly model spatial sampling bias in radio map prediction
3. Coverage-aware attention mechanism modulating diffusion by observation density
4. Physics-informed losses with warmup schedule for stable training
5. Uncertainty-aware generation with proper calibration for blind spots

---

## Implementation Progress

| Phase | Status | Tests | Key Files |
|-------|--------|-------|-----------|
| Phase 0: Setup | Done | - | configs/, environment.yaml |
| Phase 1: Data Pipeline | Done | 13 tests | src/data/ |
| Phase 2: Model Development | Done | 88 tests | src/models/, src/training/ |
| Phase 3: Physics + Architecture | Done | 98 new tests | losses.py, coverage_unet.py, attention.py |
| Phase 3.5: CVPR Audit Fixes | Done | 199 tests | 24 fixes across 14 files |
| Phase 4: Experiments | **TRAINING** | - | 16 experiment configs, SLURM scripts |
| Phase 5: Paper Writing | Not Started | - | - |

**Total Tests: 199 passing (9 test files) | Version: v0.4.0-experiment-ready**

---

## Training Log

### Wave 2 — Submitted Feb 16, 2026 ~15:42 (post-CVPR-audit code)

All jobs use FRESH=1 (no checkpoint resume from pre-audit runs).

| Job ID | Experiment | Partition / MIG | Batch | Status (Feb 17) | Notes |
|--------|-----------|-----------------|-------|--------|-------|
| 2707 | trajectory_full | gpu2 / 7g.141gb | 32 x accum=2 | Epoch 127/200, val/loss=0.00474 | ~21 min/epoch, ~3h to 48h timeout |
| 2725 | trajectory_baseline | gpu2 / 2g.35gb | 8 x accum=2 | Epoch 29/200, val/loss=0.00376 | ~35 min/epoch, ~30h to timeout |
| 2726 | uniform_baseline | gpu2 / 2g.35gb | 8 x accum=2 | Epoch 36/200, val/loss=0.00335 | ~28 min/epoch, ~30h to timeout |
| 2727 | classical baselines (CPU) | cpu / mcore-n01 | - | COMPLETED (but JSON save crashed) | Path import bug — fixed |
| 2728 | classical baselines (re-run) | cpu / mcore-n01 | - | RUNNING | With per-region dBm metrics |

**Val/loss trajectory for job 2707 (trajectory_full)**:
- Epoch 0: 1.000 → Epoch 10: 0.912 → Epoch 20: 0.314 → Epoch 30: 0.0811
- Epoch 40: 0.0255 → Epoch 50: 0.0107 → Epoch 60: 0.00684 → Epoch 72: 0.00574
- Epoch 100: 0.00507 → Epoch 126: 0.00474
- Still declining ~0.3%/epoch, projected ~0.0037-0.0042 by epoch 200
- Physics warmup gradient spike at epoch 48-49 (max grad 0.09) handled gracefully by gradient clipping

**val/loss gap analysis**:
- trajectory_full (0.00474) vs uniform_baseline (0.00335) = 41% higher
- This is EXPECTED: coverage-weighted training + physics losses optimize different objective than val/loss
- val/loss is noise prediction MSE; actual quality requires DDIM eval in dBm
- Initial HealthCheck loss: trajectory_full=0.0517, baselines=0.4982 (10x gap from coverage weighting)

**Classical baselines (Job 2727) — all-pixel results (MISLEADING)**:
- Best: RBF Multiquadric = 40.16 dBm RMSE, 0.446 SSIM
- These numbers are inflated by ~35 dBm from building pixels (70% of image)
- Free-space RMSE estimated ~7 dBm (awaiting per-region results from Job 2728)
- See `docs/metrics.md` "Fair Evaluation Methodology" for details

**Time estimates**:
- trajectory_full (2707): timeout ~Feb 17 16:00, needs resume for remaining ~73 epochs
- trajectory_baseline (2725): timeout ~Feb 18 10:00, needs resume
- uniform_baseline (2726): timeout ~Feb 18 10:00, needs resume
- classical baselines (2728): ~5h total on mcore-n01

### Wave 1 (INVALID — pre-audit code, all checkpoints discarded)

Jobs 2683-2686 ran with broken LR schedule, inverted building map convention, incorrect attention heads, and other bugs found during CVPR audit. All Wave 1 checkpoints are invalid.

---

## Published SOTA on RadioMapSeer (for reference)

These results assume **dense/full observation** (building map + TX position → predict full pathloss). Our sparse-trajectory setting is fundamentally harder.

| Method | NMSE | RMSE (norm) | SSIM | Source |
|--------|------|-------------|------|--------|
| RME-GAN | 0.0115 | 0.0303 | 0.932 | RadioDiff (2024) |
| RadioUNet | 0.0074 | 0.0244 | 0.959 | RadioDiff (2024) |
| RadioDiff | 0.0049 | 0.0190 | 0.969 | RadioDiff (2024) |
| **RMDM** | **0.0031** | **0.0125** | **0.978** | RMDM (Jan 2025) |

**IMPORTANT — NOT COMPARABLE to our results:**
- Published SOTA uses **dense/full observation** (building map + TX position → full pathloss)
- Our task uses **~0.5% trajectory coverage** (300 sparse points on walking paths)
- Do NOT put these numbers in the same table as ours in the paper
- Instead, cite them in Related Work with explicit disclaimer about evaluation protocol difference

---

## Remaining Experiment Queue

### Priority 1 — Core (needed for every table)
- [x] trajectory_full (running — job 2707)
- [x] trajectory_baseline (running — job 2725)
- [x] uniform_baseline (running — job 2726)
- [x] classical baselines (running — job 2728)
- [x] supervised_unet baseline (wired up — `experiment=supervised_unet`)
- [x] radio_unet baseline (implemented — `experiment=radio_unet`)
- [x] rmdm baseline (implemented — `experiment=rmdm_baseline`)

### Priority 2 — Ablations (needed for ablation table)
- [ ] ablation_no_physics_loss
- [ ] ablation_no_coverage_attention
- [ ] ablation_no_trajectory_mask
- [ ] ablation_no_coverage_density
- [ ] ablation_no_tx_position

### Priority 3 — Coverage sweeps (robustness figure)
- [ ] coverage_sweep_1pct
- [ ] coverage_sweep_5pct
- [ ] coverage_sweep_10pct (same as trajectory_full)
- [ ] coverage_sweep_20pct

### Priority 4 — Lower priority
- [ ] ablation_small_unet
- [ ] cross_eval_traj_to_uniform
- [ ] cross_eval_uniform_to_traj
- [ ] num_trajectories_sweep (Hydra multirun)

---

## Cluster Configuration

| Resource | Value |
|----------|-------|
| Partition (GPU) | gpu2, node=deepnet2 |
| Partition (CPU) | cpu, node=mcore-n01 (128 cores) |
| GPUs | 8x NVIDIA H200, MIG-partitioned |
| Max concurrent jobs | 4 (QOSMaxGRESPerUser) |
| Max per MIG profile | 1g.18gb=4, 2g.35gb=2, 7g.141gb=1 |
| Max time per job | 48h |
| Conda env | trajdiff |

### Batch Size Constraints (OOM-validated)
- 7g.141gb: batch=32 x accum=2 (effective 64)
- 2g.35gb: batch=8 x accum=2 (effective 16)
- 1g.18gb: batch=4 x accum=4 (effective 16)
- Always set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

---

## Architecture Overview

### Conditioning Pipeline
```
building_map (1ch)  ──┐
sparse_rss (1ch)    ──┼──► Concat (4-5ch) ──► CNN ──► condition (64ch)
trajectory_mask (1ch)─┘                                    │
coverage_density (1ch)─┘                                   │
tx_position (2)     ──► MLP ──► Spatial encoding ──────────┘
                                                           │
                                           concat with x_t │
                                                           ▼
                                              UNet (65ch input)
                                                           │
                                                    noise prediction
```

### Novel Components
1. **CoverageAwareUNet** — UNet with additive log-bias attention modulated by coverage density
2. **Physics-Informed Losses** — trajectory consistency (0.5) + distance decay (0.1) with 30-epoch warmup + 20-epoch ramp
3. **Min-SNR-gamma weighting** — stabilizes diffusion training across timesteps

### Key Fixes from CVPR Audit (Feb 15, 2026)
- Additive log-bias attention (was multiplicative — cancelled in softmax)
- DDIM uses linspace for timestep subsequence (was arange, caused index OOB)
- Physics loss gradients detached from diffusion backbone
- LR schedule uses trainer.estimated_stepping_batches (was hardcoded steps)
- Building map convention: free_space = building_map > 0.0 (was inverted)
- SSIM computed on dBm scale with skimage sliding window
- Worker seeding for reproducibility
- Per-region metrics (observed vs unobserved)

---

## Phase Summaries

### Phase 0-1: Setup + Data Pipeline
- RadioMapSeer dataset (701 maps x 80 TX = 56K radio maps)
- Three trajectory types: shortest-path (A*), random-walk, corridor-biased
- Data augmentation: rotation, flip (physics-preserving)
- 70/15/15 train/val/test split by building

### Phase 2: Model Development
- DDPM with cosine noise schedule, DDIM fast sampling (50 steps)
- UNet backbone (Small/Medium/Large) with attention at low resolutions
- TrajectoryConditionedUNet fusing 5 conditioning signals
- Lightning training module with EMA, warmup+cosine LR
- Evaluation metrics: RMSE, SSIM, PSNR + trajectory-aware variants

### Phase 3: Physics + Architecture
- TrajectoryDiffLoss (consistency + coverage-weighted + distance decay)
- CoverageAwareUNet with coverage-modulated attention
- Physics loss warmup (30 epoch delay + 20 epoch ramp)
- 199 tests across 9 test files

### Phase 3.5: CVPR Audit
- 8 critical, 13 moderate, 3 suggestion fixes
- 24 changes across 14 files + 2 new files
- All Wave 1 checkpoints invalidated

### Phase 4: Experiments (IN PROGRESS)
- 19 experiment configs ready (16 original + 3 DL baselines)
- 4 jobs running (1 GPU full model, 2 GPU baselines, 1 CPU classical baselines)
- Classical baselines: nearest neighbor, IDW, RBF, distance transform
- DL baselines: Supervised UNet, RadioUNet, RMDM (all implemented, ready to train)

### Phase 5: Paper Writing (NOT STARTED)
- Target venue: CVPR 2026 or ECCV 2026
- Figures needed: problem illustration, architecture, qualitative results, uncertainty viz, ablation plots, coverage sweep
