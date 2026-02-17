# TrajectoryDiff: Trajectory-Conditioned Radio Map Generation

## Project Overview

Research project exploring trajectory-conditioned diffusion models for indoor radio map generation. Core insight: real RSS measurements come from human walking paths (corridors, accessible areas), not uniform random sampling. We model this bias explicitly.

**Goal**: Generate complete radio maps from sparse trajectory-sampled RSS measurements, with proper uncertainty quantification for unobserved regions.

**Target**: ECCV-level publication demonstrating trajectory-aware diffusion outperforms uniform-sampling assumptions.

**Version**: v0.4.0-experiment-ready | **Tests**: 199 passing (9 test files) | **CVPR Audit**: 24 fixes applied (Feb 15, 2026)

## Key Concepts (Domain Knowledge)

- **Radio Map**: 2D spatial function R(x,y) = signal strength (dBm) at each location
- **RSS**: Received Signal Strength, what phones measure (-50 excellent, -100 poor)
- **Pathloss**: Signal attenuation between transmitter and receiver (dB)
- **Trajectory Sampling**: RSS collected along human walking paths (biased, correlated)
- **Uniform Sampling**: RSS at random locations (idealized, unrealistic)
- **Blind Spots**: Regions never visited (locked rooms, restricted areas)

## Repository Structure

```
TrajectoryDiff/
├── configs/                    # Hydra configuration files
│   ├── config.yaml            # Main config entry point
│   ├── data/
│   │   └── radiomapseer.yaml  # RadioMapSeer dataset config
│   ├── model/
│   │   └── trajectory_diffusion.yaml  # Model architecture config
│   ├── training/
│   │   └── default.yaml       # Training hyperparameters
│   └── experiment/            # 19 experiment configs
│       ├── trajectory_full.yaml          # Full model, all features
│       ├── trajectory_baseline.yaml      # Trajectory sampling baseline
│       ├── uniform_baseline.yaml         # Uniform sampling baseline
│       ├── supervised_unet.yaml          # Supervised UNet DL baseline
│       ├── radio_unet.yaml              # RadioUNet DL baseline
│       ├── rmdm_baseline.yaml           # RMDM DL baseline
│       ├── ablation_no_*.yaml            # 5 ablation configs
│       ├── cross_eval_*.yaml             # 2 cross-evaluation configs
│       ├── coverage_sweep_{1,5,10,20}pct.yaml  # Coverage sweeps
│       └── num_trajectories_sweep.yaml   # Hydra multirun sweep
├── src/                       # Source code
│   ├── data/                  # Data loading and processing
│   │   ├── dataset.py         # RadioMapDataset, UniformSamplingDataset
│   │   ├── datamodule.py      # Lightning RadioMapDataModule
│   │   ├── floor_plan.py      # Floor plan processing, dBm conversion
│   │   ├── transforms.py      # Data augmentations
│   │   └── trajectory_sampler.py  # Trajectory generation logic
│   ├── models/
│   │   ├── diffusion/         # Diffusion model components
│   │   │   ├── ddpm.py        # GaussianDiffusion, DDIMSampler
│   │   │   ├── unet.py        # UNet backbone (Small/Medium/Large)
│   │   │   ├── coverage_unet.py  # CoverageAwareUNet (novel)
│   │   │   └── attention.py   # CoverageAwareAttention (novel)
│   │   ├── encoders/          # Condition encoders
│   │   │   └── condition_encoder.py  # TrajectoryConditionedUNet
│   │   └── baselines/         # Classical + DL baselines
│   │       ├── interpolation.py  # IDW, RBF, Kriging, NN interpolation
│   │       ├── supervised_unet.py  # Supervised UNet (same arch, no diffusion)
│   │       ├── radio_unet.py     # RadioUNet (Levie et al., 2021)
│   │       └── rmdm.py           # RMDM (Xu et al., 2025)
│   ├── training/              # Training logic
│   │   ├── diffusion_module.py  # Lightning DiffusionModule (EMA, LR schedule)
│   │   ├── losses.py          # TrajectoryDiffLoss (physics-informed)
│   │   ├── inference.py       # DiffusionInference, uncertainty estimation
│   │   └── callbacks.py       # W&B logging, metrics, gradient monitoring
│   ├── evaluation/            # Evaluation metrics
│   │   └── metrics.py         # RMSE, SSIM, trajectory-aware metrics
│   └── utils/
│       └── visualization.py   # Radio map visualization
├── scripts/                   # Entry point scripts
│   ├── train.py              # Hydra training script
│   ├── evaluate.py           # Evaluation with dBm-scale metrics
│   ├── smoke_test.py         # Lightning-based smoke test (full epochs)
│   ├── smoke_test_quick.py   # Fast manual smoke test (~15s on CPU)
│   ├── run_experiments.sh    # SLURM job script (partition=gpu2, MIG gres)
│   ├── submit_experiment.sh  # Submit single experiment with MIG profile
│   ├── submit_all.sh         # Submit all 16 experiments with concurrency control
│   ├── gpu_validation.sh     # Quick GPU validation job (30 min)
│   ├── run_evaluation.sh     # SLURM/SSH: batch evaluation
│   ├── run_baselines.py      # Classical baseline evaluation (IDW, RBF, etc.)
│   ├── analyze_uncertainty.py # Uncertainty calibration analysis
│   └── generate_figures.py   # Paper figure generation
├── tests/                     # 199 tests (9 test files)
├── notebooks/                 # Jupyter notebooks for exploration
├── experiments/               # Experiment outputs (gitignored)
├── data/                      # Data directory (gitignored)
│   └── raw/                   # RadioMapSeer data (701 maps x 80 Tx)
├── docs/                      # Documentation
│   ├── PLAN.md               # Research plan and milestones
│   ├── WHAT_WE_BUILT.md      # Technical architecture deep dive
│   ├── NEXT_STEPS.md         # Experiment plans
│   ├── metrics.md            # Evaluation metrics guide
│   └── dataset_notes.md      # RadioMapSeer dataset docs
├── environment.yaml           # Conda environment
├── pyproject.toml            # Project metadata
└── README.md                 # Project README
```

## Tech Stack

- **Python**: 3.10+
- **Deep Learning**: PyTorch 2.0+, PyTorch Lightning 2.0+
- **Config Management**: Hydra 1.3+, OmegaConf
- **Experiment Tracking**: Weights & Biases (wandb)
- **Data Processing**: NumPy, SciPy, scikit-image
- **Visualization**: Matplotlib, Seaborn
- **Testing**: pytest
- **GPU Cluster**: SLURM (NVIDIA H200 via MIG profiles)

## Common Commands

```bash
# Environment setup
conda env create -f environment.yaml
conda activate trajdiff

# Quick smoke test (CPU, ~15 seconds)
python scripts/smoke_test_quick.py

# Training (local)
python scripts/train.py                                       # Default config
python scripts/train.py experiment=trajectory_full            # Full model
python scripts/train.py experiment=trajectory_baseline        # Trajectory baseline
python scripts/train.py experiment=supervised_unet            # Supervised UNet baseline
python scripts/train.py experiment=radio_unet                 # RadioUNet baseline
python scripts/train.py experiment=rmdm_baseline              # RMDM baseline
python scripts/train.py data.loader.batch_size=16 training.max_epochs=50  # Override

# Training (SLURM cluster - via submit scripts)
bash scripts/submit_experiment.sh trajectory_full 7g.141gb    # Single experiment (full GPU)
bash scripts/submit_experiment.sh trajectory_baseline 2g.35gb # Single experiment (1/4 GPU)
bash scripts/submit_all.sh 2g.35gb                            # All 16 experiments

# Monitor SLURM jobs
squeue -u $USER                                               # Job queue
sacct -j <JOBID> --format=JobID,State,Elapsed                 # Job history
tail -f experiments/logs/trajdiff_<JOBID>.out                 # Live training output

# Evaluation
python scripts/evaluate.py checkpoint=path/to/checkpoint.ckpt

# Testing
pytest tests/ -v
pytest tests/test_coverage_unet.py -v     # Specific test file
pytest tests/ -v -k "test_training"       # Keyword filter

# Linting
ruff check src/
black src/ --check
```

## Architecture Overview

### Novel Contributions

1. **CoverageAwareUNet** (`src/models/diffusion/coverage_unet.py`): UNet variant that replaces standard AttentionBlocks with CoverageAwareAttentionBlocks. Coverage density is threaded through encoder/middle/decoder, modulating attention weights so the model pays more attention to high-coverage (observed) regions.

2. **Physics-Informed Losses** (`src/training/losses.py`): TrajectoryDiffLoss combines:
   - TrajectoryConsistencyLoss: Enforces prediction accuracy along observed trajectories
   - CoverageWeightedLoss: Weights diffusion loss by coverage density
   - DistanceDecayLoss: Soft constraint that signal decreases with TX distance

### Data Normalization Convention

| Data Type | Range | Notes |
|-----------|-------|-------|
| Signal data (radio_map, sparse_rss, building_map) | [-1, 1] | Normalized from [0, 255] PNG |
| Masks (trajectory_mask, coverage_density) | [0, 1] | Binary/continuous masks |
| TX position | [0, 1] | Normalized by image size (256) |

### DiffusionModule API

- `training_step()` returns **bare loss tensor** (not dict)
- `validation_step()` returns `{'val_loss': loss}` dict
- Checkpoint monitors `val/loss` (not `val/rmse`)
- Physics losses toggled via `use_physics_losses` flag
- Coverage attention toggled via `use_coverage_attention` flag

## Code Style & Conventions

- **Formatting**: Black (line length 100), isort for imports
- **Linting**: Ruff for fast linting
- **Type Hints**: Required for all function signatures
- **Docstrings**: Google style docstrings for public functions/classes
- **Naming**:
  - `snake_case` for functions, variables, modules
  - `PascalCase` for classes
  - `UPPER_CASE` for constants
- **Imports**: Absolute imports preferred, group stdlib/third-party/local

## Key Files to Understand First

1. `configs/config.yaml` - Entry point for all Hydra configuration
2. `src/data/trajectory_sampler.py` - Core trajectory generation logic
3. `src/models/encoders/condition_encoder.py` - TrajectoryConditionedUNet (main model)
4. `src/models/diffusion/coverage_unet.py` - CoverageAwareUNet (novel contribution)
5. `src/training/diffusion_module.py` - Lightning training module
6. `src/training/losses.py` - Physics-informed losses (novel contribution)
7. `docs/PLAN.md` - Detailed research plan and milestones

## Dataset: RadioMapSeer

- **Location**: `data/raw/` (701 maps, 56K radio maps)
- **Format**: 256x256 PNG pathloss maps, binary building maps, Tx positions (JSON)
- **Encoding**: PNG 0-255 maps to dBm via `(png / 255) * 139 + (-186)`
- **Note**: We simulate trajectory sampling on top of dense ground truth
- **Caveat**: File scan of 56K files is slow on Windows; use `map_ids` subset for local testing

## Trajectory Sampling Strategy

Three trajectory types implemented in `src/data/trajectory_sampler.py`:

1. **Shortest-path**: A* navigation between random accessible points
2. **Random-walk**: Momentum-based walk respecting walls
3. **Corridor-biased**: Prefers corridor regions (realistic human behavior)

Each trajectory produces: `[(t, x, y, rss), ...]` with optional noise injection.

## Hydra Config Notes

- Model config keys live directly under `model:` group (no double nesting in YAML file)
- Data path: `data.dataset.root` resolves to `data/raw/`
- Experiment configs use `# @package _global_` for proper override scope

## Git Workflow

- **Main branch**: `main` - stable, tested code
- **Feature branches**: `feature/<name>` - new features
- **Experiment branches**: `exp/<name>` - experimental work
- **Commit messages**: Conventional commits (feat:, fix:, docs:, refactor:)

## Important Notes

- NEVER commit data files or model checkpoints to git
- Always run tests before pushing: `pytest tests/ -v`
- Log all experiments to wandb for reproducibility
- Use Hydra's `--cfg job` to inspect resolved config before running
- UNet output_conv is zero-initialized (standard for diffusion) -- use `_make_output_nonzero()` helper in tests

## SLURM Cluster Details

- **Cluster**: gpujobs.qatar.cmu.edu → deepnet2 (8x NVIDIA H200, MIG-partitioned)
- **Partition**: gpu2, nodelist=deepnet2, mcs-label=unicellular
- **MIG gres format**: `gpu:nvidia_h200_<profile>:1`
- **Max concurrent jobs**: 4 (QOSMaxGRESPerUser)
- **Conda env**: `/data1/yansari/.conda/envs/trajdiff` (PATH prepend, not `conda activate`)
- **OOM-validated batch sizes**:
  - 7g.141gb (141GB): batch=64 works; batch=32 x accum=2 is safer
  - 2g.35gb (32.5GB): batch=16 OOMs; batch=8 x accum=2 works
- **Key env vars**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `WANDB_MODE=offline`

## Current Focus

**Phase**: GPU Training (v0.4.0) | Post-CVPR-audit, fresh training wave

### Active Jobs (Feb 16, 2026):
| Job | Experiment | Resource | Batch | Status |
|-----|-----------|----------|-------|--------|
| 2707 | trajectory_full | gpu2 / 7g.141gb | 32 x accum=2 | Epoch ~73/200, val/loss=0.00574, ~21 min/epoch |
| 2725 | trajectory_baseline | gpu2 / 2g.35gb | 8 x accum=2 | Epoch 0/200, ~35 min/epoch |
| 2726 | uniform_baseline | gpu2 / 2g.35gb | 8 x accum=2 | Epoch 0/200, ~28 min/epoch |
| 2727 | classical baselines | cpu / mcore-n01 | - | 8480 test samples, ~2.2s/sample, ~5h total |

All FRESH=1 (post-CVPR-audit code). Previous Wave 1 jobs (2683-2686) used pre-audit code and are **invalid**.

### Remaining Experiment Queue:
- **Ablations**: no_physics_loss, no_coverage_attention, no_trajectory_mask, no_coverage_density, no_tx_position
- **Coverage sweeps**: 1%, 5%, 20% (10% = trajectory_full)
- **Cross-eval**: traj_to_uniform, uniform_to_traj
- **DL baselines**: supervised_unet, radio_unet, rmdm_baseline
- **Extras**: ablation_small_unet, num_trajectories_sweep

### After Training Completes:
1. Review classical baseline results: `experiments/eval_results/baselines.json`
2. Resume GPU jobs after 48h timeout (checkpoint resume, no FRESH flag)
3. Evaluate best checkpoints: `python scripts/evaluate.py checkpoint=<path>`
4. Submit ablation wave (2 at a time on 2g.35gb)
5. Analyze uncertainty + generate figures
6. Write paper
