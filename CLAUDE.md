# TrajectoryDiff: Trajectory-Conditioned Radio Map Generation

## Project Overview

Research project exploring trajectory-conditioned diffusion models for indoor radio map generation. Core insight: real RSS measurements come from human walking paths (corridors, accessible areas), not uniform random sampling. We model this bias explicitly.

**Goal**: Generate complete radio maps from sparse trajectory-sampled RSS measurements, with proper uncertainty quantification for unobserved regions.

**Target**: ECCV-level publication demonstrating trajectory-aware diffusion outperforms uniform-sampling assumptions.

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
│   └── experiment/
│       ├── trajectory_baseline.yaml  # Trajectory sampling experiment
│       └── uniform_baseline.yaml     # Uniform sampling baseline
├── src/                       # Source code (TO BE IMPLEMENTED)
│   ├── data/                  # Data loading and processing
│   │   ├── dataset.py         # PyTorch Dataset classes
│   │   ├── datamodule.py      # Lightning DataModule
│   │   ├── transforms.py      # Data augmentations
│   │   └── trajectory_sampler.py  # Trajectory generation logic
│   ├── models/
│   │   ├── diffusion/         # Diffusion model components
│   │   ├── encoders/          # Floor plan & trajectory encoders
│   │   └── baselines/         # Baseline models
│   ├── training/              # Training logic
│   ├── evaluation/            # Evaluation metrics
│   └── utils/                 # Utilities
├── scripts/                   # Entry point scripts
│   └── train.py              # Training script (skeleton)
├── notebooks/                 # Jupyter notebooks for exploration
├── experiments/               # Experiment outputs (gitignored)
├── data/                      # Data directory (gitignored)
│   ├── raw/                   # Original RadioMapSeer data
│   └── processed/             # Preprocessed data
├── tests/                     # Unit tests
├── docs/
│   └── PLAN.md               # Detailed research plan
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

## Common Commands

```bash
# Environment setup
conda env create -f environment.yaml
conda activate trajdiff

# Training
python scripts/train.py                                    # Default config
python scripts/train.py model=unet_small data.batch_size=16  # Override params
python scripts/train.py experiment=trajectory_baseline     # Named experiment
python scripts/train.py -m model=unet_small,unet_large     # Multirun sweep

# Evaluation
python scripts/evaluate.py checkpoint=path/to/checkpoint.ckpt
python scripts/evaluate.py checkpoint=path/to/checkpoint.ckpt data.sampling=trajectory

# Visualization
python scripts/visualize.py checkpoint=path/to/checkpoint.ckpt

# Testing
pytest tests/ -v
pytest tests/test_data.py -v -k "test_trajectory"

# Linting
ruff check src/
black src/ --check
mypy src/
```

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

1. `configs/config.yaml` - Entry point for all configuration
2. `src/data/trajectory_sampler.py` - Core trajectory generation logic
3. `src/models/diffusion/trajectory_diffusion.py` - Main model
4. `src/training/trainer.py` - Lightning training module
5. `docs/PLAN.md` - Detailed research plan and milestones

## Dataset: RadioMapSeer

- **Location**: `data/raw/RadioMapSeer/`
- **Format**: Dense pathloss maps (256x256 or 512x512), floor plans, Tx locations
- **Preprocessing**: Run `python scripts/preprocess_data.py` after download
- **Note**: We simulate trajectory sampling on top of dense ground truth

## Trajectory Sampling Strategy

Three trajectory types implemented in `src/data/trajectory_sampler.py`:

1. **Shortest-path**: A* navigation between random accessible points
2. **Random-walk**: Momentum-based walk respecting walls
3. **Corridor-biased**: Prefers corridor regions (realistic human behavior)

Each trajectory produces: `[(t, x, y, rss), ...]` with optional noise injection.

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

## Current Focus

See `docs/PLAN.md` for detailed weekly plan. Current phase: **Phase 0: Environment Setup**

### Immediate Next Steps:
1. Download RadioMapSeer dataset
2. Implement data loading (src/data/dataset.py, datamodule.py)
3. Implement trajectory sampling (src/data/trajectory_sampler.py)
4. Create visualization notebook for data exploration

## Contact

For questions about the research direction or implementation details, refer to `docs/radio_map_deep_dive.md` for theoretical background.
