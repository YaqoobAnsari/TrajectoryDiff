# TrajectoryDiff

**Trajectory-Conditioned Diffusion Models for Indoor Radio Map Generation**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/lightning-2.0+-purple.svg)](https://lightning.ai/)
[![Tests](https://img.shields.io/badge/tests-199%20passing-green.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

TrajectoryDiff is a research project exploring **trajectory-conditioned diffusion models** for generating complete indoor radio maps from sparse measurements collected along human walking paths.

### The Problem

Existing radio map generation methods assume **uniformly random sampling** -- measurements scattered randomly across the space. In reality, radio measurements come from **trajectories** -- paths people actually walk:

- Corridors and walkable areas have dense samples
- Locked rooms and restricted areas have **zero samples**
- Samples are temporally correlated (sequential along paths)

This mismatch between assumption and reality degrades performance.

### Our Solution

TrajectoryDiff explicitly models the **trajectory structure** of real-world sampling:

1. **Trajectory-Aware Conditioning**: 5-channel input (building map, sparse RSS, trajectory mask, coverage density, TX position)
2. **CoverageAwareUNet**: Novel attention mechanism modulated by coverage density (key ECCV contribution)
3. **Physics-Informed Training**: Trajectory consistency, coverage-weighted loss, distance decay regularization
4. **Uncertainty Quantification**: Multiple DDIM samples yield calibrated uncertainty maps

## Installation

### Option A: Virtual Environment (Recommended)

```bash
git clone https://github.com/YaqoobAnsari/TrajectoryDiff.git
cd TrajectoryDiff

python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Activate (Linux/Mac)
source venv/bin/activate

pip install -r requirements.txt
pip install -r requirements-dev.txt  # For testing/linting
```

### Option B: Conda Environment

```bash
conda env create -f environment.yaml
conda activate trajdiff
```

### Verify Installation

```bash
pytest tests/ -v                    # Run all 199 tests
python scripts/smoke_test_quick.py  # Quick pipeline check (~15s on CPU)
```

## Quick Start

### 1. Download Data

Download the RadioMapSeer dataset from [IEEE DataPort](https://ieee-dataport.org/) and place it in `data/raw/`.

### 2. Smoke Test

```bash
# Fast test: 2 maps, 2 training steps, ~15 seconds on CPU
python scripts/smoke_test_quick.py

# Full test: 2 maps, 3 epochs via Lightning Trainer
python scripts/smoke_test.py
```

### 3. Train a Model

```bash
# Default configuration
python scripts/train.py

# Full model with all features
python scripts/train.py experiment=trajectory_full

# Override parameters
python scripts/train.py data.loader.batch_size=32 training.max_epochs=50
```

### 4. Evaluate

```bash
python scripts/evaluate.py checkpoint=experiments/trajectory_full/checkpoints/best.ckpt
```

## Architecture

```
  building_map ──┐
  sparse_rss ────┤  ConditionEncoder                CoverageAwareUNet
  traj_mask ─────┤  (CNN + TX Spatial) ──►  cond ──► (coverage-modulated  ──► noise_pred
  coverage ──────┤       encoding              attention at each level)
  tx_position ───┘
```

### Novel Components

| Component | File | Description |
|-----------|------|-------------|
| CoverageAwareUNet | `src/models/diffusion/coverage_unet.py` | UNet with coverage-modulated attention blocks |
| CoverageAwareAttention | `src/models/diffusion/attention.py` | Attention weights scaled by coverage density |
| TrajectoryDiffLoss | `src/training/losses.py` | Combined physics-informed training loss |

## Project Structure

```
TrajectoryDiff/
├── configs/                    # Hydra configuration (19 experiment configs)
├── src/
│   ├── data/                  # Dataset, DataModule, trajectory sampling
│   ├── models/
│   │   ├── diffusion/         # DDPM, UNet, CoverageAwareUNet, attention
│   │   ├── encoders/          # TrajectoryConditionedUNet, ConditionEncoder
│   │   └── baselines/         # Classical (IDW, RBF) + DL (SupervisedUNet, RadioUNet, RMDM)
│   ├── training/              # DiffusionModule, losses, inference, callbacks
│   └── evaluation/            # RMSE, SSIM, trajectory-aware metrics (dBm)
├── scripts/                   # train.py, evaluate.py, smoke tests, SLURM scripts
├── tests/                     # 199 tests across 9 test files
└── docs/                      # Research plan, architecture docs, metrics guide
```

## Experiments

19 experiment configurations ready for systematic evaluation:

| Category | Experiments | Description |
|----------|-------------|-------------|
| **Main** | `trajectory_full`, `trajectory_baseline`, `uniform_baseline` | Full model vs baselines |
| **DL Baselines** | `supervised_unet`, `radio_unet`, `rmdm_baseline` | Deep learning comparison methods |
| **Ablations** | `ablation_no_{coverage_attention, physics_loss, trajectory_mask, coverage_density, tx_position}`, `ablation_small_unet` | Component contribution |
| **Cross-eval** | `cross_eval_traj_to_uniform`, `cross_eval_uniform_to_traj` | Generalization |
| **Sweeps** | `coverage_sweep_{1,5,10,20}pct`, `num_trajectories_sweep` | Coverage effects |

### DL Baselines

| Baseline | Architecture | Key Difference | Reference |
|----------|-------------|----------------|-----------|
| **Supervised UNet** | Same as ours (TrajectoryConditionedUNet) | Direct MSE, no diffusion | Ablation |
| **RadioUNet** | Standalone encoder-decoder UNet | Raw channel concat, no condition encoder | Levie et al., 2021 |
| **RMDM** | Dual-UNet diffusion + anchor fusion | Physics-conductor + detail-sculptor | Xu et al., 2025 |

```bash
# Run a specific experiment
python scripts/train.py experiment=trajectory_full
python scripts/train.py experiment=supervised_unet   # Supervised UNet baseline
python scripts/train.py experiment=radio_unet         # RadioUNet baseline
python scripts/train.py experiment=rmdm_baseline      # RMDM baseline

# Run on SLURM cluster (H200 GPUs)
sbatch scripts/run_experiments.sh trajectory_full
```

## Configuration

We use [Hydra](https://hydra.cc/) for configuration management:

```bash
python scripts/train.py \
    data.sampling.strategy=trajectory \
    data.sampling.trajectory.num_trajectories=5 \
    model.coverage_attention.enabled=true \
    model.physics.enabled=true \
    training.optimizer.lr=1e-4
```

## Key Concepts

| Term | Definition |
|------|------------|
| **Radio Map** | 2D function R(x,y) = signal strength at each location |
| **RSS** | Received Signal Strength in dBm (-50 excellent, -100 poor) |
| **Trajectory Sampling** | Samples collected along walking paths (biased, correlated) |
| **Uniform Sampling** | Samples at random locations (idealized, unrealistic) |
| **Blind Spots** | Regions never visited (locked rooms, restricted areas) |
| **Coverage Density** | Gaussian-smoothed trajectory mask indicating sample confidence |

## Citation

If you use this code in your research, please cite:

```bibtex
@article{trajectorydiff2026,
  title={TrajectoryDiff: Trajectory-Conditioned Diffusion for Radio Map Generation},
  author={Ansari, Yaqoob},
  journal={arXiv preprint},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- RadioMapSeer dataset from IEEE DataPort
- Built with [PyTorch Lightning](https://lightning.ai/) and [Hydra](https://hydra.cc/)
- GPU resources: CMUQ deepnet2 (NVIDIA H200)
