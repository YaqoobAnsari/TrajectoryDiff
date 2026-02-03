# TrajectoryDiff

**Trajectory-Conditioned Diffusion Models for Indoor Radio Map Generation**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/lightning-2.0+-purple.svg)](https://lightning.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

TrajectoryDiff is a research project exploring **trajectory-conditioned diffusion models** for generating complete indoor radio maps from sparse measurements collected along human walking paths.

### The Problem

Existing radio map generation methods assume **uniformly random sampling** — measurements scattered randomly across the space. In reality, radio measurements come from **trajectories** — paths people actually walk:

- Corridors and walkable areas have dense samples
- Locked rooms and restricted areas have **zero samples**
- Samples are temporally correlated (sequential along paths)

This mismatch between assumption and reality degrades performance.

### Our Solution

TrajectoryDiff explicitly models the **trajectory structure** of real-world sampling:

1. **Trajectory Encoder**: Captures temporal correlation and path geometry
2. **Coverage-Aware Conditioning**: Model knows where data exists vs. doesn't
3. **Physics-Informed Training**: Uses radio propagation constraints
4. **Uncertainty Quantification**: Higher uncertainty in unobserved regions

## Installation

```bash
# Clone the repository
git clone https://github.com/YaqoobAnsari/TrajectoryDiff.git
cd TrajectoryDiff

# Create conda environment
conda env create -f environment.yaml
conda activate trajdiff

# Install in development mode (optional)
pip install -e .
```

## Quick Start

### 1. Download Data

Download the RadioMapSeer dataset from [IEEE DataPort](https://ieee-dataport.org/) and place it in `data/raw/RadioMapSeer/`.

### 2. Train a Model

```bash
# Default configuration
python scripts/train.py

# With experiment config
python scripts/train.py experiment=trajectory_baseline

# Override parameters
python scripts/train.py data.loader.batch_size=32 training.max_epochs=50
```

### 3. Evaluate

```bash
python scripts/evaluate.py checkpoint=experiments/trajectory_baseline/checkpoints/best.ckpt
```

## Project Structure

```
trajectorydiff/
├── configs/                    # Hydra configuration files
│   ├── config.yaml            # Main config
│   ├── data/                  # Dataset configs
│   ├── model/                 # Model configs
│   ├── training/              # Training configs
│   └── experiment/            # Experiment-specific overrides
├── src/
│   ├── data/                  # Data loading and trajectory sampling
│   ├── models/                # Model definitions
│   ├── training/              # Training logic
│   ├── evaluation/            # Metrics and evaluation
│   └── utils/                 # Utilities
├── scripts/                   # Entry point scripts
├── notebooks/                 # Jupyter notebooks
├── tests/                     # Unit tests
└── docs/                      # Documentation
```

## Key Concepts

| Term | Definition |
|------|------------|
| **Radio Map** | 2D function R(x,y) = signal strength at each location |
| **RSS** | Received Signal Strength in dBm (-50 excellent, -100 poor) |
| **Trajectory Sampling** | Samples collected along walking paths (biased, correlated) |
| **Uniform Sampling** | Samples at random locations (idealized, unrealistic) |
| **Blind Spots** | Regions never visited (locked rooms, restricted areas) |

## Configuration

We use [Hydra](https://hydra.cc/) for configuration management. Key configuration groups:

- **data**: Dataset, sampling strategy, augmentation
- **model**: Architecture, diffusion parameters, trajectory encoder
- **training**: Optimizer, scheduler, checkpointing

Override any parameter from command line:

```bash
python scripts/train.py \
    data.sampling.strategy=trajectory \
    data.sampling.trajectory.num_trajectories=5 \
    model.unet.base_channels=64 \
    training.optimizer.lr=1e-4
```

## Experiments

See `docs/PLAN.md` for the full research plan. Key experiments:

| Experiment | Description |
|------------|-------------|
| Baseline Comparison | TrajectoryDiff vs interpolation, U-Net, standard diffusion |
| Sampling Ablation | Uniform vs trajectory sampling comparison |
| Encoder Ablation | GNN vs Transformer vs Hybrid trajectory encoder |
| Coverage Study | Performance vs sampling density |

## Citation

If you use this code in your research, please cite:

```bibtex
@article{trajectorydiff2025,
  title={TrajectoryDiff: Trajectory-Conditioned Diffusion for Radio Map Generation},
  author={Your Name},
  journal={arXiv preprint},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- RadioMapSeer dataset from IEEE DataPort
- Built with [PyTorch Lightning](https://lightning.ai/) and [Hydra](https://hydra.cc/)
