# TrajectoryDiff

**Trajectory-Conditioned Diffusion for Radio Map Generation with Calibrated Uncertainty**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Lightning 2.0+](https://img.shields.io/badge/Lightning-2.0+-792EE5.svg?logo=lightning&logoColor=white)](https://lightning.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-199%20passing-brightgreen.svg)]()

---

## Overview

TrajectoryDiff is a trajectory-conditioned diffusion model for generating complete indoor radio maps from sparse RSS (Received Signal Strength) measurements collected along human walking paths. Unlike prior methods that assume uniformly distributed sensor observations, TrajectoryDiff explicitly models the spatial bias inherent in crowdsourced trajectory data — where corridors and accessible areas are densely sampled while locked rooms and restricted zones remain entirely unobserved. Through coverage-aware attention, physics-informed training losses, and ensemble-based uncertainty quantification, TrajectoryDiff achieves state-of-the-art radio map reconstruction accuracy while providing calibrated confidence estimates for unobserved regions.

> **Paper:** M. Y. Ansari, "TrajectoryDiff: Trajectory-Conditioned Diffusion for Radio Map Generation with Calibrated Uncertainty," *IEEE Transactions on Wireless Communications*, 2026. (Under review)

---

## Key Results

Evaluated on 8,480 test samples from the RadioMapSeer dataset. RMSE is reported in dBm on free-space unobserved regions (the primary metric for practical deployment).

| Method | Type | Free-Unobs RMSE (dBm) | SSIM |
|:-------|:-----|----------------------:|-----:|
| **TrajectoryDiff (ensemble, N=10)** | Ours | **5.68** | 0.947 |
| **TrajectoryDiff (single sample)** | Ours | **7.33** | 0.935 |
| Uniform Baseline (diffusion) | DL | 7.94 | 0.895 |
| IDW (best classical) | Classical | 8.09 | 0.995 |
| Gaussian Process (Kriging) | Classical | 8.16 | 0.995 |
| RadioUNet (Levie et al., 2021) | DL | 8.69 | 0.860 |
| RMDM (Xu et al., 2025) | DL | 21.13 | 0.825 |

**Highlights:**
- TrajectoryDiff (single) reduces free-unobserved RMSE by **9.4%** over IDW and **10.2%** over Kriging.
- Ensemble averaging (N=10) yields a **29.8%** improvement over IDW (5.68 vs 8.09 dBm).
- Coverage-aware attention is the most critical component: ablation increases error by +18.92 dBm.
- Noise-robust inference: only 0.08 dBm RMSE variation across measurement noise levels (sigma = 1--8 dBm).
- Well-calibrated uncertainty: 1.97x unobserved/observed uncertainty ratio; 0.745 error-uncertainty correlation.

---

## Architecture

```
                        Conditioning Pipeline
  ┌─────────────────┐
  │  Building Map   │──┐
  │  Sparse RSS     │──┤   ┌──────────────────┐     ┌────────────────────────────┐
  │  Trajectory Mask│──┼──>│ ConditionEncoder │────>│                            │
  │  Coverage       │──┤   │  (5ch -> 64ch)   │     │   CoverageAwareUNet        │
  │  Density        │──┤   │  CNN + TX Embed  │     │                            │
  │  TX Position    │──┘   └──────────────────┘     │   65ch input (cond + x_t)  │
  │   (x, y)       │              │                 │                            │
  └─────────────────┘              │   concat        │   Encoder ──> Middle ──>   │
                                   ├────────────>    │   Decoder with coverage-   │
                     x_t ──────────┘                 │   modulated attention at   │
                  (noisy radio map)                  │   each resolution level    │
                                                     │                            │
                                                     │          ──> noise pred    │
                                                     └────────────────────────────┘

                                      │
                                      v
                          DDIM Sampling (T=50 steps)
                          Ensemble (N samples) for
                          uncertainty quantification
```

**Model specifications:**
- **Parameters:** 61.9M trainable
- **Conditioning:** 5 input channels encoded to 64-channel spatial features via a multi-scale CNN with learnable TX position embedding
- **Backbone:** CoverageAwareUNet -- standard UNet with additive log-bias attention modulated by observation density at each resolution level
- **Diffusion:** Gaussian DDPM training (1000 timesteps), DDIM inference (50 steps default)
- **Physics losses:** Trajectory consistency + distance decay regularization with 30-epoch warmup and 20-epoch linear ramp

---

## Installation

### Conda (Recommended)

```bash
git clone https://github.com/YaqoobAnsari/TrajectoryDiff.git
cd TrajectoryDiff

conda env create -f environment.yaml
conda activate trajdiff
```

### pip

```bash
git clone https://github.com/YaqoobAnsari/TrajectoryDiff.git
cd TrajectoryDiff

python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

pip install -e ".[dev]"
```

### Verify Installation

```bash
# Run the full test suite (199 tests)
pytest tests/ -v

# Quick pipeline smoke test (~15 seconds on CPU)
python scripts/smoke_test_quick.py
```

---

## Dataset Setup

TrajectoryDiff uses the [RadioMapSeer](https://ieee-dataport.org/) dataset, comprising 701 building floor plans with 80 transmitter positions each (56,080 radio maps total) at 256x256 resolution.

### Download and Placement

1. Download the RadioMapSeer dataset from [IEEE DataPort](https://ieee-dataport.org/).
2. Extract and place the data under `data/raw/`:

```
data/
└── raw/
    ├── DPM/                  # Dense pathloss maps (256x256 PNG)
    │   ├── buildings_complete/
    │   └── ...
    ├── building_footprints/  # Binary building maps
    └── Tx_pos/               # Transmitter positions (JSON)
```

The dataset provides dense ground-truth pathloss maps. Trajectory-based sparse sampling is simulated on top of these dense maps during training, using three trajectory generation strategies:

| Strategy | Description |
|:---------|:------------|
| **Shortest-path** | A* navigation between random accessible points |
| **Random-walk** | Momentum-based walk respecting wall boundaries |
| **Corridor-biased** | Prefers corridor regions, simulating realistic pedestrian behavior |

Default configuration uses approximately 1.5% free-space coverage from trajectory sampling.

---

## Quick Start

### Training

```bash
# Default configuration
python scripts/train.py

# Full model with all proposed components
python scripts/train.py experiment=trajectory_full

# Override hyperparameters via Hydra
python scripts/train.py \
    data.loader.batch_size=32 \
    training.max_epochs=200 \
    training.optimizer.lr=1e-4

# Inspect resolved configuration without running
python scripts/train.py --cfg job
```

### Evaluation

```bash
# Evaluate a trained checkpoint
python scripts/evaluate.py \
    checkpoint=experiments/trajectory_full/<run_dir>/checkpoints/best.ckpt

# Classical baselines (IDW, RBF, Kriging)
python scripts/run_baselines.py
```

### Smoke Tests

```bash
# Fast manual smoke test (~15 seconds, CPU)
python scripts/smoke_test_quick.py

# Full Lightning smoke test (2 maps, 3 epochs)
python scripts/smoke_test.py
```

---

## Experiments

TrajectoryDiff includes 19 experiment configurations for systematic evaluation, managed via [Hydra](https://hydra.cc/). All configurations are located in `configs/experiment/`.

### Core Comparisons

| Config | Description |
|:-------|:------------|
| `trajectory_full` | Full TrajectoryDiff model with all proposed components |
| `trajectory_baseline` | Trajectory-conditioned diffusion without physics losses or coverage attention |
| `uniform_baseline` | Diffusion model with uniform random sampling (standard assumption) |

```bash
python scripts/train.py experiment=trajectory_full
python scripts/train.py experiment=trajectory_baseline
python scripts/train.py experiment=uniform_baseline
```

### Deep Learning Baselines

| Config | Reference | Description |
|:-------|:----------|:------------|
| `supervised_unet` | -- | Same architecture, direct MSE regression (no diffusion) |
| `radio_unet` | Levie et al., 2021 | RadioUNet encoder-decoder with raw channel concatenation |
| `rmdm_baseline` | Xu et al., 2025 | RMDM dual-UNet diffusion with anchor fusion |

```bash
python scripts/train.py experiment=supervised_unet
python scripts/train.py experiment=radio_unet
python scripts/train.py experiment=rmdm_baseline
```

### Ablation Studies

Each ablation removes one proposed component to measure its contribution:

| Config | Component Removed |
|:-------|:------------------|
| `ablation_no_coverage_attention` | Coverage-aware attention mechanism |
| `ablation_no_physics_loss` | Physics-informed training losses |
| `ablation_no_coverage_density` | Coverage density conditioning channel |
| `ablation_no_trajectory_mask` | Trajectory mask conditioning channel |
| `ablation_no_tx_position` | Transmitter position embedding |
| `ablation_small_unet` | Reduced UNet capacity (Small vs Medium) |

```bash
python scripts/train.py experiment=ablation_no_coverage_attention
python scripts/train.py experiment=ablation_no_physics_loss
# ... etc.
```

### Coverage Sweep

Evaluates performance as a function of spatial observation density:

| Config | Effective Coverage |
|:-------|:-------------------|
| `coverage_sweep_1pct` | ~0.25% free-space |
| `coverage_sweep_5pct` | ~0.81% free-space |
| `coverage_sweep_10pct` | ~1.5% free-space (default) |
| `coverage_sweep_20pct` | ~3.8% free-space |

```bash
python scripts/train.py experiment=coverage_sweep_5pct
```

### Cross-Evaluation and Sweeps

| Config | Description |
|:-------|:------------|
| `cross_eval_traj_to_uniform` | Model trained on trajectories, tested on uniform sampling |
| `cross_eval_uniform_to_traj` | Model trained on uniform, tested on trajectory sampling |
| `num_trajectories_sweep` | Hydra multirun over number of trajectories per sample |

---

## Configuration

All configurations use [Hydra](https://hydra.cc/) with [OmegaConf](https://omegaconf.readthedocs.io/). Key configuration groups:

```
configs/
├── config.yaml            # Main entry point
├── data/
│   └── radiomapseer.yaml  # Dataset paths, sampling strategy, augmentations
├── model/
│   └── trajectory_diffusion.yaml  # Architecture, diffusion schedule, attention
├── training/
│   └── default.yaml       # Optimizer, scheduler, epochs, checkpointing
└── experiment/            # 19 experiment overrides (see above)
```

Override any parameter from the command line:

```bash
python scripts/train.py \
    data.sampling.strategy=trajectory \
    data.sampling.trajectory.num_trajectories=5 \
    model.coverage_attention.enabled=true \
    model.physics.enabled=true \
    training.optimizer.lr=1e-4
```

---

## Project Structure

```
TrajectoryDiff/
├── configs/                           # Hydra configuration
│   ├── config.yaml                   # Main config entry point
│   ├── data/                         # Dataset and sampling configs
│   ├── model/                        # Architecture configs
│   ├── training/                     # Training hyperparameters
│   └── experiment/                   # 19 experiment configs
├── src/
│   ├── data/
│   │   ├── dataset.py                # RadioMapDataset, UniformSamplingDataset
│   │   ├── datamodule.py             # Lightning DataModule
│   │   ├── floor_plan.py             # Floor plan processing, dBm conversion
│   │   ├── transforms.py            # Data augmentations
│   │   └── trajectory_sampler.py     # Trajectory generation (3 strategies)
│   ├── models/
│   │   ├── diffusion/
│   │   │   ├── ddpm.py               # GaussianDiffusion, DDIMSampler
│   │   │   ├── unet.py              # UNet backbone (Small/Medium/Large)
│   │   │   ├── coverage_unet.py     # CoverageAwareUNet (proposed)
│   │   │   └── attention.py         # CoverageAwareAttention (proposed)
│   │   ├── encoders/
│   │   │   └── condition_encoder.py  # TrajectoryConditionedUNet
│   │   └── baselines/
│   │       ├── interpolation.py      # IDW, RBF, Kriging, NN
│   │       ├── supervised_unet.py    # Supervised UNet baseline
│   │       ├── radio_unet.py        # RadioUNet (Levie et al., 2021)
│   │       └── rmdm.py              # RMDM (Xu et al., 2025)
│   ├── training/
│   │   ├── diffusion_module.py       # Lightning DiffusionModule (EMA, scheduling)
│   │   ├── losses.py                # TrajectoryDiffLoss (physics-informed)
│   │   ├── inference.py             # DiffusionInference, uncertainty estimation
│   │   └── callbacks.py            # W&B logging, metrics, gradient monitoring
│   ├── evaluation/
│   │   └── metrics.py               # RMSE, SSIM, trajectory-aware metrics (dBm)
│   └── utils/
│       └── visualization.py         # Radio map visualization
├── scripts/
│   ├── train.py                     # Hydra-based training entry point
│   ├── evaluate.py                  # Evaluation with dBm-scale metrics
│   ├── run_baselines.py             # Classical baseline evaluation
│   ├── smoke_test.py                # Full Lightning smoke test
│   ├── smoke_test_quick.py          # Fast CPU smoke test (~15s)
│   ├── analyze_uncertainty.py       # Uncertainty calibration analysis
│   └── generate_figures.py          # Paper figure generation
├── tests/                            # 199 tests across 9 test files
├── notebooks/                        # Jupyter notebooks for exploration
├── data/                             # Dataset directory (not tracked)
├── experiments/                      # Experiment outputs (not tracked)
├── environment.yaml                  # Conda environment specification
├── pyproject.toml                   # Project metadata and tool configuration
├── CITATION.bib                     # BibTeX citation
└── LICENSE                          # MIT License
```

---

## Citation

If you use this code or method in your research, please cite:

```bibtex
@article{ansari2026trajectorydiff,
  title={TrajectoryDiff: Trajectory-Conditioned Diffusion for Radio Map Generation
         with Calibrated Uncertainty},
  author={Ansari, Mohammed Yaqoob},
  journal={IEEE Transactions on Wireless Communications},
  year={2026},
  note={Under review}
}
```

---

## License

This project is released under the [MIT License](LICENSE).

---

## Acknowledgments

- The [RadioMapSeer](https://ieee-dataport.org/) dataset was provided by Yapar et al. We gratefully acknowledge the authors for making this resource publicly available.
- This work builds upon [PyTorch](https://pytorch.org/), [PyTorch Lightning](https://lightning.ai/), and [Hydra](https://hydra.cc/).
