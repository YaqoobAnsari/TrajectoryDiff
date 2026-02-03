#!/bin/bash
#SBATCH --job-name=trajdiff-eval
#SBATCH --partition=gpu2
#SBATCH --nodelist=deepnet2
#SBATCH --mcs-label=unicellular
#SBATCH --gres=gpu:nvidia_h200_2g.35gb:1
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=experiments/logs/uncertainty_ensemble_%A.out
#SBATCH --error=experiments/logs/uncertainty_ensemble_%A.err
#
# N=10 Uncertainty Ensemble: Run 10 DDIM samples per test input.
# Produces variance maps + ensemble mean improvement.
# Core argument for why diffusion > direct regression.
#
# Usage:
#   sbatch scripts/submit_uncertainty_ensemble.sh

set -euo pipefail

echo "=============================================="
echo "Uncertainty Ensemble N=10 (SLURM)"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="${CONDA_PREFIX:-$HOME/.conda/envs/trajdiff}/bin:$PATH"

cd "$(dirname "$0")/.."

CHECKPOINT="experiments/trajectory_full/2026-02-19_10-14-16/checkpoints/epoch=199-val_loss=0.0045.ckpt"

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
props = torch.cuda.get_device_properties(0)
vram = getattr(props, 'total_memory', getattr(props, 'total_mem', 0))
print(f'VRAM: {vram / 1024**3:.1f} GB')
"
echo ""

python -c "
import sys, os, json
from pathlib import Path

sys.path.insert(0, 'src')

import numpy as np
import torch
from tqdm import tqdm

from training import DiffusionInference, denormalize_radio_map
from data import RadioMapDataModule
from evaluation.metrics import compute_masked_ssim

checkpoint_path = '$CHECKPOINT'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_ENSEMBLE = 10

# Load model
print('Loading model...')
inference = DiffusionInference.from_checkpoint(checkpoint_path, device=device, use_ema=True)

# Setup data
print('Setting up data...')
dm = RadioMapDataModule(
    data_dir='data/raw/',
    batch_size=4,
    num_workers=8,
    train_ratio=0.7,
    val_ratio=0.15,
    sampling_strategy='trajectory',
    num_trajectories=3,
    points_per_trajectory=100,
    trajectory_method='mixed',
    rss_noise_std=2.0,
    position_noise_std=0.5,
)
dm.setup('test')
test_loader = dm.test_dataloader()

# Metrics accumulators
# Single-sample metrics (first run = baseline, same as T=50 eval)
single_rmse_free_unobs = []
# Ensemble mean metrics
ensemble_rmse_free_unobs = []
ensemble_rmse_free_space = []
ensemble_rmse_free_obs = []
ensemble_ssim_free_space = []
# Uncertainty metrics
mean_uncertainty_free_unobs = []
mean_uncertainty_free_obs = []
mean_uncertainty_free_space = []
# Calibration: correlation between uncertainty and error
calibration_data = []

print(f'Running {N_ENSEMBLE}-sample ensemble over {len(test_loader)} batches...')

for batch_idx, batch in enumerate(tqdm(test_loader, desc='Ensemble eval')):
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    condition = {
        'building_map': batch.get('building_map'),
        'sparse_rss': batch.get('sparse_rss'),
        'trajectory_mask': batch.get('trajectory_mask'),
        'coverage_density': batch.get('coverage_density'),
        'tx_position': batch.get('tx_position'),
    }
    condition = {k: v for k, v in condition.items() if v is not None}

    # Generate N samples
    samples_list = []
    for n in range(N_ENSEMBLE):
        s = inference.sample(condition, use_ddim=True, progress=False)
        samples_list.append(denormalize_radio_map(s))

    # Stack: (N, B, 1, H, W)
    samples_stack = torch.stack(samples_list, dim=0)
    ensemble_mean = samples_stack.mean(dim=0)     # (B, 1, H, W)
    ensemble_std = samples_stack.std(dim=0)        # (B, 1, H, W)
    single_sample = samples_list[0]                # (B, 1, H, W)

    gt_dbm = denormalize_radio_map(batch['radio_map'])

    for b in range(gt_dbm.shape[0]):
        bmap = batch['building_map'][b, 0].cpu().numpy()
        tmask = batch['trajectory_mask'][b, 0].cpu().numpy() > 0.5
        pred_ens = ensemble_mean[b, 0].cpu().numpy()
        pred_single = single_sample[b, 0].cpu().numpy()
        std_map = ensemble_std[b, 0].cpu().numpy()
        gt_np = gt_dbm[b, 0].cpu().numpy()

        free_space = bmap > 0.0
        free_obs = free_space & tmask
        free_unobs = free_space & ~tmask

        # Single-sample RMSE
        if free_unobs.any():
            diff_s = pred_single[free_unobs] - gt_np[free_unobs]
            single_rmse_free_unobs.append(float(np.sqrt(np.mean(diff_s ** 2))))

        # Ensemble mean RMSE
        if free_unobs.any():
            diff_e = pred_ens[free_unobs] - gt_np[free_unobs]
            ensemble_rmse_free_unobs.append(float(np.sqrt(np.mean(diff_e ** 2))))
        if free_space.any():
            diff_fs = pred_ens[free_space] - gt_np[free_space]
            ensemble_rmse_free_space.append(float(np.sqrt(np.mean(diff_fs ** 2))))
            ensemble_ssim_free_space.append(
                compute_masked_ssim(pred_ens, gt_np, free_space, data_range=139.0)
            )
        if free_obs.any():
            diff_fo = pred_ens[free_obs] - gt_np[free_obs]
            ensemble_rmse_free_obs.append(float(np.sqrt(np.mean(diff_fo ** 2))))

        # Uncertainty per region
        if free_unobs.any():
            mean_uncertainty_free_unobs.append(float(np.mean(std_map[free_unobs])))
        if free_obs.any():
            mean_uncertainty_free_obs.append(float(np.mean(std_map[free_obs])))
        if free_space.any():
            mean_uncertainty_free_space.append(float(np.mean(std_map[free_space])))

        # Calibration: per-pixel |error| vs uncertainty correlation
        if free_space.any():
            error_map = np.abs(pred_ens[free_space] - gt_np[free_space])
            unc_map = std_map[free_space]
            if len(error_map) > 10:
                corr = float(np.corrcoef(error_map, unc_map)[0, 1])
                calibration_data.append(corr)

results = {
    'n_ensemble': N_ENSEMBLE,
    'num_samples': len(ensemble_rmse_free_unobs),
    'single_sample': {
        'rmse_free_unobs_dbm': float(np.mean(single_rmse_free_unobs)),
        'rmse_free_unobs_dbm_std': float(np.std(single_rmse_free_unobs)),
    },
    'ensemble_mean': {
        'rmse_free_unobs_dbm': float(np.mean(ensemble_rmse_free_unobs)),
        'rmse_free_unobs_dbm_std': float(np.std(ensemble_rmse_free_unobs)),
        'rmse_free_space_dbm': float(np.mean(ensemble_rmse_free_space)),
        'rmse_free_obs_dbm': float(np.mean(ensemble_rmse_free_obs)),
        'ssim_free_space': float(np.nanmean(ensemble_ssim_free_space)),
        'per_sample_rmse_free_unobs_dbm': ensemble_rmse_free_unobs,
    },
    'uncertainty': {
        'mean_std_free_unobs_dbm': float(np.mean(mean_uncertainty_free_unobs)),
        'mean_std_free_obs_dbm': float(np.mean(mean_uncertainty_free_obs)),
        'mean_std_free_space_dbm': float(np.mean(mean_uncertainty_free_space)),
        'ratio_unobs_over_obs': float(np.mean(mean_uncertainty_free_unobs)) / max(float(np.mean(mean_uncertainty_free_obs)), 1e-8),
    },
    'calibration': {
        'mean_error_uncertainty_correlation': float(np.mean(calibration_data)) if calibration_data else None,
        'std_correlation': float(np.std(calibration_data)) if calibration_data else None,
    },
}

# Print summary
print(f'\n{\"=\"*60}')
print('Uncertainty Ensemble — Summary')
print(f'{\"=\"*60}')
s = results['single_sample']
e = results['ensemble_mean']
u = results['uncertainty']
c = results['calibration']
print(f'Single sample  RMSE (free-unobs): {s[\"rmse_free_unobs_dbm\"]:.2f} +/- {s[\"rmse_free_unobs_dbm_std\"]:.2f} dBm')
print(f'Ensemble N={N_ENSEMBLE} RMSE (free-unobs): {e[\"rmse_free_unobs_dbm\"]:.2f} +/- {e[\"rmse_free_unobs_dbm_std\"]:.2f} dBm')
improvement = s['rmse_free_unobs_dbm'] - e['rmse_free_unobs_dbm']
pct = improvement / s['rmse_free_unobs_dbm'] * 100
print(f'Ensemble improvement: {improvement:.2f} dBm ({pct:.1f}%)')
print(f'')
print(f'Ensemble RMSE (free-space): {e[\"rmse_free_space_dbm\"]:.2f} dBm')
print(f'Ensemble RMSE (free-obs):   {e[\"rmse_free_obs_dbm\"]:.2f} dBm')
print(f'Ensemble SSIM (free):       {e[\"ssim_free_space\"]:.4f}')
print(f'')
print(f'Uncertainty (mean std):')
print(f'  Free-unobs: {u[\"mean_std_free_unobs_dbm\"]:.2f} dBm')
print(f'  Free-obs:   {u[\"mean_std_free_obs_dbm\"]:.2f} dBm')
print(f'  Ratio (unobs/obs): {u[\"ratio_unobs_over_obs\"]:.2f}x')
print(f'')
if c['mean_error_uncertainty_correlation'] is not None:
    print(f'Calibration (error-uncertainty correlation): {c[\"mean_error_uncertainty_correlation\"]:.3f} +/- {c[\"std_correlation\"]:.3f}')
    print(f'  (>0 = uncertainty is higher where error is higher = well-calibrated)')

# Save
out_dir = Path('experiments/eval_results/uncertainty_ensemble')
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / 'uncertainty_ensemble.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nResults saved to: {out_dir / \"uncertainty_ensemble.json\"}')
"

echo ""
echo "=============================================="
echo "Uncertainty ensemble complete at $(date)"
echo "=============================================="
