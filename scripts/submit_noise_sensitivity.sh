#!/bin/bash
#SBATCH --job-name=trajdiff-eval
#SBATCH --partition=gpu2
#SBATCH --nodelist=deepnet2
#SBATCH --mcs-label=unicellular
#SBATCH --gres=gpu:nvidia_h200_2g.35gb:1
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=experiments/logs/noise_sensitivity_%A.out
#SBATCH --error=experiments/logs/noise_sensitivity_%A.err
#
# Noise Sensitivity: Evaluate trajectory_full under different RSS noise levels.
# Model was trained with rss_noise_std=2.0. Test with sigma={1, 2, 4, 8}.
# No retraining needed — just change test-time noise.
#
# Usage:
#   sbatch scripts/submit_noise_sensitivity.sh

set -euo pipefail

echo "=============================================="
echo "Noise Sensitivity Experiment (SLURM)"
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

# Load model once
print('Loading model...')
inference = DiffusionInference.from_checkpoint(checkpoint_path, device=device, use_ema=True)

noise_levels = [1.0, 2.0, 4.0, 8.0]  # dB (trained on 2.0)
results = {}

for noise_std in noise_levels:
    print(f'\n{\"=\"*60}')
    print(f'Evaluating with rss_noise_std={noise_std} dB')
    print(f'{\"=\"*60}')

    # Recreate datamodule with different noise level
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
        rss_noise_std=noise_std,
        position_noise_std=0.5,
    )
    dm.setup('test')
    test_loader = dm.test_dataloader()

    all_rmse_free_unobs = []
    all_rmse_free_space = []
    all_rmse_free_obs = []
    all_ssim_free_space = []

    for batch in tqdm(test_loader, desc=f'sigma={noise_std}'):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        condition = {
            'building_map': batch.get('building_map'),
            'sparse_rss': batch.get('sparse_rss'),
            'trajectory_mask': batch.get('trajectory_mask'),
            'coverage_density': batch.get('coverage_density'),
            'tx_position': batch.get('tx_position'),
        }
        condition = {k: v for k, v in condition.items() if v is not None}

        samples = inference.sample(condition, use_ddim=True, progress=False)
        samples_dbm = denormalize_radio_map(samples)
        gt_dbm = denormalize_radio_map(batch['radio_map'])

        for b in range(samples_dbm.shape[0]):
            bmap = batch['building_map'][b, 0].cpu().numpy()
            tmask = batch['trajectory_mask'][b, 0].cpu().numpy() > 0.5
            pred_np = samples_dbm[b, 0].cpu().numpy()
            gt_np = gt_dbm[b, 0].cpu().numpy()

            free_space = bmap > 0.0
            free_obs = free_space & tmask
            free_unobs = free_space & ~tmask

            if free_unobs.any():
                diff = pred_np[free_unobs] - gt_np[free_unobs]
                all_rmse_free_unobs.append(float(np.sqrt(np.mean(diff ** 2))))
            if free_space.any():
                diff = pred_np[free_space] - gt_np[free_space]
                all_rmse_free_space.append(float(np.sqrt(np.mean(diff ** 2))))
                all_ssim_free_space.append(
                    compute_masked_ssim(pred_np, gt_np, free_space, data_range=139.0)
                )
            if free_obs.any():
                diff = pred_np[free_obs] - gt_np[free_obs]
                all_rmse_free_obs.append(float(np.sqrt(np.mean(diff ** 2))))

    result = {
        'rss_noise_std': noise_std,
        'trained_noise_std': 2.0,
        'num_samples': len(all_rmse_free_unobs),
        'rmse_free_unobs_dbm': float(np.mean(all_rmse_free_unobs)),
        'rmse_free_unobs_dbm_std': float(np.std(all_rmse_free_unobs)),
        'rmse_free_space_dbm': float(np.mean(all_rmse_free_space)),
        'rmse_free_obs_dbm': float(np.mean(all_rmse_free_obs)),
        'ssim_free_space': float(np.nanmean(all_ssim_free_space)),
        'per_sample_rmse_free_unobs_dbm': all_rmse_free_unobs,
    }
    results[f'sigma={noise_std}'] = result

    print(f'  Free-unobs RMSE: {result[\"rmse_free_unobs_dbm\"]:.2f} +/- {result[\"rmse_free_unobs_dbm_std\"]:.2f} dBm')
    print(f'  Free-space RMSE: {result[\"rmse_free_space_dbm\"]:.2f} dBm')
    print(f'  Free-obs RMSE:   {result[\"rmse_free_obs_dbm\"]:.2f} dBm')
    print(f'  SSIM (free):     {result[\"ssim_free_space\"]:.4f}')

# Save results
out_dir = Path('experiments/eval_results/noise_sensitivity')
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / 'noise_sensitivity.json', 'w') as f:
    json.dump(results, f, indent=2)

# Summary table
print(f'\n{\"=\"*60}')
print('Noise Sensitivity — Summary')
print(f'{\"=\"*60}')
print(f'{\"sigma (dB)\":<12} {\"Free-unobs RMSE\":>18} {\"Free-obs RMSE\":>15} {\"SSIM (free)\":>12} {\"Note\":>12}')
print('-' * 72)
for key in sorted(results.keys(), key=lambda x: results[x]['rss_noise_std']):
    r = results[key]
    note = '<-- trained' if r['rss_noise_std'] == 2.0 else ''
    print(f'{r[\"rss_noise_std\"]:<12.1f} {r[\"rmse_free_unobs_dbm\"]:>14.2f} dBm {r[\"rmse_free_obs_dbm\"]:>11.2f} dBm {r[\"ssim_free_space\"]:>12.4f} {note:>12}')

print(f'\nResults saved to: {out_dir / \"noise_sensitivity.json\"}')
"

echo ""
echo "=============================================="
echo "Noise sensitivity complete at $(date)"
echo "=============================================="
