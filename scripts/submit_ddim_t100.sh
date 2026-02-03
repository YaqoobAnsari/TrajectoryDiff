#!/bin/bash
#SBATCH --job-name=trajdiff-eval
#SBATCH --partition=gpu2
#SBATCH --nodelist=deepnet2
#SBATCH --mcs-label=unicellular
#SBATCH --gres=gpu:nvidia_h200_2g.35gb:1
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=experiments/logs/ddim_t100_%A.out
#SBATCH --error=experiments/logs/ddim_t100_%A.err
#
# DDIM T=100 only — T=10,25,50 already completed (job 3142).
#
# Usage:
#   sbatch scripts/submit_ddim_t100.sh

set -euo pipefail

echo "=============================================="
echo "DDIM T=100 Evaluation (SLURM)"
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
import sys, os, json, time
from pathlib import Path

sys.path.insert(0, 'src')

import numpy as np
import torch
from tqdm import tqdm

from models.diffusion.ddpm import DDIMSampler
from training import DiffusionModule, DiffusionInference, denormalize_radio_map
from data import RadioMapDataModule
from evaluation.metrics import compute_masked_ssim

checkpoint_path = '$CHECKPOINT'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Loading model...')
inference = DiffusionInference.from_checkpoint(checkpoint_path, device=device, use_ema=True)
module = inference.module

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

ddim_steps = 100
print(f'\n{\"=\"*60}')
print(f'Evaluating DDIM T={ddim_steps}')
print(f'{\"=\"*60}')

module.ddim_sampler = DDIMSampler(module.diffusion, ddim_num_steps=ddim_steps)

all_rmse_free_unobs = []
all_rmse_free_space = []
all_rmse_free_obs = []
all_ssim_free_space = []
total_time = 0.0
total_samples = 0

for batch in tqdm(test_loader, desc=f'T={ddim_steps}'):
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    condition = {
        'building_map': batch.get('building_map'),
        'sparse_rss': batch.get('sparse_rss'),
        'trajectory_mask': batch.get('trajectory_mask'),
        'coverage_density': batch.get('coverage_density'),
        'tx_position': batch.get('tx_position'),
    }
    condition = {k: v for k, v in condition.items() if v is not None}

    t0 = time.time()
    samples = inference.sample(condition, use_ddim=True, progress=False)
    t1 = time.time()
    total_time += (t1 - t0)

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

    total_samples += samples_dbm.shape[0]

result = {
    'ddim_steps': ddim_steps,
    'num_samples': total_samples,
    'total_time_s': total_time,
    'time_per_sample_ms': (total_time / total_samples) * 1000,
    'rmse_free_unobs_dbm': float(np.mean(all_rmse_free_unobs)),
    'rmse_free_unobs_dbm_std': float(np.std(all_rmse_free_unobs)),
    'rmse_free_space_dbm': float(np.mean(all_rmse_free_space)),
    'rmse_free_obs_dbm': float(np.mean(all_rmse_free_obs)),
    'ssim_free_space': float(np.nanmean(all_ssim_free_space)),
    'per_sample_rmse_free_unobs_dbm': all_rmse_free_unobs,
}

print(f'  Free-unobs RMSE: {result[\"rmse_free_unobs_dbm\"]:.2f} +/- {result[\"rmse_free_unobs_dbm_std\"]:.2f} dBm')
print(f'  Free-space RMSE: {result[\"rmse_free_space_dbm\"]:.2f} dBm')
print(f'  Free-obs RMSE:   {result[\"rmse_free_obs_dbm\"]:.2f} dBm')
print(f'  SSIM (free):     {result[\"ssim_free_space\"]:.4f}')
print(f'  Time/sample:     {result[\"time_per_sample_ms\"]:.0f} ms')
print(f'  Total time:      {total_time:.0f}s')

# Save
out_dir = Path('experiments/eval_results/ddim_ablation')
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / 'ddim_t100.json', 'w') as f:
    json.dump(result, f, indent=2)
print(f'\nResults saved to: {out_dir / \"ddim_t100.json\"}')
"

echo ""
echo "=============================================="
echo "DDIM T=100 complete at $(date)"
echo "=============================================="
