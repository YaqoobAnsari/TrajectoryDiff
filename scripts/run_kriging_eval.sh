#!/bin/bash
#SBATCH --job-name=kriging_eval
#SBATCH --partition=cpu
#SBATCH --nodelist=mcore-n01
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mcs-label=unicellular
#SBATCH --output=experiments/logs/kriging_eval_%j.out
#SBATCH --error=experiments/logs/kriging_eval_%j.err

# Kriging (GP) baseline evaluation on full test set
# CPU-only, no GPU needed. ~2.8h estimated (1.2s/sample × 8480 samples).

set -euo pipefail

echo "=== Kriging Baseline Evaluation ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-8}"
echo ""

# Setup environment
export PATH="${CONDA_PREFIX:-$HOME/.conda/envs/trajdiff}/bin:$PATH"
cd "$(dirname "$0")/.."

python -c "
import sys, time, json
sys.path.insert(0, 'src')
import numpy as np
from skimage.metrics import structural_similarity
from tqdm import tqdm
from pathlib import Path

from data import RadioMapDataModule
from models.baselines.interpolation import KrigingBaseline

def denormalize_to_dbm(x, min_val=-186.0, max_val=-47.0):
    return ((x + 1) / 2) * (max_val - min_val) + min_val

print('Setting up test data...')
datamodule = RadioMapDataModule(
    data_dir='data/raw', batch_size=1, num_workers=4,
    sampling_strategy='trajectory', num_trajectories=3, points_per_trajectory=100,
)
datamodule.setup('test')
test_loader = datamodule.test_dataloader()

kriging = KrigingBaseline()
results = {
    'rmse_free_unobs_dbm': [], 'rmse_free_space_dbm': [],
    'rmse_free_obs_dbm': [], 'ssim_free_space': [],
}
timing_total = 0.0
total = 0

print(f'Running optimized Kriging on {len(test_loader)} test samples...')
for batch in tqdm(test_loader, desc='Kriging eval'):
    for i in range(batch['radio_map'].shape[0]):
        gt = batch['radio_map'][i, 0].numpy()
        sparse_rss = batch['sparse_rss'][i, 0].numpy()
        traj_mask = batch['trajectory_mask'][i, 0].numpy()
        building_map = batch['building_map'][i, 0].numpy()

        free_space_mask = building_map > 0.0
        building_mask = ~free_space_mask
        obs_mask = traj_mask > 0.5
        free_obs_mask = free_space_mask & obs_mask
        free_unobs_mask = free_space_mask & ~obs_mask

        t0 = time.time()
        pred = kriging(sparse_rss, traj_mask)
        timing_total += time.time() - t0

        pred_dbm = denormalize_to_dbm(pred)
        gt_dbm = denormalize_to_dbm(gt)
        diff_dbm = pred_dbm - gt_dbm

        if free_space_mask.any():
            results['rmse_free_space_dbm'].append(
                float(np.sqrt(np.mean(diff_dbm[free_space_mask] ** 2)))
            )
            fs_pred = pred_dbm.copy(); fs_gt = gt_dbm.copy()
            fs_pred[building_mask] = 0.0; fs_gt[building_mask] = 0.0
            ssim_fs = structural_similarity(fs_gt, fs_pred, data_range=139.0)
            results['ssim_free_space'].append(float(ssim_fs))
        if free_obs_mask.any():
            results['rmse_free_obs_dbm'].append(
                float(np.sqrt(np.mean(diff_dbm[free_obs_mask] ** 2)))
            )
        if free_unobs_mask.any():
            results['rmse_free_unobs_dbm'].append(
                float(np.sqrt(np.mean(diff_dbm[free_unobs_mask] ** 2)))
            )

        total += 1
        if total % 1000 == 0:
            avg = np.mean(results['rmse_free_unobs_dbm'])
            print(f'  [{total}] free-unobs RMSE: {avg:.2f} dBm, {timing_total/total*1000:.0f}ms/sample')

summary = {}
for key in results:
    if results[key]:
        summary[f'{key}_mean'] = float(np.nanmean(results[key]))
        summary[f'{key}_std'] = float(np.nanstd(results[key]))
summary['num_samples'] = total
summary['avg_time_ms'] = round(timing_total / max(total, 1) * 1000, 2)
summary['total_time_s'] = round(timing_total, 2)

print()
print('='*60)
print('KRIGING BASELINE RESULTS (full test set)')
print('='*60)
for k, v in summary.items():
    print(f'  {k}: {v}')

output_path = Path('experiments/eval_results/kriging.json')
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f'Saved to {output_path}')
"

echo ""
echo "=== Done ==="
echo "Date: $(date)"
