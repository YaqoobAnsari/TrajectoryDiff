#!/bin/bash
#SBATCH --job-name=trajdiff-validate
#SBATCH --partition=gpu2
#SBATCH --nodelist=deepnet2
#SBATCH --gres=gpu:nvidia_h200_2g.35gb:1
#SBATCH --mcs-label=unicellular
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=experiments/logs/validation_%j.out
#SBATCH --error=experiments/logs/validation_%j.err
#
# TrajectoryDiff: GPU Validation Job
#
# Quick validation to verify the full pipeline works on GPU before
# submitting the 16-experiment suite.
#
# Validates: CUDA, bf16, data loading, model training, checkpoint saving.
#
# Usage:
#   sbatch scripts/gpu_validation.sh

set -euo pipefail

echo "=============================================="
echo "TrajectoryDiff GPU Validation"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo ""

# Activate conda environment
# Directly prepend the trajdiff env to PATH (system conda is not accessible on compute nodes)
export PATH="/data1/yansari/.conda/envs/trajdiff/bin:$PATH"

# Navigate to project root
cd /data1/yansari/TrajectoryDiff

# Create log directory
mkdir -p experiments/logs

# ============================================================
# Test 1: GPU Check
# ============================================================
echo "--- Test 1: GPU Check ---"
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available!'
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
print(f'bf16 support: {torch.cuda.is_bf16_supported()}')
print('GPU check: PASSED')
"
echo ""

# ============================================================
# Test 2: Quick Smoke Test
# ============================================================
echo "--- Test 2: Quick Smoke Test ---"
python scripts/smoke_test_quick.py
echo "Smoke test: PASSED"
echo ""

# ============================================================
# Test 3: 1-Epoch Training with Hydra
# ============================================================
echo "--- Test 3: 1-Epoch Hydra Training ---"
export WANDB_MODE=offline

python scripts/train.py \
    training.max_epochs=1 \
    data.loader.batch_size=32 \
    data.loader.num_workers=8 \
    hardware.precision=bf16-mixed \
    hardware.slurm=true \
    logging.wandb.enabled=false \
    experiment.name=gpu_validation

echo "1-epoch training: PASSED"
echo ""

# ============================================================
# Summary
# ============================================================
echo "=============================================="
echo "All GPU validation tests PASSED!"
echo "Date: $(date)"
echo ""
echo "Next steps:"
echo "  bash scripts/submit_all.sh"
echo "=============================================="
