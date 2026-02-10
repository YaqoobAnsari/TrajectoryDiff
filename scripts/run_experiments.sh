#!/bin/bash
#SBATCH --job-name=trajdiff
#SBATCH --partition=gpu2
#SBATCH --nodelist=deepnet2
#SBATCH --mcs-label=unicellular
#SBATCH --time=36:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --output=experiments/logs/%x_%A_%a.out
#SBATCH --error=experiments/logs/%x_%A_%a.err
#
# TrajectoryDiff: SLURM Training Script
#
# Usage (single experiment):
#   sbatch --export=EXP_NAME=trajectory_full scripts/run_experiments.sh
#
# Usage (with MIG profile override via submit_experiment.sh):
#   bash scripts/submit_experiment.sh trajectory_full 7g.141gb
#
# Note: --gres and --time are passed by submit_experiment.sh as sbatch
# command-line args, which override the SBATCH headers above.
#
# Prerequisites:
#   - conda environment 'trajdiff' created
#   - Data at data/raw/
#   - wandb runs in offline mode by default (sync later with wandb sync)

set -euo pipefail

# ============================================================
# Environment Setup
# ============================================================
echo "=============================================="
echo "TrajectoryDiff Training (SLURM)"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

# wandb offline mode (sync later with: wandb login && wandb sync experiments/*/wandb/)
export WANDB_MODE=offline

# Activate conda environment
# Directly prepend the trajdiff env to PATH (system conda is not accessible on compute nodes)
export PATH="/data1/yansari/.conda/envs/trajdiff/bin:$PATH"

# Navigate to project root
cd /data1/yansari/TrajectoryDiff

# ============================================================
# Configuration
# ============================================================
EXP_NAME="${EXP_NAME:?ERROR: EXP_NAME not set. Use --export=EXP_NAME=trajectory_full}"
MIG_PROFILE="${MIG_PROFILE:-7g.141gb}"

# Set batch size based on MIG profile
case "$MIG_PROFILE" in
    7g.141gb)
        BATCH_SIZE="${BATCH_SIZE:-64}"
        NUM_WORKERS="${NUM_WORKERS:-12}"
        ;;
    2g.35gb)
        BATCH_SIZE="${BATCH_SIZE:-32}"
        NUM_WORKERS="${NUM_WORKERS:-8}"
        ;;
    1g.18gb)
        BATCH_SIZE="${BATCH_SIZE:-16}"
        NUM_WORKERS="${NUM_WORKERS:-4}"
        ;;
    *)
        echo "WARNING: Unknown MIG profile '$MIG_PROFILE', using defaults"
        BATCH_SIZE="${BATCH_SIZE:-32}"
        NUM_WORKERS="${NUM_WORKERS:-8}"
        ;;
esac

EPOCHS="${EPOCHS:-200}"
PRECISION="bf16-mixed"
WANDB_PROJECT="${WANDB_PROJECT:-trajectorydiff}"

echo "Experiment: $EXP_NAME"
echo "MIG Profile: $MIG_PROFILE"
echo "Batch Size: $BATCH_SIZE"
echo "Workers: $NUM_WORKERS"
echo "Precision: $PRECISION"
echo "Epochs: $EPOCHS"
echo ""

# ============================================================
# Auto-resume from checkpoint
# ============================================================
RESUME_ARG=""
LAST_CKPT=$(find "experiments/${EXP_NAME}" -name "last.ckpt" -type f 2>/dev/null | head -1)
if [ -n "$LAST_CKPT" ]; then
    echo "Resuming from checkpoint: $LAST_CKPT"
    RESUME_ARG="+ckpt_path=${LAST_CKPT}"
else
    echo "Starting fresh training (no checkpoint found)"
fi
echo ""

# Create log directory
mkdir -p experiments/logs

# ============================================================
# GPU Check
# ============================================================
if ! python -c "import torch; assert torch.cuda.is_available(), 'No CUDA'" 2>/dev/null; then
    echo "ERROR: CUDA not available. Aborting."
    exit 1
fi

python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
"
echo ""

# ============================================================
# Training
# ============================================================
echo "Starting training at $(date)"
echo "----------------------------------------------"

python scripts/train.py \
    experiment="$EXP_NAME" \
    training.max_epochs=$EPOCHS \
    data.loader.batch_size=$BATCH_SIZE \
    data.loader.num_workers=$NUM_WORKERS \
    hardware.precision=$PRECISION \
    hardware.slurm=true \
    logging.wandb.enabled=true \
    logging.wandb.offline=true \
    logging.wandb.project=$WANDB_PROJECT \
    $RESUME_ARG

EXIT_CODE=$?

echo ""
echo "----------------------------------------------"
echo "Training finished at $(date)"
echo "Exit code: $EXIT_CODE"

exit $EXIT_CODE
