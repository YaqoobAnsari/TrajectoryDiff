#!/bin/bash
#SBATCH --job-name=trajdiff-eval
#SBATCH --partition=gpu2
#SBATCH --nodelist=deepnet2
#SBATCH --mcs-label=unicellular
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=experiments/logs/%x_%A.out
#SBATCH --error=experiments/logs/%x_%A.err
#
# TrajectoryDiff: SLURM Evaluation Script
#
# Evaluates a trained checkpoint on the test set and runs classical baselines.
#
# Usage:
#   # With explicit checkpoint:
#   sbatch --gres=gpu:nvidia_h200_2g.35gb:1 \
#     --export=CHECKPOINT=path/to/best.ckpt,EXPERIMENT=uniform_baseline \
#     scripts/submit_evaluation.sh
#
#   # Auto-detect best checkpoint:
#   sbatch --gres=gpu:nvidia_h200_2g.35gb:1 \
#     --export=EXPERIMENT=uniform_baseline \
#     scripts/submit_evaluation.sh

set -euo pipefail

# ============================================================
# Environment Setup
# ============================================================
echo "=============================================="
echo "TrajectoryDiff Evaluation (SLURM)"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="/data1/yansari/.conda/envs/trajdiff/bin:$PATH"

cd /data1/yansari/TrajectoryDiff

# ============================================================
# Configuration
# ============================================================
EXPERIMENT="${EXPERIMENT:-uniform_baseline}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"

# Auto-detect best checkpoint if not explicitly provided
if [ -z "${CHECKPOINT:-}" ]; then
    echo "CHECKPOINT not set — auto-detecting best checkpoint for $EXPERIMENT..."

    # Search for checkpoints in experiment directories
    CKPT_FOUND=""
    for exp_dir in experiments/${EXPERIMENT}/*/checkpoints experiments/${EXPERIMENT}/checkpoints; do
        if [ ! -d "$exp_dir" ]; then
            continue
        fi

        # Find best checkpoint by lowest val_loss in filename
        # Filenames: epoch=XX-val_loss=Y.YYYY.ckpt
        BEST_CKPT=$(ls "$exp_dir"/epoch=*-val_loss=*.ckpt 2>/dev/null \
            | sed 's/.*val_loss=\([0-9.]*\)\.ckpt/\1 &/' \
            | sort -n \
            | head -1 \
            | awk '{print $2}')

        if [ -n "$BEST_CKPT" ]; then
            CKPT_FOUND="$BEST_CKPT"
            break
        fi

        # Fallback: last.ckpt
        if [ -f "$exp_dir/last.ckpt" ]; then
            CKPT_FOUND="$exp_dir/last.ckpt"
            break
        fi
    done

    if [ -z "$CKPT_FOUND" ]; then
        echo "ERROR: No checkpoint found for experiment '$EXPERIMENT'"
        echo "Searched in: experiments/${EXPERIMENT}/*/checkpoints/"
        exit 1
    fi

    CHECKPOINT="$CKPT_FOUND"
    echo "Auto-detected checkpoint: $CHECKPOINT"
fi

echo ""
echo "Experiment: $EXPERIMENT"
echo "Checkpoint: $CHECKPOINT"
echo "Max Samples: $MAX_SAMPLES (0=all)"
echo "Batch Size: $BATCH_SIZE"
echo ""

# Verify checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

# ============================================================
# GPU Check
# ============================================================
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

# ============================================================
# Create output directory
# ============================================================
EVAL_DIR="experiments/eval_results/${EXPERIMENT}"
mkdir -p "$EVAL_DIR"

# ============================================================
# Step 1: Model Evaluation (with DDIM sampling)
# ============================================================
echo "=============================================="
echo "Step 1: Model Evaluation"
echo "=============================================="
echo "Starting at $(date)"

MAX_SAMPLES_ARG=""
if [ "$MAX_SAMPLES" -gt 0 ] 2>/dev/null; then
    MAX_SAMPLES_ARG="+max_samples=$MAX_SAMPLES"
fi

python scripts/evaluate.py \
    experiment="$EXPERIMENT" \
    "+checkpoint='${CHECKPOINT}'" \
    data.loader.batch_size=$BATCH_SIZE \
    data.loader.num_workers=$NUM_WORKERS \
    +visualize=true \
    +num_vis_samples=8 \
    $MAX_SAMPLES_ARG \
    "hydra.run.dir=${EVAL_DIR}"

echo ""
echo "Model evaluation finished at $(date)"
echo ""

# ============================================================
# Step 2: Classical Baselines
# ============================================================
echo "=============================================="
echo "Step 2: Classical Baselines"
echo "=============================================="
echo "Starting at $(date)"

BASELINE_MAX_ARG=""
if [ "$MAX_SAMPLES" -gt 0 ] 2>/dev/null; then
    BASELINE_MAX_ARG="--max-samples $MAX_SAMPLES"
fi

# Save baselines both in experiment subdir and top-level for aggregator
python scripts/run_baselines.py \
    --output "$EVAL_DIR/baselines.json" \
    --batch-size 1 \
    --num-workers $NUM_WORKERS \
    $BASELINE_MAX_ARG

# Copy to top-level eval_results for aggregator/figure scripts
cp "$EVAL_DIR/baselines.json" "experiments/eval_results/baselines.json" 2>/dev/null || true

echo ""
echo "Baselines finished at $(date)"
echo ""

# ============================================================
# Step 3: Results Summary
# ============================================================
echo "=============================================="
echo "Step 3: Results Summary"
echo "=============================================="

python -c "
import json
from pathlib import Path

eval_dir = Path('$EVAL_DIR')

# Load model metrics
metrics_file = eval_dir / 'metrics.json'
if metrics_file.exists():
    with open(metrics_file) as f:
        model_metrics = json.load(f)
    print('MODEL RESULTS ($EXPERIMENT):')
    print(f\"  RMSE:  {model_metrics['rmse_dbm']:.2f} +/- {model_metrics['rmse_dbm_std']:.2f} dBm\")
    print(f\"  MAE:   {model_metrics['mae_dbm']:.2f} +/- {model_metrics['mae_dbm_std']:.2f} dBm\")
    print(f\"  PSNR:  {model_metrics['psnr']:.2f} dB\")
    print(f\"  SSIM:  {model_metrics['ssim']:.4f}\")
    if 'trajectory_rmse_dbm' in model_metrics:
        print(f\"  Traj RMSE:  {model_metrics['trajectory_rmse_dbm']:.2f} dBm\")
        print(f\"  Blind RMSE: {model_metrics['blind_spot_rmse_dbm']:.2f} dBm\")
    print()

# Load baseline metrics
baselines_file = eval_dir / 'baselines.json'
if baselines_file.exists():
    with open(baselines_file) as f:
        baselines = json.load(f)
    print('CLASSICAL BASELINES (dBm):')
    print(f\"{'Method':<25} {'RMSE(dBm)':>10} {'MAE(dBm)':>10} {'SSIM':>8} {'Time(ms)':>10}\")
    print('-' * 70)
    for name, s in baselines.items():
        ssim_val = s.get('ssim_mean', float('nan'))
        rmse_dbm = s.get('rmse_dbm_mean', float('nan'))
        mae_dbm = s.get('mae_dbm_mean', float('nan'))
        print(f\"{name:<25} {rmse_dbm:>10.2f} {mae_dbm:>10.2f} {ssim_val:>8.4f} {s['avg_time_ms']:>10.1f}\")
    print()

# SOTA reference table
print('SOTA REFERENCE (RadioMapSeer):')
print(f\"{'Method':<25} {'RMSE (dB)':>10} {'Task':>20}\")
print('-' * 60)
sota = [
    ('RadioFlow Large', '~0.82', 'Full prediction'),
    ('RMDM', '~1.00', 'Full prediction'),
    ('RadioDiff', '~1.52', 'Full prediction'),
    ('RadioUNet (SRM)', '~1.95', 'Full prediction'),
    ('RMDM Setup 3', '~0.94', 'Sparse reconstruct.'),
    ('IRDM (10% uniform)', '~4.23', 'Sparse reconstruct.'),
]
for name, rmse, task in sota:
    print(f'{name:<25} {rmse:>10} {task:>20}')
print()
print('Note: Our task (trajectory-based sparse reconstruction) is novel.')
print('Most comparable: sparse reconstruction benchmarks.')
"

echo ""

# ============================================================
# Step 4: Aggregate Results
# ============================================================
echo "=============================================="
echo "Step 4: Aggregate Results"
echo "=============================================="
echo "Starting at $(date)"

python scripts/aggregate_results.py \
    --eval-dir experiments/eval_results \
    --format all \
    --verbose

echo ""
echo "=============================================="
echo "Evaluation complete at $(date)"
echo "Results in: $EVAL_DIR"
echo "Aggregated: experiments/eval_results/summary.{json,csv,md}"
echo "LaTeX:      experiments/eval_results/tables.tex"
echo "=============================================="
