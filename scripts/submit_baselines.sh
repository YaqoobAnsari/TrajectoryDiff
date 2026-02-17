#!/bin/bash
#SBATCH --job-name=trajdiff-baselines
#SBATCH --partition=cpu
#SBATCH --nodelist=mcore-n01
#SBATCH --mcs-label=unicellular
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32
#SBATCH --output=experiments/logs/baselines_%j.out
#SBATCH --error=experiments/logs/baselines_%j.err
#
# TrajectoryDiff: Classical Baseline Evaluation (CPU-only)
#
# Usage:
#   sbatch scripts/submit_baselines.sh
#
# Runs IDW, RBF, Nearest Neighbor, Distance Transform on the test set.
# No GPU required — runs on the CPU partition.

set -euo pipefail

echo "=============================================="
echo "TrajectoryDiff Classical Baselines (CPU)"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-32}"
echo ""

# Activate conda environment
export PATH="/data1/yansari/.conda/envs/trajdiff/bin:$PATH"

cd /data1/yansari/TrajectoryDiff

mkdir -p experiments/eval_results

echo "Starting classical baseline evaluation at $(date)"
echo "----------------------------------------------"

python scripts/run_baselines.py \
    --data-dir data/raw \
    --output experiments/eval_results/baselines.json \
    --num-workers 16 \
    --batch-size 1

EXIT_CODE=$?

echo ""
echo "----------------------------------------------"
echo "Baseline evaluation finished at $(date)"
echo "Exit code: $EXIT_CODE"

exit $EXIT_CODE
