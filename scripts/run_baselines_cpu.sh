#!/bin/bash
#SBATCH --job-name=baselines_eval
#SBATCH --partition=cpu
#SBATCH --nodelist=mcore-n01
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --mcs-label=unicellular
#SBATCH --output=experiments/logs/baselines_eval_%j.out
#SBATCH --error=experiments/logs/baselines_eval_%j.err

# Classical baseline evaluation (IDW, RBF, NN, Kriging) on full test set.
# CPU-only, no GPU needed. Kriging ~2h, others are fast.

set -euo pipefail

echo "=== Classical Baselines Evaluation ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-20}"
echo ""

# Setup environment
export PATH="${CONDA_PREFIX:-$HOME/.conda/envs/trajdiff}/bin:$PATH"
cd "$(dirname "$0")/.."

mkdir -p experiments/logs experiments/eval_results

python scripts/run_baselines.py \
    --output experiments/eval_results/baselines.json \
    --num-workers 16

echo ""
echo "=== Done ==="
echo "Date: $(date)"
