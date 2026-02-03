#!/bin/bash
# TrajectoryDiff: Evaluate All Trained Models
#
# Usage:
#   bash scripts/run_evaluation.sh experiments/outputs/
#
# Finds all checkpoints in the given directory and evaluates them.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

EXPERIMENTS_DIR="${1:-experiments/outputs}"

if [ ! -d "$EXPERIMENTS_DIR" ]; then
    echo "ERROR: Experiments directory not found: $EXPERIMENTS_DIR"
    exit 1
fi

echo "=============================================="
echo "TrajectoryDiff Evaluation Suite"
echo "=============================================="
echo "Scanning: $EXPERIMENTS_DIR"
echo ""

mkdir -p experiments/eval_results

# Find all best checkpoint files
find "$EXPERIMENTS_DIR" -name "*.ckpt" -path "*/checkpoints/*" | sort | while read ckpt; do
    exp_dir="$(dirname "$(dirname "$ckpt")")"
    exp_name="$(basename "$exp_dir")"

    echo "----------------------------------------------"
    echo "Evaluating: $exp_name"
    echo "Checkpoint: $ckpt"
    echo "----------------------------------------------"

    # Standard evaluation
    python scripts/evaluate.py \
        checkpoint="$ckpt" \
        visualize=true \
        2>&1 | tee "experiments/eval_results/${exp_name}_eval.log"

    # Cross-evaluation: if trained on trajectory, also evaluate on uniform
    if echo "$exp_name" | grep -q "trajectory\|traj_to_uniform\|full\|ablation\|coverage_sweep\|num_traj"; then
        echo "  -> Cross-eval on uniform sampling..."
        python scripts/evaluate.py \
            checkpoint="$ckpt" \
            data.sampling.strategy=uniform \
            visualize=false \
            2>&1 | tee "experiments/eval_results/${exp_name}_cross_uniform.log"
    fi

    echo "Done: $exp_name"
    echo ""
done

echo "=============================================="
echo "All evaluations completed!"
echo "Results in: experiments/eval_results/"
echo "=============================================="
