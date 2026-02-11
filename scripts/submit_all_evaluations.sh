#!/bin/bash
#
# TrajectoryDiff: Submit evaluation jobs for all completed experiments
#
# Iterates over experiments/*, finds best checkpoint in each, and submits
# an evaluation job via sbatch. Skips non-experiment directories.
#
# Usage:
#   bash scripts/submit_all_evaluations.sh                      # Default 2g.35gb
#   bash scripts/submit_all_evaluations.sh 7g.141gb             # Override MIG profile
#   bash scripts/submit_all_evaluations.sh 2g.35gb --dry-run    # Preview without submitting

set -euo pipefail

cd /data1/yansari/TrajectoryDiff

MIG_PROFILE="${1:-2g.35gb}"
DRY_RUN=false
if [[ "${2:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

# Directories to skip (not experiment runs)
SKIP_DIRS="logs|eval_results|default|gpu_validation|__pycache__"

echo "=============================================="
echo "TrajectoryDiff: Batch Evaluation Submission"
echo "=============================================="
echo "MIG profile: $MIG_PROFILE"
echo "Dry run: $DRY_RUN"
echo ""

SUBMITTED=0
SKIPPED=0
NO_CKPT=0

for exp_dir in experiments/*/; do
    exp_name=$(basename "$exp_dir")

    # Skip non-experiment directories
    if echo "$exp_name" | grep -qE "^($SKIP_DIRS)$"; then
        continue
    fi

    # Find best checkpoint
    BEST_CKPT=""
    for ckpt_dir in "$exp_dir"/*/checkpoints "$exp_dir"/checkpoints; do
        if [ ! -d "$ckpt_dir" ]; then
            continue
        fi

        # Find by lowest val_loss in filename
        CANDIDATE=$(ls "$ckpt_dir"/epoch=*-val_loss=*.ckpt 2>/dev/null \
            | sed 's/.*val_loss=\([0-9.]*\)\.ckpt/\1 &/' \
            | sort -n \
            | head -1 \
            | awk '{print $2}')

        if [ -n "$CANDIDATE" ]; then
            BEST_CKPT="$CANDIDATE"
            break
        fi

        # Fallback: last.ckpt
        if [ -f "$ckpt_dir/last.ckpt" ]; then
            BEST_CKPT="$ckpt_dir/last.ckpt"
            break
        fi
    done

    if [ -z "$BEST_CKPT" ]; then
        echo "  [NO CKPT] $exp_name — skipping"
        NO_CKPT=$((NO_CKPT + 1))
        continue
    fi

    # Check if already evaluated
    EVAL_METRICS="experiments/eval_results/${exp_name}/metrics.json"
    if [ -f "$EVAL_METRICS" ]; then
        echo "  [DONE]    $exp_name — already evaluated, skipping"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    echo "  [SUBMIT]  $exp_name"
    echo "            Checkpoint: $BEST_CKPT"

    if [ "$DRY_RUN" = false ]; then
        sbatch \
            --gres="gpu:nvidia_h200_${MIG_PROFILE}:1" \
            --export="CHECKPOINT=${BEST_CKPT},EXPERIMENT=${exp_name}" \
            scripts/submit_evaluation.sh
    fi

    SUBMITTED=$((SUBMITTED + 1))
done

echo ""
echo "=============================================="
echo "Summary: $SUBMITTED submitted, $SKIPPED already done, $NO_CKPT no checkpoint"
echo "=============================================="
