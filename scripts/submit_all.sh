#!/bin/bash
#
# Submit all TrajectoryDiff experiments to SLURM with concurrency control.
#
# Respects the 4-job concurrent limit by checking squeue before each submission.
#
# Usage:
#   bash scripts/submit_all.sh [mig_profile]
#   bash scripts/submit_all.sh 7g.141gb
#   bash scripts/submit_all.sh 2g.35gb
#
# Experiments are submitted in priority order:
#   1. Main models (trajectory_full, trajectory_baseline, uniform_baseline)
#   2. Ablation studies
#   3. Coverage sweeps
#   4. Cross-evaluation and trajectory sweeps

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

MIG_PROFILE="${1:-7g.141gb}"
MAX_CONCURRENT=4
POLL_INTERVAL=60  # seconds between queue checks

# ============================================================
# Experiment Priority Order
# ============================================================
EXPERIMENTS=(
    # Priority 1: Main models
    "trajectory_full"
    "trajectory_baseline"
    "uniform_baseline"

    # Priority 2: Ablations (identify which components matter)
    "ablation_no_physics_loss"
    "ablation_no_coverage_attention"
    "ablation_no_trajectory_mask"
    "ablation_no_coverage_density"
    "ablation_no_tx_position"
    "ablation_small_unet"

    # Priority 3: Coverage sweeps
    "coverage_sweep_1pct"
    "coverage_sweep_5pct"
    "coverage_sweep_10pct"
    "coverage_sweep_20pct"

    # Priority 4: Cross-evaluation and sweeps
    "cross_eval_traj_to_uniform"
    "cross_eval_uniform_to_traj"
    "num_trajectories_sweep"
)

# ============================================================
# Helper Functions
# ============================================================
count_running_jobs() {
    squeue -u "$USER" -h -t RUNNING,PENDING -n trajdiff 2>/dev/null | wc -l
}

wait_for_slot() {
    local running
    running=$(count_running_jobs)
    while [ "$running" -ge "$MAX_CONCURRENT" ]; do
        echo "  Queue full ($running/$MAX_CONCURRENT jobs). Waiting ${POLL_INTERVAL}s..."
        sleep "$POLL_INTERVAL"
        running=$(count_running_jobs)
    done
}

# ============================================================
# Main
# ============================================================
echo "=============================================="
echo "TrajectoryDiff: Batch Experiment Submission"
echo "=============================================="
echo "MIG Profile: $MIG_PROFILE"
echo "Max Concurrent: $MAX_CONCURRENT"
echo "Total Experiments: ${#EXPERIMENTS[@]}"
echo ""

SUBMITTED=0
SKIPPED=0
FAILED=0

for EXP_NAME in "${EXPERIMENTS[@]}"; do
    # Check config exists
    CONFIG_FILE="${PROJECT_DIR}/configs/experiment/${EXP_NAME}.yaml"
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "[SKIP] $EXP_NAME (config not found)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # Check if experiment already has a running job
    EXISTING=$(squeue -u "$USER" -h -t RUNNING,PENDING -o "%j %k" 2>/dev/null | grep -c "$EXP_NAME" || true)
    if [ "$EXISTING" -gt 0 ]; then
        echo "[SKIP] $EXP_NAME (already running/pending)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # Wait for a free slot
    wait_for_slot

    # Submit
    echo "[SUBMIT] $EXP_NAME..."
    if bash "${SCRIPT_DIR}/submit_experiment.sh" "$EXP_NAME" "$MIG_PROFILE" 2>&1 | tail -1; then
        SUBMITTED=$((SUBMITTED + 1))
    else
        echo "[FAIL] $EXP_NAME"
        FAILED=$((FAILED + 1))
    fi

    # Small delay between submissions to avoid overwhelming the scheduler
    sleep 2
done

echo ""
echo "=============================================="
echo "Submission Summary"
echo "=============================================="
echo "  Submitted: $SUBMITTED"
echo "  Skipped:   $SKIPPED"
echo "  Failed:    $FAILED"
echo ""
echo "Monitor all jobs:"
echo "  squeue -u \$USER"
echo "  watch -n 30 'squeue -u \$USER'"
