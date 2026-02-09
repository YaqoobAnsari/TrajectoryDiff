#!/bin/bash
#
# Submit a single TrajectoryDiff experiment to SLURM.
#
# Usage:
#   bash scripts/submit_experiment.sh <experiment_name> [mig_profile] [extra_args...]
#
# Examples:
#   bash scripts/submit_experiment.sh trajectory_full
#   bash scripts/submit_experiment.sh trajectory_full 2g.35gb
#   bash scripts/submit_experiment.sh ablation_no_physics_loss 7g.141gb --time=48:00:00
#
# MIG Profiles:
#   7g.141gb  - Full H200 GPU (default), batch_size=64, 12 workers
#   2g.35gb   - 1/4 GPU, batch_size=32, 8 workers
#   1g.18gb   - 1/8 GPU, batch_size=16, 4 workers

set -euo pipefail

# ============================================================
# Arguments
# ============================================================
EXP_NAME="${1:?ERROR: Missing experiment name. Usage: bash scripts/submit_experiment.sh <experiment_name> [mig_profile]}"
MIG_PROFILE="${2:-7g.141gb}"
shift 2 2>/dev/null || shift 1 2>/dev/null || true
EXTRA_SBATCH_ARGS="$*"

# ============================================================
# Validate experiment config exists
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

CONFIG_FILE="${PROJECT_DIR}/configs/experiment/${EXP_NAME}.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Experiment config not found: $CONFIG_FILE"
    echo ""
    echo "Available experiments:"
    ls "${PROJECT_DIR}/configs/experiment/"*.yaml 2>/dev/null | \
        xargs -I{} basename {} .yaml | sed 's/^/  /'
    exit 1
fi

# ============================================================
# Set resources based on MIG profile
# ============================================================
case "$MIG_PROFILE" in
    7g.141gb)
        BATCH_SIZE=64
        NUM_WORKERS=12
        MEM="64G"
        CPUS=16
        ;;
    2g.35gb)
        BATCH_SIZE=32
        NUM_WORKERS=8
        MEM="32G"
        CPUS=8
        ;;
    1g.18gb)
        BATCH_SIZE=16
        NUM_WORKERS=4
        MEM="16G"
        CPUS=4
        ;;
    *)
        echo "WARNING: Unknown MIG profile '$MIG_PROFILE', using defaults for 7g.141gb"
        BATCH_SIZE=64
        NUM_WORKERS=12
        MEM="64G"
        CPUS=16
        ;;
esac

# ============================================================
# Create log directory
# ============================================================
mkdir -p "${PROJECT_DIR}/experiments/logs"

# ============================================================
# Submit job
# ============================================================
echo "=============================================="
echo "Submitting: $EXP_NAME"
echo "MIG Profile: $MIG_PROFILE"
echo "Batch Size: $BATCH_SIZE"
echo "Workers: $NUM_WORKERS"
echo "Memory: $MEM"
echo "CPUs: $CPUS"
echo "=============================================="

JOB_ID=$(sbatch \
    --parsable \
    --export=EXP_NAME="${EXP_NAME}",MIG_PROFILE="${MIG_PROFILE}",BATCH_SIZE="${BATCH_SIZE}",NUM_WORKERS="${NUM_WORKERS}" \
    --mem="${MEM}" \
    --cpus-per-task="${CPUS}" \
    ${EXTRA_SBATCH_ARGS} \
    "${PROJECT_DIR}/scripts/run_experiments.sh")

echo ""
echo "Submitted job: $JOB_ID"
echo "  Experiment: $EXP_NAME"
echo "  MIG Profile: $MIG_PROFILE"
echo "  Log: experiments/logs/trajdiff_${JOB_ID}.out"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f experiments/logs/trajdiff_${JOB_ID}.out"
