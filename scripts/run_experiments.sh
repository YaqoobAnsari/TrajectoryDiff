#!/bin/bash
# TrajectoryDiff: Full Experiment Suite
# Run all experiments for ECCV paper results.
#
# Usage:
#   bash scripts/run_experiments.sh              # Run all experiments
#   bash scripts/run_experiments.sh trajectory_full  # Run single experiment
#
# Prerequisites:
#   - conda activate trajdiff
#   - W&B logged in: wandb login
#   - Data at data/raw/RadioMapSeer/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Common settings
WANDB_PROJECT="trajectorydiff"
EPOCHS=200
BATCH_SIZE=16
NUM_WORKERS=8
PRECISION="16-mixed"

# GPU check
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "ERROR: CUDA not available. Aborting."
    exit 1
fi

echo "=============================================="
echo "TrajectoryDiff Experiment Suite"
echo "=============================================="
echo "Project dir: $PROJECT_DIR"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo ""

run_experiment() {
    local exp_name="$1"
    local extra_args="${2:-}"

    echo "----------------------------------------------"
    echo "Running: $exp_name"
    echo "----------------------------------------------"

    python scripts/train.py \
        experiment="$exp_name" \
        training.max_epochs=$EPOCHS \
        data.loader.batch_size=$BATCH_SIZE \
        data.loader.num_workers=$NUM_WORKERS \
        hardware.precision=$PRECISION \
        logging.wandb.enabled=true \
        logging.wandb.project=$WANDB_PROJECT \
        $extra_args \
        2>&1 | tee "experiments/logs/${exp_name}.log"

    echo "Completed: $exp_name"
    echo ""
}

# Create log directory
mkdir -p experiments/logs

# If a specific experiment is requested, run only that
if [ $# -gt 0 ]; then
    run_experiment "$1" "${2:-}"
    exit 0
fi

# === Main Experiments ===

# 1. Main model (all features)
run_experiment "trajectory_full"

# 2. Baselines
run_experiment "trajectory_baseline"
run_experiment "uniform_baseline"

# === Ablation Studies ===

# 3. Component ablations
run_experiment "ablation_no_coverage_attention"
run_experiment "ablation_no_physics_loss"
run_experiment "ablation_no_trajectory_mask"
run_experiment "ablation_no_coverage_density"
run_experiment "ablation_no_tx_position"
run_experiment "ablation_small_unet"

# === Cross-Evaluation ===

# 4. Cross-eval (train with one strategy, evaluate with the other)
run_experiment "cross_eval_traj_to_uniform"
run_experiment "cross_eval_uniform_to_traj"

# === Coverage Sweeps ===

# 5. Coverage level sweep
run_experiment "coverage_sweep_1pct"
run_experiment "coverage_sweep_5pct"
run_experiment "coverage_sweep_10pct"
run_experiment "coverage_sweep_20pct"

# === Trajectory Count Sweep (via multirun) ===

# 6. Vary number of trajectories
python scripts/train.py \
    -m \
    experiment=num_trajectories_sweep \
    data.sampling.trajectory.num_trajectories=1,2,3,5,8 \
    training.max_epochs=$EPOCHS \
    data.loader.batch_size=$BATCH_SIZE \
    data.loader.num_workers=$NUM_WORKERS \
    hardware.precision=$PRECISION \
    logging.wandb.enabled=true \
    logging.wandb.project=$WANDB_PROJECT \
    2>&1 | tee "experiments/logs/num_trajectories_sweep.log"

echo "=============================================="
echo "All experiments completed!"
echo "=============================================="
