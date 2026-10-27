#!/usr/bin/env bash
# Run FIR + Gaussian + sqrt + SNV for seeds 41, 43, 44, and 45.
# One seed runs at a time so the array occupies at most one H100.
# All model families use early-stopping patience 15. The practical training
# ceiling is intentionally unreachable under normal early stopping.
#SBATCH --job-name=bestpp600-p15
#SBATCH --account=init
#SBATCH --partition=gpu_top
#SBATCH --nodes=1
#SBATCH --nodelist=sanjose
#SBATCH --gres=gpu:h100pcie:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-3%1
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

seeds=(41 43 44 45)
task_id="${SLURM_ARRAY_TASK_ID}"
if [[ ! "$task_id" =~ ^[0-9]+$ ]] || ((task_id >= ${#seeds[@]})); then
    echo "Invalid SLURM_ARRAY_TASK_ID=${task_id}; expected 0-$(( ${#seeds[@]} - 1 ))." >&2
    exit 2
fi

seed="${seeds[task_id]}"
preprocessing="resample=fir,smooth=gaussian:2,repr=sqrt,norm=snv"

echo "Array task ${task_id}: seed ${seed}, preprocessing ${preprocessing}, patience 15"
export SEEDS="$seed"
export EPOCHS=1000000
export XGB_N_ESTIMATORS=1000000
export EARLY_STOPPING_PATIENCE=15

exec bash "${SLURM_SUBMIT_DIR}/cluster/run_preprocessing_600_sanjose.sh" \
    "$preprocessing" "fir_gaussian_sqrt_snv"
