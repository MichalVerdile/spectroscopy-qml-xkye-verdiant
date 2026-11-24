#!/usr/bin/env bash
# Run every non-empty combination of FIR, Gaussian smoothing, sqrt representation,
# and SNV exactly once with seed 42. Each array task runs all five modalities and
# all four model families. Completed results are validated and skipped on requeue.
#SBATCH --job-name=combo600-s42
#SBATCH --account=init
#SBATCH --partition=gpu_top
#SBATCH --nodes=1
#SBATCH --nodelist=sanjose
#SBATCH --gres=gpu:h100pcie:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-14
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

source scripts/preprocessing_combinations.sh
mapfile -t combinations < <(list_preprocessing_combinations)

task_id="${SLURM_ARRAY_TASK_ID}"
if [[ ! "$task_id" =~ ^[0-9]+$ ]] || ((task_id >= ${#combinations[@]})); then
    echo "Invalid SLURM_ARRAY_TASK_ID=${task_id}; expected 0-$(( ${#combinations[@]} - 1 ))." >&2
    exit 2
fi

IFS=$'\t' read -r combination_name preprocessing <<< "${combinations[task_id]}"

echo "Array task ${task_id}: ${combination_name} (${preprocessing}), seed 42"
export SEEDS=42
exec bash "${SLURM_SUBMIT_DIR}/cluster/run_preprocessing_600_sanjose.sh" \
    "$preprocessing" "$combination_name"
