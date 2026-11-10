#!/usr/bin/env bash
# Run FIR + Gaussian + sqrt + SNV for the configured seeds.
# Override the default list with SEED_LIST, for example SEED_LIST="42".
# Each task runs one seed/modality pair. At most five tasks occupy H100s.
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
#SBATCH --array=0-19%5
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

read -r -a seeds <<< "${SEED_LIST:-41 43 44 45}"
modalities=(cnmr hnmr ir msms_pos msms_neg)
task_id="${SLURM_ARRAY_TASK_ID}"
task_count=$(( ${#seeds[@]} * ${#modalities[@]} ))
if (( ${#seeds[@]} == 0 )); then
    echo "SEED_LIST must contain at least one seed." >&2
    exit 2
fi
for configured_seed in "${seeds[@]}"; do
    if [[ ! "$configured_seed" =~ ^[0-9]+$ ]]; then
        echo "Invalid seed '${configured_seed}' in SEED_LIST." >&2
        exit 2
    fi
done
if [[ ! "$task_id" =~ ^[0-9]+$ ]] || ((task_id >= task_count)); then
    echo "Invalid SLURM_ARRAY_TASK_ID=${task_id}; expected 0-$((task_count - 1))." >&2
    exit 2
fi

seed_index=$((task_id / ${#modalities[@]}))
modality_index=$((task_id % ${#modalities[@]}))
seed="${seeds[seed_index]}"
modality="${modalities[modality_index]}"
preprocessing="resample=fir,smooth=gaussian:2,repr=sqrt,norm=snv"

echo "Array task ${task_id}: seed ${seed}, modality ${modality}, preprocessing ${preprocessing}, patience 15"
export SEEDS="$seed"
export MODALITIES="$modality"
export EPOCHS=1000000
export XGB_N_ESTIMATORS=1000000
export EARLY_STOPPING_PATIENCE=15

exec bash "${SLURM_SUBMIT_DIR}/cluster/run_preprocessing_600_sanjose.sh" \
    "$preprocessing" "fir_gaussian_sqrt_snv"
