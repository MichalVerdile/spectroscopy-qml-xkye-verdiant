#!/usr/bin/env bash
# Re-run only XGBoost for the selected best preprocessing and record
# train/validation micro- and macro-F1 throughout boosting.
#
# Array layout: five seeds x five modalities. The output is intentionally
# separate from the completed benchmark so its original artifacts stay intact.
#SBATCH --job-name=xgb-lc-bestpp600
#SBATCH --account=init
#SBATCH --partition=gpu_top
#SBATCH --nodes=1
#SBATCH --nodelist=sanjose
#SBATCH --gres=gpu:h100pcie:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-24%5
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

requeue_before_timeout() {
    echo "Approaching the 24-hour limit; requeueing ${SLURM_JOB_ID}."
    scontrol requeue "${SLURM_JOB_ID}"
    exit 0
}
trap requeue_before_timeout USR1

seeds=(41 42 43 44 45)
modalities=(cnmr hnmr ir msms_pos msms_neg)
columns=(c_nmr_spectra h_nmr_spectra ir_spectra pos_msms neg_msms)

task_id="${SLURM_ARRAY_TASK_ID}"
seed_index=$((task_id / ${#modalities[@]}))
modality_index=$((task_id % ${#modalities[@]}))
seed="${seeds[seed_index]}"
modality="${modalities[modality_index]}"
column="${columns[modality_index]}"

preprocessing="resample=fir,smooth=gaussian:2,repr=sqrt,norm=snv"
split_path="benchmark/preprocessing_training/shared_splits_600/${modality}_seed_${seed}.npz"
output_root="benchmark/preprocessing_training/dim_600/fir_gaussian_sqrt_snv_xgb_learning_curves"
base_out_path="${output_root}/${modality}/seed_${seed}/xgboost"
mkdir -p "${base_out_path}"

export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "XGBoost learning curve: seed=${seed}, modality=${modality}, every=20"

.venv/bin/python benchmark/xgb/scripts/run_xgb_baseline.py \
    --analytical_data data/raw \
    --base_out_path "${base_out_path}" \
    --columns "${column}" \
    --seed "${seed}" \
    --input_dim 600 \
    --split_path "${split_path}" \
    --preprocessing "${preprocessing}" \
    --n-estimators 1000000 \
    --early-stopping-patience 15 \
    --device cuda \
    --no_kfold \
    --log_learning_curve \
    --learning_curve_every 20
