#!/usr/bin/env bash
# Re-run MPS and TTN with a common fixed classification threshold of 0.5.
# Array layout: two models x five seeds x five modalities.
#SBATCH --job-name=mps-ttn-thr05
#SBATCH --account=init
#SBATCH --partition=gpu_top
#SBATCH --nodes=1
#SBATCH --nodelist=sanjose
#SBATCH --gres=gpu:h100pcie:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-49%5
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

models=(mps ttn)
seeds=(41 42 43 44 45)
modalities=(cnmr hnmr ir msms_pos msms_neg)

task_id="${SLURM_ARRAY_TASK_ID}"
combinations_per_model=$((${#seeds[@]} * ${#modalities[@]}))
model_index=$((task_id / combinations_per_model))
combination_index=$((task_id % combinations_per_model))
seed_index=$((combination_index / ${#modalities[@]}))
modality_index=$((combination_index % ${#modalities[@]}))

model="${models[model_index]}"
seed="${seeds[seed_index]}"
modality="${modalities[modality_index]}"
preprocessing="resample=fir,smooth=gaussian:2,repr=sqrt,norm=snv"
split_path="benchmark/preprocessing_training/shared_splits_600/${modality}_seed_${seed}.npz"
run_dir="benchmark/preprocessing_training/dim_600/fir_gaussian_sqrt_snv_fixed_threshold_0p5/${modality}/seed_${seed}"

export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "Fixed-threshold run: model=${model}, seed=${seed}, modality=${modality}, threshold=0.5"

if [[ "${model}" == "mps" ]]; then
    .venv/bin/python scripts/run_mps_benchmark.py \
        --data-dir data/raw \
        --output-dir "${run_dir}/mps" \
        --modality "${modality}" \
        --input-dim 600 \
        --split-path "${split_path}" \
        --seed "${seed}" \
        --epochs 1000000 \
        --early-stopping-patience 15 \
        --device cuda \
        --preprocessing "${preprocessing}" \
        --fixed-threshold 0.5
else
    .venv/bin/python scripts/run_ttn_fixed_threshold.py \
        --modality "${modality}" \
        --fixed-threshold 0.5 \
        --data-dir data/raw \
        --output-dir "${run_dir}/ttn" \
        --split-path "${split_path}" \
        --input-dim 600 \
        --seed "${seed}" \
        --device cuda \
        --epochs 1000000 \
        --early-stopping-patience 15 \
        --no-amp \
        --preprocessing "${preprocessing}"
fi
