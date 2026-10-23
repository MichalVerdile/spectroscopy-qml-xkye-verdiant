#!/usr/bin/env bash
# Full PQN benchmark: five modalities, seed 41 and four models.
# Completed metrics are validated and skipped after a requeue.
#SBATCH --job-name=train-pqn-600
#SBATCH --account=init
#SBATCH --partition=gpu_top
#SBATCH --nodelist=fresko
#SBATCH --gres=gpu:a100pcie:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --open-mode=append
#SBATCH --output=logs/train-pqn-600-%j.out
#SBATCH --error=logs/train-pqn-600-%j.err

set -euo pipefail

requeue_before_timeout() {
    echo "Approaching the 24-hour limit; requeueing ${SLURM_JOB_ID}."
    scontrol requeue "${SLURM_JOB_ID}"
    exit 0
}
trap requeue_before_timeout USR1

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs benchmark/preprocessing_training

export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

.venv/bin/python -c '
import click
import numpy
import pandas
import sklearn
import torch
import xgboost
import tensorflow
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
print("Training dependencies imported successfully")
'

PREPROCESSING=pqn \
PYTHON=.venv/bin/python \
DEVICE=cuda \
EPOCHS=200 \
SEEDS="41" \
MIN_SEEDS=1 \
MODALITIES="cnmr hnmr ir msms_pos msms_neg" \
DATA_DIR=data/raw \
EXPERIMENT_DIR=benchmark/preprocessing_training/dim_600/pqn \
SPLIT_DIR=benchmark/preprocessing_training/shared_splits_600 \
SKIP_AGGREGATE=1 \
    scripts/run_multiseed_benchmark.sh 600
