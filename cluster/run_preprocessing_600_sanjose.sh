#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 PREPROCESSING [EXPERIMENT_NAME]" >&2
    exit 2
fi

preprocessing="$1"
experiment_name="${2:-$preprocessing}"

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

PREPROCESSING="$preprocessing" \
PYTHON=.venv/bin/python \
DEVICE=cuda \
EPOCHS="${EPOCHS:-1000000}" \
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-15}" \
XGB_N_ESTIMATORS="${XGB_N_ESTIMATORS:-1000000}" \
SEEDS="${SEEDS:-41}" \
MIN_SEEDS=1 \
MODALITIES="cnmr hnmr ir msms_pos msms_neg" \
DATA_DIR=data/raw \
EXPERIMENT_DIR="benchmark/preprocessing_training/dim_600/${experiment_name}" \
SPLIT_DIR=benchmark/preprocessing_training/shared_splits_600 \
SKIP_AGGREGATE=1 \
    scripts/run_multiseed_benchmark.sh 600
