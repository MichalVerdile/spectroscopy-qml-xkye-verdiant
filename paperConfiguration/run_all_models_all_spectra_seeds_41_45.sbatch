#!/usr/bin/env bash
# Paper architectures and model-specific paper preprocessing on all five spectra.
# CNN, MPS and TTN are trained until early stopping (patience 15). XGBoost is
# intentionally left unchanged. Submit from the repository root with:
#   sbatch paperConfiguration/run_all_models_all_spectra_seeds_41_45.sbatch
#SBATCH --job-name=paper-all-spectra
#SBATCH --account=init
#SBATCH --partition=gpu_top
#SBATCH --nodes=1
#SBATCH --nodelist=sanjose
#SBATCH --gres=gpu:h100pcie:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --open-mode=append
#SBATCH --output=logs/paper-all-spectra-%j.out
#SBATCH --error=logs/paper-all-spectra-%j.err

set -euo pipefail

requeue_before_timeout() {
    echo "Approaching the Slurm time limit; requeueing ${SLURM_JOB_ID}."
    scontrol requeue "${SLURM_JOB_ID}"
    exit 0
}
trap requeue_before_timeout USR1

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs benchmark/paper_all_spectra_seed_41_45

export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

.venv/bin/python -c '
import tensorflow
import torch
import xgboost
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
print("Training dependencies imported successfully")
'

# The training APIs require an integer epoch bound. This value is deliberately
# unreachable in practice; convergence is controlled by patience=15.
PRACTICALLY_UNBOUNDED_EPOCHS=2147483647

PREPROCESSING=legacy \
PYTHON=.venv/bin/python \
DEVICE=cuda \
EPOCHS="${PRACTICALLY_UNBOUNDED_EPOCHS}" \
EARLY_STOPPING_PATIENCE=15 \
TTN_THRESHOLD_MODE=global \
TTN_THRESHOLD_GRID_STEP=0.5 \
TTN_AMP=0 \
SEEDS="41 42 43 44 45" \
MIN_SEEDS=5 \
MODALITIES="cnmr hnmr ir msms_pos msms_neg" \
DATA_DIR=data/raw \
EXPERIMENT_DIR=benchmark/paper_all_spectra_seed_41_45 \
SPLIT_DIR=benchmark/paper_all_spectra_seed_41_45/splits \
    scripts/run_multiseed_benchmark.sh 600
