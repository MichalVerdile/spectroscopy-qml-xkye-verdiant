#!/usr/bin/env bash
#SBATCH --job-name=preprocess-5-spectra
#SBATCH --account=init
#SBATCH --partition=gpu
#SBATCH --nodelist=compton
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=logs/preprocess-%j.out
#SBATCH --error=logs/preprocess-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs data/cache/preprocessed

export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

.venv/bin/python scripts/precompute_preprocessing_cache.py \
    --data-dir data/raw \
    --cache-dir data/cache/preprocessed \
    --modalities ir cnmr hnmr msms_pos msms_neg \
    --input-dims 600 1800
