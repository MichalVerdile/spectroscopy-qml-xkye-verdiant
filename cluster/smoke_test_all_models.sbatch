#!/usr/bin/env bash
#SBATCH --job-name=smoke-4models
#SBATCH --account=init
#SBATCH --partition=gpu_top
#SBATCH --nodelist=sanjose
#SBATCH --gres=gpu:h100pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=logs/smoke-4models-%j.out
#SBATCH --error=logs/smoke-4models-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs
export PYTHONPATH="${PWD}:${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

.venv/bin/python scripts/smoke_test_all_models.py
