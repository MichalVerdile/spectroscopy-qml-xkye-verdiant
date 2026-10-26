#!/usr/bin/env bash
#SBATCH --job-name=train-binmax-600
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
#SBATCH --output=logs/train-binmax-600-%j.out
#SBATCH --error=logs/train-binmax-600-%j.err

exec bash "${SLURM_SUBMIT_DIR}/cluster/run_preprocessing_600_sanjose.sh" bin_max
