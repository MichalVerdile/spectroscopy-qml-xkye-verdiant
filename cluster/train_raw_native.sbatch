#!/usr/bin/env bash
# Create logs/ before sbatch. One seed/modality per task; four families serially.
#SBATCH --job-name=raw-native
#SBATCH --account=init
#SBATCH --partition=gpu_top
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-24%3
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
seeds=(41 42 43 44 45)
modalities=(cnmr hnmr ir msms_pos msms_neg)
task="${SLURM_ARRAY_TASK_ID:?}"
if (( task < 0 || task >= 25 )); then exit 2; fi
exec "${PYTHON:-python}" scripts/run_raw_native_benchmark.py \
  --data-dir "${DATA_DIR:-data/raw}" --model all \
  --modality "${modalities[task % 5]}" --seeds "${seeds[task / 5]}" \
  --experiment-id "${EXPERIMENT_ID:-B_optimized_raw}" \
  --phase "${PHASE:-tune}" --variant optimized --device cuda \
  --trials "${TRIALS:-12}" --epochs 100 --patience 15 \
  --time-limit-per-trial 1200 --parameter-limit 500000000 --batch-size 64 --threads 8 \
  --output-dir "${OUTPUT_DIR:-benchmark/raw_native_v1}" --resume --strict-raw
