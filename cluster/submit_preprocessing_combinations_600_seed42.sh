#!/usr/bin/env bash
# Submit the 15 preprocessing combinations as a Slurm array after ensuring that
# the log directory exists before Slurm opens the per-task output files.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs

exec sbatch cluster/train_preprocessing_combinations_600_seed42.sbatch
