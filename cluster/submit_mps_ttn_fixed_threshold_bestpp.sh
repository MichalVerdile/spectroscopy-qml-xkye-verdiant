#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs

exec sbatch cluster/train_mps_ttn_fixed_threshold_bestpp.sbatch
