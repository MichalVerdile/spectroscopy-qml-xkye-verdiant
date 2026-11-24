#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs

exec sbatch cluster/train_raw_native_multiseed_learning_curves.sbatch
