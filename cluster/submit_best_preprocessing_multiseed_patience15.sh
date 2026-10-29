#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs

SEED_LIST="${SEED_LIST:-41 43 44 45}"
read -r -a seeds <<< "$SEED_LIST"
if (( ${#seeds[@]} == 0 )); then
    echo "SEED_LIST must contain at least one seed." >&2
    exit 2
fi
for seed in "${seeds[@]}"; do
    if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
        echo "Invalid seed '$seed' in SEED_LIST." >&2
        exit 2
    fi
done

modalities_per_seed=5
last_task=$(( ${#seeds[@]} * modalities_per_seed - 1 ))
export SEED_LIST
exec sbatch \
    --array="0-${last_task}%5" \
    --export=ALL,SEED_LIST \
    cluster/train_best_preprocessing_multiseed_patience15.sbatch
