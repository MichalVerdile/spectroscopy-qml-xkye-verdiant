#!/usr/bin/env bash
# Run with bash; executable bits on a Global Volume are neither used nor needed.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
root="${PERSISTENT_ROOT:?Set the verified persistent mountpoint}"
local_root="${XGB_LOCAL_ROOT:-/root/runpod-work}"
python="${XGB_PYTHON:-$local_root/venv/bin/python}"
mkdir -p "$local_root" "$root/experiments/xgboost_raw_native/logs"
export PYTHONPATH="$repo/src"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
case "${1:-start}" in
  start)
    if tmux has-session -t xgb-raw-native 2>/dev/null; then
      echo 'Existing xgb-raw-native session is active; refusing a duplicate.' >&2
      exit 1
    fi
    printf -v worker 'env PERSISTENT_ROOT=%q XGB_LOCAL_ROOT=%q XGB_PYTHON=%q ACCEPT_RECONSTRUCTED_SPLITS=%q XGB_REUSE_READY=%q bash %q worker' \
      "$root" "$local_root" "$python" "${ACCEPT_RECONSTRUCTED_SPLITS:-0}" "${XGB_REUSE_READY:-0}" "$repo/runpod/run_xgboost_raw_native_runpod.sh"
    tmux new-session -d -s xgb-raw-native "$worker"
    tmux list-sessions
    ;;
  worker)
    if [[ -s "$local_root/master.log" ]]; then
      "$python" - "$repo" "$local_root/master.log" "$root/experiments/xgboost_raw_native/logs" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]) / 'runpod'))
from protocol import publish, stamp
publish(Path(sys.argv[2]), Path(sys.argv[3]) / ('master-' + stamp() + '.log'))
PY
    fi
    exec > "$local_root/master.log" 2>&1
    # Periodic closed snapshots avoid relying on remote append/fsync semantics.
    "$python" -u - "$local_root/master.log" "$root/experiments/xgboost_raw_native/logs/master.log" <<'PY' &
import pathlib, shutil, sys, time
source, target = map(pathlib.Path, sys.argv[1:])
while True:
    try:
        with source.open('rb') as src, target.open('wb') as dst:
            shutil.copyfileobj(src, dst)
    except OSError:
        pass
    time.sleep(30)
PY
    mirror_pid=$!
    trap 'kill "$mirror_pid" 2>/dev/null || true; cp --no-preserve=mode,ownership "$local_root/master.log" "$root/experiments/xgboost_raw_native/logs/master.log" || true' EXIT
    split_args=()
    if [[ "${ACCEPT_RECONSTRUCTED_SPLITS:-0}" == 1 ]]; then
      split_args+=(--accept-reconstructed-splits)
    fi
    if [[ "${XGB_REUSE_READY:-0}" == 1 ]]; then
      test -s "$root/experiments/xgboost_raw_native/configurations/ready.json"
      echo 'REUSING_COMPLETED_PREFLIGHT on this unchanged pod; runner revalidates source, audit, manifest and split hashes'
    else
      "$python" -u "$repo/runpod/prepare.py" --root "$root" --repo "$repo" --local "$local_root" "${split_args[@]}"
    fi
    "$python" -u "$repo/runpod/run_experiments.py" run --root "$root" --repo "$repo" --local "$local_root"
    ;;
  status)
    "$python" "$repo/runpod/run_experiments.py" status --root "$root"
    ;;
  *) echo "Usage: bash $0 {start|status}" >&2; exit 2 ;;
esac
