#!/usr/bin/env python3
"""Sequential, restartable orchestration for the 25 unchanged XGBoost runs."""
from __future__ import annotations

import argparse
import csv
import fcntl
import io
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

from protocol import MODALITIES, REFERENCE, SEEDS, check_science, publish, run_config, sha256, stamp, write_json

FIELDS = ["model", "modality", "seed", "input_dim", "preprocessing", "threshold",
          "train_f1_micro", "train_f1_macro", "val_f1_micro", "val_f1_macro",
          "test_f1_micro", "test_f1_macro", "best_iteration", "training_seconds",
          "status", "attempt", "split_sha256", "historical_identity_verified"]


def publish_log_snapshot(source, target):
    """Freeze a local log before hashing it; the trainer may still append."""
    with tempfile.NamedTemporaryFile(dir=source.parent, prefix=".log-snapshot-", delete=False) as snapshot:
        temporary = Path(snapshot.name)
        with source.open("rb") as active:
            shutil.copyfileobj(active, snapshot)
    try:
        return publish(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def rows_for(root):
    base = root / "experiments/xgboost_raw_native"
    rows = []
    for modality, (_, _, _, width) in MODALITIES.items():
        for seed in SEEDS:
            row = {"model": "xgboost", "modality": modality, "seed": seed,
                   "input_dim": width, "preprocessing": "none", "threshold": 0.5, "status": "pending"}
            for attempt in sorted((base / modality / f"seed_{seed}" / "attempts").glob("*")):
                status_file = attempt / "status.json"
                try:
                    state = json.loads(status_file.read_text())
                except (OSError, ValueError):
                    continue
                if (attempt / "completed.json").exists():
                    result = json.loads((attempt / "completed.json").read_text())
                    row.update(result["summary"], status="completed", attempt=str(attempt))
                    break
                row.update(status=state["status"], attempt=str(attempt))
                if row["status"] == "running" and time.time() - state.get("heartbeat_unix", 0) > 120:
                    row["status"] = "interrupted"
            rows.append(row)
    return rows


def summarize(root):
    rows = rows_for(root)
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    base = root / "experiments/xgboost_raw_native/summaries"
    base.mkdir(parents=True, exist_ok=True)
    (base / "experiments.csv").write_text(output.getvalue())
    counts = {status: sum(row["status"] == status for row in rows)
              for status in ["pending", "running", "completed", "failed", "interrupted"]}
    report = {"updated": stamp(), "counts": counts, "experiments": rows}
    write_json(base / "status.json", report)
    return report


def command(repo, data, attempt, split, cache, modality, seed, threads):
    return [sys.executable, "-u", str(repo / "runpod/observe_training.py"),
            "--analytical_data", str(data), "--base_out_path", str(attempt / "artifacts"),
            "--columns", MODALITIES[modality][0], "--seed", str(seed),
            "--input_dim", str(MODALITIES[modality][3]), "--split_path", str(split),
            "--cache_dir", str(cache), "--preprocessing", "none", "--no_kfold",
            "--device", "cuda", "--n_jobs", str(threads), "--n-estimators", "1000000",
            "--early-stopping-patience", "15", "--log_learning_curve", "--learning_curve_every", "20"]


def valid_completion(attempt, config):
    try:
        result = json.loads((attempt / "completed.json").read_text())
        if result["config"] != config:
            return False
        for relative, expected in result["file_sha256"].items():
            if sha256(attempt / relative) != expected:
                return False
        return True
    except (OSError, ValueError, KeyError):
        return False


def fit_arguments(argv):
    """Ignore the runner's executable and per-attempt storage locations."""
    args = list(argv[3:])
    for option in ("--base_out_path", "--cache_dir"):
        position = args.index(option)
        del args[position:position + 2]
    return args


def recoverable_fit(attempts, config, argv):
    for previous in sorted(attempts.glob("*"), reverse=True):
        if any(not (previous / "label_models" / f"label_{i:02d}.{ext}").is_file()
               for i in range(37) for ext in ("ubj", "json")):
            continue
        try:
            saved = json.loads((previous / "config.json").read_text())
            if (all(saved.get(key) == value for key, value in config.items())
                    and fit_arguments(saved["argv"]) == fit_arguments(argv)):
                return previous
        except (OSError, ValueError, KeyError):
            continue
    return None


def collect_result(repo, attempt, config, elapsed):
    import numpy as np
    import pickle

    modality, seed = config["modality"], config["seed"]
    output_name = MODALITIES[modality][2]
    output = attempt / "artifacts" / output_name / "original"
    metrics_path = output / "metrics.json"
    subprocess.run([sys.executable, str(repo / "scripts/validate_benchmark_result.py"),
                    "--metrics", str(metrics_path), "--model", "xgboost", "--preprocessing", "none",
                    "--input-dim", str(config["input_dim"]), "--seed", str(seed),
                    "--split-path", config["split_path"], "--n-estimators", "1000000",
                    "--early-stopping-patience", "15"], check=True)
    with (output / "training_logs.csv").open() as f:
        logs = list(csv.DictReader(f))
    finals = [r for r in logs if r["event"] == "training_finished"]
    if len(finals) != 1 or not any(r["event"] == "learning_curve" for r in logs):
        raise ValueError("Missing final metrics or learning curves")
    final = finals[0]
    with (output / f"{output_name}_xgboost_model.pickle").open("rb") as f:
        # Only the just-produced, trusted model artifact is deserialized here.
        model = pickle.load(f)
    if len(model.estimators_) != 37:
        raise ValueError("Expected 37 fitted binary estimators")
    best_iterations = []
    fitted_rounds = []
    for estimator in model.estimators_:
        p = estimator.get_params()
        for key, value in {"n_estimators": 1000000, "early_stopping_rounds": 15,
                           "learning_rate": 0.05, "max_depth": 6, "subsample": 0.8,
                           "colsample_bytree": 0.8, "eval_metric": "logloss",
                           "objective": "binary:logistic", "random_state": seed}.items():
            if p[key] != value:
                raise ValueError(f"Unexpected trained parameter {key}: {p[key]}")
        device = json.loads(estimator.get_booster().save_config())["learner"]["generic_param"]["device"]
        if not device.startswith("cuda"):
            raise ValueError("A trained label did not use CUDA")
        best_iterations.append(int(estimator.best_iteration))
        fitted_rounds.append(int(estimator.get_booster().num_boosted_rounds()))
    final_round = max(fitted_rounds)
    interval = config["learning_curve_every"]
    expected_steps = list(range(interval, final_round, interval)) + [final_round]
    # Pandas writes values such as 20.0 because the data_loaded row has no step.
    actual_steps = [float(row["step"]) for row in logs if row["event"] == "learning_curve"]
    if actual_steps != expected_steps:
        raise ValueError("Missing or reordered learning-curve points")
    summary = {key: config[key] for key in ["model", "modality", "seed", "input_dim", "preprocessing", "threshold", "split_sha256"]}
    for key in ["train_f1_micro", "train_f1_macro", "val_f1_micro", "val_f1_macro", "test_f1_micro", "test_f1_macro"]:
        value = float(final[key])
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid metric {key}")
        summary[key] = value
    summary.update(training_seconds=float(final["fit_seconds"]), wall_seconds=elapsed,
                   best_iteration=json.dumps(best_iterations), status="completed")
    provenance = json.loads(Path(config["split_path"]).with_suffix(".provenance.json").read_text())
    summary["historical_identity_verified"] = provenance["historical_identity_verified"]
    files = [p for p in (attempt / "artifacts").rglob("*") if p.is_file()]
    files += [p for p in (attempt / "label_models").rglob("*") if p.is_file()]
    files += [attempt / "config.json", attempt / "console.log"]
    if (attempt / "fit_recovery.json").exists():
        files.append(attempt / "fit_recovery.json")
    hashes = {str(p.relative_to(attempt)): sha256(p) for p in files}
    write_json(attempt / "completed.json", {"completed": stamp(), "config": config,
               "summary": summary, "file_sha256": hashes})
    return summary


def run(args):
    repo, root, local = args.repo.resolve(), args.root.resolve(), args.local.resolve()
    base = root / "experiments/xgboost_raw_native"
    local.mkdir(parents=True, exist_ok=True)
    # The lock is deliberately local: Global Volumes do not provide safe locking.
    lock = (local / "runner.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ready = json.loads((base / "configurations/ready.json").read_text())
    if ready["reference_commit"] != REFERENCE or ready["sources"] != check_science(repo):
        raise ValueError("Reference source validation failed")
    if sha256(root / "data/dataset_manifest.json") != ready["dataset_manifest_sha256"]:
        raise ValueError("Dataset manifest changed")
    audit_path = base / "configurations/data_and_split_audit.json"
    if sha256(audit_path) != ready["audit_sha256"]:
        raise ValueError("Data/split audit changed")
    audit = json.loads(audit_path.read_text())
    # Detect missing/replaced data before launching; complete OID verification is
    # performed in prepare.py and retained in the manifest.
    for entry in json.loads((root / "data/dataset_manifest.json").read_text())["files"]:
        if (root / "data/raw" / entry["name"]).stat().st_size != entry["bytes"]:
            raise ValueError("A raw data file changed")
    cache = local / "cache"
    cache.mkdir(exist_ok=True)
    persistent_cache = root / "data/cache"
    persistent_cache.mkdir(parents=True, exist_ok=True)
    for cached in persistent_cache.glob("*.npz"):
        target = cache / cached.name
        if not target.exists() or target.stat().st_size != cached.stat().st_size:
            publish(cached, target)
    env = dict(os.environ, PYTHONPATH=str(repo / "src"), PYTHONUNBUFFERED="1",
               OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
               OPENBLAS_NUM_THREADS=str(args.threads), XGB_REPO=str(repo),
               XGB_LOCAL_CACHE=str(cache), XGB_PERSISTENT_CACHE=str(persistent_cache))
    os.environ["PYTHONPATH"] = str(repo / "src")
    failures = 0
    summarize(root)
    # Modality-major order reuses each native-width cache for all five seeds.
    for modality in MODALITIES:
        for seed in SEEDS:
            split = root / "shared_splits" / f"{modality}_seed_{seed}.npz"
            split_hash = sha256(split)
            if split_hash != audit["modalities"][modality]["splits"][str(seed)]["sha256"]:
                raise ValueError("Split changed after validation")
            config = run_config(modality, seed, split, split_hash, ready["dataset_manifest_sha256"])
            attempts = base / modality / f"seed_{seed}" / "attempts"
            if any(valid_completion(p, config) for p in attempts.glob("*")):
                print(f"SKIP_VERIFIED {modality} seed={seed}", flush=True)
                continue
            attempt = attempts / (stamp() + "-" + uuid.uuid4().hex[:8])
            (attempt / "artifacts").mkdir(parents=True)
            argv = command(repo, root / "data/raw", attempt, split, cache, modality, seed, args.threads)
            saved_fit = recoverable_fit(attempts, config, argv)
            write_json(attempt / "config.json", {**config, "argv": argv, "created": stamp(),
                       "evaluation": "cuda-dmatrix-staged",
                       "saved_fit_attempt": str(saved_fit) if saved_fit is not None else None,
                       "infrastructure_commit": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()})
            log_path = local / f"{modality}-{seed}-{attempt.name}.log"
            started = time.monotonic()
            state = {"status": "running", "hostname": socket.gethostname(), "runner_pid": os.getpid(),
                     "modality": modality, "seed": seed, "started": stamp()}
            print(f"EXPERIMENT_START {modality} seed={seed} {attempt}", flush=True)
            with log_path.open("wb", buffering=0) as log:
                run_env = dict(env, XGB_ATTEMPT=str(attempt),
                               XGB_SAVED_FIT_ATTEMPT=str(saved_fit) if saved_fit is not None else "")
                process = subprocess.Popen(argv, cwd=repo, env=run_env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                state["training_pid"] = process.pid
                try:
                    while process.poll() is None:
                        state["heartbeat_unix"] = time.time()
                        write_json(attempt / "status.json", state)
                        publish_log_snapshot(log_path, attempt / "console.log")
                        summarize(root)
                        try:
                            process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            pass
                except BaseException:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=30)
                    raise
            publish_log_snapshot(log_path, attempt / "console.log")
            state.update(exit_code=process.returncode, finished=stamp(), wall_seconds=time.monotonic() - started)
            try:
                if process.returncode != 0:
                    raise RuntimeError(f"Trainer exited with status {process.returncode}")
                collect_result(repo, attempt, config, state["wall_seconds"])
                state["status"] = "completed"
                print(f"EXPERIMENT_COMPLETED {modality} seed={seed}", flush=True)
            except Exception as error:
                state.update(status="failed", error=str(error))
                failures += 1
                print(f"EXPERIMENT_FAILED {modality} seed={seed}: {error}", flush=True)
            write_json(attempt / "status.json", state)
            summarize(root)
            if failures:
                # Avoid spending on 24 more runs after a systematic first failure.
                raise SystemExit("Stopped after a failed experiment; inspect preserved attempt and resume safely")
    print(json.dumps(summarize(root)["counts"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["run", "status"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--local", type=Path, default=Path("/root/runpod-work"))
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    if args.action == "status":
        print(json.dumps(summarize(args.root), indent=2))
    else:
        run(args)
