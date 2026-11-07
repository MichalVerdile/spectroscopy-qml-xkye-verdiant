"""Raw experiment orchestration, strict reuse identity and sealed test evaluation."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .raw_evaluation import PROTOCOL_VERSION, evaluate, tune_thresholds
from .raw_experiments import trial_configs
from .raw_guard import strict_raw_guard
from .raw_models import build_model
from .raw_native import MODALITIES, RawCohort, atomic_json, digest_json, file_sha256, prepare_cohort
from .raw_smoke import write_synthetic_parquet
from .raw_splits import cohort_split
from .raw_training import (
    batches,
    load_checkpoint,
    load_xgb,
    predict_neural,
    predict_xgb,
    seed_everything,
    sync,
    train_neural,
    train_xgb,
    xgb_matrix,
)

ROOT = Path(__file__).resolve().parents[3]
FAMILIES = ("cnn", "xgb", "mps", "ttn")


def code_identity():
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        head = "unversioned"
    paths = sorted((ROOT / "src/spectroscopy_qml/benchmarking").rglob("*.py"))
    paths += [
        ROOT / "scripts/run_raw_native_benchmark.py",
        ROOT / "src/spectroscopy_qml/mps_classifier/ir/model.py",
    ]
    return {
        "git_head": head,
        "source_sha256": digest_json({str(p.relative_to(ROOT)): file_sha256(p) for p in paths}),
    }


def environment(device):
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": device,
        "device_name": torch.cuda.get_device_name()
        if device.startswith("cuda")
        else platform.machine(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "torch", "xgboost", "pyarrow", "scikit-learn", "rdkit")
        },
    }


def verify_complete(directory: Path, name="complete.json"):
    marker = json.loads((directory / name).read_text())
    for filename, expected in marker["files"].items():
        if file_sha256(directory / filename) != expected:
            raise ValueError(f"Completed artifact failed integrity check: {filename}")
    return marker


def architecture_config(config):
    return {k: v for k, v in config.items() if k not in {"learning_rate", "weight_decay"}}


def predict_partition(model, cohort, modality, indices, device, batch_size, output):
    sync(device)
    started = time.perf_counter()
    if isinstance(model, tuple):
        pieces = []
        for start in range(0, len(indices), 4096):
            rows = indices[start : start + 4096]
            matrix = cohort.features(modality, rows).reshape(len(rows), -1)
            pieces.append(predict_xgb(model[0], matrix, model[1]))
        probabilities = np.concatenate(pieces)
        truths = np.asarray(cohort.labels[indices])
    else:
        truths, probabilities = predict_neural(
            model, batches(cohort, modality, indices, batch_size), device
        )
    sync(device)
    return truths, probabilities, time.perf_counter() - started


def run_family(args, cohort, split, modality, family, code, env, overrides):
    from filelock import FileLock

    output = args.output_dir / args.experiment_id / modality / f"seed_{split['seed']}" / family
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / ".run.lock"), timeout=0):
        return _run_family_locked(args, cohort, split, modality, family, code, env, overrides)


def _run_family_locked(args, cohort, split, modality, family, code, env, overrides):
    shape = cohort.manifest["modalities"][modality]["shape"]
    configs = trial_configs(
        family, args.trials, args.search_seed, overrides.get(family), args.ablation_factor
    )
    if args.variant == "baseline" and family == "xgb":
        configs = [{**c, "subsample": 0.8, "colsample_bytree": 0.8} for c in configs]
    if args.smoke_test:
        if family == "xgb":
            configs = [{**c, "n_estimators": 3, "max_depth": 2} for c in configs]
        elif family == "cnn":
            configs = [{**c, "width": 12} for c in configs]
        elif family == "mps":
            configs = [{**c, "physical_dim": 8, "bond_dim": 8} for c in configs]
        else:
            configs = [{**c, "chi": 8} for c in configs]
    spec = {
        "dataset_hash": cohort.manifest["dataset_hash"],
        "input_hash": cohort.input_hash(modality),
        "split_hash": split["split_hash"],
        "seed": split["seed"],
        "modality": modality,
        "family": family,
        "variant": args.variant,
        "mode": "raw_native",
        "protocol": PROTOCOL_VERSION,
        "code": code,
        "environment": env,
        "trial_configs": configs,
        "experiment": args.experiment_id,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "time_limit_per_trial": args.time_limit_per_trial,
        "threads": args.threads,
        "xgb_memory_mode": args.xgb_memory_mode,
        "xgb_external_batch_size": args.xgb_external_batch_size,
        "xgb_external_cache": args.xgb_external_cache,
        "parameter_limit": args.parameter_limit,
        "threshold_mode": args.threshold_mode,
        "calibration_min_samples": args.calibration_min_samples,
        "calibration_min_support": args.calibration_min_support,
        "synthetic": args.smoke_test,
        "selection_metric": "selection_f1_micro_at_0.5",
    }
    identity = digest_json(spec)
    output = args.output_dir / args.experiment_id / modality / f"seed_{split['seed']}" / family
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / "identity.json"
    if identity_path.exists():
        previous = json.loads(identity_path.read_text())
        if previous["identity"] != identity:
            raise ValueError(
                f"Refusing incompatible output reuse: {output}. Use a new experiment ID."
            )
        if not args.resume and args.phase != "final":
            raise ValueError(f"Output already exists: {output}; use --resume")
    else:
        if args.phase == "final":
            raise ValueError("Final evaluation requires a completed, matching tuning run")
        atomic_json(identity_path, {"identity": identity, "spec": spec})
        atomic_json(output / "split.json", split)
    if (output / "complete.json").exists():
        verify_complete(output)
        print(f"Verified complete run; skip {output}", flush=True)
        return
    selection_path = output / "selection.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text())
        if selection["identity"] != identity:
            raise ValueError("Selection identity mismatch")
    elif args.phase == "final":
        raise ValueError("No completed hyperparameter selection; run --phase tune first")
    else:
        trial_results = []
        for index, config in enumerate(configs):
            trial = output / "trials" / f"trial_{index:03d}"
            trial.mkdir(parents=True, exist_ok=True)
            trial_identity = digest_json([identity, index, config])
            if (trial / "complete.json").exists():
                marker = verify_complete(trial)
                if marker["identity"] != trial_identity:
                    raise ValueError("Trial identity mismatch")
                result = json.loads((trial / "trial.json").read_text())
            else:
                trial_started = time.perf_counter()
                seed_everything(split["seed"])
                if family == "xgb":
                    _, _, statistics = train_xgb(
                        cohort,
                        modality,
                        split["partitions"],
                        trial,
                        trial_identity,
                        seed=split["seed"],
                        device=args.device,
                        config=config,
                        patience=args.patience,
                        time_limit=args.time_limit_per_trial,
                        resume=args.resume,
                        threads=args.threads,
                        memory_mode=args.xgb_memory_mode,
                        external_batch_size=args.xgb_external_batch_size,
                        external_on_host=args.xgb_external_cache == "host",
                    )
                    architecture = "xgb_raw_native_v1"
                    checkpoint_files = ["xgb.json", *[p.name for p in trial.glob("label_*.ubj")]]
                else:
                    model = build_model(
                        family,
                        shape[1],
                        shape[2] + 1,
                        architecture_config(config),
                        variant=args.variant,
                        modality=modality,
                        parameter_limit=args.parameter_limit,
                    )
                    count = sum(p.numel() for p in model.parameters())
                    if count > args.parameter_limit:
                        raise ValueError(
                            f"Model has {count} parameters, exceeds explicit limit {args.parameter_limit}"
                        )
                    architecture = model.architecture
                    statistics = train_neural(
                        model,
                        cohort,
                        modality,
                        split["partitions"],
                        trial,
                        trial_identity,
                        seed=split["seed"],
                        device=args.device,
                        batch_size=args.batch_size,
                        epochs=args.epochs,
                        patience=args.patience,
                        learning_rate=config["learning_rate"],
                        weight_decay=config["weight_decay"],
                        time_limit=args.time_limit_per_trial,
                        resume=args.resume,
                    )
                    checkpoint_files = ["best.pt", "best.sha256.json"]
                    del model
                    if args.device.startswith("cuda"):
                        torch.cuda.empty_cache()
                result = {
                    "trial": index,
                    "config": config,
                    "identity": trial_identity,
                    "architecture": architecture,
                    "trial_wall_seconds_this_invocation": time.perf_counter() - trial_started,
                    **statistics,
                }
                atomic_json(trial / "trial.json", result)
                atomic_json(
                    trial / "complete.json",
                    {
                        "identity": trial_identity,
                        "files": {
                            name: file_sha256(trial / name)
                            for name in ["trial.json", "history.json", *checkpoint_files]
                        },
                    },
                )
            trial_results.append(result)
        # Deterministic tie: first declared trial wins. No test metrics enter here.
        winner = max(trial_results, key=lambda result: result["selection_f1_micro"])
        selection = {
            "identity": identity,
            "winner": winner,
            "trials": trial_results,
            "search_training_seconds": sum(r["training_seconds"] for r in trial_results),
            "search_trial_count": len(trial_results),
            "search_fit_wall_seconds_this_invocation": sum(
                r["fit_wall_seconds_this_invocation"] for r in trial_results
            ),
        }
        atomic_json(selection_path, selection)
    if args.phase == "tune":
        print(f"Tuning complete; test untouched: {output}", flush=True)
        return
    winner = selection["winner"]
    trial = output / "trials" / f"trial_{winner['trial']:03d}"
    verify_complete(trial)
    seed_everything(split["seed"])
    if family == "xgb":
        model = load_xgb(trial, args.device, args.threads)
    else:
        model = build_model(
            family,
            shape[1],
            shape[2] + 1,
            architecture_config(winner["config"]),
            variant=args.variant,
            modality=modality,
            parameter_limit=args.parameter_limit,
        ).to(args.device)
        state = load_checkpoint(trial / "best.pt")
        if state["identity"] != winner["identity"]:
            raise ValueError("Winner checkpoint identity mismatch")
        model.load_state_dict(state["model"])
    partitions = split["partitions"]
    y_cal, p_cal, _ = predict_partition(
        model, cohort, modality, partitions["calibration"], args.device, args.batch_size, output
    )
    calibrations = {
        mode: tune_thresholds(
            y_cal,
            p_cal,
            mode,
            min_support=args.calibration_min_support,
            min_samples=args.calibration_min_samples,
        )
        for mode in ("fixed", "global", "per_class")
    }
    atomic_json(output / "thresholds.json", calibrations)
    # Persist a sealed selection/calibration record BEFORE touching test labels.
    seal = {
        "identity": identity,
        "selection_sha256": file_sha256(selection_path),
        "thresholds_sha256": file_sha256(output / "thresholds.json"),
        "primary_metric": "test_f1_micro",
        "primary_threshold_mode": args.threshold_mode,
    }
    seal_path = output / "test_seal.json"
    if seal_path.exists() and json.loads(seal_path.read_text()) != seal:
        raise ValueError("Test already opened with different selection/calibration")
    atomic_json(seal_path, seal)
    y_test, p_test, inference_seconds = predict_partition(
        model, cohort, modality, partitions["test"], args.device, args.batch_size, output
    )
    metrics = {
        mode: evaluate(y_test, p_test, calibration["thresholds"])
        for mode, calibration in calibrations.items()
    }
    prediction_path = output / "predictions.npz"
    # Final marker is only published after all artifact writes complete.
    np.savez_compressed(
        prediction_path,
        y_test=y_test,
        p_test=p_test,
        y_calibration=y_cal,
        p_calibration=p_cal,
        test_sample_ids=np.array([cohort.records[i]["sample_id"] for i in partitions["test"]]),
        calibration_sample_ids=np.array(
            [cohort.records[i]["sample_id"] for i in partitions["calibration"]]
        ),
        train_support=cohort.labels[partitions["train"]].sum(0),
        label_names=np.array(cohort.manifest["label_names"]),
    )
    metrics_primary = metrics[args.threshold_mode]
    result = {
        "identity": identity,
        "spec": spec,
        "winner": winner,
        "metrics": metrics,
        **{f"test_{k}": v for k, v in metrics_primary.items() if k != "reliability_bins"},
        "inference_seconds": inference_seconds,
        "inference_scope": "end-to-end test partition incl. host loading/conversion and device synchronization",
        "n_test": len(y_test),
        "search_training_seconds": selection["search_training_seconds"],
        "search_trial_count": selection["search_trial_count"],
        "search_fit_wall_seconds_this_invocation": selection[
            "search_fit_wall_seconds_this_invocation"
        ],
        "model_bytes": sum(
            (trial / n).stat().st_size
            for n in verify_complete(trial)["files"]
            if n.endswith((".pt", ".ubj")) or n == "xgb.json"
        ),
        "model_bytes_scope": "selected inference checkpoint(s), including XGBoost constant-label manifest; excludes optimizer/recovery checkpoints and calibration",
    }
    atomic_json(output / "result.json", result)
    files = [
        "result.json",
        "predictions.npz",
        "thresholds.json",
        "selection.json",
        "split.json",
        "test_seal.json",
        "identity.json",
    ]
    atomic_json(
        output / "complete.json",
        {"identity": identity, "files": {name: file_sha256(output / name) for name in files}},
    )
    print(
        f"Final {modality}/{family} seed={split['seed']} micro-F1={result['test_f1_micro']:.5f}",
        flush=True,
    )


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data/raw")
    p.add_argument("--cache-dir", type=Path, default=ROOT / "data/cache/raw_native_v1")
    p.add_argument(
        "--metadata-path", type=Path, default=ROOT / "data/meta_data/spectrum_dimensions.json"
    )
    p.add_argument("--model", choices=["all", *FAMILIES], default="all")
    p.add_argument("--mode", choices=["raw_native"], default="raw_native")
    p.add_argument("--modality", choices=["all", *MODALITIES], default="all")
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--experiment-id", default="B_optimized_raw")
    p.add_argument("--output-dir", type=Path, default=ROOT / "benchmark/raw_native_v1")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--patience", "--early-stopping-patience", type=int, default=15)
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--search-seed", type=int, default=2026)
    p.add_argument("--time-limit-per-trial", type=float, default=None)
    p.add_argument("--parameter-limit", type=int, default=100_000_000)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--xgb-memory-mode", choices=["auto", "ram", "external"], default="auto")
    p.add_argument("--xgb-external-batch-size", type=int, default=4096)
    p.add_argument("--xgb-external-cache", choices=["host", "disk"], default="host")
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--variant", choices=["baseline", "optimized"], default="optimized")
    p.add_argument("--phase", choices=["prepare", "tune", "final", "all"], default="tune")
    p.add_argument("--split-kind", choices=["molecule", "scaffold"], default="molecule")
    p.add_argument("--invalid-policy", choices=["error", "exclude"], default="error")
    p.add_argument("--max-samples", type=int)
    p.add_argument("--threshold-mode", choices=["fixed", "global", "per_class"], default="global")
    p.add_argument("--calibration-min-samples", type=int, default=100)
    p.add_argument("--calibration-min-support", type=int, default=20)
    p.add_argument("--config-json", type=Path)
    p.add_argument("--ablation-factor", help="Experiment C: change one TN architecture factor only")
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--strict-raw",
        action="store_true",
        default=True,
        help="Always enabled; raw invariants cannot be disabled",
    )
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
        for c in args.experiment_id
    ) or args.experiment_id in {"", ".", ".."}:
        raise ValueError("Experiment ID must be a simple directory name")
    args.epochs = args.epochs if args.epochs is not None else (2 if args.smoke_test else 100)
    args.seeds = args.seeds or ([42] if args.smoke_test else [41, 42, 43, 44, 45])
    if (
        min(
            args.batch_size,
            args.epochs,
            args.patience,
            args.trials,
            args.threads,
            args.parameter_limit,
        )
        <= 0
    ):
        raise ValueError("Training limits must be positive")
    if args.xgb_external_batch_size < 1:
        raise ValueError("XGBoost external batch size must be positive")
    if args.ablation_factor and (args.model not in {"mps", "ttn"} or args.variant != "optimized"):
        raise ValueError("One-factor ablations require --model mps or ttn and --variant optimized")
    if args.time_limit_per_trial is not None and args.time_limit_per_trial <= 0:
        raise ValueError("time-limit-per-trial must be positive")
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; no silent CPU fallback")
    torch.set_num_threads(args.threads)
    args.output_dir = args.output_dir.resolve()
    overrides = json.loads(args.config_json.read_text()) if args.config_json else {}
    if set(overrides) - set(FAMILIES):
        raise ValueError("Unknown model in config JSON")
    if args.smoke_test:
        # Segregated synthetic output namespace even if a real experiment ID was supplied.
        args.experiment_id = "synthetic_" + args.experiment_id.removeprefix("synthetic_")
        args.data_dir = args.output_dir / args.experiment_id / "synthetic_source"
        write_synthetic_parquet(args.data_dir)
    with strict_raw_guard():
        cohort = prepare_cohort(
            args.data_dir,
            args.cache_dir,
            metadata_path=None if args.smoke_test else args.metadata_path,
            invalid_policy=args.invalid_policy,
            max_samples=args.max_samples,
            synthetic=args.smoke_test,
        )
        print(
            f"Verified raw cohort: n={len(cohort.labels)} hash={cohort.manifest['dataset_hash']}",
            flush=True,
        )
        if args.phase == "prepare":
            print(cohort.root)
            return
        modalities = MODALITIES if args.modality == "all" else [args.modality]
        families = FAMILIES if args.model == "all" else [args.model]
        code, env = code_identity(), environment(args.device)
        failures = []
        for seed in args.seeds:
            split = cohort_split(cohort, seed, args.split_kind)
            for modality in modalities:
                for family in families:
                    try:
                        run_family(args, cohort, split, modality, family, code, env, overrides)
                    except (ValueError, RuntimeError, OSError) as error:
                        failure = {
                            "seed": seed,
                            "modality": modality,
                            "model": family,
                            "error": str(error),
                        }
                        failures.append(failure)
                        print(f"FAILED explicitly: {failure}", file=sys.stderr, flush=True)
        if failures:
            atomic_json(args.output_dir / args.experiment_id / "failures.json", failures)
            raise SystemExit(2)


if __name__ == "__main__":
    main()
