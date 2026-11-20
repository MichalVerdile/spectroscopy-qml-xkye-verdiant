"""Run the reference trainer with persistent fits and faster retrospective scoring.

New fits use the original XGBClassifier.fit and its original arguments.
An interrupted evaluation can reuse a complete, verified set of fits.
"""
from __future__ import annotations

import importlib.util
import faulthandler
import gc
import json
import os
import sys
import time
from pathlib import Path

from curve_evaluation import CurveEvaluation
from protocol import publish, sha256, stamp, write_json

repo = Path(os.environ["XGB_REPO"])
attempt = Path(os.environ["XGB_ATTEMPT"])
local_cache = Path(os.environ["XGB_LOCAL_CACHE"])
persistent_cache = Path(os.environ["XGB_PERSISTENT_CACHE"])
sys.path.insert(0, str(repo / "src"))
spec = importlib.util.spec_from_file_location("original_xgb_trainer", repo / "benchmark/xgb/scripts/run_xgb_baseline.py")
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)

original_fit = trainer.XGBClassifier.fit
original_log = trainer.append_training_log
recovered_fit_seconds = None
label_counter = 0


def finish_preparation():
    if not gc.isenabled():
        gc.enable()
        gc.collect()
        faulthandler.cancel_dump_traceback_later()
        print("DATA_PREPARATION_FINISHED cyclic GC restored", flush=True)


def observed_fit(self, *args, **kwargs):
    global label_counter
    finish_preparation()
    index = label_counter
    label_counter += 1
    print(f"LABEL_START index={index} time={stamp()}", flush=True)
    started = time.monotonic()
    result = original_fit(self, *args, **kwargs)
    booster = self.get_booster()
    config = json.loads(booster.save_config())
    device = config["learner"]["generic_param"]["device"]
    if not device.startswith("cuda"):
        raise RuntimeError(f"XGBoost silently used {device}, expected CUDA")
    local_model = local_cache.parent / f"label_{index:02d}.ubj"
    self.save_model(local_model)
    model_hash = publish(local_model, attempt / "label_models" / local_model.name)
    write_json(attempt / "label_models" / f"label_{index:02d}.json", {
        "label_index": index, "label_name": list(trainer.functional_groups)[index],
        "best_iteration": int(self.best_iteration), "best_score": float(self.best_score),
        "fitted_rounds": int(booster.num_boosted_rounds()), "device": device,
        "fit_seconds": time.monotonic() - started, "model_sha256": model_hash,
        "evals_result": self.evals_result(), "xgboost_config": config,
    })
    print(f"LABEL_FINISHED index={index} best_iteration={self.best_iteration} device={device}", flush=True)
    return result


original_cache_write = trainer.save_preprocessing_cache


def observed_cache_write(path, key, X, y):
    original_cache_write(path, key, X, y)
    publish(path, persistent_cache / Path(path).name)
    print(f"CACHE_PERSISTED {persistent_cache / Path(path).name}", flush=True)


def restore_fits(source, X_train, y_train, X_val, y_val, seed, device, n_jobs,
                 early_stopping_patience):
    """Build the reference estimator and restore the original trained trees."""
    global recovered_fit_seconds
    finish_preparation()
    model = trainer.build_xgboost_model(seed, device, n_jobs, early_stopping_patience)
    model.estimators_ = []
    labels = list(trainer.functional_groups)
    records = []
    for index, name in enumerate(labels):
        stem = f"label_{index:02d}"
        model_path = source / "label_models" / f"{stem}.ubj"
        meta_path = source / "label_models" / f"{stem}.json"
        meta = json.loads(meta_path.read_text())
        config = meta["xgboost_config"]
        old_device = config["learner"]["generic_param"]["device"]
        if (meta["label_index"] != index or meta["label_name"] != name
                or not old_device.startswith("cuda") or sha256(model_path) != meta["model_sha256"]):
            raise ValueError(f"Invalid saved fit: {model_path}")
        estimator = trainer.clone(model.estimator)
        estimator.load_model(model_path)
        booster = estimator.get_booster()
        booster.load_config(json.dumps(config))
        if (int(estimator.best_iteration) != meta["best_iteration"]
                or int(booster.num_boosted_rounds()) != meta["fitted_rounds"]
                or int(booster.num_features()) != X_train.shape[1]):
            raise ValueError(f"Saved fit disagrees with its metadata: {model_path}")
        estimator.evals_result_ = meta["evals_result"]
        model.estimators_.append(estimator)
        records.append((model_path, meta_path, meta))

    for model_path, meta_path, meta in records:
        if publish(model_path, attempt / "label_models" / model_path.name) != meta["model_sha256"]:
            raise ValueError(f"Saved fit changed during recovery: {model_path}")
        publish(meta_path, attempt / "label_models" / meta_path.name)
    recovered_fit_seconds = sum(meta["fit_seconds"] for _, _, meta in records)
    write_json(attempt / "fit_recovery.json", {
        "source_attempt": str(source), "restored_labels": len(records),
        "fit_seconds_from_label_records": recovered_fit_seconds,
        "models": {path.name: meta["model_sha256"] for path, _, meta in records},
    })
    print(f"FITS_RESTORED labels={len(records)} original_fit_seconds={recovered_fit_seconds:.2f}",
          flush=True)
    return model


def log_with_recovered_fit_time(logs, **fields):
    if fields["event"] == "training_finished" and recovered_fit_seconds is not None:
        fields["fit_seconds"] = recovered_fit_seconds
    original_log(logs, **fields)


evaluation = CurveEvaluation(original_log, trainer.MultiOutputClassifier.predict)


def observed_predict(self, features, **kwargs):
    return evaluation.predict(self, features, **kwargs)


trainer.XGBClassifier.fit = observed_fit
trainer.MultiOutputClassifier.predict = observed_predict
trainer.append_learning_curve_logs = evaluation.append_curve_logs
trainer.append_training_log = log_with_recovered_fit_time
trainer.save_preprocessing_cache = observed_cache_write
if saved_attempt := os.environ.get("XGB_SAVED_FIT_ATTEMPT"):
    from functools import partial
    trainer.train_xgboost_model = partial(restore_fits, Path(saved_attempt))
# Reference counting remains active. Avoid repeated scans of the hundreds of
# thousands of retained Python label lists while the original loader stacks
# the full-width matrix; restore cyclic collection before any estimator fits.
faulthandler.enable()
faulthandler.dump_traceback_later(120, repeat=True)
gc.disable()
print("DATA_PREPARATION_START cyclic GC deferred until fitting", flush=True)
trainer.main()
