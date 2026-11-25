"""Reuse CUDA data and XGBoost's staged prediction cache for full-data F1 curves."""
from __future__ import annotations

import math
import time

import numpy as np
from xgboost import DMatrix

THRESHOLD = 0.5
# Staging can change the order of float32 additions. Use the original predictor
# for scores near the threshold, where rounding can change a binary decision.
THRESHOLD_GUARD = 1e-4
MAX_INPUT_BYTES = 8 * 1024**3


def curve_steps(last_round, interval):
    steps = list(range(interval, last_round + 1, interval))
    if not steps or steps[-1] != last_round:
        steps.append(last_round)
    return steps


def staged_labels(estimator, matrix, features, ends):
    booster = estimator.get_booster()
    for end in ends:
        probabilities = booster.predict(matrix, iteration_range=(0, end))
        predicted = probabilities > THRESHOLD
        near_threshold = np.abs(probabilities - THRESHOLD) <= THRESHOLD_GUARD
        if near_threshold.any():
            predicted[near_threshold] = estimator.predict(
                features[near_threshold], iteration_range=(0, end)
            ).astype(bool)
        yield end, predicted


def f1_curves(counts, target):
    """Micro and macro F1 from per-label counts, including zero-support labels."""
    true_positive, predicted_positive = counts
    denominator = predicted_positive + np.count_nonzero(target, axis=0)
    macro = np.divide(
        2 * true_positive, denominator,
        out=np.zeros_like(true_positive, dtype=np.float64),
        where=denominator != 0,
    ).mean(axis=1)
    micro_denominator = denominator.sum(axis=1)
    micro = np.divide(
        2 * true_positive.sum(axis=1), micro_denominator,
        out=np.zeros(len(micro_denominator), dtype=np.float64),
        where=micro_denominator != 0,
    )
    return micro, macro


def evaluate_split(model, features, *, subset, target=None, steps=(), threads=16):
    """Return predictions at each label's best iteration and optional F1 curves."""
    import cupy as cp

    estimators = model.estimators_
    rounds = [int(est.get_booster().num_boosted_rounds()) for est in estimators]
    final_ends = [int(est.best_iteration) + 1 for est in estimators]
    stage_ends = [tuple(min(step, total) for step in steps) for total in rounds]
    predictions = np.empty((len(features), len(estimators)), dtype=np.uint8)
    counts = np.zeros((2, len(steps), len(estimators)), dtype=np.int64)
    free_bytes, _ = cp.cuda.Device().mem_info
    # XGBoost's exact DMatrix also stores indices and can allocate temporaries
    # while building it. Bound the dense input so these all fit on the GPU.
    input_bytes = min(MAX_INPUT_BYTES, free_bytes // 8)
    rows_per_block = max(1, input_bytes // (features.shape[1] * features.itemsize))
    blocks = math.ceil(len(features) / rows_per_block)
    started = time.monotonic()
    last_progress = started
    print(f"EVAL_START split={subset} rows={len(features)} points={len(steps)} batches={blocks}", flush=True)

    for block, start in enumerate(range(0, len(features), rows_per_block), 1):
        stop = min(start + rows_per_block, len(features))
        batch = features[start:stop]
        matrix = DMatrix(cp.asarray(batch, order="C"), nthread=threads, missing=np.nan)
        # The DMatrix now owns its copy. Release the temporary dense upload.
        cp.get_default_memory_pool().free_all_blocks()
        truth = target[start:stop].astype(bool) if len(steps) else None
        for label, estimator in enumerate(estimators):
            unique_curve_ends = set(stage_ends[label])
            final_end = final_ends[label]
            ends = sorted(unique_curve_ends | {final_end})
            by_end = {}
            for end, predicted in staged_labels(estimator, matrix, batch, ends):
                if end == final_end:
                    predictions[start:stop, label] = predicted
                if end in unique_curve_ends:
                    by_end[end] = (
                        np.count_nonzero(predicted & truth[:, label]),
                        np.count_nonzero(predicted),
                    )
                now = time.monotonic()
                if now - last_progress >= 20:
                    print(f"EVAL_PROGRESS split={subset} batch={block}/{blocks} "
                          f"label={label + 1}/{len(estimators)} round={end}/{ends[-1]} "
                          f"elapsed={now - started:.1f}s", flush=True)
                    last_progress = now
            if len(steps):
                counts[:, :, label] += np.asarray([by_end[end] for end in stage_ends[label]]).T
        del matrix
        print(f"EVAL_BATCH_FINISHED split={subset} batch={block}/{blocks} "
              f"elapsed={time.monotonic() - started:.1f}s", flush=True)

    curves = f1_curves(counts, target) if len(steps) else None
    print(f"EVAL_FINISHED split={subset} seconds={time.monotonic() - started:.2f}", flush=True)
    return predictions, curves


class CurveEvaluation:
    """Keep the trainer's existing logs, metrics, and model serialization."""

    def __init__(self, append_training_log, original_predict):
        self.append_training_log = append_training_log
        self.original_predict = original_predict
        self.model = None
        self.final_predictions = []

    def append_curve_logs(self, logs, model, X_train, y_train, X_val, y_val, *,
                          learning_curve_every=1, **fields):
        self.model = model
        self.threads = fields["n_jobs"]
        fitted_rounds = max(e.get_booster().num_boosted_rounds() for e in model.estimators_)
        steps = curve_steps(fitted_rounds, learning_curve_every)
        train_pred, (train_micro, train_macro) = evaluate_split(
            model, X_train, subset="train", target=y_train, steps=steps, threads=self.threads
        )
        val_pred, (val_micro, val_macro) = evaluate_split(
            model, X_val, subset="validation", target=y_val, steps=steps, threads=self.threads
        )
        self.final_predictions = [(X_train, train_pred), (X_val, val_pred)]
        for index, step in enumerate(steps):
            self.append_training_log(
                logs, **fields, event="learning_curve", step=step, n_estimators=fitted_rounds,
                train_f1=train_micro[index], val_f1=val_micro[index], test_f1=None,
                train_f1_micro=train_micro[index], train_f1_macro=train_macro[index],
                val_f1_micro=val_micro[index], val_f1_macro=val_macro[index],
                test_f1_micro=None, test_f1_macro=None, mean_cv_f1=None, std_cv_f1=None,
                best_fold=None, load_seconds=None, fit_seconds=None, test_seconds=None,
                model_type="xgboost_multioutput",
            )

    def predict(self, model, features, **kwargs):
        if model is not self.model or kwargs:
            return self.original_predict(model, features, **kwargs)
        for data, predictions in self.final_predictions:
            if data is features:
                return predictions
        predictions, _ = evaluate_split(model, features, subset="test", threads=self.threads)
        return predictions
