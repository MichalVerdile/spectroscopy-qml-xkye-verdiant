"""One versioned selection, calibration and metric definition for all models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import f1_score, hamming_loss, precision_score, recall_score

PROTOCOL_VERSION = "raw_eval_v1"
THRESHOLD_GRID = np.round(np.arange(0.1, 0.91, 0.05), 2)


def validate_predictions(y, probabilities):
    y, probabilities = np.asarray(y), np.asarray(probabilities)
    if y.ndim != 2 or y.shape != probabilities.shape or len(y) == 0:
        raise ValueError("Expected matching nonempty (samples, labels) arrays")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(probabilities).all():
        raise ValueError("Invalid labels/probabilities")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probabilities must be in [0,1]")
    return y, probabilities


def micro_f1(y, prediction) -> float:
    tp = np.count_nonzero((y == 1) & prediction)
    denom = np.count_nonzero(y) + np.count_nonzero(prediction)
    return float(2 * tp / denom) if denom else 0.0


def tune_thresholds(y, probabilities, mode="global", *, min_support=20, min_samples=100):
    """Bounded calibration only. Per-class optimizes each class F1 (macro objective).

    Small calibration sets fall back to fixed .5; rare classes fall back to the
    global threshold. Ties prefer .5/closest-to-global, then the lower threshold.
    """
    y, probabilities = validate_predictions(y, probabilities)
    if mode not in {"fixed", "global", "per_class"}:
        raise ValueError("Unknown threshold mode")
    if min_support < 1 or min_samples < 1:
        raise ValueError("Calibration minimums must be positive")
    fallback = len(y) < min_samples
    thresholds = np.full(y.shape[1], 0.5)
    if mode != "fixed" and not fallback:
        ordered = sorted(THRESHOLD_GRID, key=lambda t: (abs(t - 0.5), t))
        best = max(ordered, key=lambda t: micro_f1(y, probabilities >= t))
        thresholds[:] = best
        if mode == "per_class":
            ordered = sorted(THRESHOLD_GRID, key=lambda t: (abs(t - best), t))
            for j in range(y.shape[1]):
                support = int(y[:, j].sum())
                if min(support, len(y) - support) >= min_support:
                    thresholds[j] = max(
                        ordered, key=lambda t: micro_f1(y[:, j], probabilities[:, j] >= t)
                    )
    return {
        "mode": mode,
        "thresholds": thresholds.tolist(),
        "n_calibration": len(y),
        "min_support": min_support,
        "min_samples": min_samples,
        "small_set_fallback": fallback and mode != "fixed",
        "grid": THRESHOLD_GRID.tolist(),
        "objective": "per-class F1" if mode == "per_class" else "micro F1",
    }


def evaluate(y, probabilities, thresholds=0.5) -> dict:
    y, probabilities = validate_predictions(y, probabilities)
    thresholds = np.asarray(thresholds)
    if thresholds.shape not in {(), (y.shape[1],)} or not np.isfinite(thresholds).all():
        raise ValueError("Invalid threshold shape/values")
    if np.any((thresholds < 0) | (thresholds > 1)):
        raise ValueError("Thresholds must be probabilities")
    prediction = probabilities >= thresholds
    scores: dict = {
        f"f1_{a}": float(f1_score(y, prediction, average=a, zero_division=0))
        for a in ("micro", "macro", "weighted")
    }
    for name, fn in (("precision", precision_score), ("recall", recall_score)):
        for average in ("micro", "macro"):
            scores[f"{name}_{average}"] = float(fn(y, prediction, average=average, zero_division=0))
    p = np.clip(probabilities.astype(np.float64), 1e-7, 1 - 1e-7)
    scores.update(
        per_class_f1=f1_score(y, prediction, average=None, zero_division=0).tolist(),
        support=y.sum(axis=0).astype(int).tolist(),
        hamming_loss=float(hamming_loss(y, prediction)),
        bce=float(-(y * np.log(p) + (1 - y) * np.log1p(-p)).mean()),
        brier=float(np.square(probabilities - y).mean()),
    )
    # Label/sample pooled reliability bins; not a calibration model fitted on test.
    bins = []
    for i in range(10):
        mask = (probabilities >= i / 10) & (
            (probabilities < (i + 1) / 10) if i < 9 else (probabilities <= 1)
        )
        bins.append(
            {
                "count": int(mask.sum()),
                "confidence": float(probabilities[mask].mean()) if mask.any() else None,
                "frequency": float(y[mask].mean()) if mask.any() else None,
            }
        )
    scores["reliability_bins"] = bins
    return scores


@dataclass
class EarlyStopping:
    """Stop after patience consecutive selection checks without STRICT improvement."""

    patience: int = 15
    min_delta: float = 0.0
    best: float = -1.0
    best_step: int = -1
    stale: int = 0

    def __post_init__(self):
        if self.patience < 1 or self.min_delta < 0:
            raise ValueError("Invalid early stopping configuration")

    def update(self, score: float, step: int) -> bool:
        if not np.isfinite(score):
            raise ValueError("Nonfinite validation selection score")
        improved = score > self.best + self.min_delta
        if improved:
            self.best, self.best_step, self.stale = float(score), step, 0
        else:
            self.stale += 1
        return improved

    @property
    def stopped(self) -> bool:
        return self.stale >= self.patience
