from __future__ import annotations

import numpy as np

from spectroscopy_qml.ir.tree_tensor_network.data_loader import prepare_dataloaders
from spectroscopy_qml.ir.tree_tensor_network.train import (
    EarlyStopping,
    tune_thresholds,
)


def _collect_labels(dataloader) -> np.ndarray:
    return np.vstack([labels.numpy() for _, labels in dataloader])


def test_prepare_dataloaders_multilabel_stratification_preserves_rare_labels() -> None:
    rng = np.random.default_rng(42)
    x = rng.normal(size=(60, 16)).astype(np.float32)
    y = np.zeros((60, 3), dtype=np.float32)

    y[:20, 0] = 1.0
    y[20:40, 1] = 1.0
    y[40:, 2] = 1.0
    y[::6, 0] = 1.0
    y[1::6, 1] = 1.0
    y[2::6, 2] = 1.0

    train_loader, val_loader, test_loader = prepare_dataloaders(
        x,
        y,
        batch_size=8,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=7,
        stratify_multilabel=True,
    )

    for labels in (_collect_labels(train_loader), _collect_labels(val_loader), _collect_labels(test_loader)):
        assert labels.sum(axis=0).min() > 0


def test_tune_thresholds_per_class_improves_macro_f1_over_default_threshold() -> None:
    y_true = np.array(
        [
            [1, 0],
            [1, 0],
            [0, 0],
            [0, 1],
            [0, 1],
            [0, 1],
        ],
        dtype=int,
    )
    y_probs = np.array(
        [
            [0.90, 0.15],
            [0.65, 0.25],
            [0.45, 0.35],
            [0.40, 0.55],
            [0.30, 0.65],
            [0.20, 0.85],
        ],
        dtype=float,
    )

    default_preds = (y_probs >= 0.5).astype(int)
    default_macro = np.mean(
        [
            2 * ((default_preds[:, i] & y_true[:, i]).sum())
            / max(default_preds[:, i].sum() + y_true[:, i].sum(), 1)
            for i in range(y_true.shape[1])
        ]
    )

    thresholds = tune_thresholds(y_true, y_probs, metric="per_class_f1")
    tuned_preds = (y_probs >= thresholds).astype(int)
    tuned_macro = np.mean(
        [
            2 * ((tuned_preds[:, i] & y_true[:, i]).sum())
            / max(tuned_preds[:, i].sum() + y_true[:, i].sum(), 1)
            for i in range(y_true.shape[1])
        ]
    )

    assert thresholds.shape == (2,)
    assert tuned_macro >= default_macro


def test_early_stopping_stops_after_patience_without_improvement() -> None:
    early_stopping = EarlyStopping(patience=2, min_delta=0.01, mode="max", verbose=False)

    assert early_stopping(0.60) is False
    assert early_stopping(0.605) is False
    assert early_stopping(0.604) is True


def test_tune_thresholds_f1_micro_returns_shared_threshold() -> None:
    y_true = np.array(
        [
            [1, 0],
            [1, 0],
            [0, 1],
            [0, 1],
        ],
        dtype=int,
    )
    y_probs = np.array(
        [
            [0.9, 0.2],
            [0.7, 0.3],
            [0.4, 0.8],
            [0.2, 0.6],
        ],
        dtype=float,
    )

    thresholds = tune_thresholds(y_true, y_probs, metric="f1_micro")

    assert thresholds.shape == (2,)
    assert np.allclose(thresholds, thresholds[0])
