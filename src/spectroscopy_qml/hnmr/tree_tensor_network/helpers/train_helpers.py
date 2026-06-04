"""Training entry point for experiment 5."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = Path(__file__).resolve().parents[5]
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.spectroscopy_qml.hnmr.tree_tensor_network.helpers.data_loader import (  # noqa: E402
    FUNCTIONAL_GROUPS,
    load_cnmr_data,
    load_or_create_split_indices,
    prepare_dataloaders_from_split_indices,
)
from src.spectroscopy_qml.hnmr.tree_tensor_network.helpers.losses import build_loss  # noqa: E402
from src.spectroscopy_qml.hnmr.tree_tensor_network.helpers.isometric_helpers import TTNCnmrClassifier5  # noqa: E402


class EarlyStopping:
    """Early stopping on a configurable validation score."""

    def __init__(self, patience: int = 20, min_delta: float = 1e-4, mode: str = "max") -> None:
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.counter = 0
        self.best_score: float | None = None

    def _is_improvement(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "min":
            return score < self.best_score - self.min_delta
        return score > self.best_score + self.min_delta

    def __call__(self, score: float) -> bool:
        if self._is_improvement(score):
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        print(f"EarlyStopping counter: {self.counter}/{self.patience}")
        return self.counter >= self.patience

def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if device_arg == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def count_available_data_files(data_dir: Path) -> int:
    return len(list(Path(data_dir).glob("*.parquet")))


def resolve_used_file_count(total_files: int, max_files: int | None) -> int:
    if max_files is None:
        return total_files
    return min(total_files, max(int(max_files), 0))


def threshold_predictions(y_prob: np.ndarray, thresholds: float | np.ndarray = 0.5) -> np.ndarray:
    return (y_prob >= thresholds).astype(int)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | np.ndarray]:
    return {
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_micro": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "recall_micro": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "per_class_f1": f1_score(y_true, y_pred, average=None, zero_division=0),
    }


def build_threshold_grid(step: float) -> np.ndarray:
    if not 0.0 < step < 1.0:
        raise ValueError("threshold_grid_step must be in (0, 1).")
    grid = np.arange(step, 1.0, step, dtype=np.float32)
    if grid.size == 0:
        raise ValueError("threshold grid is empty.")
    return np.unique(np.clip(grid, step, 0.95))


def get_pos_weight(
    labels: np.ndarray,
    device: torch.device,
    power: float = 1.0,
    max_value: float | None = None,
) -> torch.Tensor:
    positives = labels.sum(axis=0)
    negatives = labels.shape[0] - positives
    pos_weight = np.ones_like(positives, dtype=np.float32)
    np.divide(negatives, positives, out=pos_weight, where=positives > 0)
    if power <= 0.0:
        raise ValueError("pos_weight_power must be positive.")
    if power != 1.0:
        pos_weight = np.power(pos_weight, power, dtype=np.float32)
    if max_value is not None:
        pos_weight = np.clip(pos_weight, 1.0, max_value)
    return torch.as_tensor(pos_weight, dtype=torch.float32, device=device)


def score_threshold_metrics(metrics: dict[str, float | np.ndarray], target_metric: str) -> float:
    if target_metric == "f1_micro":
        return float(metrics["f1_micro"])
    if target_metric == "f1_macro":
        return float(metrics["f1_macro"])
    if target_metric == "per_class_f1":
        return float(np.mean(np.asarray(metrics["per_class_f1"], dtype=np.float32)))
    raise ValueError(f"Unsupported threshold target metric: {target_metric}")


def tune_thresholds(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold_mode: str,
    target_metric: str,
    threshold_grid: np.ndarray,
) -> np.ndarray:
    n_classes = y_true.shape[1]

    if threshold_mode == "global":
        best_threshold = 0.5
        best_score = -1.0
        for threshold in threshold_grid:
            y_pred = threshold_predictions(y_probs, threshold)
            metrics = compute_metrics(y_true, y_pred)
            score = score_threshold_metrics(metrics, target_metric)
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)
        return np.full(n_classes, best_threshold, dtype=np.float32)

    if threshold_mode != "per_class":
        raise ValueError(f"Unsupported threshold mode: {threshold_mode}")
    if target_metric == "f1_micro":
        raise ValueError("threshold_mode='per_class' requires threshold_target_metric != 'f1_micro'.")

    thresholds = np.full(n_classes, 0.5, dtype=np.float32)
    return thresholds


def select_early_stopping_score(
    metrics: dict[str, float | np.ndarray],
    metric_name: str,
    blend_alpha: float,
) -> float:
    if metric_name == "f1_micro":
        return float(metrics["f1_micro"])
    if metric_name == "f1_macro":
        return float(metrics["f1_macro"])
    if metric_name == "blended_f1":
        if not 0.0 <= blend_alpha <= 1.0:
            raise ValueError("early_stopping_blend_alpha must be in [0, 1].")
        macro_score = float(metrics["f1_macro"])
        micro_score = float(metrics["f1_micro"])
        return blend_alpha * macro_score + (1.0 - blend_alpha) * micro_score
    raise ValueError(f"Unsupported early stopping metric: {metric_name}")


def run_synthetic_preflight(model: TTNCnmrClassifier5, args: argparse.Namespace, device: torch.device) -> None:
    print("\n[Preflight] Synthetic forward/backward check")
    model = model.to(device)
    model.train()

    spectra = torch.randn(args.preflight_batch_size, args.input_dim, device=device)
    labels = torch.randint(
        0,
        2,
        (args.preflight_batch_size, args.num_labels),
        device=device,
        dtype=torch.float32,
    )

    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    criterion = build_loss(args.loss_type)
    optimizer.zero_grad(set_to_none=True)
    logits = model(spectra)
    loss = criterion(logits, labels)
    if logits.shape != labels.shape:
        raise RuntimeError(f"Synthetic preflight shape mismatch: {tuple(logits.shape)} != {tuple(labels.shape)}")
    if not torch.isfinite(logits).all() or not torch.isfinite(loss):
        raise RuntimeError("Synthetic preflight produced non-finite outputs.")
    loss.backward()
    optimizer.step()
    print(f"  OK - logits {tuple(logits.shape)}, loss={loss.item():.4f}")


def run_real_batch_preflight(
    model: TTNCnmrClassifier5,
    dataloader: DataLoader,
    device: torch.device,
    criterion: torch.nn.Module,
) -> None:
    print("\n[Preflight] Real-data batch check")
    spectra, labels = next(iter(dataloader))
    spectra = spectra.to(device)
    labels = labels.to(device)

    model = model.to(device)
    model.train()
    optimizer = Adam(model.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    logits = model(spectra)
    loss = criterion(logits, labels)
    if not torch.isfinite(logits).all() or not torch.isfinite(loss):
        raise RuntimeError("Real-data preflight produced non-finite outputs.")
    loss.backward()
    optimizer.step()
    print(f"  OK - batch={tuple(spectra.shape)}, labels={tuple(labels.shape)}, loss={loss.item():.4f}")

def write_epoch_details(
    handle,
    epoch: int,
    label_names: list[str],
    thresholds: np.ndarray,
    train_metrics: dict[str, float | np.ndarray],
    val_metrics: dict[str, float | np.ndarray],
    early_stopping_score: float,
) -> None:
    record = {
        "epoch": epoch,
        "early_stopping_score": float(early_stopping_score),
        "thresholds": {name: float(value) for name, value in zip(label_names, thresholds.tolist(), strict=False)},
        "train": {
            "f1_micro": float(train_metrics["f1_micro"]),
            "f1_macro": float(train_metrics["f1_macro"]),
        },
        "val": {
            "f1_micro": float(val_metrics["f1_micro"]),
            "f1_macro": float(val_metrics["f1_macro"]),
            "precision_micro": float(val_metrics["precision_micro"]),
            "recall_micro": float(val_metrics["recall_micro"]),
            "per_class_f1": {
                name: float(value)
                for name, value in zip(
                    label_names,
                    np.asarray(val_metrics["per_class_f1"], dtype=np.float32).tolist(),
                    strict=False,
                )
            },
        },
    }
    handle.write(json.dumps(record) + "\n")
    handle.flush()


def write_threshold_artifact(
    threshold_path: Path,
    label_names: list[str],
    thresholds: np.ndarray,
    threshold_mode: str,
    threshold_target_metric: str,
    best_epoch: int,
) -> None:
    payload = {
        "best_epoch": int(best_epoch),
        "threshold_mode": threshold_mode,
        "threshold_target_metric": threshold_target_metric,
        "threshold_mean": float(np.mean(thresholds)),
        "threshold_std": float(np.std(thresholds)),
        "thresholds": {name: float(value) for name, value in zip(label_names, thresholds.tolist(), strict=False)},
    }
    threshold_path.write_text(json.dumps(payload, indent=2) + "\n")
    