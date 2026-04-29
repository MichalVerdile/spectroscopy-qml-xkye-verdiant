"""Detailed error analysis for TTN IR functional-group classifiers.

This script is designed for your experiment10 / experiment10_2 style models, but is
configurable enough to work with other models that use the same architecture API.

It produces:
  - per_class_metrics.csv
  - per_class_diagnosis.csv
  - global_label_distribution.csv
  - split_label_distribution.csv
  - switch_matrix_false_negative_vs_false_positive.csv
  - top_confusions.csv
  - threshold_sensitivity.csv
  - sample_errors.csv
  - probability_summary.csv
  - analysis_summary.json
  - plots/*.png

Example for experiment10_2:

python scripts/analyse_ttn_errors.py \
  --data-dir data/raw \
  --checkpoint src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment10_2/results_600/ttn_ir_best.pt \
  --run-config src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment10_2/results_600/run_config.json \
  --thresholds-json src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment10_2/results_600/selected_thresholds.json \
  --split-path src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment10_2/results_600/data_split_seed42_all.npz \
  --model-module spectroscopy_qml.ir.tree_tensor_network.experiment.experiment10_2.model \
  --model-class TTNIRClassifier10_2 \
  --output-dir error_analysis_10_2_test \
  --split test
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset


# -----------------------------------------------------------------------------
# Project imports
# -----------------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = Path(__file__).resolve().parents[5]
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment5.data_loader import (  # noqa: E402
    FUNCTIONAL_GROUPS,
    load_ir_data,
    load_or_create_split_indices,
)
from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment5.train import (  # noqa: E402
    build_threshold_grid,
    get_pos_weight,
    threshold_predictions,
    tune_thresholds,
)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def parse_int_list(value: str | None) -> list[int] | None:
    if value is None or str(value).strip() == "":
        return None
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


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


def strip_compile_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remove torch.compile/DataParallel prefixes when present."""
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("_orig_mod.", "module."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        cleaned[new_key] = value
    return cleaned


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    """Support raw state_dicts and experiment-style checkpoint dictionaries."""
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        if all(torch.is_tensor(v) for v in checkpoint.values()):
            return checkpoint
    raise ValueError(
        "Unsupported checkpoint format. Expected raw state_dict or dict containing "
        "'model_state_dict' / 'state_dict'."
    )


def safe_float(value: Any) -> float | None:
    try:
        value = float(value)
    except Exception:
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_matrix_csv(path: Path, matrix: np.ndarray, row_names: list[str], col_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["false_negative_true_label__vs__false_positive_predicted_label"] + col_names)
        for name, row in zip(row_names, matrix, strict=False):
            writer.writerow([name] + [int(x) for x in row])


# -----------------------------------------------------------------------------
# Model/data loading
# -----------------------------------------------------------------------------


def merge_args(cli_args: argparse.Namespace, run_config: dict[str, Any]) -> dict[str, Any]:
    """Combine explicit CLI args and run_config values.

    Explicit CLI values win. For model construction, we only pass keys that the
    model __init__ actually accepts.
    """
    merged = dict(run_config)

    # Generic core settings.
    for key in [
        "input_dim",
        "num_labels",
        "chi",
        "segment_window_size",
        "segment_stride",
        "segment_mode",
        "segment_offset",
        "segment_state_normalize",
        "merge_mode",
        "merge_residual_weight",
        "merge_renormalize_output",
        "readout_hidden_dim",
        "readout_dropout",
        "lorentz_gamma",
        "lorentz_kernel_half_width",
        "lorentz_norm_mode",
    ]:
        value = getattr(cli_args, key, None)
        if value is not None:
            merged[key] = value

    return merged


def build_model_from_config(args: argparse.Namespace, config: dict[str, Any], num_labels: int):
    module = importlib.import_module(args.model_module)
    model_cls = getattr(module, args.model_class)

    import inspect

    signature = inspect.signature(model_cls.__init__)
    valid_params = set(signature.parameters.keys()) - {"self"}

    candidate_kwargs = dict(config)
    candidate_kwargs["num_labels"] = num_labels

    # Paths, training-only settings, and unsupported run_config keys are filtered out.
    model_kwargs = {key: value for key, value in candidate_kwargs.items() if key in valid_params}
    return model_cls(**model_kwargs)


def load_thresholds(
    threshold_json: Path | None,
    label_names: list[str],
    fallback: float = 0.5,
) -> np.ndarray:
    if threshold_json is None:
        return np.full(len(label_names), fallback, dtype=np.float32)
    payload = read_json(threshold_json)
    thresholds_payload = payload.get("thresholds")
    if isinstance(thresholds_payload, dict):
        return np.asarray([thresholds_payload.get(name, fallback) for name in label_names], dtype=np.float32)
    if isinstance(thresholds_payload, list):
        values = np.asarray(thresholds_payload, dtype=np.float32)
        if values.shape[0] == len(label_names):
            return values
    raise ValueError(f"Could not parse thresholds from {threshold_json}")


def load_label_names(args: argparse.Namespace, run_config: dict[str, Any]) -> tuple[list[str], list[int] | None]:
    all_names = list(FUNCTIONAL_GROUPS.keys())
    specialist_indices = parse_int_list(args.specialist_indices)
    if specialist_indices is None:
        specialist_indices = parse_int_list(run_config.get("specialist_indices"))

    if specialist_indices is not None:
        return [all_names[i] for i in specialist_indices], specialist_indices
    return all_names[: args.num_labels if args.num_labels is not None else len(all_names)], None


@dataclass
class PredictionBundle:
    labels: np.ndarray
    probs: np.ndarray
    preds: np.ndarray
    sample_indices: np.ndarray


@torch.no_grad()
def predict_probs(
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    sample_indices: np.ndarray,
    thresholds: np.ndarray,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> PredictionBundle:
    dataset = TensorDataset(
        torch.as_tensor(X[sample_indices], dtype=torch.float32),
        torch.as_tensor(y[sample_indices], dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model.eval()
    probs_parts: list[np.ndarray] = []
    labels_parts: list[np.ndarray] = []

    for spectra, labels in loader:
        spectra = spectra.to(device, non_blocking=True)
        logits = model(spectra)
        probs_parts.append(torch.sigmoid(logits.float()).cpu().numpy())
        labels_parts.append(labels.cpu().numpy())

    labels_np = np.concatenate(labels_parts, axis=0)
    probs_np = np.concatenate(probs_parts, axis=0)
    probs_np = np.nan_to_num(probs_np, nan=0.5, posinf=1.0, neginf=0.0)
    probs_np = np.clip(probs_np, 0.0, 1.0).astype(np.float32)
    labels_np = (np.nan_to_num(labels_np, nan=0.0) >= 0.5).astype(np.int32)
    preds_np = threshold_predictions(probs_np, thresholds).astype(np.int32)
    return PredictionBundle(labels=labels_np, probs=probs_np, preds=preds_np, sample_indices=sample_indices)


# -----------------------------------------------------------------------------
# Analysis
# -----------------------------------------------------------------------------


def class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray,
    label_names: list[str],
    train_labels: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n_samples, n_classes = y_true.shape

    train_prevalence = None
    if train_labels is not None:
        train_prevalence = train_labels.mean(axis=0)

    for c in range(n_classes):
        yt = y_true[:, c].astype(int)
        yp = y_pred[:, c].astype(int)
        pr = y_prob[:, c]

        tp = int(((yt == 1) & (yp == 1)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        tn = int(((yt == 0) & (yp == 0)).sum())
        support = int(yt.sum())
        pred_pos = int(yp.sum())

        precision = precision_score(yt, yp, zero_division=0)
        recall = recall_score(yt, yp, zero_division=0)
        f1 = f1_score(yt, yp, zero_division=0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        roc_auc = None
        ap = None
        if len(np.unique(yt)) == 2:
            roc_auc = safe_float(roc_auc_score(yt, pr))
            ap = safe_float(average_precision_score(yt, pr))

        pos_probs = pr[yt == 1]
        neg_probs = pr[yt == 0]
        rows.append(
            {
                "class_index": c,
                "label": label_names[c],
                "support_true": support,
                "support_true_pct": support / n_samples,
                "predicted_positive": pred_pos,
                "predicted_positive_pct": pred_pos / n_samples,
                "prediction_minus_truth_pct": (pred_pos - support) / n_samples,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "specificity": specificity,
                "false_positive_rate": fpr,
                "false_negative_rate": fnr,
                "threshold": float(thresholds[c]),
                "mean_prob_all": float(pr.mean()),
                "mean_prob_true_positive_samples": float(pos_probs.mean()) if pos_probs.size else None,
                "mean_prob_true_negative_samples": float(neg_probs.mean()) if neg_probs.size else None,
                "prob_gap_pos_minus_neg": float(pos_probs.mean() - neg_probs.mean()) if pos_probs.size and neg_probs.size else None,
                "roc_auc": roc_auc,
                "average_precision": ap,
                "train_prevalence": float(train_prevalence[c]) if train_prevalence is not None else None,
                "eval_prevalence": support / n_samples,
                "eval_minus_train_prevalence": float((support / n_samples) - train_prevalence[c])
                if train_prevalence is not None
                else None,
            }
        )
    return rows


def global_distribution_rows(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
) -> list[dict[str, Any]]:
    n = y_true.shape[0]
    true_counts = y_true.sum(axis=0)
    pred_counts = y_pred.sum(axis=0)
    rows = []
    for i, name in enumerate(label_names):
        rows.append(
            {
                "class_index": i,
                "label": name,
                "true_count": int(true_counts[i]),
                "true_pct": float(true_counts[i] / n),
                "pred_count": int(pred_counts[i]),
                "pred_pct": float(pred_counts[i] / n),
                "pred_minus_true_count": int(pred_counts[i] - true_counts[i]),
                "pred_minus_true_pct": float((pred_counts[i] - true_counts[i]) / n),
                "over_prediction_ratio": float(pred_counts[i] / true_counts[i]) if true_counts[i] > 0 else None,
            }
        )
    return rows


def split_distribution_rows(
    y: np.ndarray,
    split_indices: dict[str, np.ndarray],
    label_names: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name, idx in split_indices.items():
        labels = y[idx]
        n = labels.shape[0]
        counts = labels.sum(axis=0)
        for c, name in enumerate(label_names):
            rows.append(
                {
                    "split": split_name,
                    "class_index": c,
                    "label": name,
                    "count": int(counts[c]),
                    "pct": float(counts[c] / n) if n else 0.0,
                    "num_samples": int(n),
                }
            )
    return rows


def switch_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Count likely label switches in multilabel predictions.

    Entry [i, j] counts samples where true label i was missed (FN) while label j
    was predicted although absent (FP). This does not prove causal confusion, but
    it is a useful multilabel confusion proxy.
    """
    fn = (y_true == 1) & (y_pred == 0)
    fp = (y_true == 0) & (y_pred == 1)
    return fn.astype(np.int64).T @ fp.astype(np.int64)


def top_confusion_rows(matrix: np.ndarray, label_names: list[str], limit: int) -> list[dict[str, Any]]:
    rows = []
    for true_i in range(matrix.shape[0]):
        for pred_j in range(matrix.shape[1]):
            if true_i == pred_j:
                continue
            count = int(matrix[true_i, pred_j])
            if count > 0:
                rows.append(
                    {
                        "missed_true_label_index": true_i,
                        "missed_true_label": label_names[true_i],
                        "wrong_predicted_label_index": pred_j,
                        "wrong_predicted_label": label_names[pred_j],
                        "count": count,
                    }
                )
    rows.sort(key=lambda row: row["count"], reverse=True)
    return rows[:limit]


def probability_summary_rows(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray,
    label_names: list[str],
) -> list[dict[str, Any]]:
    rows = []
    for c, name in enumerate(label_names):
        probs = y_prob[:, c]
        true_probs = probs[y_true[:, c] == 1]
        false_probs = probs[y_true[:, c] == 0]
        rows.append(
            {
                "class_index": c,
                "label": name,
                "threshold": float(thresholds[c]),
                "prob_p01": float(np.quantile(probs, 0.01)),
                "prob_p05": float(np.quantile(probs, 0.05)),
                "prob_p25": float(np.quantile(probs, 0.25)),
                "prob_p50": float(np.quantile(probs, 0.50)),
                "prob_p75": float(np.quantile(probs, 0.75)),
                "prob_p95": float(np.quantile(probs, 0.95)),
                "prob_p99": float(np.quantile(probs, 0.99)),
                "true_sample_prob_p50": float(np.quantile(true_probs, 0.50)) if true_probs.size else None,
                "negative_sample_prob_p50": float(np.quantile(false_probs, 0.50)) if false_probs.size else None,
            }
        )
    return rows


def threshold_sensitivity_rows(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_names: list[str],
    grid_step: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grid = build_threshold_grid(grid_step)
    for c, name in enumerate(label_names):
        yt = y_true[:, c]
        if yt.sum() == 0:
            continue
        best = {"f1": -1.0, "threshold": 0.5, "precision": 0.0, "recall": 0.0}
        for threshold in grid:
            yp = (y_prob[:, c] >= threshold).astype(int)
            precision = precision_score(yt, yp, zero_division=0)
            recall = recall_score(yt, yp, zero_division=0)
            f1 = f1_score(yt, yp, zero_division=0)
            rows.append(
                {
                    "class_index": c,
                    "label": name,
                    "threshold": float(threshold),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                }
            )
            if f1 > best["f1"]:
                best = {
                    "f1": float(f1),
                    "threshold": float(threshold),
                    "precision": float(precision),
                    "recall": float(recall),
                }
    return rows


def diagnosis_rows(per_class: list[dict[str, Any]], switch: np.ndarray, label_names: list[str]) -> list[dict[str, Any]]:
    rows = []
    f1_values = np.asarray([row["f1"] for row in per_class], dtype=float)
    support_values = np.asarray([row["support_true"] for row in per_class], dtype=float)
    median_f1 = float(np.median(f1_values))
    median_support = float(np.median(support_values))

    for row in per_class:
        c = int(row["class_index"])
        reasons: list[str] = []
        if row["support_true"] < median_support:
            reasons.append("low support / rare class")
        if row["recall"] < 0.5 and row["false_negative_rate"] > row["false_positive_rate"]:
            reasons.append("mostly under-detected: many false negatives")
        if row["precision"] < 0.5 and row["false_positive_rate"] > 0:
            reasons.append("over-predicted: many false positives")
        if row["prob_gap_pos_minus_neg"] is not None and row["prob_gap_pos_minus_neg"] < 0.15:
            reasons.append("weak probability separation between positives and negatives")
        if row["threshold"] > 0.65 and row["recall"] < 0.6:
            reasons.append("high selected threshold may suppress recall")
        if row["threshold"] < 0.25 and row["precision"] < 0.6:
            reasons.append("low selected threshold may inflate false positives")
        if row["eval_minus_train_prevalence"] is not None and abs(row["eval_minus_train_prevalence"]) > 0.02:
            reasons.append("eval prevalence differs from train prevalence")

        top_switched_to = []
        if switch[c].sum() > 0:
            top_indices = np.argsort(-switch[c])[:3]
            top_switched_to = [
                f"{label_names[j]} ({int(switch[c, j])})" for j in top_indices if switch[c, j] > 0 and j != c
            ]
            if top_switched_to:
                reasons.append("often missed while another label is predicted")

        if row["f1"] >= max(0.75, median_f1):
            reasons.append("strong class: good balance of precision and recall")
        if not reasons:
            reasons.append("no single dominant error pattern; inspect sample_errors and probability histograms")

        rows.append(
            {
                "class_index": c,
                "label": row["label"],
                "f1": row["f1"],
                "precision": row["precision"],
                "recall": row["recall"],
                "support_true": row["support_true"],
                "predicted_positive": row["predicted_positive"],
                "main_diagnosis": "; ".join(reasons),
                "top_wrong_predicted_when_this_was_missed": " | ".join(top_switched_to),
            }
        )
    rows.sort(key=lambda row: row["f1"])
    return rows


def sample_error_rows(
    bundle: PredictionBundle,
    label_names: list[str],
    max_samples: int,
) -> list[dict[str, Any]]:
    y_true, y_pred, y_prob = bundle.labels, bundle.preds, bundle.probs
    rows: list[dict[str, Any]] = []

    for local_i in range(y_true.shape[0]):
        fn_idx = np.flatnonzero((y_true[local_i] == 1) & (y_pred[local_i] == 0))
        fp_idx = np.flatnonzero((y_true[local_i] == 0) & (y_pred[local_i] == 1))
        if fn_idx.size == 0 and fp_idx.size == 0:
            continue
        severity = int(fn_idx.size + fp_idx.size)
        true_idx = np.flatnonzero(y_true[local_i] == 1)
        pred_idx = np.flatnonzero(y_pred[local_i] == 1)
        rows.append(
            {
                "global_sample_index": int(bundle.sample_indices[local_i]),
                "num_false_negatives": int(fn_idx.size),
                "num_false_positives": int(fp_idx.size),
                "error_count": severity,
                "true_labels": " | ".join(label_names[j] for j in true_idx),
                "predicted_labels": " | ".join(label_names[j] for j in pred_idx),
                "false_negative_labels": " | ".join(label_names[j] for j in fn_idx),
                "false_positive_labels": " | ".join(label_names[j] for j in fp_idx),
                "false_negative_probs": " | ".join(f"{label_names[j]}={y_prob[local_i, j]:.4f}" for j in fn_idx),
                "false_positive_probs": " | ".join(f"{label_names[j]}={y_prob[local_i, j]:.4f}" for j in fp_idx),
            }
        )
    rows.sort(key=lambda row: row["error_count"], reverse=True)
    return rows[:max_samples]


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------


def plot_bar(path: Path, names: list[str], values: list[float], title: str, ylabel: str, top_n: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if top_n is not None and len(values) > top_n:
        order = np.argsort(values)[-top_n:]
        names = [names[i] for i in order]
        values = [values[i] for i in order]
    plt.figure(figsize=(max(10, len(names) * 0.35), 6))
    plt.bar(range(len(names)), values)
    plt.xticks(range(len(names)), names, rotation=90)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_heatmap(path: Path, matrix: np.ndarray, label_names: list[str], title: str, top_n: int = 25) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    totals = matrix.sum(axis=1) + matrix.sum(axis=0)
    if matrix.shape[0] > top_n:
        selected = np.argsort(-totals)[:top_n]
        matrix = matrix[np.ix_(selected, selected)]
        label_names = [label_names[i] for i in selected]

    plt.figure(figsize=(12, 10))
    plt.imshow(matrix, aspect="auto")
    plt.colorbar(label="count")
    plt.xticks(range(len(label_names)), label_names, rotation=90)
    plt.yticks(range(len(label_names)), label_names)
    plt.xlabel("wrong predicted label / false positive")
    plt.ylabel("missed true label / false negative")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_probability_histograms(
    out_dir: Path,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_names: list[str],
    worst_class_indices: list[int],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for c in worst_class_indices:
        true_probs = y_prob[y_true[:, c] == 1, c]
        negative_probs = y_prob[y_true[:, c] == 0, c]
        plt.figure(figsize=(8, 5))
        if negative_probs.size:
            plt.hist(negative_probs, bins=40, alpha=0.65, label="true negative samples")
        if true_probs.size:
            plt.hist(true_probs, bins=40, alpha=0.65, label="true positive samples")
        plt.title(f"Probability separation: {label_names[c]}")
        plt.xlabel("predicted probability")
        plt.ylabel("sample count")
        plt.legend()
        plt.tight_layout()
        safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in label_names[c])
        plt.savefig(out_dir / f"prob_hist_{c:02d}_{safe_name}.png", dpi=180)
        plt.close()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detailed error analysis for TTN IR classifiers.")

    parser.add_argument("--data-dir", type=Path, default="data/raw")
    parser.add_argument("--checkpoint", type=Path, default="src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment10_2/results/ttn_ir_best.pt")
    parser.add_argument("--output-dir", type=Path, default="src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment10_2/evaluation/plots")
    parser.add_argument("--run-config", type=Path, default="src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment10_2/results/run_config.json")
    parser.add_argument("--split-path", type=Path, default="src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment10_2/results/data_split_seed42_all.npz")
    parser.add_argument("--thresholds-json", type=Path, default="src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment10_2/results/selected_thresholds.json")

    parser.add_argument("--model-module", type=str, default="spectroscopy_qml.ir.tree_tensor_network.experiment.experiment10_2.model")
    parser.add_argument("--model-class", type=str, default="TTNIRClassifier10_2")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")

    parser.add_argument("--input-dim", type=int, default=None)
    parser.add_argument("--num-labels", type=int, default=None)
    parser.add_argument("--chi", type=int, default=None)
    parser.add_argument("--segment-window-size", type=int, default=None)
    parser.add_argument("--segment-stride", type=int, default=None)
    parser.add_argument("--segment-mode", choices=["overlap", "dual_offset"], default=None)
    parser.add_argument("--segment-offset", type=int, default=None)
    parser.add_argument("--segment-state-normalize", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--merge-mode", choices=["strict", "relaxed"], default=None)
    parser.add_argument("--merge-residual-weight", type=float, default=None)
    parser.add_argument("--merge-renormalize-output", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--readout-hidden-dim", type=int, default=None)
    parser.add_argument("--readout-dropout", type=float, default=None)
    parser.add_argument("--lorentz-gamma", type=float, default=None)
    parser.add_argument("--lorentz-kernel-half-width", type=int, default=None)
    parser.add_argument("--lorentz-norm-mode", choices=["max_abs", "z_score", "percentile"], default=None)

    parser.add_argument("--specialist-indices", type=str, default=None)
    parser.add_argument("--apply-snv", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--test-ratio", type=float, default=None)
    parser.add_argument("--overwrite-split", action="store_true")

    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--threshold-grid-step", type=float, default=0.02)
    parser.add_argument("--retune-thresholds-on", choices=["none", "train", "val", "test", "selected"], default="none")
    parser.add_argument("--threshold-mode", choices=["global", "per_class"], default="per_class")
    parser.add_argument("--threshold-target-metric", choices=["f1_micro", "f1_macro", "per_class_f1"], default="per_class_f1")
    parser.add_argument("--top-confusions", type=int, default=100)
    parser.add_argument("--max-sample-errors", type=int, default=500)
    parser.add_argument("--num-worst-probability-plots", type=int, default=12)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.output_dir / "plots"

    run_config = read_json(args.run_config)
    config = merge_args(args, run_config)

    input_dim = int(config.get("input_dim", 1800))
    apply_snv = bool(config.get("apply_snv", True))
    max_files = args.max_files if args.max_files is not None else config.get("max_files")
    if max_files is not None:
        max_files = int(max_files)
    seed = int(config.get("seed", 42))
    train_ratio = float(config.get("train_ratio", 0.8))
    val_ratio = float(config.get("val_ratio", 0.1))
    test_ratio = float(config.get("test_ratio", 0.1))

    label_names, specialist_indices = load_label_names(args, run_config)
    num_labels = len(label_names)
    config["num_labels"] = num_labels

    print("Loading data...")
    X, y = load_ir_data(
        data_dir=args.data_dir,
        target_length=input_dim,
        max_files=max_files,
        apply_snv=apply_snv,
        cache_path=args.cache_path,
        overwrite_cache=args.overwrite_cache,
    )
    if specialist_indices is not None:
        y = y[:, specialist_indices]

    if y.shape[1] != num_labels:
        raise RuntimeError(f"Label width mismatch: y has {y.shape[1]}, expected {num_labels}.")

    split_path = args.split_path
    if split_path is None:
        used_suffix = "all" if max_files is None else f"files{max_files}"
        split_path = args.output_dir / f"data_split_seed{seed}_{used_suffix}.npz"

    split_indices = load_or_create_split_indices(
        labels=y,
        split_path=split_path,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=seed,
        stratify_multilabel=True,
        overwrite=args.overwrite_split,
    )

    if args.split == "all":
        selected_indices = np.arange(len(X), dtype=np.int64)
    else:
        selected_indices = split_indices[args.split]

    print("Building and loading model...")
    device = resolve_device(args.device)
    model = build_model_from_config(args, config, num_labels=num_labels)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = strip_compile_prefix(extract_state_dict(checkpoint))
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)

    thresholds = load_thresholds(args.thresholds_json, label_names, fallback=0.5)

    if args.retune_thresholds_on != "none":
        tune_split = args.split if args.retune_thresholds_on == "selected" else args.retune_thresholds_on
        tune_indices = np.arange(len(X), dtype=np.int64) if tune_split == "all" else split_indices[tune_split]
        print(f"Retuning thresholds on {tune_split} split...")
        tune_bundle = predict_probs(
            model=model,
            X=X,
            y=y,
            sample_indices=tune_indices,
            thresholds=np.full(num_labels, 0.5, dtype=np.float32),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        thresholds = tune_thresholds(
            tune_bundle.labels,
            tune_bundle.probs,
            threshold_mode=args.threshold_mode,
            target_metric=args.threshold_target_metric,
            threshold_grid=build_threshold_grid(args.threshold_grid_step),
        )

    print(f"Running inference on split={args.split} ({len(selected_indices)} samples)...")
    bundle = predict_probs(
        model=model,
        X=X,
        y=y,
        sample_indices=selected_indices,
        thresholds=thresholds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    train_labels = y[split_indices["train"]]
    pos_weight = get_pos_weight(train_labels, torch.device("cpu"), power=float(config.get("pos_weight_power", 0.5)))

    print("Computing analysis tables...")
    per_class = class_metrics(
        bundle.labels,
        bundle.preds,
        bundle.probs,
        thresholds,
        label_names,
        train_labels=train_labels,
    )
    for row in per_class:
        row["pos_weight"] = float(pos_weight[int(row["class_index"])] )

    distribution = global_distribution_rows(bundle.labels, bundle.preds, label_names)
    split_distribution = split_distribution_rows(y, split_indices, label_names)
    switch = switch_matrix(bundle.labels, bundle.preds)
    top_confusions = top_confusion_rows(switch, label_names, args.top_confusions)
    diagnosis = diagnosis_rows(per_class, switch, label_names)
    prob_summary = probability_summary_rows(bundle.labels, bundle.probs, thresholds, label_names)
    threshold_sensitivity = threshold_sensitivity_rows(
        bundle.labels,
        bundle.probs,
        label_names,
        grid_step=args.threshold_grid_step,
    )
    sample_errors = sample_error_rows(bundle, label_names, args.max_sample_errors)

    save_csv(args.output_dir / "per_class_metrics.csv", per_class)
    save_csv(args.output_dir / "per_class_diagnosis.csv", diagnosis)
    save_csv(args.output_dir / "global_label_distribution.csv", distribution)
    save_csv(args.output_dir / "split_label_distribution.csv", split_distribution)
    save_matrix_csv(args.output_dir / "switch_matrix_false_negative_vs_false_positive.csv", switch, label_names, label_names)
    save_csv(args.output_dir / "top_confusions.csv", top_confusions)
    save_csv(args.output_dir / "probability_summary.csv", prob_summary)
    save_csv(args.output_dir / "threshold_sensitivity.csv", threshold_sensitivity)
    save_csv(args.output_dir / "sample_errors.csv", sample_errors)

    f1_micro = f1_score(bundle.labels, bundle.preds, average="micro", zero_division=0)
    f1_macro = f1_score(bundle.labels, bundle.preds, average="macro", zero_division=0)
    precision_micro = precision_score(bundle.labels, bundle.preds, average="micro", zero_division=0)
    recall_micro = recall_score(bundle.labels, bundle.preds, average="micro", zero_division=0)

    worst_by_f1 = sorted(per_class, key=lambda row: row["f1"])[: args.num_worst_probability_plots]
    worst_indices = [int(row["class_index"]) for row in worst_by_f1]

    print("Creating plots...")
    plot_bar(
        plots_dir / "per_class_f1.png",
        label_names,
        [float(row["f1"]) for row in per_class],
        "Per-class F1",
        "F1",
    )
    plot_bar(
        plots_dir / "support_true.png",
        label_names,
        [float(row["support_true"]) for row in per_class],
        "True support per functional group",
        "count",
    )
    plot_bar(
        plots_dir / "pred_minus_true_pct.png",
        label_names,
        [float(row["pred_minus_true_pct"]) for row in distribution],
        "Prediction distribution bias: predicted % - true %",
        "percentage points as fraction",
    )
    plot_heatmap(
        plots_dir / "switch_matrix_heatmap_top25.png",
        switch,
        label_names,
        "Multilabel switch proxy: FN true label vs FP predicted label",
        top_n=25,
    )
    plot_probability_histograms(plots_dir / "probability_histograms_worst_classes", bundle.labels, bundle.probs, label_names, worst_indices)

    summary = {
        "split": args.split,
        "num_samples": int(bundle.labels.shape[0]),
        "num_labels": int(num_labels),
        "f1_micro": float(f1_micro),
        "f1_macro": float(f1_macro),
        "precision_micro": float(precision_micro),
        "recall_micro": float(recall_micro),
        "mean_threshold": float(np.mean(thresholds)),
        "std_threshold": float(np.std(thresholds)),
        "worst_classes_by_f1": [
            {
                "label": row["label"],
                "f1": float(row["f1"]),
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "support_true": int(row["support_true"]),
            }
            for row in worst_by_f1
        ],
        "most_over_predicted": sorted(distribution, key=lambda row: row["pred_minus_true_pct"], reverse=True)[:10],
        "most_under_predicted": sorted(distribution, key=lambda row: row["pred_minus_true_pct"])[:10],
        "top_confusions": top_confusions[:20],
        "artifacts": {
            "per_class_metrics": str(args.output_dir / "per_class_metrics.csv"),
            "per_class_diagnosis": str(args.output_dir / "per_class_diagnosis.csv"),
            "global_label_distribution": str(args.output_dir / "global_label_distribution.csv"),
            "switch_matrix": str(args.output_dir / "switch_matrix_false_negative_vs_false_positive.csv"),
            "top_confusions": str(args.output_dir / "top_confusions.csv"),
            "sample_errors": str(args.output_dir / "sample_errors.csv"),
            "plots": str(plots_dir),
        },
    }
    (args.output_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("\nDone.")
    print(f"Micro F1:  {f1_micro:.4f}")
    print(f"Macro F1:  {f1_macro:.4f}")
    print(f"Precision: {precision_micro:.4f}")
    print(f"Recall:    {recall_micro:.4f}")
    print(f"Outputs:   {args.output_dir}")


if __name__ == "__main__":
    main()
