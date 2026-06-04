"""Detailed error analysis for FunctionalGroupCNN models.

This script mirrors the TTN error-analysis script, but is adapted for the CNN
architecture and output artifacts from the Guwon Jung-style PyTorch training
script.

It supports two workflows:

1) Fast artifact-based analysis, recommended when you already have results.pickle:
   - original mode results.pickle with keys: pred, tgt, thresholds
   - k_fold mode results.pickle with keys: test_predictions, test_targets,
     test_probabilities, best_thresholds

2) Full reload + inference, useful when you want to analyze another split or you
   only have the saved .pt model. In this mode the script rebuilds the same data
   split from parquet files and runs the CNN.

Outputs are intentionally the same style as the TTN analyzer:
  - per_class_metrics.csv
  - per_class_diagnosis.csv
  - global_label_distribution.csv
  - split_label_distribution.csv, when raw data is available
  - switch_matrix_false_negative_vs_false_positive.csv
  - top_confusions.csv
  - threshold_sensitivity.csv, when probabilities are available
  - sample_errors.csv
  - probability_summary.csv, when probabilities are available
  - analysis_summary.json
  - plots/*.png

Examples
--------

Analyze original-mode results.pickle directly:

python scripts/analyse_cnn_errors.py \
  --results-pickle outputs/ir/original/results.pickle \
  --output-dir outputs/ir/original/error_analysis

Analyze k-fold results.pickle directly:

python scripts/analyse_cnn_errors.py \
  --results-pickle outputs/ir/k_fold/results.pickle \
  --output-dir outputs/ir/k_fold/error_analysis

Analyze from a saved model and parquet data:

python scripts/analyse_cnn_errors.py \
  --analytical-data data/raw \
  --model-checkpoint outputs/ir/original/ir_model.pt \
  --column ir_spectra \
  --mode original \
  --output-dir outputs/ir/original/error_analysis_from_model
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from scipy.interpolate import interp1d
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# -----------------------------------------------------------------------------
# Functional-group definitions, kept identical to the training script.
# -----------------------------------------------------------------------------

functional_groups = {
    "Acid anhydride": Chem.MolFromSmarts("[CX3](=[OX1])[OX2][CX3](=[OX1])"),
    "Acyl halide": Chem.MolFromSmarts("[CX3](=[OX1])[F,Cl,Br,I]"),
    "Alcohol": Chem.MolFromSmarts("[#6][OX2H]"),
    "Aldehyde": Chem.MolFromSmarts("[CX3H1](=O)[#6,H]"),
    "Alkane": Chem.MolFromSmarts("[CX4;H3,H2]"),
    "Alkene": Chem.MolFromSmarts("[CX3]=[CX3]"),
    "Alkyne": Chem.MolFromSmarts("[CX2]#[CX2]"),
    "Amide": Chem.MolFromSmarts("[NX3][CX3](=[OX1])[#6]"),
    "Amine": Chem.MolFromSmarts("[NX3;H2,H1,H0;!$(NC=O)]"),
    "Arene": Chem.MolFromSmarts("[cX3]1[cX3][cX3][cX3][cX3][cX3]1"),
    "Azo compound": Chem.MolFromSmarts("[#6][NX2]=[NX2][#6]"),
    "Carbamate": Chem.MolFromSmarts("[NX3][CX3](=[OX1])[OX2H0]"),
    "Carboxylic acid": Chem.MolFromSmarts("[CX3](=O)[OX2H]"),
    "Enamine": Chem.MolFromSmarts("[NX3][CX3]=[CX3]"),
    "Enol": Chem.MolFromSmarts("[OX2H][#6X3]=[#6]"),
    "Ester": Chem.MolFromSmarts("[#6][CX3](=O)[OX2H0][#6]"),
    "Ether": Chem.MolFromSmarts("[OD2]([#6])[#6]"),
    "Haloalkane": Chem.MolFromSmarts("[#6][F,Cl,Br,I]"),
    "Hydrazine": Chem.MolFromSmarts("[NX3][NX3]"),
    "Hydrazone": Chem.MolFromSmarts("[NX3][NX2]=[#6]"),
    "Imide": Chem.MolFromSmarts("[CX3](=[OX1])[NX3][CX3](=[OX1])"),
    "Imine": Chem.MolFromSmarts(
        "[$([CX3]([#6])[#6]),$([CX3H][#6])]=[$([NX2][#6]),$([NX2H])]"
    ),
    "Isocyanate": Chem.MolFromSmarts("[NX2]=[C]=[O]"),
    "Isothiocyanate": Chem.MolFromSmarts("[NX2]=[C]=[S]"),
    "Ketone": Chem.MolFromSmarts("[#6][CX3](=O)[#6]"),
    "Nitrile": Chem.MolFromSmarts("[NX1]#[CX2]"),
    "Phenol": Chem.MolFromSmarts("[OX2H][cX3]:[c]"),
    "Phosphine": Chem.MolFromSmarts("[PX3]"),
    "Sulfide": Chem.MolFromSmarts("[#16X2H0]"),
    "Sulfonamide": Chem.MolFromSmarts("[#16X4]([NX3])(=[OX1])(=[OX1])[#6]"),
    "Sulfonate": Chem.MolFromSmarts("[#16X4](=[OX1])(=[OX1])([#6])[OX2H0]"),
    "Sulfone": Chem.MolFromSmarts("[#16X4](=[OX1])(=[OX1])([#6])[#6]"),
    "Sulfonic acid": Chem.MolFromSmarts("[#16X4](=[OX1])(=[OX1])([#6])[OX2H]"),
    "Sulfoxide": Chem.MolFromSmarts("[#16X3]=[OX1]"),
    "Thial": Chem.MolFromSmarts("[CX3H1](=S)[#6,H]"),
    "Thioamide": Chem.MolFromSmarts("[NX3][CX3]=[SX1]"),
    "Thiol": Chem.MolFromSmarts("[#16X2H]"),
}

LABEL_NAMES = list(functional_groups.keys())

COLUMN_MAPPING = {
    "h_nmr_spectra": ("h_nmr_spectra", "hnmr"),
    "c_nmr_spectra": ("c_nmr_spectra", "cnmr"),
    "ir_spectra": ("ir_spectra", "ir"),
    "pos_msms": ("msms_positive_40ev", "pos_msms"),
    "neg_msms": ("msms_negative_40ev", "neg_msms"),
}


# -----------------------------------------------------------------------------
# CNN architecture, kept identical to the training script.
# -----------------------------------------------------------------------------

class FunctionalGroupCNN(nn.Module):
    def __init__(self, input_length: int, num_fgs: int):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=31, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm1d(31),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=31, out_channels=62, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm1d(62),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_length)
            flat_size = self.features(dummy).view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 4927),
            nn.ReLU(),
            nn.Dropout(0.48599073736368),
            nn.Linear(4927, 2785),
            nn.ReLU(),
            nn.Dropout(0.48599073736368),
            nn.Linear(2785, 1574),
            nn.ReLU(),
            nn.Dropout(0.48599073736368),
            nn.Linear(1574, num_fgs),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


# -----------------------------------------------------------------------------
# Data preprocessing, kept compatible with the training script.
# -----------------------------------------------------------------------------


def match_group(mol: Chem.Mol, func_group) -> int:
    if type(func_group) is Chem.Mol:
        n = len(mol.GetSubstructMatches(func_group))
    else:
        n = func_group(mol)
    return 0 if n == 0 else 1


def get_functional_groups(smiles: str) -> list[int] | None:
    RDLogger.DisableLog("rdApp.*")
    smiles = smiles.strip().replace(" ", "")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return [match_group(mol, smarts) for smarts in functional_groups.values()]


def interpolate_to_length(spec, target_length: int) -> np.ndarray:
    old_x = np.arange(len(spec))
    new_x = np.linspace(old_x.min(), old_x.max(), target_length)
    interp = interp1d(old_x, spec, bounds_error=False, fill_value=0)
    return interp(new_x)


def make_msms_spectrum(spectrum) -> np.ndarray:
    msms_spectrum = np.zeros(10000)
    for peak in spectrum:
        peak_pos = int(peak[0] * 10)
        if peak_pos >= 10000:
            peak_pos = 9999
        msms_spectrum[peak_pos] = peak[1]
    return msms_spectrum


def target_length_for_column(column: str) -> int:
    return 1800 if column == "ir_spectra" else 10000


def actual_column_name(column: str) -> str:
    if column not in COLUMN_MAPPING:
        valid = ", ".join(COLUMN_MAPPING)
        raise ValueError(f"Unknown column '{column}'. Valid options: {valid}")
    return COLUMN_MAPPING[column][0]


def load_column_data(
    analytical_data: Path,
    column: str,
    max_files: int | None = None,
    cache_path: Path | None = None,
    overwrite_cache: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load spectra and labels exactly like the training script.

    Returns:
        X: shape (n_samples, target_length)
        y: shape (n_samples, 37)
        original_row_ids: integer ids in loaded order, useful for sample_errors.csv
    """
    if cache_path is not None and cache_path.exists() and not overwrite_cache:
        payload = np.load(cache_path, allow_pickle=False)
        return payload["X"], payload["y"], payload["row_ids"]

    actual_col = actual_column_name(column)
    target_length = target_length_for_column(column)
    columns_to_load = ["smiles", actual_col]
    parquet_files = sorted(analytical_data.glob("*.parquet"))
    if max_files is not None:
        parquet_files = parquet_files[:max_files]
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {analytical_data}")

    frames = []
    loaded_rows = 0
    for i, parquet_file in enumerate(parquet_files, start=1):
        print(f"Loading {i}/{len(parquet_files)}: {parquet_file.name}")
        data = pd.read_parquet(parquet_file, columns=columns_to_load)
        data["_loaded_row_id"] = np.arange(loaded_rows, loaded_rows + len(data), dtype=np.int64)
        loaded_rows += len(data)

        if actual_col in ["msms_positive_40ev", "msms_negative_40ev"]:
            data[actual_col] = [make_msms_spectrum(s) for s in data[actual_col]]

        data["func_group"] = [get_functional_groups(s) for s in data["smiles"]]
        data = data.dropna(subset=["func_group", actual_col])
        data[actual_col] = [interpolate_to_length(s, target_length) for s in data[actual_col]]
        frames.append(data[[actual_col, "func_group", "_loaded_row_id"]])

    training_data = pd.concat(frames, ignore_index=True)
    X = np.stack(training_data[actual_col].to_list()).astype(np.float32)
    y = np.stack(training_data["func_group"].to_list()).astype(np.int32)
    row_ids = training_data["_loaded_row_id"].to_numpy(dtype=np.int64)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, X=X, y=y, row_ids=row_ids, column=np.asarray(column))
        print(f"Saved cache to {cache_path}")

    return X, y, row_ids


def rebuild_splits(
    X: np.ndarray,
    y: np.ndarray,
    row_ids: np.ndarray,
    seed: int,
    mode: str,
    n_folds: int,
    fold_index: int,
) -> dict[str, dict[str, np.ndarray]]:
    """Recreate train/val/test splits from the training script.

    Returns a dictionary where each split maps to arrays for X/y/row_ids.
    """
    X_train_full, X_test, y_train_full, y_test, rows_train_full, rows_test = train_test_split(
        X,
        y,
        row_ids,
        test_size=0.1,
        random_state=seed,
        shuffle=True,
    )

    splits = {
        "train_full": {"X": X_train_full, "y": y_train_full, "row_ids": rows_train_full},
        "test": {"X": X_test, "y": y_test, "row_ids": rows_test},
    }

    if mode == "original":
        X_train, X_val, y_train, y_val, rows_train, rows_val = train_test_split(
            X_train_full,
            y_train_full,
            rows_train_full,
            test_size=0.11,
            random_state=seed,
            shuffle=True,
        )
        splits["train"] = {"X": X_train, "y": y_train, "row_ids": rows_train}
        splits["val"] = {"X": X_val, "y": y_val, "row_ids": rows_val}
        return splits

    if mode == "k_fold":
        if fold_index < 1 or fold_index > n_folds:
            raise ValueError(f"fold_index must be in [1, {n_folds}].")
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        selected = None
        for current_fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_full), start=1):
            if current_fold == fold_index:
                selected = train_idx, val_idx
                break
        assert selected is not None
        train_idx, val_idx = selected
        splits["train"] = {
            "X": X_train_full[train_idx],
            "y": y_train_full[train_idx],
            "row_ids": rows_train_full[train_idx],
        }
        splits["val"] = {
            "X": X_train_full[val_idx],
            "y": y_train_full[val_idx],
            "row_ids": rows_train_full[val_idx],
        }
        return splits

    raise ValueError("mode must be 'original' or 'k_fold'.")


# -----------------------------------------------------------------------------
# Prediction helpers
# -----------------------------------------------------------------------------


def resolve_device(device_arg: str, cuda_device: int | None = None) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        if cuda_device is not None:
            torch.cuda.set_device(cuda_device)
        return torch.device("cuda")
    if torch.cuda.is_available():
        if cuda_device is not None:
            torch.cuda.set_device(cuda_device)
        return torch.device("cuda")
    return torch.device("cpu")


def clean_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("module.", "_orig_mod."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        cleaned[new_key] = value
    return cleaned


def load_cnn_model(checkpoint_path: Path, input_length: int | None, num_fgs: int | None, device: torch.device) -> FunctionalGroupCNN:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        input_length = int(checkpoint.get("target_length", input_length or 1800))
        num_fgs = int(checkpoint.get("num_fgs", num_fgs or 37))
    elif isinstance(checkpoint, dict) and all(torch.is_tensor(v) for v in checkpoint.values()):
        state_dict = checkpoint
        if input_length is None or num_fgs is None:
            raise ValueError("Raw state_dict checkpoint requires --target-length and --num-fgs.")
    else:
        raise ValueError("Unsupported checkpoint format.")

    model = FunctionalGroupCNN(input_length=input_length, num_fgs=num_fgs).to(device)
    model.load_state_dict(clean_state_dict(state_dict), strict=True)
    model.eval()
    return model


@torch.no_grad()
def predict_probs(model: nn.Module, X: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    X_input = X.reshape(X.shape[0], 1, X.shape[1]).astype(np.float32)
    loader = DataLoader(TensorDataset(torch.tensor(X_input, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
    parts = []
    model.eval()
    for (xb,) in loader:
        xb = xb.to(device)
        probs = model(xb)
        parts.append(probs.detach().float().cpu().numpy())
    probs_np = np.concatenate(parts, axis=0)
    probs_np = np.nan_to_num(probs_np, nan=0.5, posinf=1.0, neginf=0.0)
    return np.clip(probs_np, 0.0, 1.0).astype(np.float32)


def tune_per_label_thresholds(y_true: np.ndarray, y_prob: np.ndarray, grid_step: float = 0.01) -> np.ndarray:
    thresholds = np.zeros(y_true.shape[1], dtype=np.float32)
    grid = np.arange(0.05, 0.95 + 1e-12, grid_step)
    for label_idx in range(y_true.shape[1]):
        best_threshold = 0.5
        best_f1 = -1.0
        for threshold in grid:
            y_pred_label = (y_prob[:, label_idx] >= threshold).astype(int)
            score = f1_score(y_true[:, label_idx], y_pred_label, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(threshold)
        thresholds[label_idx] = best_threshold
    return thresholds


def apply_thresholds(y_prob: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return (y_prob >= thresholds.reshape(1, -1)).astype(np.int32)


@dataclass
class PredictionBundle:
    labels: np.ndarray
    probs: np.ndarray | None
    preds: np.ndarray
    sample_indices: np.ndarray


# -----------------------------------------------------------------------------
# CSV / JSON utilities
# -----------------------------------------------------------------------------


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
# Analysis tables, same semantics as TTN analyzer.
# -----------------------------------------------------------------------------


def class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    thresholds: np.ndarray | None,
    label_names: list[str],
    train_labels: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n_samples, n_classes = y_true.shape
    train_prevalence = train_labels.mean(axis=0) if train_labels is not None else None

    for c in range(n_classes):
        yt = y_true[:, c].astype(int)
        yp = y_pred[:, c].astype(int)
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

        row = {
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
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "specificity": float(specificity),
            "false_positive_rate": float(fpr),
            "false_negative_rate": float(fnr),
            "threshold": float(thresholds[c]) if thresholds is not None else None,
            "train_prevalence": float(train_prevalence[c]) if train_prevalence is not None else None,
            "eval_prevalence": support / n_samples,
            "eval_minus_train_prevalence": float((support / n_samples) - train_prevalence[c])
            if train_prevalence is not None
            else None,
        }

        if y_prob is not None:
            pr = y_prob[:, c]
            pos_probs = pr[yt == 1]
            neg_probs = pr[yt == 0]
            row.update(
                {
                    "mean_prob_all": float(pr.mean()),
                    "mean_prob_true_positive_samples": float(pos_probs.mean()) if pos_probs.size else None,
                    "mean_prob_true_negative_samples": float(neg_probs.mean()) if neg_probs.size else None,
                    "prob_gap_pos_minus_neg": float(pos_probs.mean() - neg_probs.mean())
                    if pos_probs.size and neg_probs.size
                    else None,
                    "roc_auc": safe_float(roc_auc_score(yt, pr)) if len(np.unique(yt)) == 2 else None,
                    "average_precision": safe_float(average_precision_score(yt, pr)) if len(np.unique(yt)) == 2 else None,
                }
            )
        else:
            row.update(
                {
                    "mean_prob_all": None,
                    "mean_prob_true_positive_samples": None,
                    "mean_prob_true_negative_samples": None,
                    "prob_gap_pos_minus_neg": None,
                    "roc_auc": None,
                    "average_precision": None,
                }
            )
        rows.append(row)
    return rows


def global_distribution_rows(y_true: np.ndarray, y_pred: np.ndarray, label_names: list[str]) -> list[dict[str, Any]]:
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


def split_distribution_rows(splits: dict[str, dict[str, np.ndarray]], label_names: list[str]) -> list[dict[str, Any]]:
    rows = []
    for split_name, payload in splits.items():
        labels = payload["y"]
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
    thresholds: np.ndarray | None,
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
                "threshold": float(thresholds[c]) if thresholds is not None else None,
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
    rows = []
    grid = np.arange(0.05, 0.95 + 1e-12, grid_step)
    for c, name in enumerate(label_names):
        yt = y_true[:, c]
        if yt.sum() == 0:
            continue
        for threshold in grid:
            yp = (y_prob[:, c] >= threshold).astype(int)
            rows.append(
                {
                    "class_index": c,
                    "label": name,
                    "threshold": float(threshold),
                    "precision": float(precision_score(yt, yp, zero_division=0)),
                    "recall": float(recall_score(yt, yp, zero_division=0)),
                    "f1": float(f1_score(yt, yp, zero_division=0)),
                }
            )
    return rows


def diagnosis_rows(per_class: list[dict[str, Any]], switch: np.ndarray, label_names: list[str]) -> list[dict[str, Any]]:
    rows = []
    f1_values = np.asarray([row["f1"] for row in per_class], dtype=float)
    support_values = np.asarray([row["support_true"] for row in per_class], dtype=float)
    median_f1 = float(np.median(f1_values))
    median_support = float(np.median(support_values))

    for row in per_class:
        c = int(row["class_index"])
        reasons = []
        if row["support_true"] < median_support:
            reasons.append("low support / rare functional group")
        if row["recall"] < 0.5 and row["false_negative_rate"] > row["false_positive_rate"]:
            reasons.append("mostly under-detected: many false negatives")
        if row["precision"] < 0.5 and row["false_positive_rate"] > 0:
            reasons.append("over-predicted: many false positives")
        if row.get("prob_gap_pos_minus_neg") is not None and row["prob_gap_pos_minus_neg"] < 0.15:
            reasons.append("weak probability separation between positive and negative samples")
        if row.get("threshold") is not None and row["threshold"] > 0.65 and row["recall"] < 0.6:
            reasons.append("high threshold may suppress recall")
        if row.get("threshold") is not None and row["threshold"] < 0.25 and row["precision"] < 0.6:
            reasons.append("low threshold may inflate false positives")
        if row.get("eval_minus_train_prevalence") is not None and abs(row["eval_minus_train_prevalence"]) > 0.02:
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
            reasons.append("strong class: good precision/recall balance")
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


def sample_error_rows(bundle: PredictionBundle, label_names: list[str], max_samples: int) -> list[dict[str, Any]]:
    y_true, y_pred, y_prob = bundle.labels, bundle.preds, bundle.probs
    rows = []
    for i in range(y_true.shape[0]):
        fn_idx = np.flatnonzero((y_true[i] == 1) & (y_pred[i] == 0))
        fp_idx = np.flatnonzero((y_true[i] == 0) & (y_pred[i] == 1))
        if fn_idx.size == 0 and fp_idx.size == 0:
            continue
        true_idx = np.flatnonzero(y_true[i] == 1)
        pred_idx = np.flatnonzero(y_pred[i] == 1)
        row = {
            "sample_index": int(bundle.sample_indices[i]),
            "num_false_negatives": int(fn_idx.size),
            "num_false_positives": int(fp_idx.size),
            "error_count": int(fn_idx.size + fp_idx.size),
            "true_labels": " | ".join(label_names[j] for j in true_idx),
            "predicted_labels": " | ".join(label_names[j] for j in pred_idx),
            "false_negative_labels": " | ".join(label_names[j] for j in fn_idx),
            "false_positive_labels": " | ".join(label_names[j] for j in fp_idx),
        }
        if y_prob is not None:
            row["false_negative_probs"] = " | ".join(f"{label_names[j]}={y_prob[i, j]:.4f}" for j in fn_idx)
            row["false_positive_probs"] = " | ".join(f"{label_names[j]}={y_prob[i, j]:.4f}" for j in fp_idx)
        rows.append(row)
    rows.sort(key=lambda row: row["error_count"], reverse=True)
    return rows[:max_samples]


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------


def plot_bar(path: Path, names: list[str], values: list[float], title: str, ylabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
# Artifact loading
# -----------------------------------------------------------------------------


def load_bundle_from_results_pickle(results_pickle: Path) -> tuple[PredictionBundle, np.ndarray | None, dict[str, Any]]:
    with results_pickle.open("rb") as handle:
        results = pickle.load(handle)

    if "test_predictions" in results and "test_targets" in results:
        y_pred = np.asarray(results["test_predictions"]).astype(np.int32)
        y_true = np.asarray(results["test_targets"]).astype(np.int32)
        y_prob = np.asarray(results.get("test_probabilities"), dtype=np.float32) if "test_probabilities" in results else None
        thresholds = np.asarray(results.get("best_thresholds"), dtype=np.float32) if "best_thresholds" in results else None
    elif "pred" in results and "tgt" in results:
        y_pred = np.asarray(results["pred"]).astype(np.int32)
        y_true = np.asarray(results["tgt"]).astype(np.int32)
        y_prob = np.asarray(results.get("prob"), dtype=np.float32) if "prob" in results else None
        thresholds = np.asarray(results.get("thresholds"), dtype=np.float32) if "thresholds" in results else None
    else:
        raise ValueError("Unsupported results.pickle format. Expected original or k_fold keys.")

    sample_indices = np.arange(y_true.shape[0], dtype=np.int64)
    return PredictionBundle(labels=y_true, probs=y_prob, preds=y_pred, sample_indices=sample_indices), thresholds, results


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detailed error analysis for FunctionalGroupCNN.")

    parser.add_argument("--output-dir", type=Path, default="benchmark/cnn/results/evaluation")

    # Fast path: use saved results.pickle directly.
    parser.add_argument("--results-pickle", type=Path, default="benchmark/cnn/models/ir/original/results.pickle")

    # Full reload path: reconstruct data + load saved model.
    parser.add_argument("--analytical-data", type=Path, default=None)
    parser.add_argument("--model-checkpoint", type=Path, default=None)
    parser.add_argument("--column", choices=list(COLUMN_MAPPING.keys()), default="ir_spectra")
    parser.add_argument("--mode", choices=["original", "k_fold"], default="original")
    parser.add_argument("--split", choices=["train", "val", "test", "train_full"], default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--fold-index", type=int, default=1, help="Fold to reconstruct for k_fold mode, 1-based.")
    parser.add_argument("--target-length", type=int, default=None)
    parser.add_argument("--num-fgs", type=int, default=37)
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--max-files", type=int, default=None)

    # Threshold handling.
    parser.add_argument("--thresholds-npy", type=Path, default=None)
    parser.add_argument("--thresholds-json", type=Path, default=None)
    parser.add_argument("--retune-thresholds-on", choices=["none", "train", "val", "test", "selected"], default="none")
    parser.add_argument("--threshold-grid-step", type=float, default=0.01)

    # Runtime.
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--cuda-device", type=int, default=None)
    parser.add_argument("--top-confusions", type=int, default=100)
    parser.add_argument("--max-sample-errors", type=int, default=500)
    parser.add_argument("--num-worst-probability-plots", type=int, default=12)

    return parser


def load_thresholds_from_args(args: argparse.Namespace, fallback_len: int) -> np.ndarray | None:
    if args.thresholds_npy is not None:
        return np.load(args.thresholds_npy).astype(np.float32)
    if args.thresholds_json is not None:
        payload = json.loads(args.thresholds_json.read_text())
        if isinstance(payload, list):
            return np.asarray(payload, dtype=np.float32)
        if isinstance(payload, dict) and "thresholds" in payload:
            th = payload["thresholds"]
            if isinstance(th, list):
                return np.asarray(th, dtype=np.float32)
            if isinstance(th, dict):
                return np.asarray([th.get(name, 0.5) for name in LABEL_NAMES[:fallback_len]], dtype=np.float32)
    return None


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.output_dir / "plots"

    label_names = LABEL_NAMES[: args.num_fgs]
    thresholds = load_thresholds_from_args(args, fallback_len=args.num_fgs)
    results_metadata: dict[str, Any] = {}
    splits: dict[str, dict[str, np.ndarray]] | None = None
    train_labels: np.ndarray | None = None

    if args.results_pickle is not None:
        print(f"Loading predictions from {args.results_pickle}")
        bundle, pickle_thresholds, results_metadata = load_bundle_from_results_pickle(args.results_pickle)
        if thresholds is None:
            thresholds = pickle_thresholds
        if bundle.probs is None:
            print("Warning: results.pickle has no probabilities. Probability plots, AP/AUC, and threshold sensitivity are skipped.")
        if thresholds is None and bundle.probs is not None:
            thresholds = np.full(bundle.labels.shape[1], 0.5, dtype=np.float32)
        label_names = LABEL_NAMES[: bundle.labels.shape[1]]

    else:
        if args.analytical_data is None or args.model_checkpoint is None:
            raise ValueError("Either provide --results-pickle or both --analytical-data and --model-checkpoint.")

        print("Loading raw column data...")
        X, y, row_ids = load_column_data(
            analytical_data=args.analytical_data,
            column=args.column,
            max_files=args.max_files,
            cache_path=args.cache_path,
            overwrite_cache=args.overwrite_cache,
        )
        label_names = LABEL_NAMES[: y.shape[1]]
        splits = rebuild_splits(
            X=X,
            y=y,
            row_ids=row_ids,
            seed=args.seed,
            mode=args.mode,
            n_folds=args.n_folds,
            fold_index=args.fold_index,
        )
        train_labels = splits["train"]["y"] if "train" in splits else splits["train_full"]["y"]

        selected = splits[args.split]
        device = resolve_device(args.device, cuda_device=args.cuda_device)
        input_length = args.target_length or int(selected["X"].shape[1])
        print(f"Loading CNN model on {device}...")
        model = load_cnn_model(args.model_checkpoint, input_length=input_length, num_fgs=y.shape[1], device=device)

        if thresholds is None:
            thresholds = np.full(y.shape[1], 0.5, dtype=np.float32)

        if args.retune_thresholds_on != "none":
            tune_split = args.split if args.retune_thresholds_on == "selected" else args.retune_thresholds_on
            print(f"Retuning thresholds on {tune_split} split...")
            tune_payload = splits[tune_split]
            tune_probs = predict_probs(model, tune_payload["X"], batch_size=args.batch_size, device=device)
            thresholds = tune_per_label_thresholds(tune_payload["y"], tune_probs, grid_step=args.threshold_grid_step)

        print(f"Running inference on split={args.split} ({len(selected['X'])} samples)...")
        probs = predict_probs(model, selected["X"], batch_size=args.batch_size, device=device)
        preds = apply_thresholds(probs, thresholds)
        bundle = PredictionBundle(
            labels=selected["y"].astype(np.int32),
            probs=probs,
            preds=preds,
            sample_indices=selected["row_ids"].astype(np.int64),
        )

    if thresholds is not None and len(thresholds) != bundle.labels.shape[1]:
        raise ValueError(f"Threshold length {len(thresholds)} does not match label width {bundle.labels.shape[1]}.")

    print("Computing analysis tables...")
    per_class = class_metrics(bundle.labels, bundle.preds, bundle.probs, thresholds, label_names, train_labels=train_labels)
    distribution = global_distribution_rows(bundle.labels, bundle.preds, label_names)
    switch = switch_matrix(bundle.labels, bundle.preds)
    top_confusions = top_confusion_rows(switch, label_names, args.top_confusions)
    diagnosis = diagnosis_rows(per_class, switch, label_names)
    sample_errors = sample_error_rows(bundle, label_names, args.max_sample_errors)

    save_csv(args.output_dir / "per_class_metrics.csv", per_class)
    save_csv(args.output_dir / "per_class_diagnosis.csv", diagnosis)
    save_csv(args.output_dir / "global_label_distribution.csv", distribution)
    save_matrix_csv(args.output_dir / "switch_matrix_false_negative_vs_false_positive.csv", switch, label_names, label_names)
    save_csv(args.output_dir / "top_confusions.csv", top_confusions)
    save_csv(args.output_dir / "sample_errors.csv", sample_errors)

    if splits is not None:
        save_csv(args.output_dir / "split_label_distribution.csv", split_distribution_rows(splits, label_names))

    if bundle.probs is not None:
        save_csv(args.output_dir / "probability_summary.csv", probability_summary_rows(bundle.labels, bundle.probs, thresholds, label_names))
        save_csv(
            args.output_dir / "threshold_sensitivity.csv",
            threshold_sensitivity_rows(bundle.labels, bundle.probs, label_names, grid_step=args.threshold_grid_step),
        )

    f1_micro = f1_score(bundle.labels, bundle.preds, average="micro", zero_division=0)
    f1_macro = f1_score(bundle.labels, bundle.preds, average="macro", zero_division=0)
    precision_micro = precision_score(bundle.labels, bundle.preds, average="micro", zero_division=0)
    recall_micro = recall_score(bundle.labels, bundle.preds, average="micro", zero_division=0)

    print("Creating plots...")
    plot_bar(plots_dir / "per_class_f1.png", label_names, [float(row["f1"]) for row in per_class], "Per-class F1", "F1")
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

    worst_by_f1 = sorted(per_class, key=lambda row: row["f1"])[: args.num_worst_probability_plots]
    worst_indices = [int(row["class_index"]) for row in worst_by_f1]
    if bundle.probs is not None:
        plot_probability_histograms(
            plots_dir / "probability_histograms_worst_classes",
            bundle.labels,
            bundle.probs,
            label_names,
            worst_indices,
        )

    summary = {
        "num_samples": int(bundle.labels.shape[0]),
        "num_labels": int(bundle.labels.shape[1]),
        "f1_micro": float(f1_micro),
        "f1_macro": float(f1_macro),
        "precision_micro": float(precision_micro),
        "recall_micro": float(recall_micro),
        "mean_threshold": float(np.mean(thresholds)) if thresholds is not None else None,
        "std_threshold": float(np.std(thresholds)) if thresholds is not None else None,
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
        "source_results_metadata_keys": sorted(list(results_metadata.keys())) if results_metadata else [],
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
