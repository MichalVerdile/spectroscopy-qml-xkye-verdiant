"""Training entry point for experiment 10."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = Path(__file__).resolve().parents[5]
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.spectroscopy_qml.msms_neg.tree_tensor_network.helpers.data_loader import (  # noqa: E402
    FUNCTIONAL_GROUPS,
    load_cnmr_data,
    load_or_create_split_indices,
    prepare_dataloaders_from_split_indices,
)
from src.spectroscopy_qml.msms_neg.tree_tensor_network.helpers.losses import build_loss  # noqa: E402
from src.spectroscopy_qml.msms_neg.tree_tensor_network.helpers.train_helpers import (  # noqa: E402
    EarlyStopping,
    build_threshold_grid,
    compute_metrics,
    count_available_data_files,
    get_pos_weight,
    resolve_device,
    resolve_used_file_count,
    run_real_batch_preflight,
    run_synthetic_preflight,
    select_early_stopping_score,
    threshold_predictions,
    tune_thresholds,
    write_epoch_details,
    write_threshold_artifact,
)
from src.spectroscopy_qml.msms_neg.tree_tensor_network.helpers.merge_helpers import (  # noqa: E402
    DEFAULT_SEGMENT_STRIDE,
    DEFAULT_SEGMENT_WINDOW_SIZE,
)


def resolve_cache_path(args: argparse.Namespace) -> Path:
    if args.cache_path is not None:
        return args.cache_path

    cache_dir = Path("data/cache")
    file_suffix = "all" if args.max_files is None else f"files{int(args.max_files)}"
    snv_suffix = "snv" if args.apply_snv else "raw"
    return cache_dir / f"cnmr_spectra_len{args.input_dim}_{snv_suffix}_{file_suffix}.npz"


def write_summary(
    summary_path: Path,
    elapsed_seconds: float,
    completed_epochs: int,
    requested_epochs: int,
    used_data_files: int,
    total_data_files: int,
    split_path: Path,
    best_epoch: int,
    best_score: float,
    best_metric_name: str,
    best_val_loss: float,
    test_loss: float,
    test_metrics: dict[str, float | np.ndarray],
    final_thresholds: np.ndarray,
) -> None:
    with summary_path.open("w") as handle:
        handle.write("TTN C-NMR Experiment10 Summary\n")
        handle.write("=" * 80 + "\n")
        handle.write("Leaf encoder:              none (segments enter TTN directly)\n")
        handle.write("Feature channels:          raw + first_derivative + second_derivative\n")
        handle.write(f"Elapsed seconds:           {elapsed_seconds:.2f}\n")
        handle.write(f"Epochs completed:          {completed_epochs}/{requested_epochs}\n")
        handle.write(f"Data files used:           {used_data_files}/{total_data_files}\n")
        handle.write(f"Fixed split artifact:      {split_path}\n")
        handle.write(f"Best epoch:                {best_epoch}\n")
        handle.write(f"Best early-stop score:     {best_score:.6f}\n")
        handle.write(f"Best score metric:         {best_metric_name}\n")
        handle.write(f"Best val loss:             {best_val_loss:.6f}\n")
        handle.write(f"Threshold mean:            {float(np.mean(final_thresholds)):.6f}\n")
        handle.write(f"Threshold std:             {float(np.std(final_thresholds)):.6f}\n")
        handle.write(f"Test loss:                 {test_loss:.6f}\n")
        handle.write(f"Test f1_micro:             {float(test_metrics['f1_micro']):.6f}\n")
        handle.write(f"Test f1_macro:             {float(test_metrics['f1_macro']):.6f}\n")
        handle.write(f"Test precision_micro:      {float(test_metrics['precision_micro']):.6f}\n")
        handle.write(f"Test recall_micro:         {float(test_metrics['recall_micro']):.6f}\n")


def sanitize_binary_targets_and_probs(labels: np.ndarray, probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    safe_labels = np.nan_to_num(labels, nan=0.0, posinf=1.0, neginf=0.0)
    safe_labels = np.clip(safe_labels, 0.0, 1.0)
    safe_labels = (safe_labels >= 0.5).astype(np.float32)

    safe_probs = np.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
    safe_probs = np.clip(safe_probs, 0.0, 1.0).astype(np.float32)
    return safe_labels, safe_probs


def ensure_finite_tensor(tensor: torch.Tensor, name: str, stage: str, batch_index: int) -> None:
    """Fail fast when numerics break instead of masking them in later metrics."""
    if torch.isfinite(tensor).all():
        return
    raise RuntimeError(
        f"Non-finite {name} detected during {stage} at batch {batch_index} on device {tensor.device}."
    )


def resolve_compile_enabled(requested_compile: bool, device: torch.device) -> bool:
    """Disable torch.compile automatically on backends where it is unstable here."""
    return bool(requested_compile and device.type != "mps")


def train_epoch_amp(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    grad_clip_norm: float | None = None,
    scaler: GradScaler | None = None,
):
    model.train()
    total_loss = 0.0
    labels_list: list[np.ndarray] = []
    probs_list: list[np.ndarray] = []
    use_amp = scaler is not None

    for batch_index, (spectra, labels) in enumerate(dataloader, start=1):
        spectra = spectra.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(spectra)
            loss = criterion(logits, labels)
        ensure_finite_tensor(logits, name="logits", stage="train", batch_index=batch_index)
        ensure_finite_tensor(loss.detach(), name="loss", stage="train", batch_index=batch_index)

        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        total_loss += loss.item() * spectra.size(0)
        labels_list.append(labels.detach().cpu().numpy())
        probs_list.append(torch.sigmoid(logits.detach().float()).cpu().numpy())

    all_labels = np.concatenate(labels_list, axis=0)
    all_probs = np.concatenate(probs_list, axis=0)
    all_labels, all_probs = sanitize_binary_targets_and_probs(all_labels, all_probs)
    preds = threshold_predictions(all_probs, 0.5)
    metrics = compute_metrics(all_labels, preds)
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss, metrics


@torch.no_grad()
def evaluate_with_probs_amp(model, dataloader, criterion, device, use_amp: bool = False):
    model.eval()
    total_loss = 0.0
    labels_list: list[np.ndarray] = []
    probs_list: list[np.ndarray] = []

    for batch_index, (spectra, labels) in enumerate(dataloader, start=1):
        spectra = spectra.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(spectra)
            loss = criterion(logits, labels)
        ensure_finite_tensor(logits, name="logits", stage="eval", batch_index=batch_index)
        ensure_finite_tensor(loss.detach(), name="loss", stage="eval", batch_index=batch_index)

        total_loss += loss.item() * spectra.size(0)
        labels_list.append(labels.cpu().numpy())
        probs_list.append(torch.sigmoid(logits.float()).cpu().numpy())

    all_labels = np.concatenate(labels_list, axis=0)
    all_probs = np.concatenate(probs_list, axis=0)
    all_labels, all_probs = sanitize_binary_targets_and_probs(all_labels, all_probs)
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss, all_labels, all_probs

