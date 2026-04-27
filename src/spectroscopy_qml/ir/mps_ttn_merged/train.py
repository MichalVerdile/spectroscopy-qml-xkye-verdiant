from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = Path(__file__).resolve().parents[3]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spectroscopy_qml.ir.mps_classifier.model import MPSFunctionalGroupClassifier  # noqa: E402
from spectroscopy_qml.ir.mps_classifier.train import tune_thresholds as tune_thresholds_mps  # noqa: E402
from spectroscopy_qml.ir.mps_ttn_merged.config import (  # noqa: E402
    DATA_CONFIG,
    MPS_CONFIG,
    MPS_DATA_CONFIG,
    MPS_TRAINING_CONFIG,
    PATH_CONFIG,
    SPLIT_CONFIG,
    TTN_CONFIG,
    TTN_DATA_CONFIG,
    TTN_TRAINING_CONFIG,
)
from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment10.train import (  # noqa: E402
    ensure_finite_tensor,
    resolve_compile_enabled,
)
from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment10_2.model import TTNIRClassifier10_2  # noqa: E402
from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment5.data_loader import (  # noqa: E402
    FUNCTIONAL_GROUPS,
    load_ir_data,
    load_or_create_split_indices,
    prepare_dataloaders_from_split_indices,
)
from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment5.losses import build_loss  # noqa: E402
from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment5.train import (  # noqa: E402
    EarlyStopping,
    build_threshold_grid,
    compute_metrics,
    count_available_data_files,
    get_pos_weight,
    resolve_device,
    resolve_used_file_count,
    select_early_stopping_score,
    threshold_predictions,
    tune_thresholds,
    write_threshold_artifact,
)

# Functional groups exclusively trained by the TTN branch.
# The MPS branch trains on all remaining groups.
TTN_BRANCH_GROUPS: frozenset[str] = frozenset({
    "Alkane",
    "Arene",
    "Ether",
    "Haloalkane",
    "Isocyanate",
    "Nitrile",
    "Sulfonic acid",
    "Thial",
})


class ClassSubsetLoss(nn.Module):
    """Compute loss on a fixed subset of class columns only."""

    def __init__(self, inner: nn.Module, class_indices: list[int]) -> None:
        super().__init__()
        self.inner = inner
        self._class_indices = class_indices

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.inner(logits[:, self._class_indices], labels[:, self._class_indices])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train MPS and TTN as separate models with branch-specific configs, then "
            "merge their saved validation/test predictions only after both trainings complete."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path(DATA_CONFIG.data_dir))
    parser.add_argument("--output-dir", type=Path, default=Path(PATH_CONFIG.output_dir))
    parser.add_argument("--split-path", type=Path, default=None)
    parser.add_argument("--overwrite-split", action="store_true")
    parser.add_argument("--input-dim", type=int, default=DATA_CONFIG.input_dim)
    parser.add_argument("--num-labels", type=int, default=DATA_CONFIG.num_classes)
    parser.add_argument("--max-files", type=int, default=DATA_CONFIG.max_files)
    parser.add_argument("--train-ratio", type=float, default=SPLIT_CONFIG.train_ratio)
    parser.add_argument("--val-ratio", type=float, default=SPLIT_CONFIG.val_ratio)
    parser.add_argument("--test-ratio", type=float, default=SPLIT_CONFIG.test_ratio)
    parser.add_argument("--num-workers", type=int, default=SPLIT_CONFIG.num_workers)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default=SPLIT_CONFIG.device)
    parser.add_argument("--seed", type=int, default=SPLIT_CONFIG.seed)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force deterministic kernels where possible to make reruns stable.",
    )

    parser.add_argument("--mps-apply-snv", action=argparse.BooleanOptionalAction, default=MPS_DATA_CONFIG.apply_snv)
    parser.add_argument("--mps-apply-savgol", action=argparse.BooleanOptionalAction, default=MPS_DATA_CONFIG.apply_savgol)
    parser.add_argument("--mps-savgol-window-length", type=int, default=MPS_DATA_CONFIG.savgol_window_length)
    parser.add_argument("--mps-savgol-polyorder", type=int, default=MPS_DATA_CONFIG.savgol_polyorder)
    parser.add_argument("--mps-cache-path", type=Path, default=None)
    parser.add_argument("--mps-overwrite-cache", action="store_true")
    parser.add_argument("--mps-num-sites", type=int, default=MPS_CONFIG.num_sites)
    parser.add_argument("--mps-physical-dim", type=int, default=MPS_CONFIG.physical_dim)
    parser.add_argument("--mps-bond-dim", type=int, default=MPS_CONFIG.bond_dim)
    parser.add_argument("--mps-dropout-rate", type=float, default=MPS_CONFIG.dropout_rate)
    parser.add_argument("--mps-classifier-head", choices=["mps", "cnn"], default=MPS_CONFIG.classifier_head)
    parser.add_argument("--mps-num-sites-2", type=int, default=MPS_CONFIG.num_sites_2)
    parser.add_argument("--mps-physical-dim-2", type=int, default=MPS_CONFIG.physical_dim_2)
    parser.add_argument("--mps-bond-dim-2", type=int, default=MPS_CONFIG.bond_dim_2)
    parser.add_argument("--mps-batch-size", type=int, default=MPS_TRAINING_CONFIG.batch_size)
    parser.add_argument("--mps-epochs", type=int, default=MPS_TRAINING_CONFIG.epochs)
    parser.add_argument("--mps-learning-rate", type=float, default=MPS_TRAINING_CONFIG.learning_rate)
    parser.add_argument("--mps-weight-decay", type=float, default=MPS_TRAINING_CONFIG.weight_decay)
    parser.add_argument("--mps-lr-scheduler-factor", type=float, default=MPS_TRAINING_CONFIG.lr_scheduler_factor)
    parser.add_argument("--mps-lr-scheduler-patience", type=int, default=MPS_TRAINING_CONFIG.lr_scheduler_patience)
    parser.add_argument("--mps-lr-scheduler-min-lr", type=float, default=MPS_TRAINING_CONFIG.lr_scheduler_min_lr)
    parser.add_argument("--mps-early-stopping-patience", type=int, default=MPS_TRAINING_CONFIG.early_stopping_patience)
    parser.add_argument("--mps-early-stopping-min-delta", type=float, default=MPS_TRAINING_CONFIG.early_stopping_min_delta)
    parser.add_argument("--mps-grad-clip-norm", type=float, default=MPS_TRAINING_CONFIG.grad_clip_norm)
    parser.add_argument("--mps-amp", action=argparse.BooleanOptionalAction, default=MPS_TRAINING_CONFIG.amp)
    parser.add_argument("--mps-compile", action=argparse.BooleanOptionalAction, default=MPS_TRAINING_CONFIG.compile)
    parser.add_argument("--mps-threshold-mode", choices=["global", "per_class"], default=MPS_TRAINING_CONFIG.threshold_mode)
    parser.add_argument(
        "--mps-threshold-target-metric",
        choices=["f1_micro", "f1_macro", "per_class_f1"],
        default=MPS_TRAINING_CONFIG.threshold_target_metric,
    )
    parser.add_argument("--mps-threshold-grid-step", type=float, default=MPS_TRAINING_CONFIG.threshold_grid_step)

    parser.add_argument("--ttn-apply-snv", action=argparse.BooleanOptionalAction, default=TTN_DATA_CONFIG.apply_snv)
    parser.add_argument("--ttn-apply-savgol", action=argparse.BooleanOptionalAction, default=TTN_DATA_CONFIG.apply_savgol)
    parser.add_argument("--ttn-savgol-window-length", type=int, default=TTN_DATA_CONFIG.savgol_window_length)
    parser.add_argument("--ttn-savgol-polyorder", type=int, default=TTN_DATA_CONFIG.savgol_polyorder)
    parser.add_argument("--ttn-cache-path", type=Path, default=None)
    parser.add_argument("--ttn-overwrite-cache", action="store_true")
    parser.add_argument("--ttn-chi", type=int, default=TTN_CONFIG.chi)
    parser.add_argument("--ttn-segment-window-size", type=int, default=TTN_CONFIG.segment_window_size)
    parser.add_argument("--ttn-segment-stride", type=int, default=TTN_CONFIG.segment_stride)
    parser.add_argument("--ttn-segment-mode", choices=["overlap", "dual_offset"], default=TTN_CONFIG.segment_mode)
    parser.add_argument("--ttn-segment-offset", type=int, default=TTN_CONFIG.segment_offset)
    parser.add_argument(
        "--ttn-segment-state-normalize",
        action=argparse.BooleanOptionalAction,
        default=TTN_CONFIG.segment_state_normalize,
    )
    parser.add_argument("--ttn-merge-mode", choices=["strict", "relaxed"], default=TTN_CONFIG.merge_mode)
    parser.add_argument("--ttn-merge-residual-weight", type=float, default=TTN_CONFIG.merge_residual_weight)
    parser.add_argument(
        "--ttn-merge-renormalize-output",
        action=argparse.BooleanOptionalAction,
        default=TTN_CONFIG.merge_renormalize_output,
    )
    parser.add_argument("--ttn-lorentz-gamma", type=float, default=TTN_CONFIG.lorentz_gamma)
    parser.add_argument("--ttn-lorentz-kernel-half-width", type=int, default=TTN_CONFIG.lorentz_kernel_half_width)
    parser.add_argument(
        "--ttn-lorentz-norm-mode",
        choices=["max_abs", "z_score", "percentile"],
        default=TTN_CONFIG.lorentz_norm_mode,
    )
    parser.add_argument("--ttn-batch-size", type=int, default=TTN_TRAINING_CONFIG.batch_size)
    parser.add_argument("--ttn-epochs", type=int, default=TTN_TRAINING_CONFIG.epochs)
    parser.add_argument("--ttn-learning-rate", type=float, default=TTN_TRAINING_CONFIG.learning_rate)
    parser.add_argument("--ttn-weight-decay", type=float, default=TTN_TRAINING_CONFIG.weight_decay)
    parser.add_argument("--ttn-lr-scheduler-factor", type=float, default=TTN_TRAINING_CONFIG.lr_scheduler_factor)
    parser.add_argument("--ttn-lr-scheduler-patience", type=int, default=TTN_TRAINING_CONFIG.lr_scheduler_patience)
    parser.add_argument("--ttn-lr-scheduler-min-lr", type=float, default=TTN_TRAINING_CONFIG.lr_scheduler_min_lr)
    parser.add_argument("--ttn-threshold-mode", choices=["global", "per_class"], default=TTN_TRAINING_CONFIG.threshold_mode)
    parser.add_argument(
        "--ttn-threshold-target-metric",
        choices=["f1_micro", "f1_macro", "per_class_f1"],
        default=TTN_TRAINING_CONFIG.threshold_target_metric,
    )
    parser.add_argument("--ttn-threshold-grid-step", type=float, default=TTN_TRAINING_CONFIG.threshold_grid_step)
    parser.add_argument(
        "--ttn-early-stopping-metric",
        choices=["f1_micro", "f1_macro", "blended_f1"],
        default=TTN_TRAINING_CONFIG.early_stopping_metric,
    )
    parser.add_argument(
        "--ttn-early-stopping-blend-alpha",
        type=float,
        default=TTN_TRAINING_CONFIG.early_stopping_blend_alpha,
    )
    parser.add_argument("--ttn-early-stopping-patience", type=int, default=TTN_TRAINING_CONFIG.early_stopping_patience)
    parser.add_argument(
        "--ttn-early-stopping-min-delta",
        type=float,
        default=TTN_TRAINING_CONFIG.early_stopping_min_delta,
    )
    parser.add_argument(
        "--ttn-min-epochs-before-stopping",
        type=int,
        default=TTN_TRAINING_CONFIG.min_epochs_before_stopping,
    )
    parser.add_argument("--ttn-grad-clip-norm", type=float, default=TTN_TRAINING_CONFIG.grad_clip_norm)
    parser.add_argument("--ttn-amp", action=argparse.BooleanOptionalAction, default=TTN_TRAINING_CONFIG.amp)
    parser.add_argument("--ttn-compile", action=argparse.BooleanOptionalAction, default=TTN_TRAINING_CONFIG.compile)
    parser.add_argument("--ttn-pos-weight-power", type=float, default=TTN_TRAINING_CONFIG.pos_weight_power)
    parser.add_argument("--ttn-pos-weight-max", type=float, default=TTN_TRAINING_CONFIG.pos_weight_max)
    parser.add_argument("--ttn-loss-type", choices=["bce", "focal"], default=TTN_TRAINING_CONFIG.loss_type)
    parser.add_argument("--ttn-focal-gamma", type=float, default=TTN_TRAINING_CONFIG.focal_gamma)

    parser.add_argument("--preflight-batch-size", type=int, default=4)
    parser.add_argument("--check-only", action="store_true")
    return parser


def resolve_branch_cache_path(
    branch_name: str,
    explicit_path: Path | None,
    input_dim: int,
    max_files: int | None,
    apply_snv: bool,
    apply_savgol: bool,
) -> Path:
    if explicit_path is not None:
        return explicit_path

    cache_dir = Path("data/cache")
    file_suffix = "all" if max_files is None else f"files{int(max_files)}"
    snv_suffix = "snv" if apply_snv else "raw"
    savgol_suffix = "sg" if apply_savgol else "nosg"
    return cache_dir / f"ir_{branch_name}_len{input_dim}_{snv_suffix}_{savgol_suffix}_{file_suffix}.npz"


def build_mps_model(args: argparse.Namespace) -> MPSFunctionalGroupClassifier:
    return MPSFunctionalGroupClassifier(
        input_dim=args.input_dim,
        num_sites=args.mps_num_sites,
        physical_dim=args.mps_physical_dim,
        bond_dim=args.mps_bond_dim,
        num_classes=args.num_labels,
        dropout_rate=args.mps_dropout_rate,
        classifier_head=args.mps_classifier_head,
        num_sites_2=args.mps_num_sites_2,
        physical_dim_2=args.mps_physical_dim_2,
        bond_dim_2=args.mps_bond_dim_2,
    )


def build_ttn_model(args: argparse.Namespace) -> TTNIRClassifier10_2:
    return TTNIRClassifier10_2(
        num_labels=args.num_labels,
        chi=args.ttn_chi,
        input_dim=args.input_dim,
        segment_window_size=args.ttn_segment_window_size,
        segment_stride=args.ttn_segment_stride,
        segment_mode=args.ttn_segment_mode,
        segment_offset=args.ttn_segment_offset,
        segment_state_normalize=args.ttn_segment_state_normalize,
        merge_mode=args.ttn_merge_mode,
        merge_residual_weight=args.ttn_merge_residual_weight,
        merge_renormalize_output=args.ttn_merge_renormalize_output,
        lorentz_gamma=args.ttn_lorentz_gamma,
        lorentz_kernel_half_width=args.ttn_lorentz_kernel_half_width,
        lorentz_norm_mode=args.ttn_lorentz_norm_mode,
    )


def describe_args(args: argparse.Namespace, split_path: Path, mps_cache_path: Path, ttn_cache_path: Path) -> None:
    print("=" * 80)
    print("MPS + TTN Separate Training")
    print("=" * 80)
    print(f"Data dir:                 {args.data_dir}")
    print(f"Output dir:               {args.output_dir}")
    print(f"Split path:               {split_path}")
    print(f"Input dim:                {args.input_dim}")
    print(f"Num labels:               {args.num_labels}")
    print(f"Train/Val/Test:           {args.train_ratio:.2f}/{args.val_ratio:.2f}/{args.test_ratio:.2f}")
    print(f"MPS cache path:           {mps_cache_path}")
    print(f"TTN cache path:           {ttn_cache_path}")
    print(f"MPS preprocessing:        snv={args.mps_apply_snv}, savgol={args.mps_apply_savgol}")
    print(f"TTN preprocessing:        snv={args.ttn_apply_snv}, savgol={args.ttn_apply_savgol}")
    print(f"MPS batch/epochs:         {args.mps_batch_size}/{args.mps_epochs}")
    print(f"TTN batch/epochs:         {args.ttn_batch_size}/{args.ttn_epochs}")
    print(f"Deterministic mode:       {args.deterministic}")
    print("Training mode:            direct branch training, no shared wrapper forward pass")
    print("Merge rule:               choose validation-best branch per label after both trainings")


def configure_reproducibility(seed: int, deterministic: bool) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.use_deterministic_algorithms(False)


def write_selector_report(
    report_path: Path,
    label_names: list[str],
    selector: dict[str, object],
) -> None:
    mps_f1 = np.asarray(selector["per_class_f1_mps"], dtype=np.float32)
    ttn_f1 = np.asarray(selector["per_class_f1_ttn"], dtype=np.float32)
    selected_branch_names = list(selector["selected_branch_names"])

    with report_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "f1_mps", "f1_ttn", "delta_mps_minus_ttn", "selected_branch"])
        for label_name, mps_score, ttn_score, branch_name in zip(label_names, mps_f1, ttn_f1, selected_branch_names):
            writer.writerow(
                [
                    label_name,
                    float(mps_score),
                    float(ttn_score),
                    float(mps_score - ttn_score),
                    branch_name,
                ]
            )


def write_summary(
    summary_path: Path,
    elapsed_seconds: float,
    split_path: Path,
    mps_result: dict[str, object],
    ttn_result: dict[str, object],
    selector_val_metrics: dict[str, float | np.ndarray],
    selector_test_metrics: dict[str, float | np.ndarray],
    final_thresholds: np.ndarray,
    selected_branch_names: list[str],
) -> None:
    with summary_path.open("w") as handle:
        handle.write("MPS + TTN Separate Training Selector Summary\n")
        handle.write("=" * 80 + "\n")
        handle.write("Training mode:            models trained separately and merged after training\n")
        handle.write(f"Elapsed seconds:           {elapsed_seconds:.2f}\n")
        handle.write(f"Fixed split artifact:      {split_path}\n")
        handle.write(f"Best MPS epoch:            {int(mps_result['best_epoch'])}\n")
        handle.write(f"Best TTN epoch:            {int(ttn_result['best_epoch'])}\n")
        handle.write(f"MPS val f1_micro:          {float(mps_result['val_metrics']['f1_micro']):.6f}\n")
        handle.write(f"TTN val f1_micro:          {float(ttn_result['val_metrics']['f1_micro']):.6f}\n")
        handle.write(f"Selector val f1_micro:     {float(selector_val_metrics['f1_micro']):.6f}\n")
        handle.write(f"Selector val f1_macro:     {float(selector_val_metrics['f1_macro']):.6f}\n")
        handle.write(f"Threshold mean:            {float(np.mean(final_thresholds)):.6f}\n")
        handle.write(f"Threshold std:             {float(np.std(final_thresholds)):.6f}\n")
        handle.write(f"Labels assigned to MPS:    {sum(name == 'mps' for name in selected_branch_names)}\n")
        handle.write(f"Labels assigned to TTN:    {sum(name == 'ttn' for name in selected_branch_names)}\n")
        handle.write(f"Selector test f1_micro:    {float(selector_test_metrics['f1_micro']):.6f}\n")
        handle.write(f"Selector test f1_macro:    {float(selector_test_metrics['f1_macro']):.6f}\n")
        handle.write(f"Selector precision_micro:  {float(selector_test_metrics['precision_micro']):.6f}\n")
        handle.write(f"Selector recall_micro:     {float(selector_test_metrics['recall_micro']):.6f}\n")


def compute_branch_metrics(y_true: np.ndarray, probs: np.ndarray, thresholds: np.ndarray) -> dict[str, float | np.ndarray]:
    preds = threshold_predictions(probs, thresholds)
    return compute_metrics(y_true, preds)


def select_best_branch_per_label(
    y_true: np.ndarray,
    probs_mps: np.ndarray,
    probs_ttn: np.ndarray,
    thresholds_mps: np.ndarray,
    thresholds_ttn: np.ndarray,
) -> dict[str, object]:
    preds_mps = threshold_predictions(probs_mps, thresholds_mps)
    preds_ttn = threshold_predictions(probs_ttn, thresholds_ttn)
    metrics_mps = compute_metrics(y_true, preds_mps)
    metrics_ttn = compute_metrics(y_true, preds_ttn)

    per_class_f1_mps = np.asarray(metrics_mps["per_class_f1"], dtype=np.float32)
    per_class_f1_ttn = np.asarray(metrics_ttn["per_class_f1"], dtype=np.float32)
    use_mps = per_class_f1_mps >= per_class_f1_ttn

    selected_probs = np.where(use_mps[None, :], probs_mps, probs_ttn)
    selected_thresholds = np.where(use_mps, thresholds_mps, thresholds_ttn).astype(np.float32, copy=False)
    selected_preds = threshold_predictions(selected_probs, selected_thresholds)
    selected_metrics = compute_metrics(y_true, selected_preds)
    selected_branch_names = ["mps" if flag else "ttn" for flag in use_mps.tolist()]

    return {
        "selected_probs": selected_probs,
        "selected_thresholds": selected_thresholds,
        "selected_preds": selected_preds,
        "selected_metrics": selected_metrics,
        "selected_branch_names": selected_branch_names,
        "per_class_f1_mps": per_class_f1_mps,
        "per_class_f1_ttn": per_class_f1_ttn,
    }


def clone_module_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def run_branch_preflight(
    model: nn.Module,
    branch_name: str,
    input_dim: int,
    num_labels: int,
    batch_size: int,
    device: torch.device,
    criterion: nn.Module,
    use_amp: bool,
) -> None:
    print(f"\n[Preflight] {branch_name} synthetic forward/backward check")
    model = model.to(device)
    model.train()
    spectra = torch.randn(batch_size, input_dim, device=device)
    labels = torch.randint(0, 2, (batch_size, num_labels), device=device).float()

    with autocast(device_type=device.type, enabled=use_amp):
        logits = model(spectra)
        loss = criterion(logits, labels)
    ensure_finite_tensor(logits, name=f"{branch_name}_logits", stage="preflight", batch_index=1)
    ensure_finite_tensor(loss.detach(), name=f"{branch_name}_loss", stage="preflight", batch_index=1)
    loss.backward()
    model.zero_grad(set_to_none=True)
    print(f"  OK: {tuple(spectra.shape)} -> {tuple(logits.shape)}")


def train_epoch_collect_probs(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer,
    device: torch.device,
    grad_clip_norm: float | None,
    scaler: GradScaler | None,
) -> tuple[float, np.ndarray, np.ndarray]:
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

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss, np.concatenate(labels_list, axis=0), np.concatenate(probs_list, axis=0)


@torch.no_grad()
def evaluate_with_probs(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, np.ndarray, np.ndarray]:
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

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss, np.concatenate(labels_list, axis=0), np.concatenate(probs_list, axis=0)


def save_branch_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    thresholds: np.ndarray,
    best_epoch: int,
    best_score: float,
    args: argparse.Namespace,
    branch_name: str,
) -> None:
    torch.save(
        {
            "branch": branch_name,
            "model_state_dict": model.state_dict(),
            "thresholds": thresholds,
            "best_epoch": best_epoch,
            "best_score": best_score,
            "args": vars(args),
        },
        checkpoint_path,
    )


def train_mps_branch(
    model: nn.Module,
    train_loader,
    val_loader,
    criterion: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    checkpoint_path: Path,
    log_path: Path,
) -> dict[str, object]:
    use_amp = args.mps_amp and device.type == "cuda"
    scaler = GradScaler() if use_amp else None
    compile_enabled = resolve_compile_enabled(args.mps_compile, device)
    training_model = torch.compile(model) if compile_enabled else model
    optimizer = Adam(model.parameters(), lr=args.mps_learning_rate, weight_decay=args.mps_weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.mps_lr_scheduler_factor,
        patience=args.mps_lr_scheduler_patience,
        min_lr=args.mps_lr_scheduler_min_lr,
    )
    early_stopping = EarlyStopping(
        patience=args.mps_early_stopping_patience,
        min_delta=args.mps_early_stopping_min_delta,
        mode="max",
    )
    threshold_grid = build_threshold_grid(args.mps_threshold_grid_step)

    best_state = clone_module_state(model)
    best_thresholds = np.full(args.num_labels, 0.5, dtype=np.float32)
    best_score = -1.0
    best_epoch = 0

    with log_path.open("w", newline="") as log_handle:
        writer = csv.writer(log_handle)
        writer.writerow(["epoch", "train_loss", "train_f1_micro", "val_loss", "val_f1_micro", "threshold_mean", "threshold_std", "lr"])

        for epoch in range(1, args.mps_epochs + 1):
            train_loss, train_labels, train_probs = train_epoch_collect_probs(
                training_model,
                train_loader,
                criterion,
                optimizer,
                device,
                grad_clip_norm=args.mps_grad_clip_norm,
                scaler=scaler,
            )
            train_metrics = compute_branch_metrics(train_labels, train_probs, best_thresholds)
            val_loss, val_labels, val_probs = evaluate_with_probs(training_model, val_loader, criterion, device, use_amp=use_amp)
            current_thresholds = tune_thresholds(
                val_labels,
                val_probs,
                threshold_mode=args.mps_threshold_mode,
                target_metric=args.mps_threshold_target_metric,
                threshold_grid=threshold_grid,
            )
            current_metrics = compute_branch_metrics(val_labels, val_probs, current_thresholds)

            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]
            writer.writerow(
                [
                    epoch,
                    train_loss,
                    float(train_metrics["f1_micro"]),
                    val_loss,
                    float(current_metrics["f1_micro"]),
                    float(np.mean(current_thresholds)),
                    float(np.std(current_thresholds)),
                    current_lr,
                ]
            )
            log_handle.flush()

            print(
                f"MPS Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | val_f1_micro={current_metrics['f1_micro']:.4f}"
            )

            if float(current_metrics["f1_micro"]) > best_score:
                best_score = float(current_metrics["f1_micro"])
                best_epoch = epoch
                best_thresholds = current_thresholds.copy()
                best_state = clone_module_state(model)

            if early_stopping(float(current_metrics["f1_micro"])):
                print(f"MPS early stopping at epoch {epoch}.")
                break

    model.load_state_dict(best_state)
    save_branch_checkpoint(checkpoint_path, model, best_thresholds, best_epoch, best_score, args, "mps")
    val_loss, val_labels, val_probs = evaluate_with_probs(model, val_loader, criterion, device, use_amp=use_amp)
    val_metrics = compute_branch_metrics(val_labels, val_probs, best_thresholds)

    return {
        "model": model,
        "thresholds": best_thresholds,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "val_loss": val_loss,
        "val_labels": val_labels,
        "val_probs": val_probs,
        "val_metrics": val_metrics,
    }


def train_ttn_branch(
    model: nn.Module,
    train_loader,
    val_loader,
    criterion: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    checkpoint_path: Path,
    log_path: Path,
) -> dict[str, object]:
    use_amp = args.ttn_amp and device.type == "cuda"
    scaler = GradScaler() if use_amp else None
    compile_enabled = resolve_compile_enabled(args.ttn_compile, device)
    training_model = torch.compile(model) if compile_enabled else model
    optimizer = Adam(model.parameters(), lr=args.ttn_learning_rate, weight_decay=args.ttn_weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.ttn_lr_scheduler_factor,
        patience=args.ttn_lr_scheduler_patience,
        min_lr=args.ttn_lr_scheduler_min_lr,
    )
    early_stopping = EarlyStopping(
        patience=args.ttn_early_stopping_patience,
        min_delta=args.ttn_early_stopping_min_delta,
        mode="max",
    )
    threshold_grid = build_threshold_grid(args.ttn_threshold_grid_step)

    best_state = clone_module_state(model)
    best_thresholds = np.full(args.num_labels, 0.5, dtype=np.float32)
    best_score = -1.0
    best_epoch = 0

    with log_path.open("w", newline="") as log_handle:
        writer = csv.writer(log_handle)
        writer.writerow(["epoch", "train_loss", "train_f1_micro", "val_loss", "val_f1_micro", "score", "threshold_mean", "threshold_std", "lr"])

        for epoch in range(1, args.ttn_epochs + 1):
            train_loss, train_labels, train_probs = train_epoch_collect_probs(
                training_model,
                train_loader,
                criterion,
                optimizer,
                device,
                grad_clip_norm=args.ttn_grad_clip_norm,
                scaler=scaler,
            )
            train_metrics = compute_metrics(train_labels, threshold_predictions(train_probs, 0.5))
            val_loss, val_labels, val_probs = evaluate_with_probs(training_model, val_loader, criterion, device, use_amp=use_amp)
            current_thresholds = tune_thresholds(
                val_labels,
                val_probs,
                threshold_mode=args.ttn_threshold_mode,
                target_metric=args.ttn_threshold_target_metric,
                threshold_grid=threshold_grid,
            )
            current_metrics = compute_branch_metrics(val_labels, val_probs, current_thresholds)
            current_score = select_early_stopping_score(
                current_metrics,
                metric_name=args.ttn_early_stopping_metric,
                blend_alpha=args.ttn_early_stopping_blend_alpha,
            )

            scheduler.step(current_score)
            current_lr = optimizer.param_groups[0]["lr"]
            writer.writerow(
                [
                    epoch,
                    train_loss,
                    float(train_metrics["f1_micro"]),
                    val_loss,
                    float(current_metrics["f1_micro"]),
                    current_score,
                    float(np.mean(current_thresholds)),
                    float(np.std(current_thresholds)),
                    current_lr,
                ]
            )
            log_handle.flush()

            print(
                f"TTN Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | val_f1_micro={current_metrics['f1_micro']:.4f} | score={current_score:.4f}"
            )

            if current_score > best_score:
                best_score = current_score
                best_epoch = epoch
                best_thresholds = current_thresholds.astype(np.float32, copy=True)
                best_state = clone_module_state(model)

            if epoch >= args.ttn_min_epochs_before_stopping and early_stopping(current_score):
                print(f"TTN early stopping at epoch {epoch}.")
                break

    model.load_state_dict(best_state)
    save_branch_checkpoint(checkpoint_path, model, best_thresholds, best_epoch, best_score, args, "ttn")
    val_loss, val_labels, val_probs = evaluate_with_probs(model, val_loader, criterion, device, use_amp=use_amp)
    val_metrics = compute_branch_metrics(val_labels, val_probs, best_thresholds)

    return {
        "model": model,
        "thresholds": best_thresholds,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "val_loss": val_loss,
        "val_labels": val_labels,
        "val_probs": val_probs,
        "val_metrics": val_metrics,
    }


def load_branch_dataset(
    data_dir: Path,
    input_dim: int,
    max_files: int | None,
    apply_snv: bool,
    apply_savgol: bool,
    savgol_window_length: int,
    savgol_polyorder: int,
    cache_path: Path,
    overwrite_cache: bool,
) -> tuple[np.ndarray, np.ndarray]:
    return load_ir_data(
        data_dir=data_dir,
        target_length=input_dim,
        max_files=max_files,
        apply_snv=apply_snv,
        apply_savgol=apply_savgol,
        savgol_window_length=savgol_window_length,
        savgol_polyorder=savgol_polyorder,
        cache_path=cache_path,
        overwrite_cache=overwrite_cache,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0.")
    if not args.data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {args.data_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / Path(PATH_CONFIG.selector_model_path).name
    mps_model_path = args.output_dir / Path(PATH_CONFIG.mps_model_path).name
    ttn_model_path = args.output_dir / Path(PATH_CONFIG.ttn_model_path).name
    summary_path = args.output_dir / Path(PATH_CONFIG.summary_path).name
    threshold_path = args.output_dir / Path(PATH_CONFIG.selector_threshold_path).name
    prediction_artifact_path = args.output_dir / Path(PATH_CONFIG.prediction_artifact_path).name
    mps_log_path = args.output_dir / Path(PATH_CONFIG.mps_log_path).name
    ttn_log_path = args.output_dir / Path(PATH_CONFIG.ttn_log_path).name
    selector_report_path = args.output_dir / "selector_label_report.csv"
    config_path = args.output_dir / "run_config.json"

    mps_cache_path = resolve_branch_cache_path(
        "mps",
        args.mps_cache_path,
        args.input_dim,
        args.max_files,
        args.mps_apply_snv,
        args.mps_apply_savgol,
    )
    ttn_cache_path = resolve_branch_cache_path(
        "ttn",
        args.ttn_cache_path,
        args.input_dim,
        args.max_files,
        args.ttn_apply_snv,
        args.ttn_apply_savgol,
    )

    total_data_files = count_available_data_files(args.data_dir)
    used_data_files = resolve_used_file_count(total_data_files, args.max_files)
    split_suffix = "all" if args.max_files is None else f"files{used_data_files}"
    split_path = args.split_path or (args.output_dir / f"data_split_seed{args.seed}_{split_suffix}.npz")
    describe_args(args, split_path, mps_cache_path, ttn_cache_path)
    config_path.write_text(json.dumps(vars(args), indent=2, default=str) + "\n")

    configure_reproducibility(args.seed, args.deterministic)
    device = resolve_device(args.device)
    print(f"Device: {device}")

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    print("\nLoading MPS dataset...")
    X_mps, y_mps = load_branch_dataset(
        data_dir=args.data_dir,
        input_dim=args.input_dim,
        max_files=args.max_files,
        apply_snv=args.mps_apply_snv,
        apply_savgol=args.mps_apply_savgol,
        savgol_window_length=args.mps_savgol_window_length,
        savgol_polyorder=args.mps_savgol_polyorder,
        cache_path=mps_cache_path,
        overwrite_cache=args.mps_overwrite_cache,
    )
    print("\nLoading TTN dataset...")
    X_ttn, y_ttn = load_branch_dataset(
        data_dir=args.data_dir,
        input_dim=args.input_dim,
        max_files=args.max_files,
        apply_snv=args.ttn_apply_snv,
        apply_savgol=args.ttn_apply_savgol,
        savgol_window_length=args.ttn_savgol_window_length,
        savgol_polyorder=args.ttn_savgol_polyorder,
        cache_path=ttn_cache_path,
        overwrite_cache=args.ttn_overwrite_cache,
    )

    if X_mps.shape != X_ttn.shape:
        raise RuntimeError(f"Branch datasets have different shapes: MPS {X_mps.shape}, TTN {X_ttn.shape}.")
    if y_mps.shape != y_ttn.shape or not np.array_equal(y_mps, y_ttn):
        raise RuntimeError("Branch label arrays do not match exactly; cannot use a shared split safely.")
    if X_mps.shape[1] != args.input_dim:
        raise RuntimeError(f"Loaded spectra have width {X_mps.shape[1]}, expected {args.input_dim}.")
    if y_mps.shape[1] != args.num_labels:
        raise RuntimeError(f"Loaded labels have width {y_mps.shape[1]}, expected {args.num_labels}.")

    split_indices = load_or_create_split_indices(
        labels=y_mps,
        split_path=split_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
        stratify_multilabel=True,
        overwrite=args.overwrite_split,
    )
    train_loader_mps, val_loader_mps, test_loader_mps = prepare_dataloaders_from_split_indices(
        X_mps,
        y_mps,
        split_indices,
        batch_size=args.mps_batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    train_loader_ttn, val_loader_ttn, test_loader_ttn = prepare_dataloaders_from_split_indices(
        X_ttn,
        y_ttn,
        split_indices,
        batch_size=args.ttn_batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    train_labels = y_mps[split_indices["train"]]

    all_group_names = list(FUNCTIONAL_GROUPS.keys())
    ttn_class_indices = [i for i, name in enumerate(all_group_names) if name in TTN_BRANCH_GROUPS]
    mps_class_indices = [i for i, name in enumerate(all_group_names) if name not in TTN_BRANCH_GROUPS]
    print(f"TTN assigned classes ({len(ttn_class_indices)}): {[all_group_names[i] for i in ttn_class_indices]}")
    print(f"MPS assigned classes ({len(mps_class_indices)}): {[all_group_names[i] for i in mps_class_indices]}")

    mps_pos_weight_full = get_pos_weight(train_labels, device, power=1.0, max_value=None)
    ttn_pos_weight_full = get_pos_weight(
        train_labels,
        device,
        power=args.ttn_pos_weight_power,
        max_value=args.ttn_pos_weight_max,
    )
    _idx_mps = torch.tensor(mps_class_indices, dtype=torch.long)
    _idx_ttn = torch.tensor(ttn_class_indices, dtype=torch.long)

    mps_model = build_mps_model(args).to(device)
    ttn_model = build_ttn_model(args).to(device)
    mps_criterion = ClassSubsetLoss(
        nn.BCEWithLogitsLoss(pos_weight=mps_pos_weight_full[_idx_mps]),
        mps_class_indices,
    )
    ttn_criterion = ClassSubsetLoss(
        build_loss(args.ttn_loss_type, pos_weight=ttn_pos_weight_full[_idx_ttn], focal_gamma=args.ttn_focal_gamma),
        ttn_class_indices,
    )

    run_branch_preflight(
        mps_model,
        branch_name="MPS",
        input_dim=args.input_dim,
        num_labels=args.num_labels,
        batch_size=args.preflight_batch_size,
        device=device,
        criterion=mps_criterion,
        use_amp=args.mps_amp and device.type == "cuda",
    )
    run_branch_preflight(
        ttn_model,
        branch_name="TTN",
        input_dim=args.input_dim,
        num_labels=args.num_labels,
        batch_size=args.preflight_batch_size,
        device=device,
        criterion=ttn_criterion,
        use_amp=args.ttn_amp and device.type == "cuda",
    )

    if args.check_only:
        print("\nCheck-only mode finished successfully.")
        return

    start_time = time.time()
    print("\nStarting MPS branch training...")
    mps_result = train_mps_branch(
        mps_model,
        train_loader_mps,
        val_loader_mps,
        mps_criterion,
        device,
        args,
        checkpoint_path=mps_model_path,
        log_path=mps_log_path,
    )

    print("\nStarting TTN branch training...")
    ttn_result = train_ttn_branch(
        ttn_model,
        train_loader_ttn,
        val_loader_ttn,
        ttn_criterion,
        device,
        args,
        checkpoint_path=ttn_model_path,
        log_path=ttn_log_path,
    )

    mps_test_loss, mps_test_labels, mps_test_probs = evaluate_with_probs(
        mps_result["model"],
        test_loader_mps,
        mps_criterion,
        device,
        use_amp=args.mps_amp and device.type == "cuda",
    )
    ttn_test_loss, ttn_test_labels, ttn_test_probs = evaluate_with_probs(
        ttn_result["model"],
        test_loader_ttn,
        ttn_criterion,
        device,
        use_amp=args.ttn_amp and device.type == "cuda",
    )

    if not np.array_equal(mps_test_labels, ttn_test_labels):
        raise RuntimeError("Test label arrays differ across branches; selector merge is invalid.")

    final_selector = select_best_branch_per_label(
        np.asarray(mps_result["val_labels"]),
        np.asarray(mps_result["val_probs"]),
        np.asarray(ttn_result["val_probs"]),
        thresholds_mps=np.asarray(mps_result["thresholds"]),
        thresholds_ttn=np.asarray(ttn_result["thresholds"]),
    )
    best_thresholds = np.asarray(final_selector["selected_thresholds"], dtype=np.float32)
    best_branch_names = list(final_selector["selected_branch_names"])
    selector_val_metrics = final_selector["selected_metrics"]
    write_selector_report(selector_report_path, list(FUNCTIONAL_GROUPS.keys()), final_selector)

    use_mps = np.asarray([branch_name == "mps" for branch_name in best_branch_names], dtype=bool)
    selector_test_probs = np.where(use_mps[None, :], mps_test_probs, ttn_test_probs)
    selector_test_preds = threshold_predictions(selector_test_probs, best_thresholds)
    selector_test_metrics = compute_metrics(mps_test_labels, selector_test_preds)

    torch.save(
        {
            "thresholds": best_thresholds,
            "selected_branch_names": best_branch_names,
            "mps_best_epoch": mps_result["best_epoch"],
            "ttn_best_epoch": ttn_result["best_epoch"],
            "args": vars(args),
        },
        model_path,
    )
    write_threshold_artifact(
        threshold_path,
        list(FUNCTIONAL_GROUPS.keys()),
        best_thresholds,
        "per_class",
        "per_class_f1",
        max(int(mps_result["best_epoch"]), int(ttn_result["best_epoch"])),
    )
    np.savez_compressed(
        prediction_artifact_path,
        val_labels=np.asarray(mps_result["val_labels"], dtype=np.float32),
        val_probs_mps=np.asarray(mps_result["val_probs"], dtype=np.float32),
        val_probs_ttn=np.asarray(ttn_result["val_probs"], dtype=np.float32),
        test_labels=np.asarray(mps_test_labels, dtype=np.float32),
        test_probs_mps=np.asarray(mps_test_probs, dtype=np.float32),
        test_probs_ttn=np.asarray(ttn_test_probs, dtype=np.float32),
        thresholds_mps=np.asarray(mps_result["thresholds"], dtype=np.float32),
        thresholds_ttn=np.asarray(ttn_result["thresholds"], dtype=np.float32),
        thresholds_selector=best_thresholds,
    )

    elapsed_seconds = time.time() - start_time
    print("\nFinal selector evaluation on test set...")
    print(f"MPS best epoch:  {int(mps_result['best_epoch'])}")
    print(f"TTN best epoch:  {int(ttn_result['best_epoch'])}")
    print(f"MPS val F1µ:     {float(mps_result['val_metrics']['f1_micro']):.4f}")
    print(f"TTN val F1µ:     {float(ttn_result['val_metrics']['f1_micro']):.4f}")
    print(f"Selector val F1µ:{selector_val_metrics['f1_micro']:.4f}")
    print(f"MPS test loss:   {mps_test_loss:.4f}")
    print(f"TTN test loss:   {ttn_test_loss:.4f}")
    print(f"Test f1_micro:   {selector_test_metrics['f1_micro']:.4f}")
    print(f"Test f1_macro:   {selector_test_metrics['f1_macro']:.4f}")
    print(f"MPS labels:      {sum(branch == 'mps' for branch in best_branch_names)}")
    print(f"TTN labels:      {sum(branch == 'ttn' for branch in best_branch_names)}")
    print(f"Selector report: {selector_report_path}")

    write_summary(
        summary_path=summary_path,
        elapsed_seconds=elapsed_seconds,
        split_path=split_path,
        mps_result=mps_result,
        ttn_result=ttn_result,
        selector_val_metrics=selector_val_metrics,
        selector_test_metrics=selector_test_metrics,
        final_thresholds=best_thresholds,
        selected_branch_names=best_branch_names,
    )
    print(f"Summary:         {summary_path}")


if __name__ == "__main__":
    main()