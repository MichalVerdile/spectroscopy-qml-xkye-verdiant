"""TTN 10.2 training branch for the MPS + TTN joint pipeline.

Loads data via :mod:`shared_data` so that TTN 10.2 trains on exactly the same
preprocessed spectra and the same train/val/test split as the MPS model.

All training-utility functions are inlined here so that this module does not
depend on the problematic ``mps_encoder`` → ``experiment5.data_loader`` import
chain.  Only the TTN 10.2 model and the loss builder are imported from the
existing experiment modules.

Usage (standalone)::

    python train_ttn102.py \\
        --data-dir data/raw \\
        --split-path src/spectroscopy_qml/ir/MPS_TTN/results/shared_split.npz \\
        --spectra-cache src/spectroscopy_qml/ir/MPS_TTN/results/shared_spectra.npz \\
        --output-dir src/spectroscopy_qml/ir/MPS_TTN/results/ttn102

Or call :func:`run_ttn102_training` from ``run.py`` with pre-loaded data.
"""

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
from torch.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = Path(__file__).resolve().parents[4]
for _p in (str(CURRENT_DIR), str(SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared_data import (  # noqa: E402
    FUNCTIONAL_GROUPS,
    load_or_create_split_indices,
    load_shared_data,
    prepare_dataloaders_from_split_indices,
)
from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment5.losses import (  # noqa: E402
    build_loss,
)
from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment10_2.model import (  # noqa: E402
    DEFAULT_SEGMENT_STRIDE,
    DEFAULT_SEGMENT_WINDOW_SIZE,
    TTNIRClassifier10_2,
)

# ---------------------------------------------------------------------------
# Inlined training utilities (sourced from experiment5/train.py and
# experiment10/train.py to avoid the mps_encoder import dependency).
# ---------------------------------------------------------------------------

class EarlyStopping:
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


def resolve_compile_enabled(requested_compile: bool, device: torch.device) -> bool:
    return bool(requested_compile and device.type != "mps")


def build_threshold_grid(step: float) -> np.ndarray:
    if not 0.0 < step < 1.0:
        raise ValueError("threshold_grid_step must be in (0, 1).")
    grid = np.arange(step, 1.0, step, dtype=np.float32)
    return np.unique(np.clip(grid, step, 0.95))


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
    if power != 1.0:
        pos_weight = np.power(pos_weight, power, dtype=np.float32)
    if max_value is not None:
        pos_weight = np.clip(pos_weight, 1.0, max_value)
    return torch.as_tensor(pos_weight, dtype=torch.float32, device=device)


def _score_threshold_metric(metrics: dict, target_metric: str) -> float:
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
        best_threshold, best_score = 0.5, -1.0
        for thr in threshold_grid:
            score = _score_threshold_metric(
                compute_metrics(y_true, threshold_predictions(y_probs, thr)), target_metric
            )
            if score > best_score:
                best_score = score
                best_threshold = float(thr)
        return np.full(n_classes, best_threshold, dtype=np.float32)

    if target_metric == "f1_micro":
        raise ValueError("threshold_mode='per_class' requires target_metric != 'f1_micro'.")
    thresholds = np.full(n_classes, 0.5, dtype=np.float32)
    for idx in range(n_classes):
        class_labels = y_true[:, idx]
        if class_labels.sum() == 0:
            thresholds[idx] = 0.95
            continue
        best_thr, best_score = 0.5, -1.0
        for thr in threshold_grid:
            score = float(f1_score(class_labels, threshold_predictions(y_probs[:, idx], thr), zero_division=0))
            if score > best_score:
                best_score = score
                best_thr = float(thr)
        thresholds[idx] = best_thr
    return thresholds


def select_early_stopping_score(
    metrics: dict,
    metric_name: str,
    blend_alpha: float,
) -> float:
    if metric_name == "f1_micro":
        return float(metrics["f1_micro"])
    if metric_name == "f1_macro":
        return float(metrics["f1_macro"])
    if metric_name == "blended_f1":
        return blend_alpha * float(metrics["f1_macro"]) + (1.0 - blend_alpha) * float(metrics["f1_micro"])
    raise ValueError(f"Unsupported early stopping metric: {metric_name}")


def sanitize_binary_targets_and_probs(
    labels: np.ndarray, probs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    safe_labels = np.clip(np.nan_to_num(labels, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    safe_labels = (safe_labels >= 0.5).astype(np.float32)
    safe_probs = np.clip(np.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0).astype(np.float32)
    return safe_labels, safe_probs


def ensure_finite_tensor(tensor: torch.Tensor, name: str, stage: str, batch_index: int) -> None:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(
            f"Non-finite {name} detected during {stage} at batch {batch_index} on {tensor.device}."
        )


def train_epoch_amp(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float | None = None,
    scaler: GradScaler | None = None,
) -> tuple[float, dict]:
    model.train()
    total_loss = 0.0
    labels_list: list[np.ndarray] = []
    probs_list: list[np.ndarray] = []
    use_amp = scaler is not None

    for batch_idx, (spectra, labels) in enumerate(dataloader, start=1):
        spectra = spectra.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(spectra)
            loss = criterion(logits, labels)

        ensure_finite_tensor(logits, "logits", "train", batch_idx)
        ensure_finite_tensor(loss.detach(), "loss", "train", batch_idx)

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
    metrics = compute_metrics(all_labels, threshold_predictions(all_probs, 0.5))
    return total_loss / len(dataloader.dataset), metrics


@torch.no_grad()
def evaluate_with_probs_amp(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    use_amp: bool = False,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    labels_list: list[np.ndarray] = []
    probs_list: list[np.ndarray] = []

    for batch_idx, (spectra, labels) in enumerate(dataloader, start=1):
        spectra = spectra.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(spectra)
            loss = criterion(logits, labels)

        ensure_finite_tensor(logits, "logits", "eval", batch_idx)
        ensure_finite_tensor(loss.detach(), "loss", "eval", batch_idx)

        total_loss += loss.item() * spectra.size(0)
        labels_list.append(labels.cpu().numpy())
        probs_list.append(torch.sigmoid(logits.float()).cpu().numpy())

    all_labels = np.concatenate(labels_list, axis=0)
    all_probs = np.concatenate(probs_list, axis=0)
    all_labels, all_probs = sanitize_binary_targets_and_probs(all_labels, all_probs)
    return total_loss / len(dataloader.dataset), all_labels, all_probs


def write_epoch_details(
    handle,
    epoch: int,
    label_names: list[str],
    thresholds: np.ndarray,
    train_metrics: dict,
    val_metrics: dict,
    early_stopping_score: float,
) -> None:
    record = {
        "epoch": epoch,
        "early_stopping_score": float(early_stopping_score),
        "thresholds": {n: float(v) for n, v in zip(label_names, thresholds.tolist(), strict=False)},
        "train": {"f1_micro": float(train_metrics["f1_micro"]), "f1_macro": float(train_metrics["f1_macro"])},
        "val": {
            "f1_micro": float(val_metrics["f1_micro"]),
            "f1_macro": float(val_metrics["f1_macro"]),
            "precision_micro": float(val_metrics["precision_micro"]),
            "recall_micro": float(val_metrics["recall_micro"]),
            "per_class_f1": {
                n: float(v)
                for n, v in zip(
                    label_names,
                    np.asarray(val_metrics["per_class_f1"], dtype=np.float32).tolist(),
                    strict=False,
                )
            },
        },
    }
    handle.write(json.dumps(record) + "\n")


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
        "thresholds": {n: float(v) for n, v in zip(label_names, thresholds.tolist(), strict=False)},
    }
    threshold_path.write_text(json.dumps(payload, indent=2) + "\n")


def write_summary(
    summary_path: Path,
    elapsed_seconds: float,
    completed_epochs: int,
    requested_epochs: int,
    split_path: Path,
    best_epoch: int,
    best_score: float,
    best_metric_name: str,
    best_val_loss: float,
    test_loss: float,
    test_metrics: dict,
    final_thresholds: np.ndarray,
) -> None:
    with summary_path.open("w") as fh:
        fh.write("TTN 10.2 (MPS_TTN branch) Summary\n")
        fh.write("=" * 80 + "\n")
        fh.write(f"Elapsed seconds:           {elapsed_seconds:.2f}\n")
        fh.write(f"Epochs completed:          {completed_epochs}/{requested_epochs}\n")
        fh.write(f"Fixed split artifact:      {split_path}\n")
        fh.write(f"Best epoch:                {best_epoch}\n")
        fh.write(f"Best early-stop score:     {best_score:.6f}\n")
        fh.write(f"Best score metric:         {best_metric_name}\n")
        fh.write(f"Best val loss:             {best_val_loss:.6f}\n")
        fh.write(f"Threshold mean:            {float(np.mean(final_thresholds)):.6f}\n")
        fh.write(f"Threshold std:             {float(np.std(final_thresholds)):.6f}\n")
        fh.write(f"Test loss:                 {test_loss:.6f}\n")
        fh.write(f"Test f1_micro:             {float(test_metrics['f1_micro']):.6f}\n")
        fh.write(f"Test f1_macro:             {float(test_metrics['f1_macro']):.6f}\n")
        fh.write(f"Test precision_micro:      {float(test_metrics['precision_micro']):.6f}\n")
        fh.write(f"Test recall_micro:         {float(test_metrics['recall_micro']):.6f}\n")


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

def build_model(args: argparse.Namespace) -> TTNIRClassifier10_2:
    return TTNIRClassifier10_2(
        num_labels=args.num_labels,
        chi=args.chi,
        input_dim=args.input_dim,
        segment_window_size=args.segment_window_size,
        segment_stride=args.segment_stride,
        segment_mode=args.segment_mode,
        segment_offset=args.segment_offset,
        segment_state_normalize=args.segment_state_normalize,
        merge_mode=args.merge_mode,
        merge_residual_weight=args.merge_residual_weight,
        merge_renormalize_output=args.merge_renormalize_output,
        lorentz_gamma=args.lorentz_gamma,
        lorentz_kernel_half_width=args.lorentz_kernel_half_width,
        lorentz_norm_mode=args.lorentz_norm_mode,
    )


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train TTN 10.2 on the shared MPS+TTN data split. "
            "Architecture: Lorentzian feature map + segmented TTN + linear readout."
        )
    )
    # --- Shared data ---
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--spectra-cache",
        type=Path,
        default=Path("src/spectroscopy_qml/ir/MPS_TTN/results/shared_spectra.npz"),
    )
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument(
        "--split-path",
        type=Path,
        default=Path("src/spectroscopy_qml/ir/MPS_TTN/results/shared_split.npz"),
    )
    parser.add_argument("--overwrite-split", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/spectroscopy_qml/ir/MPS_TTN/results/ttn102"),
    )
    # --- Preprocessing ---
    parser.add_argument("--apply-snv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--apply-savgol", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-files", type=int, default=None)
    # --- Split ratios ---
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    # --- Model ---
    parser.add_argument("--input-dim", type=int, default=1800)
    parser.add_argument("--num-labels", type=int, default=len(FUNCTIONAL_GROUPS))
    parser.add_argument("--chi", type=int, default=64)
    parser.add_argument("--segment-window-size", type=int, default=DEFAULT_SEGMENT_WINDOW_SIZE)
    parser.add_argument("--segment-stride", type=int, default=DEFAULT_SEGMENT_STRIDE)
    parser.add_argument("--segment-mode", choices=["overlap", "dual_offset"], default="overlap")
    parser.add_argument("--segment-offset", type=int, default=None)
    parser.add_argument("--segment-state-normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--merge-mode", choices=["strict", "relaxed"], default="relaxed")
    parser.add_argument("--merge-residual-weight", type=float, default=0.1)
    parser.add_argument("--merge-renormalize-output", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lorentz-gamma", type=float, default=3.0)
    parser.add_argument("--lorentz-kernel-half-width", type=int, default=15)
    parser.add_argument(
        "--lorentz-norm-mode",
        choices=["max_abs", "z_score", "percentile"],
        default="max_abs",
    )
    # --- Loss ---
    parser.add_argument("--loss-type", choices=["bce", "focal"], default="bce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--pos-weight-power", type=float, default=0.5)
    parser.add_argument("--pos-weight-max", type=float, default=None)
    parser.add_argument("--hard-class-boost", type=float, default=1.0)
    parser.add_argument("--hard-class-indices", type=str, default=None)
    # --- Training ---
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--lr-scheduler-factor", type=float, default=0.9)
    parser.add_argument("--lr-scheduler-patience", type=int, default=5)
    parser.add_argument("--lr-scheduler-min-lr", type=float, default=1e-6)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    # --- Thresholds ---
    parser.add_argument("--threshold-mode", choices=["global", "per_class"], default="per_class")
    parser.add_argument(
        "--threshold-target-metric",
        choices=["f1_micro", "f1_macro", "per_class_f1"],
        default="per_class_f1",
    )
    parser.add_argument("--threshold-grid-step", type=float, default=0.02)
    # --- Early stopping ---
    parser.add_argument(
        "--early-stopping-metric",
        choices=["f1_micro", "f1_macro", "blended_f1"],
        default="blended_f1",
    )
    parser.add_argument("--early-stopping-blend-alpha", type=float, default=0.5)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--min-epochs-before-stopping", type=int, default=30)
    # --- Misc ---
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    return parser


# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------

def run_ttn102_training(
    X: np.ndarray,
    y: np.ndarray,
    split_indices: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> None:
    """Train TTN 10.2 on pre-loaded data and a pre-computed split.

    Args:
        X: Preprocessed spectra (n_samples, spectrum_length).
        y: Label matrix (n_samples, num_classes).
        split_indices: Dict with keys "train", "val", "test" → index arrays.
        args: Parsed argument namespace (see :func:`build_parser`).
    """
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = args.output_dir / "ttn_ir_best.pt"
    log_path = args.output_dir / "training_log.csv"
    details_path = args.output_dir / "training_details.jsonl"
    summary_path = args.output_dir / "summary.txt"
    threshold_path = args.output_dir / "selected_thresholds.json"
    config_path = args.output_dir / "run_config.json"
    config_path.write_text(json.dumps(vars(args), indent=2, default=str) + "\n")

    print("\n" + "=" * 80)
    print("TTN 10.2 Training Branch")
    print("=" * 80)
    print(f"Output dir:               {args.output_dir}")
    print(f"Split path:               {args.split_path}")
    print(f"Input dim:                {args.input_dim}")
    print(f"Num labels:               {args.num_labels}")
    print(f"Chi:                      {args.chi}")
    print(f"Lorentz gamma:            {args.lorentz_gamma}")
    print(f"Lorentz kernel hw:        {args.lorentz_kernel_half_width}")
    print(f"Lorentz norm mode:        {args.lorentz_norm_mode}")
    print(f"Window size:              {args.segment_window_size}")
    print(f"Stride:                   {args.segment_stride}")
    print(f"Batch size:               {args.batch_size}")
    print(f"Epochs:                   {args.epochs}")
    print(f"AMP:                      {args.amp}")
    print(f"compile:                  {args.compile}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    print(f"Device:                   {device}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    # Build dataloaders from the shared split
    train_loader, val_loader, test_loader = prepare_dataloaders_from_split_indices(
        X,
        y,
        split_indices,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    # Class weights from training labels only
    train_labels = y[split_indices["train"]]
    pos_weight = get_pos_weight(
        train_labels, device, power=args.pos_weight_power, max_value=args.pos_weight_max
    )
    if args.hard_class_boost != 1.0 and args.hard_class_indices:
        for idx in (int(i) for i in args.hard_class_indices.split(",")):
            pos_weight[idx] *= args.hard_class_boost

    criterion = build_loss(args.loss_type, pos_weight=pos_weight, focal_gamma=args.focal_gamma)

    model = build_model(args).to(device)
    raw_model = model

    # Synthetic preflight
    print("\n[Preflight] Synthetic forward check")
    model.eval()
    with torch.no_grad():
        dummy = torch.randn(4, args.input_dim, device=device)
        dummy_logits = model(dummy)
        if not torch.isfinite(dummy_logits).all():
            raise RuntimeError("Preflight: non-finite logits on synthetic batch.")
    print(f"  OK - output shape {tuple(dummy_logits.shape)}")

    use_amp = args.amp and device.type == "cuda"
    scaler: GradScaler | None = GradScaler() if use_amp else None
    if use_amp:
        print("Mixed precision (AMP) enabled.")

    compile_enabled = resolve_compile_enabled(args.compile, device)
    if compile_enabled:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)
        print("Compilation done.")

    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.lr_scheduler_factor,
        patience=args.lr_scheduler_patience,
        min_lr=args.lr_scheduler_min_lr,
    )
    early_stopping = EarlyStopping(
        patience=args.early_stopping_patience,
        min_delta=args.early_stopping_min_delta,
        mode="max",
    )
    threshold_grid = build_threshold_grid(args.threshold_grid_step)
    label_names = list(FUNCTIONAL_GROUPS.keys())

    best_score = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    completed_epochs = 0
    best_thresholds = np.full(args.num_labels, 0.5, dtype=np.float32)

    print("\nStarting training...")
    start_time = time.time()

    with log_path.open("w", newline="") as log_handle, details_path.open("w") as details_handle:
        writer = csv.writer(log_handle)
        writer.writerow([
            "epoch", "train_loss", "train_f1_micro", "train_f1_macro",
            "val_loss", "val_f1_micro", "val_f1_macro",
            "early_stopping_score", "threshold_mean", "threshold_std", "lr",
        ])

        for epoch in range(1, args.epochs + 1):
            train_loss, train_metrics = train_epoch_amp(
                model, train_loader, criterion, optimizer, device,
                grad_clip_norm=args.grad_clip_norm, scaler=scaler,
            )
            val_loss, val_labels, val_probs = evaluate_with_probs_amp(
                model, val_loader, criterion, device, use_amp=use_amp,
            )
            current_thresholds = tune_thresholds(
                val_labels, val_probs,
                threshold_mode=args.threshold_mode,
                target_metric=args.threshold_target_metric,
                threshold_grid=threshold_grid,
            )
            val_preds = threshold_predictions(val_probs, current_thresholds)
            val_metrics = compute_metrics(val_labels, val_preds)
            es_score = select_early_stopping_score(
                val_metrics,
                metric_name=args.early_stopping_metric,
                blend_alpha=args.early_stopping_blend_alpha,
            )

            scheduler.step(es_score)
            current_lr = optimizer.param_groups[0]["lr"]

            writer.writerow([
                epoch,
                train_loss, float(train_metrics["f1_micro"]), float(train_metrics["f1_macro"]),
                val_loss, float(val_metrics["f1_micro"]), float(val_metrics["f1_macro"]),
                es_score, float(np.mean(current_thresholds)), float(np.std(current_thresholds)),
                current_lr,
            ])
            log_handle.flush()

            write_epoch_details(
                details_handle, epoch=epoch, label_names=label_names,
                thresholds=current_thresholds, train_metrics=train_metrics,
                val_metrics=val_metrics, early_stopping_score=es_score,
            )
            details_handle.flush()

            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"val_f1_micro={val_metrics['f1_micro']:.4f} | "
                f"val_f1_macro={val_metrics['f1_macro']:.4f} | "
                f"score={es_score:.4f} | lr={current_lr:.2e}"
            )

            completed_epochs = epoch
            if es_score > best_score:
                best_score = es_score
                best_val_loss = val_loss
                best_epoch = epoch
                best_thresholds = current_thresholds.astype(np.float32, copy=True)
                # Strip _orig_mod. prefixes added by torch.compile so that the
                # checkpoint can be loaded with plain load_state_dict later.
                raw_state = {
                    k.replace("_orig_mod.", ""): v
                    for k, v in model.state_dict().items()
                }
                torch.save(raw_state, checkpoint_path)
                write_threshold_artifact(
                    threshold_path, label_names, best_thresholds,
                    args.threshold_mode, args.threshold_target_metric, best_epoch,
                )

            if epoch >= args.min_epochs_before_stopping and early_stopping(es_score):
                print(f"Stopping early at epoch {epoch}.")
                break
            if epoch >= args.min_epochs_before_stopping and early_stopping.counter > 0:
                print(f"EarlyStopping counter: {early_stopping.counter}/{early_stopping.patience}")

    print("\nTraining complete. Evaluating on test set...")

    best_state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    raw_model.load_state_dict(best_state)

    test_loss, test_labels, test_probs = evaluate_with_probs_amp(
        model, test_loader, criterion, device, use_amp=use_amp,
    )
    test_preds = threshold_predictions(test_probs, best_thresholds)
    test_metrics = compute_metrics(test_labels, test_preds)

    elapsed = time.time() - start_time
    print(f"Best epoch:      {best_epoch}")
    print(f"Best score:      {best_score:.4f} ({args.early_stopping_metric})")
    print(f"Test f1_micro:   {test_metrics['f1_micro']:.4f}")
    print(f"Test f1_macro:   {test_metrics['f1_macro']:.4f}")

    write_summary(
        summary_path=summary_path,
        elapsed_seconds=elapsed,
        completed_epochs=completed_epochs,
        requested_epochs=args.epochs,
        split_path=args.split_path,
        best_epoch=best_epoch,
        best_score=best_score,
        best_metric_name=args.early_stopping_metric,
        best_val_loss=best_val_loss,
        test_loss=test_loss,
        test_metrics=test_metrics,
        final_thresholds=best_thresholds,
    )
    print(f"Summary saved:   {summary_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_parser().parse_args()

    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0.")
    if args.threshold_mode == "per_class" and args.threshold_target_metric == "f1_micro":
        raise ValueError("--threshold-mode per_class is incompatible with --threshold-target-metric f1_micro.")

    X, y = load_shared_data(
        data_dir=args.data_dir,
        apply_snv=args.apply_snv,
        apply_savgol=args.apply_savgol,
        max_files=args.max_files,
        cache_path=args.spectra_cache,
        overwrite_cache=args.overwrite_cache,
    )

    split_indices = load_or_create_split_indices(
        labels=y,
        split_path=args.split_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
        overwrite=args.overwrite_split,
    )

    run_ttn102_training(X=X, y=y, split_indices=split_indices, args=args)


if __name__ == "__main__":
    main()
