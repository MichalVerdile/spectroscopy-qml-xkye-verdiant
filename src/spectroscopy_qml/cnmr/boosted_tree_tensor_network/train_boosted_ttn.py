"""Training entry point for boosted TTN C-NMR classifiers.

This keeps the existing preprocessing and fixed split logic, but replaces the
single multi-output TTN with one XGBoost-like boosted TTN ensemble per functional
group.

For functional group j:

    F_j(x) = base_logit_j + eta * sum_t h_{j,t}(x)

Each h_{j,t} is a scalar-output Tree Tensor Network weak learner. At boosting
round t, the weak learner is trained on the current logistic pseudo-residuals.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = Path(__file__).resolve().parents[4]
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.spectroscopy_qml.cnmr.boosted_tree_tensor_network.boosted_model import (  # noqa: E402
    BoostedTTNBinaryEnsemble,
    BoostedTTNMultiLabelClassifier,
    build_ttn_weak_learner,
)
from src.spectroscopy_qml.cnmr.boosted_tree_tensor_network.helpers.data_loader import (  # noqa: E402
    FUNCTIONAL_GROUPS,
    load_cnmr_data,
    load_or_create_split_indices,
)
from src.spectroscopy_qml.cnmr.boosted_tree_tensor_network.helpers.train_helpers import (  # noqa: E402
    build_threshold_grid,
    compute_metrics,
    count_available_data_files,
    resolve_device,
    resolve_used_file_count,
    threshold_predictions,
    write_threshold_artifact,
)
from src.spectroscopy_qml.cnmr.boosted_tree_tensor_network.model import (  # noqa: E402
    DEFAULT_SEGMENT_STRIDE,
    DEFAULT_SEGMENT_WINDOW_SIZE,
)


class ResidualRegressionDataset(Dataset):
    """Dataset for one boosting round.

    It returns:
        spectrum, pseudo_target, sample_weight

    pseudo_target can be a gradient residual or Newton target.
    """

    def __init__(
        self,
        X: np.ndarray,
        targets: np.ndarray,
        weights: np.ndarray | None = None,
        indices: np.ndarray | None = None,
    ) -> None:
        if indices is None:
            indices = np.arange(len(X), dtype=np.int64)

        self.X = X
        self.targets = targets.astype(np.float32, copy=False)
        self.weights = (
            np.ones_like(self.targets, dtype=np.float32)
            if weights is None
            else weights.astype(np.float32, copy=False)
        )
        self.indices = indices.astype(np.int64, copy=False)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, item: int):
        index = int(self.indices[item])
        spectrum = torch.as_tensor(self.X[index], dtype=torch.float32)
        target = torch.tensor([self.targets[index]], dtype=torch.float32)
        weight = torch.tensor([self.weights[index]], dtype=torch.float32)
        return spectrum, target, weight


def resolve_cache_path(args: argparse.Namespace) -> Path:
    if args.cache_path is not None:
        return args.cache_path

    cache_dir = Path("data/cache")
    file_suffix = "all" if args.max_files is None else f"files{int(args.max_files)}"
    snv_suffix = "snv" if args.apply_snv else "raw"
    return cache_dir / f"cnmr_spectra_len{args.input_dim}_{snv_suffix}_{file_suffix}.npz"


def sigmoid_np(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    out = np.empty_like(logits, dtype=np.float64)
    positive = logits >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_x = np.exp(logits[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out.astype(np.float32)


def compute_base_logit(labels: np.ndarray, eps: float = 1e-4) -> float:
    """Initial logit for one binary label."""
    #positive_rate = float(np.mean(labels))
    #positive_rate = min(max(positive_rate, eps), 1.0 - eps)
    #return float(math.log(positive_rate / (1.0 - positive_rate)))
    return 0.5


def make_boosting_targets(
    labels: np.ndarray,
    logits: np.ndarray,
    *,
    mode: str,
    newton_eps: float,
    newton_clip: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Create pseudo-targets for the next TTN weak learner.

    gradient:
        target = y - sigmoid(F)

    newton:
        target = (y - sigmoid(F)) / (sigmoid(F) * (1 - sigmoid(F)))
        weight = sigmoid(F) * (1 - sigmoid(F))

    The newton version is closer to XGBoost's second-order idea, while the
    gradient version is often more stable for neural weak learners.
    """
    probs = sigmoid_np(logits)
    gradients = labels.astype(np.float32) - probs

    if mode == "gradient":
        return gradients.astype(np.float32), np.ones_like(gradients, dtype=np.float32)

    if mode != "newton":
        raise ValueError(f"Unsupported boost target mode: {mode}")

    hessians = np.maximum(probs * (1.0 - probs), newton_eps).astype(np.float32)
    targets = gradients / hessians
    targets = np.clip(targets, -float(newton_clip), float(newton_clip)).astype(np.float32)
    return targets, hessians


def weighted_mse_loss(prediction: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    loss = (prediction - target).pow(2)
    return (loss * weight).mean()


def build_model_kwargs(args: argparse.Namespace) -> dict:
    """Architecture config for scalar TTN weak learners."""
    return {
        "chi": int(args.chi),
        "input_dim": int(args.input_dim),
        "segment_window_size": int(args.segment_window_size),
        "segment_stride": int(args.segment_stride),
        "segment_mode": str(args.segment_mode),
        "segment_offset": args.segment_offset,
        "segment_state_normalize": bool(args.segment_state_normalize),
        "merge_mode": str(args.merge_mode),
        "merge_residual_weight": float(args.merge_residual_weight),
        "merge_renormalize_output": bool(args.merge_renormalize_output),
        "lorentz_gamma": float(args.lorentz_gamma),
        "lorentz_kernel_half_width": int(args.lorentz_kernel_half_width),
        "lorentz_norm_mode": str(args.lorentz_norm_mode),
    }


def build_weak_learner(args: argparse.Namespace) -> nn.Module:
    return build_ttn_weak_learner(build_model_kwargs(args))


def train_one_weak_learner(
    learner: nn.Module,
    *,
    X_train: np.ndarray,
    train_targets: np.ndarray,
    train_weights: np.ndarray,
    train_indices: np.ndarray,
    X_val: np.ndarray,
    val_targets: np.ndarray,
    val_weights: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, float]:
    """Train one TTN weak learner on one boosting pseudo-target."""
    train_dataset = ResidualRegressionDataset(
        X_train,
        train_targets,
        train_weights,
        indices=train_indices,
    )
    val_dataset = ResidualRegressionDataset(
        X_val,
        val_targets,
        val_weights,
        indices=None,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=True if args.num_workers > 0 else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=True if args.num_workers > 0 else False,
    )

    learner = learner.to(device)
    optimizer = Adam(
        learner.parameters(),
        lr=args.weak_learning_rate,
        weight_decay=args.weak_weight_decay,
    )
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = GradScaler() if use_amp else None

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, args.weak_epochs + 1):
        print(epoch)
        learner.train()
        total_loss = 0.0

        for spectra, target, weight in train_loader:
            spectra = spectra.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            weight = weight.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, enabled=use_amp):
                prediction = learner(spectra)
                loss = weighted_mse_loss(prediction, target, weight)

            if not torch.isfinite(loss.detach()):
                raise RuntimeError("Non-finite weak learner loss detected.")

            if scaler is not None:
                scaler.scale(loss).backward()
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(learner.parameters(), args.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(learner.parameters(), args.grad_clip_norm)
                optimizer.step()

            total_loss += float(loss.item()) * spectra.size(0)

        learner.eval()
        val_loss_total = 0.0
        val_count = 0

        with torch.no_grad():
            for spectra, target, weight in val_loader:
                spectra = spectra.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                weight = weight.to(device, non_blocking=True)

                with autocast(device_type=device.type, enabled=use_amp):
                    prediction = learner(spectra)
                    loss = weighted_mse_loss(prediction, target, weight)

                val_loss_total += float(loss.item()) * spectra.size(0)
                val_count += int(spectra.size(0))

        val_loss = val_loss_total / max(1, val_count)

        if val_loss < best_val_loss - args.weak_early_stopping_min_delta:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in learner.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if args.weak_early_stopping_patience > 0 and patience_counter >= args.weak_early_stopping_patience:
            break

    if best_state is not None:
        learner.load_state_dict(best_state)

    learner.eval()
    return learner, best_val_loss


@torch.no_grad()
def predict_single_learner_logits(
    learner: nn.Module,
    X: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
) -> np.ndarray:
    learner = learner.to(device)
    learner.eval()

    outputs: list[np.ndarray] = []
    dataset = ResidualRegressionDataset(
        X,
        np.zeros(len(X), dtype=np.float32),
        np.ones(len(X), dtype=np.float32),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    for spectra, _, _ in loader:
        spectra = spectra.to(device)
        with autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            logits = learner(spectra)
        outputs.append(logits.detach().float().cpu().numpy().reshape(-1))

    return np.concatenate(outputs, axis=0).astype(np.float32)


@torch.no_grad()
def predict_boosted_model_probs(
    model: BoostedTTNMultiLabelClassifier,
    X: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model = model.to(device)
    model.eval()

    outputs: list[np.ndarray] = []
    dataset = ResidualRegressionDataset(
        X,
        np.zeros(len(X), dtype=np.float32),
        np.ones(len(X), dtype=np.float32),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    for spectra, _, _ in loader:
        spectra = spectra.to(device)
        probs = model(spectra, apply_sigmoid=True)
        outputs.append(probs.detach().float().cpu().numpy())

    return np.concatenate(outputs, axis=0).astype(np.float32)


def single_label_f1(labels: np.ndarray, logits: np.ndarray) -> float:
    probs = sigmoid_np(logits)
    preds = (probs >= 0.5).astype(np.int32)
    return float(f1_score(labels.astype(np.int32), preds, zero_division=0))


def single_label_f1_at_threshold(labels: np.ndarray, logits: np.ndarray, threshold: float) -> float:
    probs = sigmoid_np(logits)
    preds = (probs >= float(threshold)).astype(np.int32)
    return float(f1_score(labels.astype(np.int32), preds, zero_division=0))


def tune_binary_threshold(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold_grid: np.ndarray,
) -> tuple[float, float]:
    """Return the best threshold and F1 for one binary label.

    This is important for imbalanced labels. A rare functional group can have
    useful probabilities while still producing F1=0 at the fixed 0.5 threshold.
    """
    labels_i = labels.astype(np.int32)
    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in threshold_grid:
        preds = (probs >= float(threshold)).astype(np.int32)
        score = float(f1_score(labels_i, preds, zero_division=0))
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold, max(best_f1, 0.0)


def tune_thresholds(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold_mode: str,
    target_metric: str,
    threshold_grid: np.ndarray,
) -> np.ndarray:
    """Tune thresholds locally instead of relying on the placeholder helper.

    The earlier helper version in some experiments returned 0.5 for
    ``threshold_mode='per_class'``. For boosted rare-label models, this makes
    the final metric look like zero even when the ranking is useful.
    """
    n_classes = y_true.shape[1]

    if threshold_mode == "global":
        best_threshold = 0.5
        best_score = -1.0
        for threshold in threshold_grid:
            y_pred = threshold_predictions(y_probs, float(threshold))
            metrics = compute_metrics(y_true, y_pred)
            score = float(metrics[target_metric]) if target_metric in {"f1_micro", "f1_macro"} else float(
                np.mean(np.asarray(metrics["per_class_f1"], dtype=np.float32))
            )
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)
        return np.full(n_classes, best_threshold, dtype=np.float32)

    if threshold_mode != "per_class":
        raise ValueError(f"Unsupported threshold mode: {threshold_mode}")

    thresholds = np.full(n_classes, 0.5, dtype=np.float32)
    for class_index in range(n_classes):
        thresholds[class_index], _ = tune_binary_threshold(
            y_true[:, class_index],
            y_probs[:, class_index],
            threshold_grid,
        )
    return thresholds


def compute_residual_sample_weights(
    labels: np.ndarray,
    *,
    positive_weight_power: float,
    positive_weight_max: float | None,
) -> np.ndarray:
    """Return class-balance weights for residual regression.

    A TTN trained with weighted MSE can otherwise minimize loss by mostly
    learning the many negative samples, especially for labels with <1% positives.
    This is analogous in spirit to using a softer ``scale_pos_weight`` for the
    residual learner.
    """
    if positive_weight_power < 0.0:
        raise ValueError("residual_pos_weight_power must be >= 0.")

    labels_i = (labels >= 0.5).astype(np.float32)
    positives = float(labels_i.sum())
    negatives = float(len(labels_i) - positives)

    if positives <= 0.0 or positive_weight_power == 0.0:
        positive_weight = 1.0
    else:
        positive_weight = (negatives / max(positives, 1.0)) ** float(positive_weight_power)

    if positive_weight_max is not None:
        positive_weight = min(float(positive_weight), float(positive_weight_max))

    weights = np.ones_like(labels_i, dtype=np.float32)
    weights[labels_i >= 0.5] = float(positive_weight)
    return weights


def make_stratified_boost_subsample_indices(
    labels: np.ndarray,
    subsample: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Subsample rows while keeping rare positives whenever possible.

    Pure random 80% subsampling can drop most positives for ultra-rare labels.
    For one-vs-rest boosted specialists, keeping all positives is usually a
    better default.
    """
    n_samples = int(len(labels))
    if subsample >= 1.0:
        return np.arange(n_samples, dtype=np.int64)

    sample_size = max(1, int(round(n_samples * float(subsample))))
    positive_indices = np.flatnonzero(labels >= 0.5)
    negative_indices = np.flatnonzero(labels < 0.5)

    if len(positive_indices) >= sample_size:
        return np.sort(rng.choice(positive_indices, size=sample_size, replace=False)).astype(np.int64)

    remaining = sample_size - len(positive_indices)
    if remaining <= 0 or len(negative_indices) == 0:
        return np.sort(positive_indices).astype(np.int64)

    chosen_negatives = rng.choice(negative_indices, size=min(remaining, len(negative_indices)), replace=False)
    return np.sort(np.concatenate((positive_indices, chosen_negatives))).astype(np.int64)


def parse_specialist_indices(raw: str | None) -> list[int] | None:
    if raw in (None, "", "None"):
        return None
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]


def write_summary(
    path: Path,
    *,
    args: argparse.Namespace,
    elapsed_seconds: float,
    label_names: list[str],
    base_scores: list[float],
    estimators_per_label: list[int],
    val_metrics: dict,
    test_metrics: dict,
    thresholds: np.ndarray,
    split_path: Path,
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("Boosted TTN C-NMR Summary\n")
        handle.write("=" * 80 + "\n")
        handle.write("Model: one boosted scalar-output TTN ensemble per functional group\n")
        handle.write(f"Elapsed seconds:           {elapsed_seconds:.2f}\n")
        handle.write(f"Fixed split artifact:      {split_path}\n")
        handle.write(f"Functional groups:         {len(label_names)}\n")
        handle.write(f"Boosting rounds requested: {args.n_estimators}\n")
        handle.write(f"Weak learner epochs:       {args.weak_epochs}\n")
        handle.write(f"Boost learning rate:       {args.boost_learning_rate}\n")
        handle.write(f"Boost target:              {args.boost_target}\n")
        handle.write(f"Threshold mean:            {float(np.mean(thresholds)):.6f}\n")
        handle.write(f"Threshold std:             {float(np.std(thresholds)):.6f}\n")
        handle.write("\nValidation metrics:\n")
        for key, value in val_metrics.items():
            if not isinstance(value, np.ndarray):
                handle.write(f"  {key}: {float(value):.6f}\n")
        handle.write("\nTest metrics:\n")
        for key, value in test_metrics.items():
            if not isinstance(value, np.ndarray):
                handle.write(f"  {key}: {float(value):.6f}\n")
        handle.write("\nPer-label ensemble sizes:\n")
        for name, base_score, count in zip(label_names, base_scores, estimators_per_label, strict=False):
            handle.write(f"  {name}: base_logit={base_score:.6f}, estimators={count}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train XGBoost-like boosted TTN ensembles, one per functional group."
    )

    # Data / split arguments kept compatible with your existing experiment 10.2 script.
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("c_nmr/boosted_tree_tensor_network/boosted_results"))
    parser.add_argument("--split-path", type=Path, default=None)
    parser.add_argument("--overwrite-split", action="store_true")
    parser.add_argument("--input-dim", type=int, default=600)
    parser.add_argument("--num-labels", type=int, default=len(FUNCTIONAL_GROUPS))
    parser.add_argument("--specialist-indices", type=str, default=None)
    parser.add_argument("--apply-snv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)

    # TTN weak learner architecture.
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
        default="percentile",
    )

    # Boosting arguments.
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--boost-learning-rate", type=float, default=0.05)
    parser.add_argument("--boost-subsample", type=float, default=0.8)
    parser.add_argument("--boost-target", choices=["gradient", "newton"], default="newton")
    parser.add_argument("--newton-eps", type=float, default=1e-3)
    parser.add_argument("--newton-clip", type=float, default=10.0)
    parser.add_argument(
        "--residual-pos-weight-power",
        type=float,
        default=0.5,
        help=(
            "Positive-class weight power for residual MSE. "
            "0 disables it; 0.5 is a safe sqrt(neg/pos) weighting."
        ),
    )
    parser.add_argument(
        "--residual-pos-weight-max",
        type=float,
        default=50.0,
        help="Maximum positive residual weight. Use None only by editing the script.",
    )

    # Weak learner training arguments.
    parser.add_argument("--weak-epochs", type=int, default=10)
    parser.add_argument("--weak-learning-rate", type=float, default=3e-4)
    parser.add_argument("--weak-weight-decay", type=float, default=1e-6)
    parser.add_argument("--weak-early-stopping-patience", type=int, default=10)
    parser.add_argument("--weak-early-stopping-min-delta", type=float, default=1e-5)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)

    # Runtime / metrics.
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--predict-batch-size", type=int, default=8192)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--threshold-mode", choices=["global", "per_class"], default="per_class")
    parser.add_argument(
        "--threshold-target-metric",
        choices=["f1_micro", "f1_macro", "per_class_f1"],
        default="per_class_f1",
    )
    parser.add_argument("--threshold-grid-step", type=float, default=0.02)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {args.data_dir}")
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0.")
    if not 0.0 < args.boost_learning_rate <= 1.0:
        raise ValueError("--boost-learning-rate must be in (0, 1].")
    if not 0.0 < args.boost_subsample <= 1.0:
        raise ValueError("--boost-subsample must be in (0, 1].")
    if args.threshold_mode == "per_class" and args.threshold_target_metric == "f1_micro":
        raise ValueError("--threshold-mode per_class is incompatible with --threshold-target-metric f1_micro.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "run_config.json"
    checkpoint_path = args.output_dir / "boosted_ttn_cnmr.pt"
    log_path = args.output_dir / "boosting_log.csv"
    threshold_path = args.output_dir / "selected_thresholds.json"
    summary_path = args.output_dir / "summary.txt"
    results_path = args.output_dir / "results.npz"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    total_data_files = count_available_data_files(args.data_dir)
    used_data_files = resolve_used_file_count(total_data_files, args.max_files)
    split_suffix = "all" if args.max_files is None else f"files{used_data_files}"
    split_path = args.split_path or (args.output_dir / f"data_split_seed{args.seed}_{split_suffix}.npz")

    print("=" * 80)
    print("Boosted TTN C-NMR Training")
    print("=" * 80)
    print(f"Device:                 {device}")
    print(f"Data dir:               {args.data_dir}")
    print(f"Output dir:             {args.output_dir}")
    print(f"Split path:             {split_path}")
    print(f"Input dim:              {args.input_dim}")
    print(f"Boosting rounds:        {args.n_estimators}")
    print(f"Boost learning rate:    {args.boost_learning_rate}")
    print(f"Boost target:           {args.boost_target}")
    print(f"Newton clip:            {args.newton_clip}")
    print(f"Residual pos wt power:  {args.residual_pos_weight_power}")
    print(f"Residual pos wt max:    {args.residual_pos_weight_max}")
    print(f"Weak learner epochs:    {args.weak_epochs}")
    print(f"One ensemble per label: yes")

    config_payload = vars(args).copy()
    config_payload["split_path"] = str(split_path)
    config_payload["model_kind"] = "boosted_ttn_per_functional_group"
    config_path.write_text(json.dumps(config_payload, indent=2, default=str) + "\n", encoding="utf-8")

    print("\nLoading data...")
    X, y = load_cnmr_data(
        data_dir=args.data_dir,
        target_length=args.input_dim,
        max_files=args.max_files,
        apply_snv=args.apply_snv,
        cache_path=resolve_cache_path(args),
        overwrite_cache=args.overwrite_cache,
    )
    if X.shape[1] != args.input_dim:
        raise RuntimeError(f"Loaded spectra have width {X.shape[1]}, expected {args.input_dim}.")

    split_indices = load_or_create_split_indices(
        labels=y,
        split_path=split_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
        stratify_multilabel=True,
        overwrite=args.overwrite_split,
    )

    all_label_names = list(FUNCTIONAL_GROUPS.keys())
    specialist_indices = parse_specialist_indices(args.specialist_indices)
    if specialist_indices is not None:
        y = y[:, specialist_indices]
        label_names = [all_label_names[index] for index in specialist_indices]
        args.num_labels = len(specialist_indices)
        print(f"Specialist mode: {args.num_labels} labels -> {specialist_indices}")
    else:
        label_names = all_label_names
        args.num_labels = y.shape[1]

    X_train = X[split_indices["train"]].astype(np.float32, copy=False)
    X_val = X[split_indices["val"]].astype(np.float32, copy=False)
    X_test = X[split_indices["test"]].astype(np.float32, copy=False)
    y_train = y[split_indices["train"]].astype(np.float32, copy=False)
    y_val = y[split_indices["val"]].astype(np.float32, copy=False)
    y_test = y[split_indices["test"]].astype(np.float32, copy=False)

    print("\nData split:")
    print(f"  Train: {len(X_train)}")
    print(f"  Val:   {len(X_val)}")
    print(f"  Test:  {len(X_test)}")
    print(f"  Labels:{args.num_labels}")

    boosted_model = BoostedTTNMultiLabelClassifier()
    model_kwargs = build_model_kwargs(args)

    train_logits_all = np.zeros_like(y_train, dtype=np.float32)
    val_logits_all = np.zeros_like(y_val, dtype=np.float32)
    test_logits_all = np.zeros_like(y_test, dtype=np.float32)

    base_scores: list[float] = []
    estimators_per_label: list[int] = []
    threshold_grid = build_threshold_grid(args.threshold_grid_step)

    start_time = time.time()

    with log_path.open("w", newline="", encoding="utf-8") as log_handle:
        writer = csv.writer(log_handle)
        writer.writerow(
            [
                "label_index",
                "label_name",
                "stage",
                "base_logit",
                "train_positive_rate",
                "weak_val_residual_loss",
                "stage_threshold",
                "train_f1_stage_threshold",
                "val_f1_stage_threshold",
                "test_f1_stage_threshold",
                "train_f1_0p5",
                "val_f1_0p5",
                "test_f1_0p5",
                "elapsed_seconds",
            ]
        )

        for label_index, label_name in enumerate(label_names):
            print("\n" + "=" * 80)
            print(f"Training boosted TTN for label {label_index + 1}/{len(label_names)}: {label_name}")
            print("=" * 80)

            y_train_label = y_train[:, label_index]
            y_val_label = y_val[:, label_index]
            y_test_label = y_test[:, label_index]

            base_score = compute_base_logit(y_train_label)
            train_logits = np.full(len(y_train_label), base_score, dtype=np.float32)
            val_logits = np.full(len(y_val_label), base_score, dtype=np.float32)
            test_logits = np.full(len(y_test_label), base_score, dtype=np.float32)

            base_scores.append(float(base_score))
            label_ensemble = BoostedTTNBinaryEnsemble(
                base_score=base_score,
                learning_rate=args.boost_learning_rate,
            )

            train_positive_rate = float(np.mean(y_train_label))
            residual_balance_train = compute_residual_sample_weights(
                y_train_label,
                positive_weight_power=args.residual_pos_weight_power,
                positive_weight_max=args.residual_pos_weight_max,
            )
            residual_balance_val = compute_residual_sample_weights(
                y_val_label,
                positive_weight_power=args.residual_pos_weight_power,
                positive_weight_max=args.residual_pos_weight_max,
            )
            positive_residual_weight = float(residual_balance_train[y_train_label >= 0.5][0]) if np.any(y_train_label >= 0.5) else 1.0
            print(f"Positive rate:        {train_positive_rate:.4f}")
            print(f"Base logit:           {base_score:.4f}")
            print(f"Positive resid weight:{positive_residual_weight:.4f}")

            for stage in range(1, args.n_estimators + 1):
                stage_seed = args.seed + 10_000 * label_index + stage
                torch.manual_seed(stage_seed)

                train_targets, train_weights = make_boosting_targets(
                    y_train_label,
                    train_logits,
                    mode=args.boost_target,
                    newton_eps=args.newton_eps,
                    newton_clip=args.newton_clip,
                )
                val_targets, val_weights = make_boosting_targets(
                    y_val_label,
                    val_logits,
                    mode=args.boost_target,
                    newton_eps=args.newton_eps,
                    newton_clip=args.newton_clip,
                )

                train_weights = train_weights * residual_balance_train
                val_weights = val_weights * residual_balance_val

                train_indices = make_stratified_boost_subsample_indices(
                    y_train_label,
                    args.boost_subsample,
                    rng,
                )

                learner = build_weak_learner(args)
                learner, weak_val_loss = train_one_weak_learner(
                    learner,
                    X_train=X_train,
                    train_targets=train_targets,
                    train_weights=train_weights,
                    train_indices=train_indices,
                    X_val=X_val,
                    val_targets=val_targets,
                    val_weights=val_weights,
                    args=args,
                    device=device,
                )

                use_amp = bool(args.amp and device.type == "cuda")
                pred_train = predict_single_learner_logits(
                    learner,
                    X_train,
                    batch_size=args.predict_batch_size,
                    device=device,
                    use_amp=use_amp,
                )
                pred_val = predict_single_learner_logits(
                    learner,
                    X_val,
                    batch_size=args.predict_batch_size,
                    device=device,
                    use_amp=use_amp,
                )
                pred_test = predict_single_learner_logits(
                    learner,
                    X_test,
                    batch_size=args.predict_batch_size,
                    device=device,
                    use_amp=use_amp,
                )

                train_logits += args.boost_learning_rate * pred_train
                val_logits += args.boost_learning_rate * pred_val
                test_logits += args.boost_learning_rate * pred_test

                stage_threshold, val_f1 = tune_binary_threshold(
                    y_val_label,
                    sigmoid_np(val_logits),
                    threshold_grid,
                )
                train_f1 = single_label_f1_at_threshold(y_train_label, train_logits, stage_threshold)
                test_f1 = single_label_f1_at_threshold(y_test_label, test_logits, stage_threshold)

                train_f1_0p5 = single_label_f1(y_train_label, train_logits)
                val_f1_0p5 = single_label_f1(y_val_label, val_logits)
                test_f1_0p5 = single_label_f1(y_test_label, test_logits)

                learner = learner.to("cpu")
                label_ensemble.add_learner(learner, freeze=True)

                if device.type == "cuda":
                    torch.cuda.empty_cache()

                elapsed = time.time() - start_time
                writer.writerow(
                    [
                        label_index,
                        label_name,
                        stage,
                        base_score,
                        train_positive_rate,
                        weak_val_loss,
                        stage_threshold,
                        train_f1,
                        val_f1,
                        test_f1,
                        train_f1_0p5,
                        val_f1_0p5,
                        test_f1_0p5,
                        elapsed,
                    ]
                )
                log_handle.flush()

                print(
                    f"Label {label_index:02d} {label_name:<24} | "
                    f"stage {stage:03d}/{args.n_estimators} | "
                    f"weak_val_mse={weak_val_loss:.6f} | "
                    f"thr={stage_threshold:.2f} | "
                    f"train_f1={train_f1:.4f} | "
                    f"val_f1={val_f1:.4f} | "
                    f"test_f1={test_f1:.4f} | "
                    f"val_f1@0.5={val_f1_0p5:.4f}"
                )

            estimators_per_label.append(len(label_ensemble.learners))
            boosted_model.add_label_ensemble(label_ensemble)

            train_logits_all[:, label_index] = train_logits
            val_logits_all[:, label_index] = val_logits
            test_logits_all[:, label_index] = test_logits

    print("\nTuning thresholds on validation probabilities...")
    threshold_grid = build_threshold_grid(args.threshold_grid_step)
    train_probs = sigmoid_np(train_logits_all)
    val_probs = sigmoid_np(val_logits_all)
    test_probs = sigmoid_np(test_logits_all)

    thresholds = tune_thresholds(
        y_val,
        val_probs,
        threshold_mode=args.threshold_mode,
        target_metric=args.threshold_target_metric,
        threshold_grid=threshold_grid,
    )

    train_preds = threshold_predictions(train_probs, thresholds)
    val_preds = threshold_predictions(val_probs, thresholds)
    test_preds = threshold_predictions(test_probs, thresholds)

    train_metrics = compute_metrics(y_train, train_preds)
    val_metrics = compute_metrics(y_val, val_preds)
    test_metrics = compute_metrics(y_test, test_preds)

    print("\n" + "=" * 80)
    print("Final boosted TTN results")
    print("=" * 80)
    print(f"Train F1 micro: {train_metrics['f1_micro']:.4f}")
    print(f"Train F1 macro: {train_metrics['f1_macro']:.4f}")
    print(f"Val F1 micro:   {val_metrics['f1_micro']:.4f}")
    print(f"Val F1 macro:   {val_metrics['f1_macro']:.4f}")
    print(f"Test F1 micro:  {test_metrics['f1_micro']:.4f}")
    print(f"Test F1 macro:  {test_metrics['f1_macro']:.4f}")

    write_threshold_artifact(
        threshold_path,
        label_names,
        thresholds.astype(np.float32),
        args.threshold_mode,
        args.threshold_target_metric,
        best_epoch=args.n_estimators,
    )

    checkpoint_payload = {
        "state_dict": boosted_model.state_dict(),
        "model_kwargs": model_kwargs,
        "base_scores": base_scores,
        "learning_rate": float(args.boost_learning_rate),
        "estimators_per_label": estimators_per_label,
        "label_names": label_names,
        "specialist_indices": specialist_indices,
        "run_config": config_payload,
    }
    torch.save(checkpoint_payload, checkpoint_path)

    np.savez_compressed(
        results_path,
        train_probs=train_probs,
        val_probs=val_probs,
        test_probs=test_probs,
        train_preds=train_preds,
        val_preds=val_preds,
        test_preds=test_preds,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        thresholds=thresholds.astype(np.float32),
    )

    elapsed_seconds = time.time() - start_time
    write_summary(
        summary_path,
        args=args,
        elapsed_seconds=elapsed_seconds,
        label_names=label_names,
        base_scores=base_scores,
        estimators_per_label=estimators_per_label,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        thresholds=thresholds,
        split_path=split_path,
    )

    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"Thresholds: {threshold_path}")
    print(f"Results:    {results_path}")
    print(f"Log:        {log_path}")
    print(f"Summary:    {summary_path}")


if __name__ == "__main__":
    main()
