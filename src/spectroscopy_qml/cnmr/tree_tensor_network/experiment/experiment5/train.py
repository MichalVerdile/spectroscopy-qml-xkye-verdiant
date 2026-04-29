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

from data_loader import (  # noqa: E402
    FUNCTIONAL_GROUPS,
    load_ir_data,
    load_or_create_split_indices,
    prepare_dataloaders_from_split_indices,
)
from losses import build_loss  # noqa: E402
from model import TTNIRClassifier5  # noqa: E402


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train experiment5: structured TTN-inspired IR classifier with fixed splits."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment5/results"),
    )
    parser.add_argument("--split-path", type=Path, default=None)
    parser.add_argument("--overwrite-split", action="store_true")
    parser.add_argument("--input-dim", type=int, default=1800)
    parser.add_argument("--num-labels", type=int, default=len(FUNCTIONAL_GROUPS))
    parser.add_argument("--chi", type=int, default=64)
    parser.add_argument("--segment-window-size", type=int, default=64)
    parser.add_argument("--segment-stride", type=int, default=32)
    parser.add_argument("--segment-mode", choices=["overlap", "dual_offset"], default="overlap")
    parser.add_argument("--segment-offset", type=int, default=None)
    parser.add_argument("--embedding-scale", type=float, default=0.1)
    parser.add_argument("--x-max-mode", choices=["per_sample", "global"], default="per_sample")
    parser.add_argument("--global-x-max", type=float, default=None)
    parser.add_argument("--leaf-hidden-dim", type=int, default=None)
    parser.add_argument("--leaf-dropout", type=float, default=0.1)
    parser.add_argument("--leaf-renormalize-output", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--merge-mode", choices=["strict", "relaxed"], default="relaxed")
    parser.add_argument("--merge-residual-weight", type=float, default=0.15)
    parser.add_argument("--merge-renormalize-output", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--readout-hidden-dim", type=int, default=None)
    parser.add_argument("--readout-dropout", type=float, default=0.1)
    parser.add_argument("--apply-snv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--lr-scheduler-factor", type=float, default=0.9)
    parser.add_argument("--lr-scheduler-patience", type=int, default=5)
    parser.add_argument("--lr-scheduler-min-lr", type=float, default=1e-6)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--loss-type", choices=["bce"], default="bce")
    parser.add_argument("--pos-weight-power", type=float, default=0.5)
    parser.add_argument("--pos-weight-max", type=float, default=None)
    parser.add_argument("--threshold-mode", choices=["global", "per_class"], default="per_class")
    parser.add_argument(
        "--threshold-target-metric",
        choices=["f1_micro", "f1_macro", "per_class_f1"],
        default="per_class_f1",
    )
    parser.add_argument("--threshold-grid-step", type=float, default=0.05)
    parser.add_argument(
        "--early-stopping-metric",
        choices=["f1_micro", "f1_macro", "blended_f1"],
        default="blended_f1",
    )
    parser.add_argument(
        "--early-stopping-blend-alpha",
        type=float,
        default=0.5,
        help="Macro weight for blended_f1; micro uses (1 - alpha).",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--min-epochs-before-stopping", type=int, default=30)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--preflight-batch-size", type=int, default=4)
    return parser


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


def build_model(args: argparse.Namespace) -> TTNIRClassifier5:
    return TTNIRClassifier5(
        num_labels=args.num_labels,
        chi=args.chi,
        input_dim=args.input_dim,
        segment_window_size=args.segment_window_size,
        segment_stride=args.segment_stride,
        segment_mode=args.segment_mode,
        segment_offset=args.segment_offset,
        embedding_scale=args.embedding_scale,
        x_max_mode=args.x_max_mode,
        global_x_max=args.global_x_max,
        leaf_hidden_dim=args.leaf_hidden_dim,
        leaf_dropout=args.leaf_dropout,
        leaf_renormalize_output=args.leaf_renormalize_output,
        merge_mode=args.merge_mode,
        merge_residual_weight=args.merge_residual_weight,
        merge_renormalize_output=args.merge_renormalize_output,
        readout_hidden_dim=args.readout_hidden_dim,
        readout_dropout=args.readout_dropout,
    )


def describe_args(args: argparse.Namespace, split_path: Path) -> None:
    print("=" * 80)
    print("TTN IR Experiment5 Training")
    print("=" * 80)
    print(f"Data dir:                 {args.data_dir}")
    print(f"Output dir:               {args.output_dir}")
    print(f"Split path:               {split_path}")
    print(f"Input dim:                {args.input_dim}")
    print(f"Num labels:               {args.num_labels}")
    print(f"Chi:                      {args.chi}")
    print(f"Segment mode:             {args.segment_mode}")
    print(f"Window size:              {args.segment_window_size}")
    print(f"Stride:                   {args.segment_stride}")
    print(f"Offset:                   {args.segment_offset}")
    print(f"Embedding scale:          {args.embedding_scale}")
    print(f"x_max mode:               {args.x_max_mode}")
    print(f"Leaf hidden dim:          {args.leaf_hidden_dim}")
    print(f"Merge mode:               {args.merge_mode}")
    print(f"Merge residual weight:    {args.merge_residual_weight}")
    print(f"Merge renormalize output: {args.merge_renormalize_output}")
    print(f"Threshold mode:           {args.threshold_mode}")
    print(f"Threshold target metric:  {args.threshold_target_metric}")
    print(f"Early stop metric:        {args.early_stopping_metric}")
    print(f"Min epochs before stop:   {args.min_epochs_before_stopping}")
    print(f"Loss type:                {args.loss_type}")
    print(f"Apply SNV:                {args.apply_snv}")
    print(f"Batch size:               {args.batch_size}")
    print(f"Epochs:                   {args.epochs}")
    print(f"Max files:                {args.max_files}")


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
    #for class_index in range(n_classes):
    #    class_labels = y_true[:, class_index]
    #    if class_labels.sum() == 0:
    #        thresholds[class_index] = 0.95
    #        continue
    #
    #    best_score = -1.0
    #    best_threshold = 0.5
    #    for threshold in threshold_grid:
    #        class_pred = threshold_predictions(y_probs[:, class_index], threshold)
    #        score = f1_score(class_labels, class_pred, zero_division=0)
    #        if score > best_score:
    #            best_score = score
    #            best_threshold = float(threshold)
    #    thresholds[class_index] = best_threshold
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


def run_synthetic_preflight(model: TTNIRClassifier5, args: argparse.Namespace, device: torch.device) -> None:
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
    model: TTNIRClassifier5,
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


def train_epoch(
    model: TTNIRClassifier5,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: Adam,
    device: torch.device,
    grad_clip_norm: float | None = None,
) -> tuple[float, dict[str, float | np.ndarray]]:
    model.train()
    total_loss = 0.0
    labels_list: list[np.ndarray] = []
    probs_list: list[np.ndarray] = []

    for spectra, labels in dataloader:
        spectra = spectra.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(spectra)
        loss = criterion(logits, labels)
        loss.backward()
        if grad_clip_norm is not None and grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()

        total_loss += loss.item() * spectra.size(0)
        labels_list.append(labels.detach().cpu().numpy())
        probs_list.append(torch.sigmoid(logits).detach().cpu().numpy())

    labels = np.concatenate(labels_list, axis=0)
    probs = np.concatenate(probs_list, axis=0)
    metrics = compute_metrics(labels, threshold_predictions(probs, 0.5))
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss, metrics


@torch.no_grad()
def evaluate_with_probs(
    model: TTNIRClassifier5,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    labels_list: list[np.ndarray] = []
    probs_list: list[np.ndarray] = []

    for spectra, labels in dataloader:
        spectra = spectra.to(device)
        labels = labels.to(device)

        logits = model(spectra)
        loss = criterion(logits, labels)
        total_loss += loss.item() * spectra.size(0)

        labels_list.append(labels.cpu().numpy())
        probs_list.append(torch.sigmoid(logits).cpu().numpy())

    labels = np.concatenate(labels_list, axis=0)
    probs = np.concatenate(probs_list, axis=0)
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss, labels, probs


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
        handle.write("TTN IR Experiment5 Summary\n")
        handle.write("=" * 80 + "\n")
        handle.write(f"Elapsed seconds:           {elapsed_seconds:.2f}\n")
        handle.write(f"Epochs completed:          {completed_epochs}/{requested_epochs}\n")
        handle.write(f"Data files used:           {used_data_files}/{total_data_files}\n")
        handle.write(f"Fixed split artifact:      {split_path}\n")
        handle.write(f"Best epoch:                {best_epoch}\n")
        handle.write(f"Best early-stop score:     {best_score:.6f}\n")
        handle.write(f"Best score metric:         {best_metric_name}\n")
        handle.write(f"Best val loss:             {best_val_loss:.6f}\n")
        handle.write(f"Threshold mean:            {final_thresholds.mean():.6f}\n")
        handle.write(f"Threshold std:             {final_thresholds.std():.6f}\n")
        handle.write(f"Test loss:                 {test_loss:.6f}\n")
        handle.write(f"Test f1_micro:             {float(test_metrics['f1_micro']):.6f}\n")
        handle.write(f"Test f1_macro:             {float(test_metrics['f1_macro']):.6f}\n")
        handle.write(f"Test precision_micro:      {float(test_metrics['precision_micro']):.6f}\n")
        handle.write(f"Test recall_micro:         {float(test_metrics['recall_micro']):.6f}\n")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0.")
    if args.x_max_mode == "global" and args.global_x_max is None:
        raise ValueError("--global-x-max is required when --x-max-mode global is used.")
    if args.threshold_mode == "per_class" and args.threshold_target_metric == "f1_micro":
        raise ValueError("--threshold-mode per_class is incompatible with --threshold-target-metric f1_micro.")
    if not args.data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {args.data_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    total_data_files = count_available_data_files(args.data_dir)
    used_data_files = resolve_used_file_count(total_data_files, args.max_files)
    split_suffix = "all" if args.max_files is None else f"files{used_data_files}"
    split_path = args.split_path or (args.output_dir / f"data_split_seed{args.seed}_{split_suffix}.npz")
    describe_args(args, split_path)

    config_path = args.output_dir / "run_config.json"
    checkpoint_path = args.output_dir / "ttn_ir_best.pt"
    log_path = args.output_dir / "training_log.csv"
    details_path = args.output_dir / "training_details.jsonl"
    summary_path = args.output_dir / "summary.txt"
    threshold_path = args.output_dir / "selected_thresholds.json"

    config_path.write_text(json.dumps(vars(args), indent=2, default=str) + "\n")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    print(f"Device: {device}")

    model = build_model(args)
    run_synthetic_preflight(model, args, device)

    print("\nLoading data...")
    X, y = load_ir_data(
        data_dir=args.data_dir,
        target_length=args.input_dim,
        max_files=args.max_files,
        apply_snv=args.apply_snv,
    )
    if X.shape[1] != args.input_dim:
        raise RuntimeError(f"Loaded spectra have width {X.shape[1]}, expected {args.input_dim}.")
    if y.shape[1] != args.num_labels:
        raise RuntimeError(f"Loaded labels have width {y.shape[1]}, expected {args.num_labels}.")

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
    train_loader, val_loader, test_loader = prepare_dataloaders_from_split_indices(
        X,
        y,
        split_indices,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    train_labels = y[split_indices["train"]]
    pos_weight = get_pos_weight(
        train_labels,
        device,
        power=args.pos_weight_power,
        max_value=args.pos_weight_max,
    )
    print(
        "Class weights: "
        f"min={pos_weight.min().item():.2f}, "
        f"max={pos_weight.max().item():.2f}, "
        f"mean={pos_weight.mean().item():.2f}"
    )

    criterion = build_loss(args.loss_type, pos_weight=pos_weight)
    run_real_batch_preflight(model, train_loader, device, criterion)

    if args.check_only:
        print("\nCheck-only mode finished successfully.")
        return

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
    model = model.to(device)

    print("\nStarting training...")
    start_time = time.time()
    best_score = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    completed_epochs = 0
    best_thresholds = np.full(args.num_labels, 0.5, dtype=np.float32)

    with log_path.open("w", newline="") as log_handle, details_path.open("w") as details_handle:
        writer = csv.writer(log_handle)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_f1_micro",
                "train_f1_macro",
                "val_loss",
                "val_f1_micro",
                "val_f1_macro",
                "early_stopping_score",
                "threshold_mean",
                "threshold_std",
                "lr",
            ]
        )

        for epoch in range(1, args.epochs + 1):
            train_loss, train_metrics = train_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                grad_clip_norm=args.grad_clip_norm,
            )
            val_loss, val_labels, val_probs = evaluate_with_probs(model, val_loader, criterion, device)
            current_thresholds = tune_thresholds(
                val_labels,
                val_probs,
                threshold_mode=args.threshold_mode,
                target_metric=args.threshold_target_metric,
                threshold_grid=threshold_grid,
            )
            val_preds = threshold_predictions(val_probs, current_thresholds)
            val_metrics = compute_metrics(val_labels, val_preds)
            early_stopping_score = select_early_stopping_score(
                val_metrics,
                metric_name=args.early_stopping_metric,
                blend_alpha=args.early_stopping_blend_alpha,
            )

            scheduler.step(early_stopping_score)
            current_lr = optimizer.param_groups[0]["lr"]

            writer.writerow(
                [
                    epoch,
                    train_loss,
                    float(train_metrics["f1_micro"]),
                    float(train_metrics["f1_macro"]),
                    val_loss,
                    float(val_metrics["f1_micro"]),
                    float(val_metrics["f1_macro"]),
                    early_stopping_score,
                    float(np.mean(current_thresholds)),
                    float(np.std(current_thresholds)),
                    current_lr,
                ]
            )
            log_handle.flush()
            write_epoch_details(
                details_handle,
                epoch=epoch,
                label_names=label_names,
                thresholds=current_thresholds,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                early_stopping_score=early_stopping_score,
            )
            completed_epochs = epoch

            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_f1_micro={float(val_metrics['f1_micro']):.4f} | "
                f"val_f1_macro={float(val_metrics['f1_macro']):.4f} | "
                f"score={early_stopping_score:.4f} | "
                f"lr={current_lr:.2e}"
            )

            if early_stopping_score > best_score + args.early_stopping_min_delta:
                best_score = early_stopping_score
                best_val_loss = val_loss
                best_epoch = epoch
                best_thresholds = current_thresholds.copy()
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_score": best_score,
                        "best_val_loss": best_val_loss,
                        "val_metrics": {
                            "f1_micro": float(val_metrics["f1_micro"]),
                            "f1_macro": float(val_metrics["f1_macro"]),
                            "precision_micro": float(val_metrics["precision_micro"]),
                            "recall_micro": float(val_metrics["recall_micro"]),
                            "per_class_f1": np.asarray(val_metrics["per_class_f1"], dtype=np.float32),
                        },
                        "thresholds": best_thresholds,
                        "split_path": str(split_path),
                        "args": vars(args),
                    },
                    checkpoint_path,
                )

            if epoch >= args.min_epochs_before_stopping and early_stopping(early_stopping_score):
                print(f"Early stopping triggered at epoch {epoch:03d}.")
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, val_labels, val_probs = evaluate_with_probs(model, val_loader, criterion, device)
    final_thresholds = tune_thresholds(
        val_labels,
        val_probs,
        threshold_mode=args.threshold_mode,
        target_metric=args.threshold_target_metric,
        threshold_grid=threshold_grid,
    )
    test_loss, test_labels, test_probs = evaluate_with_probs(model, test_loader, criterion, device)
    test_metrics = compute_metrics(test_labels, threshold_predictions(test_probs, final_thresholds))

    write_threshold_artifact(
        threshold_path=threshold_path,
        label_names=label_names,
        thresholds=final_thresholds,
        threshold_mode=args.threshold_mode,
        threshold_target_metric=args.threshold_target_metric,
        best_epoch=best_epoch,
    )

    elapsed_seconds = time.time() - start_time
    write_summary(
        summary_path=summary_path,
        elapsed_seconds=elapsed_seconds,
        completed_epochs=completed_epochs,
        requested_epochs=args.epochs,
        used_data_files=used_data_files,
        total_data_files=total_data_files,
        split_path=split_path,
        best_epoch=best_epoch,
        best_score=best_score,
        best_metric_name=args.early_stopping_metric,
        best_val_loss=best_val_loss,
        test_loss=test_loss,
        test_metrics=test_metrics,
        final_thresholds=final_thresholds,
    )

    print("\nTraining finished.")
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Config:          {config_path}")
    print(f"Split artifact:  {split_path}")
    print(f"Training log:    {log_path}")
    print(f"Details log:     {details_path}")
    print(f"Thresholds:      {threshold_path}")
    print(f"Summary:         {summary_path}")
    print(
        f"Best epoch={best_epoch}, best_score={best_score:.4f}, "
        f"test_f1_micro={float(test_metrics['f1_micro']):.4f}, "
        f"test_f1_macro={float(test_metrics['f1_macro']):.4f}"
    )


if __name__ == "__main__":
    main()
