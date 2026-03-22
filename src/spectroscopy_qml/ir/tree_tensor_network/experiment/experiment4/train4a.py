"""Training entry point for Experiment4A with stricter isometric TTN merges."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import nn
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
    prepare_dataloaders,
)
from model4a import TTNIRClassifier4A  # noqa: E402


class EarlyStopping:
    """Early stopping on a validation metric."""

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
        description="Train Experiment4A with stricter isometric TTN merges."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment4/results/4a"),
    )
    parser.add_argument("--input-dim", type=int, default=1800)
    parser.add_argument("--num-labels", type=int, default=len(FUNCTIONAL_GROUPS))
    parser.add_argument("--chi", type=int, default=64)
    parser.add_argument("--num-segments", type=int, default=32)
    parser.add_argument("--embedding-scale", type=float, default=0.1)
    parser.add_argument("--x-max-mode", choices=["per_sample", "global"], default="per_sample")
    parser.add_argument("--global-x-max", type=float, default=None)
    parser.add_argument("--position-scale", type=float, default=0.25)
    parser.add_argument("--position-strength", type=float, default=0.1)
    parser.add_argument("--merge-renormalize-output", action=argparse.BooleanOptionalAction, default=True)
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
    parser.add_argument("--pos-weight-power", type=float, default=0.5)
    parser.add_argument("--pos-weight-max", type=float, default=None)
    parser.add_argument(
        "--threshold-metric",
        choices=["f1_micro", "f1_macro", "per_class_f1"],
        default="f1_micro",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
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


def build_model(args: argparse.Namespace) -> TTNIRClassifier4A:
    return TTNIRClassifier4A(
        num_labels=args.num_labels,
        chi=args.chi,
        num_segments=args.num_segments,
        embedding_scale=args.embedding_scale,
        x_max_mode=args.x_max_mode,
        global_x_max=args.global_x_max,
        position_scale=args.position_scale,
        position_strength=args.position_strength,
        input_dim=args.input_dim,
        merge_renormalize_output=args.merge_renormalize_output,
    )


def describe_args(args: argparse.Namespace) -> None:
    print("=" * 80)
    print("TTN IR Experiment4A Training")
    print("=" * 80)
    print(f"Data dir:               {args.data_dir}")
    print(f"Output dir:             {args.output_dir}")
    print(f"Input dim:              {args.input_dim}")
    print(f"Num labels:             {args.num_labels}")
    print(f"Chi:                    {args.chi}")
    print(f"Num segments:           {args.num_segments}")
    print(f"Embedding scale:        {args.embedding_scale}")
    print(f"x_max mode:             {args.x_max_mode}")
    print(f"Position scale:         {args.position_scale}")
    print(f"Position strength:      {args.position_strength}")
    print(f"Merge renormalization:  {args.merge_renormalize_output}")
    print(f"Apply SNV:              {args.apply_snv}")
    print(f"Batch size:             {args.batch_size}")
    print(f"Epochs:                 {args.epochs}")
    print(f"Threshold metric:       {args.threshold_metric}")
    print(f"Max files:              {args.max_files}")


def count_available_data_files(data_dir: Path) -> int:
    return len(list(Path(data_dir).glob("*.parquet")))


def resolve_used_file_count(total_files: int, max_files: int | None) -> int:
    if max_files is None:
        return total_files
    return min(total_files, max(int(max_files), 0))


def threshold_predictions(y_prob: np.ndarray, thresholds: float | np.ndarray = 0.5) -> np.ndarray:
    return (y_prob >= thresholds).astype(int)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_micro": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "recall_micro": recall_score(y_true, y_pred, average="micro", zero_division=0),
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
    if power <= 0.0:
        raise ValueError("pos_weight_power must be positive.")
    if power != 1.0:
        pos_weight = np.power(pos_weight, power, dtype=np.float32)
    if max_value is not None:
        pos_weight = np.clip(pos_weight, 1.0, max_value)
    return torch.as_tensor(pos_weight, dtype=torch.float32, device=device)


def tune_thresholds(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    metric: str = "f1_micro",
) -> np.ndarray:
    n_classes = y_true.shape[1]
    threshold_grid = np.arange(0.05, 1.0, 0.05)

    if metric == "f1_micro":
        best_score = -1.0
        best_threshold = 0.5
        for threshold in threshold_grid:
            y_pred = threshold_predictions(y_probs, threshold)
            score = f1_score(y_true, y_pred, average="micro", zero_division=0)
            if score > best_score:
                best_score = score
                best_threshold = threshold
        return np.full(n_classes, best_threshold)

    thresholds = np.full(n_classes, 0.5)
    for class_index in range(n_classes):
        class_labels = y_true[:, class_index]
        if class_labels.sum() == 0:
            thresholds[class_index] = 0.95
            continue

        best_score = -1.0
        best_threshold = 0.5
        for threshold in threshold_grid:
            y_pred = threshold_predictions(y_probs[:, class_index], threshold)
            score = f1_score(class_labels, y_pred, zero_division=0)
            if score > best_score:
                best_score = score
                best_threshold = threshold
        thresholds[class_index] = best_threshold

    return thresholds


def run_synthetic_preflight(model: TTNIRClassifier4A, args: argparse.Namespace, device: torch.device) -> None:
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
    logits = model(spectra)
    loss = nn.BCEWithLogitsLoss()(logits, labels)
    loss.backward()
    print(f"  OK - logits {tuple(logits.shape)}, loss={loss.item():.4f}")
    model.zero_grad(set_to_none=True)
    model.cpu()


def run_real_batch_preflight(
    model: TTNIRClassifier4A,
    train_loader: DataLoader,
    device: torch.device,
    pos_weight: torch.Tensor,
) -> None:
    print("\n[Preflight] Real-data batch check")
    spectra, labels = next(iter(train_loader))
    spectra = spectra.to(device)
    labels = labels.to(device)

    model = model.to(device)
    model.train()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    logits = model(spectra)
    loss = criterion(logits, labels)
    loss.backward()
    print(
        f"  OK - batch={tuple(spectra.shape)}, labels={tuple(labels.shape)}, "
        f"loss={loss.item():.4f}"
    )
    model.zero_grad(set_to_none=True)


def train_epoch(
    model: TTNIRClassifier4A,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    thresholds: np.ndarray,
    grad_clip_norm: float,
) -> tuple[float, dict[str, float]]:
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
        if grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        total_loss += loss.item() * spectra.size(0)
        labels_list.append(labels.detach().cpu().numpy())
        probs_list.append(torch.sigmoid(logits).detach().cpu().numpy())

    labels = np.concatenate(labels_list, axis=0)
    probs = np.concatenate(probs_list, axis=0)
    preds = threshold_predictions(probs, thresholds)
    metrics = compute_metrics(labels, preds)
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss, metrics


@torch.no_grad()
def evaluate(
    model: TTNIRClassifier4A,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    thresholds: np.ndarray | None = None,
    return_probs: bool = False,
) -> tuple[float, dict[str, float]] | tuple[float, dict[str, float], np.ndarray, np.ndarray]:
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
    if thresholds is None:
        preds = threshold_predictions(probs, 0.5)
    else:
        preds = threshold_predictions(probs, thresholds)
    avg_loss = total_loss / len(dataloader.dataset)
    metrics = compute_metrics(labels, preds)
    if return_probs:
        return avg_loss, metrics, labels, probs
    return avg_loss, metrics


@torch.no_grad()
def tune_thresholds_on_best_checkpoint(
    model: TTNIRClassifier4A,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    metric: str,
) -> np.ndarray:
    _, _, labels, probs = evaluate(
        model,
        dataloader,
        criterion,
        device,
        thresholds=None,
        return_probs=True,
    )
    return tune_thresholds(labels, probs, metric=metric)


def write_summary(
    summary_path: Path,
    elapsed_seconds: float,
    completed_epochs: int,
    requested_epochs: int,
    used_data_files: int,
    total_data_files: int,
    best_epoch: int,
    best_val_f1: float,
    best_val_loss: float,
    test_loss: float,
    test_metrics: dict[str, float],
    best_thresholds: np.ndarray,
) -> None:
    with summary_path.open("w") as handle:
        handle.write("TTN IR Experiment4A Summary\n")
        handle.write("=" * 80 + "\n")
        handle.write(f"Elapsed seconds: {elapsed_seconds:.2f}\n")
        handle.write(f"Epochs completed:{completed_epochs}/{requested_epochs}\n")
        handle.write(f"Data files used: {used_data_files}/{total_data_files}\n")
        handle.write(f"Best epoch:      {best_epoch}\n")
        handle.write(f"Best val F1:     {best_val_f1:.6f}\n")
        handle.write(f"Best val loss:   {best_val_loss:.6f}\n")
        handle.write(f"Threshold mean:  {best_thresholds.mean():.6f}\n")
        handle.write(f"Threshold std:   {best_thresholds.std():.6f}\n")
        handle.write(f"Test loss:       {test_loss:.6f}\n")
        for key, value in test_metrics.items():
            handle.write(f"{key}: {value:.6f}\n")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    describe_args(args)

    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0.")
    if args.x_max_mode == "global" and args.global_x_max is None:
        raise ValueError("--global-x-max is required when --x-max-mode global is used.")
    if not args.data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {args.data_dir}")

    total_data_files = count_available_data_files(args.data_dir)
    used_data_files = resolve_used_file_count(total_data_files, args.max_files)
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

    train_loader, val_loader, test_loader = prepare_dataloaders(
        X,
        y,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        stratify_multilabel=True,
    )

    train_labels = np.vstack([labels.numpy() for _, labels in train_loader])
    pos_weight = get_pos_weight(train_labels, device, power=args.pos_weight_power, max_value=args.pos_weight_max)
    print(
        "Class weights: "
        f"min={pos_weight.min().item():.2f}, "
        f"max={pos_weight.max().item():.2f}, "
        f"mean={pos_weight.mean().item():.2f}"
    )

    run_real_batch_preflight(model, train_loader, device, pos_weight)

    if args.check_only:
        print("\nCheck-only mode finished successfully.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "ttn_ir_best.pt"
    log_path = args.output_dir / "training_log.csv"
    summary_path = args.output_dir / "summary.txt"

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
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
    model = model.to(device)

    print("\nStarting training...")
    start_time = time.time()
    best_val_f1 = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    selection_thresholds = np.full(args.num_labels, 0.5)
    best_thresholds = selection_thresholds.copy()
    completed_epochs = 0

    with log_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_f1_micro",
                "train_f1_macro",
                "val_loss",
                "val_f1_micro",
                "val_f1_macro",
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
                selection_thresholds,
                grad_clip_norm=args.grad_clip_norm,
            )
            val_loss, val_metrics = evaluate(
                model,
                val_loader,
                criterion,
                device,
                thresholds=selection_thresholds,
            )
            scheduler.step(val_metrics["f1_micro"])
            current_lr = optimizer.param_groups[0]["lr"]

            writer.writerow(
                [
                    epoch,
                    train_loss,
                    train_metrics["f1_micro"],
                    train_metrics["f1_macro"],
                    val_loss,
                    val_metrics["f1_micro"],
                    val_metrics["f1_macro"],
                    selection_thresholds.mean(),
                    selection_thresholds.std(),
                    current_lr,
                ]
            )
            handle.flush()
            completed_epochs = epoch

            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_loss:.4f} | train_f1_micro={train_metrics['f1_micro']:.4f} | "
                f"val_loss={val_loss:.4f} | val_f1_micro={val_metrics['f1_micro']:.4f} | "
                f"thr={selection_thresholds.mean():.2f} | lr={current_lr:.2e}"
            )

            if val_metrics["f1_micro"] > best_val_f1:
                best_val_f1 = val_metrics["f1_micro"]
                best_val_loss = val_loss
                best_epoch = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_loss,
                        "val_f1_micro": best_val_f1,
                        "val_metrics": val_metrics,
                        "thresholds": best_thresholds,
                        "args": vars(args),
                    },
                    checkpoint_path,
                )

            if early_stopping(val_metrics["f1_micro"]):
                print(f"Early stopping triggered at epoch {epoch:03d}.")
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    best_thresholds = tune_thresholds_on_best_checkpoint(
        model,
        val_loader,
        criterion,
        device,
        metric=args.threshold_metric,
    )
    test_loss, test_metrics = evaluate(
        model,
        test_loader,
        criterion,
        device,
        thresholds=best_thresholds,
    )

    elapsed_seconds = time.time() - start_time
    write_summary(
        summary_path,
        elapsed_seconds,
        completed_epochs,
        args.epochs,
        used_data_files,
        total_data_files,
        best_epoch,
        best_val_f1,
        best_val_loss,
        test_loss,
        test_metrics,
        best_thresholds,
    )

    print("\nTraining finished.")
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Training log:    {log_path}")
    print(f"Summary:         {summary_path}")
    print(
        f"Best epoch={best_epoch}, best_val_f1_micro={best_val_f1:.4f}, "
        f"best_val_loss={best_val_loss:.4f}, test_f1_micro={test_metrics['f1_micro']:.4f}"
    )
    print(f"Epochs completed={completed_epochs}/{args.epochs}, data files={used_data_files}/{total_data_files}")


if __name__ == "__main__":
    main()
