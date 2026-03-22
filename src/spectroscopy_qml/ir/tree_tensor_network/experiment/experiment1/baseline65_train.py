"""Training entry point for the restored 0.65 TTN baseline."""

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

from baseline65_data_loader import (  # noqa: E402
    FUNCTIONAL_GROUPS,
    load_ir_data,
    prepare_dataloaders,
)
from baseline65_model import Baseline65TTNIRClassifier  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the restored 0.65 TTN baseline with preflight checks."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/spectroscopy_qml/ir/tree_tensor_network/baseline65_results"),
    )
    parser.add_argument("--input-dim", type=int, default=1800)
    parser.add_argument("--num-labels", type=int, default=len(FUNCTIONAL_GROUPS))
    parser.add_argument("--chi", type=int, default=64)
    parser.add_argument("--num-segments", type=int, default=32)
    parser.add_argument("--embedding-scale", type=float, default=0.1)
    parser.add_argument("--x-max-mode", choices=["per_sample", "global"], default="per_sample")
    parser.add_argument("--global-x-max", type=float, default=None)
    parser.add_argument("--use-bias", action=argparse.BooleanOptionalAction, default=True)
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


def build_model(args: argparse.Namespace) -> Baseline65TTNIRClassifier:
    return Baseline65TTNIRClassifier(
        num_labels=args.num_labels,
        chi=args.chi,
        num_segments=args.num_segments,
        embedding_scale=args.embedding_scale,
        x_max_mode=args.x_max_mode,
        global_x_max=args.global_x_max,
        use_bias=args.use_bias,
        input_dim=args.input_dim,
    )


def describe_args(args: argparse.Namespace) -> None:
    print("=" * 80)
    print("TTN IR Baseline65 Training")
    print("=" * 80)
    print(f"Data dir:        {args.data_dir}")
    print(f"Output dir:      {args.output_dir}")
    print(f"Input dim:       {args.input_dim}")
    print(f"Num labels:      {args.num_labels}")
    print(f"Chi:             {args.chi}")
    print(f"Num segments:    {args.num_segments}")
    print(f"Embedding scale: {args.embedding_scale}")
    print(f"x_max mode:      {args.x_max_mode}")
    print(f"Apply SNV:       {args.apply_snv}")
    print(f"Batch size:      {args.batch_size}")
    print(f"Epochs:          {args.epochs}")
    print(f"Max files:       {args.max_files}")


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

    if metric == "f1_micro":
        best_score = 0.0
        best_threshold = 0.5

        for threshold in np.arange(0.1, 0.9, 0.05):
            y_pred = threshold_predictions(y_probs, threshold)
            score = f1_score(y_true, y_pred, average="micro", zero_division=0)
            if score > best_score:
                best_score = score
                best_threshold = threshold

        return np.full(n_classes, best_threshold)

    thresholds = np.zeros(n_classes)
    for class_index in range(n_classes):
        best_score = 0.0
        best_threshold = 0.5
        for threshold in np.arange(0.1, 0.9, 0.05):
            y_pred = threshold_predictions(y_probs[:, class_index], threshold)
            score = f1_score(y_true[:, class_index], y_pred, zero_division=0)
            if score > best_score:
                best_score = score
                best_threshold = threshold
        thresholds[class_index] = best_threshold

    return thresholds


def run_synthetic_preflight(
    model: Baseline65TTNIRClassifier,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
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
    criterion = nn.BCEWithLogitsLoss()

    optimizer.zero_grad()
    logits = model(spectra)
    if logits.shape != labels.shape:
        raise RuntimeError(f"Synthetic preflight shape mismatch: {tuple(logits.shape)} != {tuple(labels.shape)}")
    if not torch.isfinite(logits).all():
        raise RuntimeError("Synthetic preflight produced non-finite logits.")

    loss = criterion(logits, labels)
    if not torch.isfinite(loss):
        raise RuntimeError("Synthetic preflight produced a non-finite loss.")
    loss.backward()
    optimizer.step()

    print(f"  OK - logits {tuple(logits.shape)}, loss={loss.item():.4f}")


def run_real_batch_preflight(
    model: Baseline65TTNIRClassifier,
    dataloader: DataLoader,
    device: torch.device,
    pos_weight: torch.Tensor,
) -> None:
    print("\n[Preflight] Real-data batch check")
    model = model.to(device)
    model.train()

    spectra, labels = next(iter(dataloader))
    spectra = spectra.to(device)
    labels = labels.to(device)

    if spectra.ndim != 2:
        raise RuntimeError(f"Expected batch spectra to be 2D, got shape {tuple(spectra.shape)}.")
    if spectra.size(1) != model.input_dim:
        raise RuntimeError(f"Expected input_dim={model.input_dim}, got {spectra.size(1)}.")
    if labels.size(1) != model.num_labels:
        raise RuntimeError(f"Expected num_labels={model.num_labels}, got {labels.size(1)}.")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = Adam(model.parameters(), lr=1e-4)

    optimizer.zero_grad()
    logits = model(spectra)
    loss = criterion(logits, labels)
    if not torch.isfinite(logits).all():
        raise RuntimeError("Real-data preflight produced non-finite logits.")
    if not torch.isfinite(loss):
        raise RuntimeError("Real-data preflight produced a non-finite loss.")
    loss.backward()
    optimizer.step()

    print(f"  OK - batch={tuple(spectra.shape)}, labels={tuple(labels.shape)}, loss={loss.item():.4f}")


def train_epoch(
    model: Baseline65TTNIRClassifier,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: Adam,
    device: torch.device,
    thresholds: np.ndarray,
) -> tuple[float, dict[str, float]]:
    model.train()
    total_loss = 0.0
    all_labels: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []

    for spectra, labels in dataloader:
        spectra = spectra.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(spectra)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * spectra.size(0)
        all_labels.append(labels.detach().cpu().numpy())
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.append(threshold_predictions(probs, thresholds))

    avg_loss = total_loss / len(dataloader.dataset)
    metrics = compute_metrics(np.vstack(all_labels), np.vstack(all_preds))
    return avg_loss, metrics


@torch.no_grad()
def evaluate(
    model: Baseline65TTNIRClassifier,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    thresholds: np.ndarray | None = None,
    return_probs: bool = False,
) -> tuple[float, dict[str, float]] | tuple[float, dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_labels: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []

    if thresholds is None:
        thresholds = np.full(model.num_labels, 0.5)

    for spectra, labels in dataloader:
        spectra = spectra.to(device)
        labels = labels.to(device)

        logits = model(spectra)
        loss = criterion(logits, labels)

        total_loss += loss.item() * spectra.size(0)
        all_labels.append(labels.cpu().numpy())
        all_probs.append(torch.sigmoid(logits).cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    labels = np.vstack(all_labels)
    probs = np.vstack(all_probs)
    preds = threshold_predictions(probs, thresholds)
    metrics = compute_metrics(labels, preds)
    if return_probs:
        return avg_loss, metrics, labels, probs
    return avg_loss, metrics


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
        handle.write("TTN IR Training Summary\n")
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

    if X.shape[1] != args.input_dim:
        raise RuntimeError(f"Loaded spectra have width {X.shape[1]}, expected {args.input_dim}.")
    if y.shape[1] != args.num_labels:
        raise RuntimeError(f"Loaded labels have width {y.shape[1]}, expected {args.num_labels}.")

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
    )

    train_labels = np.vstack([labels.numpy() for _, labels in train_loader])
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
        mode="min",
        factor=args.lr_scheduler_factor,
        patience=args.lr_scheduler_patience,
        min_lr=args.lr_scheduler_min_lr,
    )
    model = model.to(device)

    print("\nStarting training...")
    start_time = time.time()
    best_val_f1 = 0.0
    best_val_loss = float("inf")
    best_epoch = 0
    current_thresholds = np.full(args.num_labels, 0.5)
    best_thresholds = np.full(args.num_labels, 0.5)
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
                current_thresholds,
            )
            val_loss, _, val_labels, val_probs = evaluate(
                model,
                val_loader,
                criterion,
                device,
                thresholds=None,
                return_probs=True,
            )
            tuned_thresholds = tune_thresholds(val_labels, val_probs, metric="f1_micro")
            val_preds = threshold_predictions(val_probs, tuned_thresholds)
            val_metrics = compute_metrics(val_labels, val_preds)
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]
            current_thresholds = tuned_thresholds

            writer.writerow(
                [
                    epoch,
                    train_loss,
                    train_metrics["f1_micro"],
                    train_metrics["f1_macro"],
                    val_loss,
                    val_metrics["f1_micro"],
                    val_metrics["f1_macro"],
                    tuned_thresholds.mean(),
                    tuned_thresholds.std(),
                    current_lr,
                ]
            )
            handle.flush()
            completed_epochs = epoch

            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_loss:.4f} | train_f1_micro={train_metrics['f1_micro']:.4f} | "
                f"val_loss={val_loss:.4f} | val_f1_micro={val_metrics['f1_micro']:.4f} | "
                f"thr={tuned_thresholds.mean():.2f} | lr={current_lr:.2e}"
            )

            if val_metrics["f1_micro"] > best_val_f1:
                best_val_f1 = val_metrics["f1_micro"]
                best_val_loss = val_loss
                best_epoch = epoch
                best_thresholds = tuned_thresholds.copy()
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_loss,
                        "val_f1_micro": best_val_f1,
                        "thresholds": best_thresholds,
                        "args": vars(args),
                    },
                    checkpoint_path,
                )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    best_thresholds = checkpoint.get("thresholds", np.full(args.num_labels, 0.5))
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
