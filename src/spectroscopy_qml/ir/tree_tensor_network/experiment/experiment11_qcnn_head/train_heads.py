"""Train linear / MLP / QCNN specialist heads on frozen CNN features."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = Path(__file__).resolve().parents[5]
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment5.data_loader import (  # noqa: E402
    FUNCTIONAL_GROUPS,
    load_or_create_split_indices,
)
from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment5.losses import build_loss  # noqa: E402
from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment5.train import (  # noqa: E402
    EarlyStopping,
    build_threshold_grid,
    compute_metrics,
    get_pos_weight,
    resolve_device,
    select_early_stopping_score,
    threshold_predictions,
    tune_thresholds,
    write_epoch_details,
    write_threshold_artifact,
)
from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment11_qcnn_head.model import (  # noqa: E402
    CNN_FEATURE_DIM,
    SpecialistHeadEnsemble,
)

ALL_LABEL_NAMES = list(FUNCTIONAL_GROUPS.keys())
DEFAULT_SPECIALIST_INDICES = [17, 16, 28, 24, 5]


def parse_indices(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",")]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train specialist linear / MLP / QCNN heads on frozen CNN features."
    )
    parser.add_argument(
        "--features-path",
        type=Path,
        default=Path("benchmark/cnn/features/cnn_ir_features.npz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment11_qcnn_head/results"),
    )
    parser.add_argument("--split-path", type=Path, default=None)
    parser.add_argument("--overwrite-split", action="store_true")
    parser.add_argument(
        "--specialist-indices",
        type=parse_indices,
        default=DEFAULT_SPECIALIST_INDICES,
    )
    parser.add_argument("--head-type", choices=["linear", "mlp", "qcnn"], default="qcnn")
    parser.add_argument("--cnn-feature-dim", type=int, default=CNN_FEATURE_DIM)
    parser.add_argument("--mlp-hidden-dim", type=int, default=128)
    parser.add_argument("--qcnn-qubits", type=int, default=8)
    parser.add_argument("--qcnn-projection-hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--lr-scheduler-factor", type=float, default=0.9)
    parser.add_argument("--lr-scheduler-patience", type=int, default=5)
    parser.add_argument("--lr-scheduler-min-lr", type=float, default=1e-6)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--loss-type", choices=["bce", "focal"], default="focal")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--pos-weight-power", type=float, default=1.0)
    parser.add_argument("--pos-weight-max", type=float, default=20.0)
    parser.add_argument("--threshold-mode", choices=["global", "per_class"], default="per_class")
    parser.add_argument(
        "--threshold-target-metric",
        choices=["f1_micro", "f1_macro", "per_class_f1"],
        default="per_class_f1",
    )
    parser.add_argument("--threshold-grid-step", type=float, default=0.02)
    parser.add_argument(
        "--early-stopping-metric",
        choices=["f1_micro", "f1_macro", "blended_f1"],
        default="blended_f1",
    )
    parser.add_argument("--early-stopping-blend-alpha", type=float, default=0.5)
    parser.add_argument("--early-stopping-patience", type=int, default=16)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--min-epochs-before-stopping", type=int, default=20)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--check-only", action="store_true")
    return parser


def resolve_head_device(requested_device: str, head_type: str) -> torch.device:
    device = resolve_device(requested_device)
    if head_type == "qcnn" and device.type != "cpu":
        print("QCNN head uses PennyLane default.qubit; falling back to CPU.")
        return torch.device("cpu")
    return device


def make_feature_dataloaders(
    features: np.ndarray,
    labels: np.ndarray,
    split_indices: dict[str, np.ndarray],
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    def loader(idx: np.ndarray, shuffle: bool) -> DataLoader:
        x_t = torch.from_numpy(features[idx]).float()
        y_t = torch.from_numpy(labels[idx]).float()
        return DataLoader(
            TensorDataset(x_t, y_t),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return (
        loader(split_indices["train"], shuffle=True),
        loader(split_indices["val"], shuffle=False),
        loader(split_indices["test"], shuffle=False),
    )


def train_epoch(
    model: SpecialistHeadEnsemble,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float,
) -> tuple[float, dict]:
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for x_batch, y_batch in loader:
        y_np = y_batch.numpy().astype(np.int32)
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        total_loss += loss.item() * x_batch.size(0)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.append((probs >= 0.5).astype(np.int32))
        all_labels.append(y_np)

    avg_loss = total_loss / sum(len(chunk) for chunk in all_labels)
    preds_cat = np.concatenate(all_preds, axis=0)
    labels_cat = np.concatenate(all_labels, axis=0)
    return avg_loss, compute_metrics(labels_cat, preds_cat)


@torch.no_grad()
def evaluate(
    model: SpecialistHeadEnsemble,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []

    for x_batch, y_batch in loader:
        y_np = y_batch.numpy().astype(np.int32)
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        total_loss += loss.item() * x_batch.size(0)
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(y_np)

    n = sum(len(chunk) for chunk in all_labels)
    return (
        total_loss / n,
        np.concatenate(all_labels, axis=0),
        np.concatenate(all_probs, axis=0).astype(np.float32),
    )


def write_summary(
    summary_path: Path,
    *,
    args: argparse.Namespace,
    specialist_names: list[str],
    best_epoch: int,
    best_score: float,
    completed_epochs: int,
    elapsed_seconds: float,
    test_loss: float,
    test_metrics: dict,
) -> None:
    lines = [
        "Experiment11 QCNN Head Summary",
        "=" * 80,
        f"Head type:                 {args.head_type}",
        f"Specialist classes:        {', '.join(specialist_names)}",
        f"Completed epochs:          {completed_epochs}/{args.epochs}",
        f"Elapsed seconds:           {elapsed_seconds:.2f}",
        f"Best epoch:                {best_epoch}",
        f"Best score:                {best_score:.6f}",
        f"Test loss:                 {test_loss:.6f}",
        f"Test f1_micro:             {float(test_metrics['f1_micro']):.6f}",
        f"Test f1_macro:             {float(test_metrics['f1_macro']):.6f}",
        f"Test precision_micro:      {float(test_metrics['precision_micro']):.6f}",
        f"Test recall_micro:         {float(test_metrics['recall_micro']):.6f}",
        "",
        "Per-class F1:",
    ]
    for name, f1 in zip(specialist_names, test_metrics["per_class_f1"], strict=False):
        lines.append(f"  {name:20s}: {float(f1):.6f}")
    summary_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    specialist_indices = sorted(set(args.specialist_indices))
    specialist_names = [ALL_LABEL_NAMES[i] for i in specialist_indices]
    num_labels = len(specialist_indices)

    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0.")
    if not args.features_path.exists():
        raise FileNotFoundError(f"Features file not found: {args.features_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_path = args.split_path or (args.output_dir / f"data_split_seed{args.seed}_all.npz")
    checkpoint_path = args.output_dir / "specialist_head_best.pt"
    threshold_path = args.output_dir / "selected_thresholds.json"
    summary_path = args.output_dir / "summary.txt"
    log_path = args.output_dir / "training_log.csv"
    details_path = args.output_dir / "training_details.jsonl"
    specialist_map_path = args.output_dir / "specialist_map.json"
    run_config_path = args.output_dir / "run_config.json"

    run_config_path.write_text(json.dumps(vars(args), indent=2, default=str) + "\n")
    specialist_map_path.write_text(
        json.dumps(
            {
                "specialist_indices": specialist_indices,
                "specialist_names": specialist_names,
                "head_type": args.head_type,
            },
            indent=2,
        )
        + "\n"
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_head_device(args.device, args.head_type)
    print(f"Device: {device}")
    print(f"Specialist classes: {specialist_names}")

    data = np.load(args.features_path)
    features = data["features"].astype(np.float32)
    labels_full = data["labels"].astype(np.int32)
    specialist_labels = labels_full[:, specialist_indices]

    split_indices = load_or_create_split_indices(
        labels=labels_full,
        split_path=split_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
        stratify_multilabel=True,
        overwrite=args.overwrite_split,
    )
    train_loader, val_loader, test_loader = make_feature_dataloaders(
        features,
        specialist_labels,
        split_indices,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    train_labels = specialist_labels[split_indices["train"]]
    pos_weight = get_pos_weight(
        train_labels,
        device,
        power=args.pos_weight_power,
        max_value=args.pos_weight_max,
    )

    model = SpecialistHeadEnsemble(
        num_specialist_classes=num_labels,
        head_type=args.head_type,
        input_dim=args.cnn_feature_dim,
        mlp_hidden_dim=args.mlp_hidden_dim,
        qcnn_qubits=args.qcnn_qubits,
        qcnn_projection_hidden_dim=args.qcnn_projection_hidden_dim,
        dropout=args.dropout,
    ).to(device)

    if args.check_only:
        xb, yb = next(iter(train_loader))
        logits = model(xb.to(device))
        print(
            f"Check-only OK: features={tuple(xb.shape)}, labels={tuple(yb.shape)}, logits={tuple(logits.shape)}"
        )
        return

    criterion = build_loss(
        loss_name=args.loss_type,
        pos_weight=pos_weight,
        focal_gamma=args.focal_gamma,
    )
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.lr_scheduler_factor,
        patience=args.lr_scheduler_patience,
        min_lr=args.lr_scheduler_min_lr,
    )
    threshold_grid = build_threshold_grid(args.threshold_grid_step)
    early_stopping = EarlyStopping(
        patience=args.early_stopping_patience,
        min_delta=args.early_stopping_min_delta,
    )

    best_state_dict = None
    best_thresholds = None
    best_score = float("-inf")
    best_epoch = 0
    completed_epochs = 0
    start_time = time.time()

    with log_path.open("w", newline="") as csv_file, details_path.open("w") as details_file:
        writer = csv.writer(csv_file)
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
                "lr",
            ]
        )

        for epoch in range(1, args.epochs + 1):
            train_loss, train_metrics = train_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                grad_clip_norm=args.grad_clip_norm,
            )
            val_loss, val_labels, val_probs = evaluate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
            )
            thresholds = tune_thresholds(
                y_true=val_labels,
                y_probs=val_probs,
                threshold_mode=args.threshold_mode,
                target_metric=args.threshold_target_metric,
                threshold_grid=threshold_grid,
            )
            val_metrics = compute_metrics(val_labels, threshold_predictions(val_probs, thresholds))
            early_score = select_early_stopping_score(
                val_metrics,
                args.early_stopping_metric,
                args.early_stopping_blend_alpha,
            )
            scheduler.step(early_score)

            writer.writerow(
                [
                    epoch,
                    float(train_loss),
                    float(train_metrics["f1_micro"]),
                    float(train_metrics["f1_macro"]),
                    float(val_loss),
                    float(val_metrics["f1_micro"]),
                    float(val_metrics["f1_macro"]),
                    float(early_score),
                    float(optimizer.param_groups[0]["lr"]),
                ]
            )
            csv_file.flush()
            write_epoch_details(
                details_file,
                epoch=epoch,
                label_names=specialist_names,
                thresholds=thresholds,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                early_stopping_score=early_score,
            )

            print(
                f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"val_f1_micro={val_metrics['f1_micro']:.4f} | val_f1_macro={val_metrics['f1_macro']:.4f}"
            )

            if early_score > best_score:
                best_score = float(early_score)
                best_epoch = epoch
                best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_thresholds = thresholds.copy()
                torch.save(best_state_dict, checkpoint_path)

            completed_epochs = epoch
            if epoch >= args.min_epochs_before_stopping and early_stopping(early_score):
                break

    if best_state_dict is None or best_thresholds is None:
        raise RuntimeError("Training finished without a valid checkpoint.")

    model.load_state_dict(best_state_dict)
    model.to(device)
    test_loss, test_labels, test_probs = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )
    test_metrics = compute_metrics(test_labels, threshold_predictions(test_probs, best_thresholds))

    write_threshold_artifact(
        threshold_path=threshold_path,
        label_names=specialist_names,
        thresholds=best_thresholds,
        threshold_mode=args.threshold_mode,
        threshold_target_metric=args.threshold_target_metric,
        best_epoch=best_epoch,
    )
    write_summary(
        summary_path,
        args=args,
        specialist_names=specialist_names,
        best_epoch=best_epoch,
        best_score=best_score,
        completed_epochs=completed_epochs,
        elapsed_seconds=time.time() - start_time,
        test_loss=test_loss,
        test_metrics=test_metrics,
    )

    print("\nTest metrics")
    print(f"  f1_micro:        {float(test_metrics['f1_micro']):.4f}")
    print(f"  f1_macro:        {float(test_metrics['f1_macro']):.4f}")


if __name__ == "__main__":
    main()
