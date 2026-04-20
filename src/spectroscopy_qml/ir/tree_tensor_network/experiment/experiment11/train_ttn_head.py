"""Experiment 11: Train TTN head on frozen CNN features.

Workflow:
  1. Run extract_cnn_features.py once to produce cnn_ir_features.npz.
  2. Run this script to train the TTN head on the extracted features.
  3. Run ensemble_predict.py to evaluate CNN + TTN head ensemble.

The TTN head is a purely PyTorch model that receives the 1574-dim frozen CNN
feature vectors and predicts the specialist (hard) functional group classes.
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
from torch.amp import GradScaler
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
from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment10.train import (  # noqa: E402
    write_summary,
)
from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment11.model_ttn_head import (  # noqa: E402
    CNN_FEATURE_DIM,
    TTNHead,
)

ALL_LABEL_NAMES = list(FUNCTIONAL_GROUPS.keys())
DEFAULT_SPECIALIST_INDICES = [17, 16, 28, 24, 5]


def parse_indices(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",")]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train TTN head on frozen CNN features (Experiment 11)."
    )
    parser.add_argument(
        "--features-path",
        type=Path,
        default=Path("benchmark/cnn/features/cnn_ir_features.npz"),
        help="Path to cnn_ir_features.npz produced by extract_cnn_features.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment11/results"),
    )
    parser.add_argument("--split-path", type=Path, default=None)
    parser.add_argument("--overwrite-split", action="store_true")
    parser.add_argument(
        "--specialist-indices",
        type=parse_indices,
        default=DEFAULT_SPECIALIST_INDICES,
        help=f"Comma-separated label indices. Default: {DEFAULT_SPECIALIST_INDICES}",
    )
    parser.add_argument("--cnn-feature-dim", type=int, default=CNN_FEATURE_DIM)
    parser.add_argument("--chi", type=int, default=64)
    parser.add_argument("--num-segments", type=int, default=16,
                        help="Number of virtual TTN segments (must be power of 2).")
    parser.add_argument("--segment-dim", type=int, default=None,
                        help="Segment state dimension. Defaults to chi.")
    parser.add_argument("--merge-mode", choices=["strict", "relaxed"], default="relaxed")
    parser.add_argument("--merge-residual-weight", type=float, default=0.1)
    parser.add_argument("--merge-renormalize-output", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--segment-normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=200)
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
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--min-epochs-before-stopping", type=int, default=30)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-compile", dest="compile", action="store_false")
    parser.add_argument("--compile", dest="compile", action="store_true", default=True)
    return parser


def make_feature_dataloaders(
    features: np.ndarray,
    labels: np.ndarray,
    split_indices: dict,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    def loader(idx: np.ndarray, shuffle: bool) -> DataLoader:
        X_t = torch.from_numpy(features[idx]).float()
        y_t = torch.from_numpy(labels[idx]).float()  # float for loss; collected as numpy before device move
        return DataLoader(
            TensorDataset(X_t, y_t),
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
    model: TTNHead,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float,
    scaler: GradScaler | None,
    use_amp: bool,
) -> tuple[float, dict]:
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for X_batch, y_batch in loader:
        y_np = y_batch.numpy().astype(np.int32)  # collect before MPS transfer to avoid int corruption
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(X_batch)
            loss = criterion(logits, y_batch)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.append((probs >= 0.5).astype(np.int32))
        all_labels.append(y_np)

    avg_loss = total_loss / sum(len(l) for l in all_labels)
    preds_cat = np.concatenate(all_preds, axis=0)
    labels_cat = np.concatenate(all_labels, axis=0)
    metrics = compute_metrics(labels_cat, preds_cat)
    return avg_loss, metrics


@torch.no_grad()
def evaluate(
    model: TTNHead,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []

    for X_batch, y_batch in loader:
        y_np = y_batch.numpy().astype(np.int32)  # collect before MPS transfer to avoid int corruption
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(X_batch)
            loss = criterion(logits, y_batch)

        total_loss += loss.item() * X_batch.size(0)
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(y_np)

    n = sum(len(l) for l in all_labels)
    return (
        total_loss / n,
        np.concatenate(all_labels, axis=0),
        np.concatenate(all_probs, axis=0).astype(np.float32),
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    specialist_indices = sorted(set(args.specialist_indices))
    specialist_names = [ALL_LABEL_NAMES[i] for i in specialist_indices]
    num_labels = len(specialist_indices)

    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0.")
    if not args.features_path.exists():
        raise FileNotFoundError(
            f"Features file not found: {args.features_path}\n"
            "Run extract_cnn_features.py first."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load pre-extracted features ─────────────────────────────────────────────
    print(f"Loading CNN features from {args.features_path} ...")
    data = np.load(args.features_path)
    features = data["features"].astype(np.float32)    # (N, 1574)
    labels_full = data["labels"].astype(np.int32)     # (N, 37)
    labels = labels_full[:, specialist_indices]        # (N, num_specialist)
    nan_count = np.isnan(features).sum()
    if nan_count > 0:
        print(f"WARNING: {nan_count} NaN values in CNN features — replacing with 0.")
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    # Z-score normalise CNN features per dimension so TTN layers stay numerically stable.
    feat_mean = features.mean(axis=0, keepdims=True)
    feat_std  = features.std(axis=0, keepdims=True) + 1e-8
    features  = ((features - feat_mean) / feat_std).astype(np.float32)
    print(f"Features z-score normalised: mean≈{feat_mean.mean():.3f}, std≈{feat_std.mean():.3f}")
    print(f"Features: {features.shape}, Labels (specialist): {labels.shape}")
    print("Specialist class positive rates:")
    for i, name in zip(specialist_indices, specialist_names):
        rate = labels_full[:, i].mean()
        print(f"  {name:20s} (idx={i}): {rate*100:.2f}%")

    # ── Data split ──────────────────────────────────────────────────────────────
    split_suffix = "all"
    split_path = args.split_path or (
        args.output_dir / f"data_split_seed{args.seed}_{split_suffix}.npz"
    )
    split_indices = load_or_create_split_indices(
        labels=labels,
        split_path=split_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
        stratify_multilabel=True,
        overwrite=args.overwrite_split,
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    print(f"Device: {device}")
    use_amp = args.amp and device.type == "cuda"

    train_loader, val_loader, test_loader = make_feature_dataloaders(
        features, labels, split_indices,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    # ── Model ───────────────────────────────────────────────────────────────────
    model = TTNHead(
        cnn_feature_dim=args.cnn_feature_dim,
        num_specialist_classes=num_labels,
        chi=args.chi,
        num_segments=args.num_segments,
        segment_dim=args.segment_dim,
        merge_mode=args.merge_mode,
        merge_residual_weight=args.merge_residual_weight,
        merge_renormalize_output=args.merge_renormalize_output,
        segment_normalize=args.segment_normalize,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTTN Head parameters: {total_params:,}")
    print(f"Specialist classes: {specialist_names}")

    # ── Loss / optimizer ────────────────────────────────────────────────────────
    train_labels = labels[split_indices["train"]]
    pos_weight = get_pos_weight(
        train_labels, device, power=args.pos_weight_power, max_value=args.pos_weight_max
    )
    print(f"Class weights: {np.round(pos_weight.cpu().numpy(), 2).tolist()}")
    criterion = build_loss(args.loss_type, pos_weight=pos_weight, focal_gamma=args.focal_gamma)

    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max",
        factor=args.lr_scheduler_factor,
        patience=args.lr_scheduler_patience,
        min_lr=args.lr_scheduler_min_lr,
    )
    early_stopping = EarlyStopping(
        patience=args.early_stopping_patience,
        min_delta=args.early_stopping_min_delta,
        mode="max",
    )
    scaler = GradScaler() if use_amp else None
    threshold_grid = build_threshold_grid(args.threshold_grid_step)

    # ── Save config ─────────────────────────────────────────────────────────────
    config = {**vars(args), "specialist_indices": specialist_indices, "specialist_names": specialist_names}
    checkpoint_path = args.output_dir / "ttn_head_best.pt"
    log_path = args.output_dir / "training_log.csv"
    details_path = args.output_dir / "training_details.jsonl"
    summary_path = args.output_dir / "summary.txt"
    threshold_path = args.output_dir / "selected_thresholds.json"
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, default=str) + "\n"
    )
    (args.output_dir / "specialist_map.json").write_text(
        json.dumps({
            "specialist_indices": specialist_indices,
            "specialist_names": specialist_names,
            "all_label_names": ALL_LABEL_NAMES,
            "cnn_feature_dim": args.cnn_feature_dim,
            "model_type": "TTNHead",
        }, indent=2) + "\n"
    )

    # ── Training loop ───────────────────────────────────────────────────────────
    print("\nStarting training...")
    print("=" * 80)
    start_time = time.time()
    best_score = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    completed_epochs = 0
    best_thresholds = np.full(num_labels, 0.5, dtype=np.float32)

    with log_path.open("w", newline="") as log_handle, details_path.open("w") as det_handle:
        writer = csv.writer(log_handle)
        writer.writerow([
            "epoch", "train_loss", "train_f1_micro", "train_f1_macro",
            "val_loss", "val_f1_micro", "val_f1_macro",
            "early_stopping_score", "threshold_mean", "threshold_std", "lr",
        ])

        for epoch in range(1, args.epochs + 1):
            train_loss, train_metrics = train_epoch(
                model, train_loader, criterion, optimizer, device,
                args.grad_clip_norm, scaler, use_amp,
            )
            val_loss, val_labels_raw, val_probs = evaluate(
                model, val_loader, criterion, device, use_amp,
            )
            val_labels = val_labels_raw.astype(np.int32)
            val_probs = np.nan_to_num(val_probs, nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)
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
                epoch, train_loss,
                float(train_metrics["f1_micro"]), float(train_metrics["f1_macro"]),
                val_loss,
                float(val_metrics["f1_micro"]), float(val_metrics["f1_macro"]),
                es_score, float(np.mean(current_thresholds)),
                float(np.std(current_thresholds)), current_lr,
            ])
            log_handle.flush()
            write_epoch_details(
                det_handle, epoch=epoch, label_names=specialist_names,
                thresholds=current_thresholds, train_metrics=train_metrics,
                val_metrics=val_metrics, early_stopping_score=es_score,
            )
            det_handle.flush()

            print(
                f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
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
                torch.save(model.state_dict(), checkpoint_path)
                write_threshold_artifact(
                    threshold_path, specialist_names, best_thresholds,
                    args.threshold_mode, args.threshold_target_metric, best_epoch,
                )

            if epoch >= args.min_epochs_before_stopping and early_stopping(es_score):
                print(f"Stopping early at epoch {epoch}.")
                break
            if epoch >= args.min_epochs_before_stopping and early_stopping.counter > 0:
                print(f"EarlyStopping counter: {early_stopping.counter}/{early_stopping.patience}")

    # ── Test evaluation ─────────────────────────────────────────────────────────
    print("\nTraining complete. Evaluating on test set...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_loss, test_labels, test_probs = evaluate(model, test_loader, criterion, device, use_amp)
    test_preds = threshold_predictions(test_probs, best_thresholds).astype(np.int32)
    test_metrics = compute_metrics(test_labels.astype(np.int32), test_preds)

    elapsed = time.time() - start_time
    print(f"\nBest epoch:    {best_epoch}")
    print(f"Best score:    {best_score:.4f} ({args.early_stopping_metric})")
    print(f"Test f1_micro: {test_metrics['f1_micro']:.4f}")
    print(f"Test f1_macro: {test_metrics['f1_macro']:.4f}")
    print("\nPer-class test F1 (specialist):")
    if "per_class_f1" in test_metrics:
        for name, f1 in zip(specialist_names, test_metrics["per_class_f1"]):
            print(f"  {name:20s}: {f1:.4f}")

    write_summary(
        summary_path=summary_path,
        elapsed_seconds=elapsed,
        completed_epochs=completed_epochs,
        requested_epochs=args.epochs,
        used_data_files=0,
        total_data_files=0,
        split_path=split_path,
        best_epoch=best_epoch,
        best_score=best_score,
        best_metric_name=args.early_stopping_metric,
        best_val_loss=best_val_loss,
        test_loss=test_loss,
        test_metrics=test_metrics,
        final_thresholds=best_thresholds,
    )
    print(f"Summary: {summary_path}")

    # ── Auto-run ensemble evaluation ────────────────────────────────────────────
    ensemble_script = CURRENT_DIR / "ensemble_predict.py"
    cnn_pickle = Path("benchmark/cnn/models/ir/k_fold/results.pickle")
    if ensemble_script.exists() and cnn_pickle.exists():
        import subprocess
        print("\n" + "=" * 80)
        print("Running ensemble evaluation automatically...")
        print("=" * 80)
        cmd = [
            sys.executable, str(ensemble_script),
            "--specialist-dir", str(args.output_dir),
            "--cnn-pickle", str(cnn_pickle),
            "--features-path", str(args.features_path),
            "--split-path", str(split_path),
            "--device", args.device,
            "--seed", str(args.seed),
        ]
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"WARNING: ensemble_predict.py exited with code {result.returncode}")
    else:
        missing = []
        if not ensemble_script.exists():
            missing.append(str(ensemble_script))
        if not cnn_pickle.exists():
            missing.append(str(cnn_pickle))
        print(f"\nSkipping ensemble evaluation (missing: {', '.join(missing)})")
        print(f"Run manually:\n  python {ensemble_script} --specialist-dir {args.output_dir} --cnn-pickle {cnn_pickle}")


if __name__ == "__main__":
    main()
