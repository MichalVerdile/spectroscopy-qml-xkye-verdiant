"""Experiment 11: TTN specialist training for high-impact functional groups.

Trains a TTN model on a subset of the 37 functional group labels — those that
contribute strongly to the CNN baseline's micro-F1 loss in absolute error count.
The specialist is later combined with the CNN in ensemble_predict.py.

Default specialist classes (indices into FUNCTIONAL_GROUPS order):
    17  Haloalkane
    16  Ether
    28  Sulfide
    24  Ketone
     5  Alkene
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

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = Path(__file__).resolve().parents[5]
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

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
    run_real_batch_preflight,
    run_synthetic_preflight,
    select_early_stopping_score,
    threshold_predictions,
    tune_thresholds,
    write_epoch_details,
    write_threshold_artifact,
)
from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment10_4.model import (  # noqa: E402
    DEFAULT_SEGMENT_STRIDE,
    DEFAULT_SEGMENT_WINDOW_SIZE,
    TTNIRClassifier10_4,
)
from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment10.train import (  # noqa: E402
    evaluate_with_probs_amp,
    resolve_cache_path,
    resolve_compile_enabled,
    train_epoch_amp,
    write_summary,
)

# Default specialist class indices chosen by absolute error contribution to
# micro-F1 loss (high FN / total-error load in the CNN baseline).
DEFAULT_SPECIALIST_INDICES = [17, 16, 28, 24, 5]

ALL_LABEL_NAMES = list(FUNCTIONAL_GROUPS.keys())


def parse_indices(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",")]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train TTN specialist model for hard functional group classes."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
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
        help=(
            "Comma-separated label indices to specialise on. "
            f"Default: {DEFAULT_SPECIALIST_INDICES} "
            f"({', '.join(ALL_LABEL_NAMES[i] for i in DEFAULT_SPECIALIST_INDICES)})"
        ),
    )
    parser.add_argument("--input-dim", type=int, default=1800)
    parser.add_argument("--chi", type=int, default=64)
    parser.add_argument("--segment-window-size", type=int, default=DEFAULT_SEGMENT_WINDOW_SIZE)
    parser.add_argument("--segment-stride", type=int, default=DEFAULT_SEGMENT_STRIDE)
    parser.add_argument("--segment-mode", choices=["overlap", "dual_offset"], default="overlap")
    parser.add_argument("--segment-offset", type=int, default=None)
    parser.add_argument("--segment-state-normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--merge-mode", choices=["strict", "relaxed"], default="relaxed")
    parser.add_argument("--merge-residual-weight", type=float, default=0.1)
    parser.add_argument("--merge-renormalize-output", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sg-window-length", type=int, default=11)
    parser.add_argument("--sg-polyorder", type=int, default=3)
    parser.add_argument(
        "--sg-norm-mode", choices=["max_abs", "z_score", "percentile"], default="max_abs"
    )
    parser.add_argument("--apply-snv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
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
    parser.add_argument("--loss-type", choices=["bce", "focal"], default="focal")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    # Higher pos-weight-power for rare class focus
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
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--preflight-batch-size", type=int, default=4)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-compile", dest="compile", action="store_false")
    return parser


def build_model(args: argparse.Namespace, num_labels: int) -> TTNIRClassifier10_4:
    return TTNIRClassifier10_4(
        num_labels=num_labels,
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
        sg_window_length=args.sg_window_length,
        sg_polyorder=args.sg_polyorder,
        sg_norm_mode=args.sg_norm_mode,
    )


def describe_args(
    args: argparse.Namespace,
    specialist_indices: list[int],
    specialist_names: list[str],
    split_path: Path,
) -> None:
    print("=" * 80)
    print("TTN Specialist Training (Experiment 11)")
    print("=" * 80)
    print(f"Specialist indices:       {specialist_indices}")
    print(f"Specialist classes:       {specialist_names}")
    print(f"Data dir:                 {args.data_dir}")
    print(f"Output dir:               {args.output_dir}")
    print(f"Split path:               {split_path}")
    print(f"Input dim:                {args.input_dim}")
    print(f"Num labels (specialist):  {len(specialist_indices)}")
    print(f"Chi:                      {args.chi}")
    print(f"SG window length:         {args.sg_window_length}")
    print(f"SG polynomial order:      {args.sg_polyorder}")
    print(f"SG norm mode:             {args.sg_norm_mode}")
    print(f"Loss type:                {args.loss_type} (gamma={args.focal_gamma})")
    print(f"Pos weight power:         {args.pos_weight_power}")
    print(f"Pos weight max:           {args.pos_weight_max}")
    print(f"Early stop metric:        {args.early_stopping_metric}")
    print(f"Apply SNV:                {args.apply_snv}")
    print(f"Batch size:               {args.batch_size}")
    print(f"Epochs:                   {args.epochs}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    specialist_indices = sorted(set(args.specialist_indices))
    if not specialist_indices:
        raise ValueError("--specialist-indices must not be empty.")
    n_all = len(FUNCTIONAL_GROUPS)
    if any(i < 0 or i >= n_all for i in specialist_indices):
        raise ValueError(f"All specialist indices must be in [0, {n_all}).")

    specialist_names = [ALL_LABEL_NAMES[i] for i in specialist_indices]
    num_labels = len(specialist_indices)

    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0.")
    if not args.data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Attach num_labels to args so helper functions work (e.g. resolve_cache_path)
    args.num_labels = num_labels

    cache_path = resolve_cache_path(args)
    total_data_files = count_available_data_files(args.data_dir)
    used_data_files = resolve_used_file_count(total_data_files, args.max_files)
    split_suffix = "all" if args.max_files is None else f"files{used_data_files}"
    split_path = args.split_path or (
        args.output_dir / f"data_split_seed{args.seed}_{split_suffix}.npz"
    )

    describe_args(args, specialist_indices, specialist_names, split_path)

    # Save config including specialist indices
    config = vars(args).copy()
    config["specialist_indices"] = specialist_indices
    config["specialist_names"] = specialist_names
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, default=str) + "\n"
    )

    # Also save a standalone specialist mapping for use during ensemble inference
    specialist_map = {
        "specialist_indices": specialist_indices,
        "specialist_names": specialist_names,
        "all_label_names": ALL_LABEL_NAMES,
    }
    (args.output_dir / "specialist_map.json").write_text(
        json.dumps(specialist_map, indent=2) + "\n"
    )

    checkpoint_path = args.output_dir / "ttn_specialist_best.pt"
    log_path = args.output_dir / "training_log.csv"
    details_path = args.output_dir / "training_details.jsonl"
    summary_path = args.output_dir / "summary.txt"
    threshold_path = args.output_dir / "selected_thresholds.json"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    print(f"Device: {device}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    model = build_model(args, num_labels=num_labels)
    run_synthetic_preflight(model, args, device)

    print("\nLoading data...")
    X, y_full = load_ir_data(
        data_dir=args.data_dir,
        target_length=args.input_dim,
        max_files=args.max_files,
        apply_snv=args.apply_snv,
        cache_path=cache_path,
        overwrite_cache=args.overwrite_cache,
    )

    # Subset labels to specialist classes only
    y = y_full[:, specialist_indices]
    print(f"Label matrix shape after specialist subset: {y.shape}")
    print("Specialist class positive rates:")
    for i, name in zip(specialist_indices, specialist_names):
        rate = y_full[:, i].mean()
        print(f"  {name:20s} (idx={i}): {rate*100:.2f}%")

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
        X, y, split_indices,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    train_labels = y[split_indices["train"]]
    pos_weight = get_pos_weight(
        train_labels, device, power=args.pos_weight_power, max_value=args.pos_weight_max
    )
    print(
        f"Class weights: min={pos_weight.min():.2f}, max={pos_weight.max():.2f}, "
        f"mean={pos_weight.mean():.2f}"
    )

    criterion = build_loss(args.loss_type, pos_weight=pos_weight, focal_gamma=args.focal_gamma)
    run_real_batch_preflight(model, train_loader, device, criterion)

    if args.check_only:
        print("\nCheck-only mode finished.")
        return

    use_amp = args.amp and device.type == "cuda"
    scaler = GradScaler() if use_amp else None
    if use_amp:
        print("Mixed precision (AMP) enabled.")
    elif args.amp and device.type != "cuda":
        print(f"AMP requested but device is {device.type}; using fp32.")

    compile_enabled = resolve_compile_enabled(args.compile, device)
    if compile_enabled:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)
        print("Compilation done.")

    optimizer = Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
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
    threshold_grid = build_threshold_grid(args.threshold_grid_step)
    model = model.to(device)

    print("\nStarting training...")
    start_time = time.time()
    best_score = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    completed_epochs = 0
    best_thresholds = np.full(num_labels, 0.5, dtype=np.float32)

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
            early_stopping_score = select_early_stopping_score(
                val_metrics,
                metric_name=args.early_stopping_metric,
                blend_alpha=args.early_stopping_blend_alpha,
            )

            scheduler.step(early_stopping_score)
            current_lr = optimizer.param_groups[0]["lr"]

            writer.writerow([
                epoch, train_loss,
                float(train_metrics["f1_micro"]), float(train_metrics["f1_macro"]),
                val_loss,
                float(val_metrics["f1_micro"]), float(val_metrics["f1_macro"]),
                early_stopping_score,
                float(np.mean(current_thresholds)), float(np.std(current_thresholds)),
                current_lr,
            ])
            log_handle.flush()
            write_epoch_details(
                details_handle, epoch=epoch, label_names=specialist_names,
                thresholds=current_thresholds, train_metrics=train_metrics,
                val_metrics=val_metrics, early_stopping_score=early_stopping_score,
            )
            details_handle.flush()

            print(
                f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"val_f1_micro={val_metrics['f1_micro']:.4f} | "
                f"val_f1_macro={val_metrics['f1_macro']:.4f} | "
                f"score={early_stopping_score:.4f} | lr={current_lr:.2e}"
            )

            completed_epochs = epoch
            if early_stopping_score > best_score:
                best_score = early_stopping_score
                best_val_loss = val_loss
                best_epoch = epoch
                best_thresholds = current_thresholds.astype(np.float32, copy=True)
                _raw = model._orig_mod if hasattr(model, "_orig_mod") else model
                torch.save(_raw.state_dict(), checkpoint_path)
                write_threshold_artifact(
                    threshold_path, specialist_names, best_thresholds,
                    args.threshold_mode, args.threshold_target_metric, best_epoch,
                )

            if epoch >= args.min_epochs_before_stopping and early_stopping(early_stopping_score):
                print(f"Stopping early at epoch {epoch}.")
                break
            if epoch >= args.min_epochs_before_stopping and early_stopping.counter > 0:
                print(f"EarlyStopping counter: {early_stopping.counter}/{early_stopping.patience}")

    print("\nTraining complete.")
    best_state = torch.load(checkpoint_path, map_location=device)
    _raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    _raw.load_state_dict(best_state)

    test_loss, test_labels, test_probs = evaluate_with_probs_amp(
        model, test_loader, criterion, device, use_amp=use_amp,
    )
    test_preds = threshold_predictions(test_probs, best_thresholds)
    test_metrics = compute_metrics(test_labels, test_preds)

    elapsed_seconds = time.time() - start_time
    print(f"Best epoch:    {best_epoch}")
    print(f"Best score:    {best_score:.4f} ({args.early_stopping_metric})")
    print(f"Test f1_micro: {test_metrics['f1_micro']:.4f}")
    print(f"Test f1_macro: {test_metrics['f1_macro']:.4f}")
    print("\nPer-class test F1 (specialist classes):")
    if "per_class_f1" in test_metrics:
        for name, f1 in zip(specialist_names, test_metrics["per_class_f1"]):
            print(f"  {name:20s}: {f1:.4f}")

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
        final_thresholds=best_thresholds,
    )
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
