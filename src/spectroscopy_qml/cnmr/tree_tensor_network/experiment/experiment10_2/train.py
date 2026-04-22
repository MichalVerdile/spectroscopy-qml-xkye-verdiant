"""Training entry point for C-NMR experiment 10.2 with quantile normalization."""

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

from spectroscopy_qml.ir.mps_encoder.data_loader import (  # noqa: E402
    FUNCTIONAL_GROUPS,
    load_cnmr_data,
)
from spectroscopy_qml.cnmr.tree_tensor_network.experiment.experiment10_2.data_loader import (  # noqa: E402
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
from spectroscopy_qml.cnmr.tree_tensor_network.experiment.experiment10_2.model import (  # noqa: E402
    DEFAULT_SEGMENT_STRIDE,
    DEFAULT_SEGMENT_WINDOW_SIZE,
    TTNCNMRClassifier10_2,
)
from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment10.train import (  # noqa: E402
    ensure_finite_tensor,
    evaluate_with_probs_amp,
    resolve_cache_path,
    resolve_compile_enabled,
    sanitize_binary_targets_and_probs,
    train_epoch_amp,
    write_summary,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train experiment10.2 for C-NMR: TTN classifier with Lorentzian-smoothed feature "
            "channels, direct segmented states, linear readout, and quantile normalization."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/spectroscopy_qml/cnmr/tree_tensor_network/experiment/experiment10_2/results"),
    )
    parser.add_argument("--split-path", type=Path, default=None)
    parser.add_argument("--overwrite-split", action="store_true")
    parser.add_argument("--input-dim", type=int, default=10000)
    parser.add_argument("--num-labels", type=int, default=len(FUNCTIONAL_GROUPS))
    parser.add_argument(
        "--specialist-indices",
        type=str,
        default=None,
        help="Comma-separated class indices to train on (e.g. '19,33,21,13,28'). Subsets labels.",
    )
    parser.add_argument("--chi", type=int, default=64)
    parser.add_argument("--segment-window-size", type=int, default=DEFAULT_SEGMENT_WINDOW_SIZE)
    parser.add_argument("--segment-stride", type=int, default=DEFAULT_SEGMENT_STRIDE)
    parser.add_argument("--segment-mode", choices=["overlap", "dual_offset"], default="overlap")
    parser.add_argument("--segment-offset", type=int, default=None)
    parser.add_argument("--segment-state-normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--merge-mode", choices=["strict", "relaxed"], default="relaxed")
    parser.add_argument("--merge-residual-weight", type=float, default=0.1)
    parser.add_argument("--merge-renormalize-output", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--lorentz-gamma",
        type=float,
        default=3.0,
        help="Half-width at half-maximum of the Lorentzian smoothing kernel (in data points).",
    )
    parser.add_argument(
        "--lorentz-kernel-half-width",
        type=int,
        default=15,
        help="Kernel half-width in data points (kernel is truncated at ±this value).",
    )
    parser.add_argument(
        "--lorentz-norm-mode", choices=["max_abs", "z_score", "percentile"], default="max_abs",
        help="Per-channel normalisation: max_abs (original), z_score (÷std), percentile (÷99th pct).",
    )
    parser.add_argument("--apply-snv", action=argparse.BooleanOptionalAction, default=False,
        help="Apply SNV normalization (C-NMR uses quantile normalization, so this is False by default).")
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
    parser.add_argument("--loss-type", choices=["bce", "focal"], default="bce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--pos-weight-power", type=float, default=0.5)
    parser.add_argument("--pos-weight-max", type=float, default=None)
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
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable mixed-precision (AMP) training for ~2x speedup.",
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use torch.compile to fuse ops (requires PyTorch 2.0+).",
    )
    return parser


def build_model(args: argparse.Namespace) -> TTNCNMRClassifier10_2:
    return TTNCNMRClassifier10_2(
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


def describe_args(args: argparse.Namespace, split_path: Path) -> None:
    print("=" * 80)
    print("TTN C-NMR Experiment10.2 Training")
    print("=" * 80)
    print(f"Data dir:                 {args.data_dir}")
    print(f"Output dir:               {args.output_dir}")
    print(f"Split path:               {split_path}")
    print(f"Input dim:                {args.input_dim}")
    print(f"Num labels:               {args.num_labels}")
    print(f"Chi:                      {args.chi}")
    print("Leaf encoder:             none (segments enter TTN directly)")
    print("Readout:                  linear (chi -> num_labels, no hidden expansion)")
    print(f"Segment state normalize:  {args.segment_state_normalize}")
    print(f"Feature channels:         raw + lorentz_d1 + lorentz_d2")
    print(f"Lorentz gamma:            {args.lorentz_gamma}")
    print(f"Lorentz kernel half-width:{args.lorentz_kernel_half_width}")
    print(f"Window size:              {args.segment_window_size}")
    print(f"Stride:                   {args.segment_stride}")
    print(f"Segment mode:             {args.segment_mode}")
    print(f"Segment offset:           {args.segment_offset}")
    print(f"Merge mode:               {args.merge_mode}")
    print(f"Merge residual weight:    {args.merge_residual_weight}")
    print(f"Merge renormalize output: {args.merge_renormalize_output}")
    print(f"Threshold mode:           {args.threshold_mode}")
    print(f"Threshold target metric:  {args.threshold_target_metric}")
    print(f"Early stop metric:        {args.early_stopping_metric}")
    print(f"Min epochs before stop:   {args.min_epochs_before_stopping}")
    print(f"Loss type:                {args.loss_type}")
    print(f"Preprocessing:            Quantile normalization (no SNV applied)")
    print(f"Cache path:               {resolve_cache_path(args)}")
    print(f"Batch size:               {args.batch_size}")
    print(f"Epochs:                   {args.epochs}")
    print(f"Max files:                {args.max_files}")
    print(f"Mixed precision (AMP):    {args.amp}")
    print(f"torch.compile:            {args.compile}")
    print(f"Num workers:              {args.num_workers}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0.")
    if args.threshold_mode == "per_class" and args.threshold_target_metric == "f1_micro":
        raise ValueError("--threshold-mode per_class is incompatible with --threshold-target-metric f1_micro.")
    if not args.data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {args.data_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = resolve_cache_path(args)
    total_data_files = count_available_data_files(args.data_dir)
    used_data_files = resolve_used_file_count(total_data_files, args.max_files)
    split_suffix = "all" if args.max_files is None else f"files{used_data_files}"
    split_path = args.split_path or (args.output_dir / f"data_split_seed{args.seed}_{split_suffix}.npz")
    describe_args(args, split_path)

    config_path = args.output_dir / "run_config.json"
    checkpoint_path = args.output_dir / "ttn_cnmr_best.pt"
    log_path = args.output_dir / "training_log.csv"
    details_path = args.output_dir / "training_details.jsonl"
    summary_path = args.output_dir / "summary.txt"
    threshold_path = args.output_dir / "selected_thresholds.json"

    config_path.write_text(json.dumps(vars(args), indent=2, default=str) + "\n")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    print(f"Device: {device}")

    if args.specialist_indices is not None:
        args.num_labels = len(args.specialist_indices.split(","))

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    model = build_model(args)
    run_synthetic_preflight(model, args, device)

    print("\nLoading C-NMR data (interpolated, NOT quantile-normalized yet)...")
    print("⚠️  WARNING: Quantile normalization should be applied during data preprocessing.")
    print("⚠️  Currently using: linear interpolation only (no QN applied)")
    X, y = load_cnmr_data(
        data_dir=args.data_dir,
        target_length=args.input_dim,
        max_files=args.max_files,
        apply_snv=False,  # C-NMR uses quantile normalization in preprocessing, not SNV
        cache_path=cache_path,
        overwrite_cache=args.overwrite_cache,
    )
    if X.shape[1] != args.input_dim:
        raise RuntimeError(f"Loaded spectra have width {X.shape[1]}, expected {args.input_dim}.")
    specialist_indices = None
    if args.specialist_indices is not None:
        specialist_indices = [int(i) for i in args.specialist_indices.split(",")]

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
    if specialist_indices is not None:
        y = y[:, specialist_indices]
        args.num_labels = len(specialist_indices)
        print(f"Specialist mode: using {args.num_labels} classes at indices {specialist_indices}")

    if y.shape[1] != args.num_labels:
        raise RuntimeError(f"Loaded labels have width {y.shape[1]}, expected {args.num_labels}.")

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

    criterion = build_loss(args.loss_type, pos_weight=pos_weight, focal_gamma=args.focal_gamma)
    run_real_batch_preflight(model, train_loader, device, criterion)

    if args.check_only:
        print("\nCheck-only mode finished successfully.")
        return

    use_amp = args.amp and device.type == "cuda"
    scaler = GradScaler() if use_amp else None
    if use_amp:
        print("Mixed precision (AMP) enabled.")
    elif args.amp and device.type != "cuda":
        print(f"AMP requested but device is {device.type}; falling back to fp32.")

    compile_enabled = resolve_compile_enabled(args.compile, device)
    if args.compile and not compile_enabled:
        print(f"torch.compile requested but disabled on device {device.type}; using eager mode.")

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
    _all_label_names = list(FUNCTIONAL_GROUPS.keys())
    label_names = (
        [_all_label_names[i] for i in specialist_indices]
        if specialist_indices is not None
        else _all_label_names
    )
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
            train_loss, train_metrics = train_epoch_amp(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                grad_clip_norm=args.grad_clip_norm,
                scaler=scaler,
            )
            val_loss, val_labels, val_probs = evaluate_with_probs_amp(
                model,
                val_loader,
                criterion,
                device,
                use_amp=use_amp,
            )
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
            details_handle.flush()

            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_f1_micro={val_metrics['f1_micro']:.4f} | "
                f"val_f1_macro={val_metrics['f1_macro']:.4f} | "
                f"score={early_stopping_score:.4f} | "
                f"lr={current_lr:.2e}"
            )

            completed_epochs = epoch
            if early_stopping_score > best_score:
                best_score = early_stopping_score
                best_val_loss = val_loss
                best_epoch = epoch
                best_thresholds = current_thresholds.astype(np.float32, copy=True)
                torch.save(model.state_dict(), checkpoint_path)
                write_threshold_artifact(
                    threshold_path,
                    label_names,
                    best_thresholds,
                    args.threshold_mode,
                    args.threshold_target_metric,
                    best_epoch,
                )

            if epoch >= args.min_epochs_before_stopping and early_stopping(early_stopping_score):
                print(f"Stopping early at epoch {epoch}.")
                break
            if epoch >= args.min_epochs_before_stopping and early_stopping.counter > 0:
                print(f"EarlyStopping counter: {early_stopping.counter}/{early_stopping.patience}")

    print("\nTraining complete.")

    best_state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(best_state)

    test_loss, test_labels, test_probs = evaluate_with_probs_amp(
        model,
        test_loader,
        criterion,
        device,
        use_amp=use_amp,
    )
    test_preds = threshold_predictions(test_probs, best_thresholds)
    test_metrics = compute_metrics(test_labels, test_preds)

    elapsed_seconds = time.time() - start_time
    print(f"Best epoch:      {best_epoch}")
    print(f"Best score:      {best_score:.4f} ({args.early_stopping_metric})")
    print(f"Best val loss:   {best_val_loss:.4f}")
    print(f"Test loss:       {test_loss:.4f}")
    print(f"Test f1_micro:   {test_metrics['f1_micro']:.4f}")
    print(f"Test f1_macro:   {test_metrics['f1_macro']:.4f}")

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
    print(f"Summary:         {summary_path}")


if __name__ == "__main__":
    main()
