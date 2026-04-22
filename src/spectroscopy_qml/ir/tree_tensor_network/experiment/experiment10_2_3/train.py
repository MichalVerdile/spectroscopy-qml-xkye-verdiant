"""Experiment 10.2.3: Train specialist heads with partial TTN fine-tuning.

Top N TTN merge levels + output_norm are unfrozen and trained with a small
backbone LR. Specialist heads use the normal learning rate.
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
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

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
from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment10_2_3.model import (  # noqa: E402
    SPECIALIST_INDICES,
    TTN102CombinedEnsemble,
    load_ttn102,
)

ALL_LABEL_NAMES = list(FUNCTIONAL_GROUPS.keys())

DEFAULT_TTN_CHECKPOINT = Path(
    "src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment10_2"
    "/results/full_dataset_run_20260417_173715_percentile/ttn_ir_best.pt"
)
DEFAULT_TTN_CONFIG = Path(
    "src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment10_2"
    "/results/full_dataset_run_20260417_173715_percentile/run_config.json"
)


def parse_indices(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",")]


def parse_hidden_dims(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",")]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train specialist heads with partial TTN fine-tuning (Exp 10.2.3)."
    )
    parser.add_argument("--spectra-cache", type=Path,
                        default=Path("data/cache/ir_spectra_len1800_snv_all.npz"))
    parser.add_argument("--ttn-checkpoint", type=Path, default=DEFAULT_TTN_CHECKPOINT)
    parser.add_argument("--ttn-config", type=Path, default=DEFAULT_TTN_CONFIG)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("src/spectroscopy_qml/ir/tree_tensor_network"
                                     "/experiment/experiment10_2_3/results"))
    parser.add_argument("--split-path", type=Path, default=None)
    parser.add_argument("--overwrite-split", action="store_true")
    parser.add_argument("--specialist-indices", type=parse_indices,
                        default=SPECIALIST_INDICES)
    parser.add_argument("--hidden-dims", type=parse_hidden_dims, default=[128, 64])
    parser.add_argument("--window-proj-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--finetune-ttn-layers", type=int, default=1,
                        help="Number of top TTN merge levels to unfreeze (0 = fully frozen).")
    parser.add_argument("--oversample", action="store_true", default=True)
    parser.add_argument("--no-oversample", dest="oversample", action="store_false")
    parser.add_argument("--oversample-epoch-multiplier", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5,
                        help="LR for the unfrozen TTN layers (much smaller than head LR).")
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
    parser.add_argument("--pos-weight-max", type=float, default=50.0)
    parser.add_argument("--threshold-mode", choices=["global", "per_class"], default="per_class")
    parser.add_argument("--threshold-target-metric",
                        choices=["f1_micro", "f1_macro", "per_class_f1"],
                        default="per_class_f1")
    parser.add_argument("--threshold-grid-step", type=float, default=0.02)
    parser.add_argument("--early-stopping-metric",
                        choices=["f1_micro", "f1_macro", "blended_f1"], default="blended_f1")
    parser.add_argument("--early-stopping-blend-alpha", type=float, default=0.5)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--min-epochs-before-stopping", type=int, default=30)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--check-only", action="store_true")
    return parser


def make_oversampled_weights(labels: np.ndarray) -> np.ndarray:
    n, c = labels.shape
    class_pos_counts = labels.sum(axis=0).clip(min=1)
    class_weights = n / class_pos_counts
    sample_weights = np.zeros(n, dtype=np.float32)
    for i in range(n):
        pos_mask = labels[i] == 1
        if pos_mask.any():
            sample_weights[i] = class_weights[pos_mask].max()
        else:
            sample_weights[i] = class_weights.min()
    return sample_weights


def make_train_loader(X, y, train_idx, batch_size, num_workers, pin_memory,
                      oversample, epoch_multiplier):
    dataset = TensorDataset(
        torch.from_numpy(X[train_idx]).float(),
        torch.from_numpy(y[train_idx]).float(),
    )
    if oversample:
        weights = make_oversampled_weights(y[train_idx])
        num_samples = int(len(train_idx) * epoch_multiplier)
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(weights),
            num_samples=num_samples,
            replacement=True,
        )
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                          num_workers=num_workers, pin_memory=pin_memory)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=pin_memory)


def make_eval_loader(X, y, idx, batch_size, num_workers, pin_memory):
    return DataLoader(
        TensorDataset(
            torch.from_numpy(X[idx]).float(),
            torch.from_numpy(y[idx]).float(),
        ),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )


def train_epoch(model, loader, criterion, optimizer, device, grad_clip):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []
    for x_batch, y_batch in loader:
        y_np = y_batch.numpy().astype(np.int32)
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * x_batch.size(0)
        all_preds.append((torch.sigmoid(logits).detach().cpu().numpy() >= 0.5).astype(np.int32))
        all_labels.append(y_np)
    n = sum(len(c) for c in all_labels)
    return total_loss / n, compute_metrics(np.concatenate(all_labels), np.concatenate(all_preds))


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_probs, all_labels = 0.0, [], []
    for x_batch, y_batch in loader:
        y_np = y_batch.numpy().astype(np.int32)
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        logits = model(x_batch)
        total_loss += criterion(logits, y_batch).item() * x_batch.size(0)
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(y_np)
    n = sum(len(c) for c in all_labels)
    return total_loss / n, np.concatenate(all_labels), np.concatenate(all_probs).astype(np.float32)


def write_summary(path, *, specialist_names, best_epoch, best_score,
                  completed_epochs, elapsed, test_loss, test_metrics, args):
    lines = [
        "Experiment 10.2.3 — TTN 10.2 Partial Fine-tune + Specialist Head Summary",
        "=" * 80,
        f"Specialist classes:    {', '.join(specialist_names)}",
        f"Hidden dims:           {args.hidden_dims}",
        f"Window proj dim:       {args.window_proj_dim}",
        f"Finetune TTN layers:   {args.finetune_ttn_layers}",
        f"Backbone LR:           {args.backbone_lr}",
        f"Head LR:               {args.learning_rate}",
        f"Oversampling:          {args.oversample}",
        f"Completed epochs:      {completed_epochs}/{args.epochs}",
        f"Elapsed seconds:       {elapsed:.2f}",
        f"Best epoch:            {best_epoch}",
        f"Best score:            {best_score:.6f}",
        f"Test loss:             {test_loss:.6f}",
        f"Test f1_micro:         {float(test_metrics['f1_micro']):.6f}",
        f"Test f1_macro:         {float(test_metrics['f1_macro']):.6f}",
        "", "Per-class F1:",
    ]
    for name, f1 in zip(specialist_names, test_metrics["per_class_f1"], strict=False):
        lines.append(f"  {name:20s}: {float(f1):.6f}")
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = build_parser()
    args = parser.parse_args()

    specialist_indices = sorted(set(args.specialist_indices))
    specialist_names = [ALL_LABEL_NAMES[i] for i in specialist_indices]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_path      = args.split_path or (args.output_dir / f"data_split_seed{args.seed}_all.npz")
    checkpoint_path = args.output_dir / "specialist_best.pt"
    threshold_path  = args.output_dir / "selected_thresholds.json"
    summary_path    = args.output_dir / "summary.txt"
    log_path        = args.output_dir / "training_log.csv"
    details_path    = args.output_dir / "training_details.jsonl"

    (args.output_dir / "run_config.json").write_text(
        json.dumps(vars(args), indent=2, default=str) + "\n"
    )
    (args.output_dir / "specialist_map.json").write_text(
        json.dumps({"specialist_indices": specialist_indices,
                    "specialist_names": specialist_names}, indent=2) + "\n"
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    print(f"Device: {device}")
    print(f"Specialist classes: {specialist_names}")
    print(f"Finetune TTN layers: {args.finetune_ttn_layers}")

    print("\nLoading TTN 10.2 backbone...")
    ttn_config = json.loads(args.ttn_config.read_text())
    ttn = load_ttn102(args.ttn_checkpoint, ttn_config, device)
    print(f"TTN loaded — feature dim: {ttn.output_norm.normalized_shape[0]}")

    print("Loading spectra from cache...")
    cache   = np.load(args.spectra_cache)
    X       = cache["X"].astype(np.float32)
    y_full  = cache["y"].astype(np.int32)
    print(f"Loaded X={X.shape}, y={y_full.shape}")

    specialist_labels = y_full[:, specialist_indices]

    split_indices = load_or_create_split_indices(
        labels=y_full, split_path=split_path,
        train_ratio=args.train_ratio, val_ratio=args.val_ratio,
        test_ratio=args.test_ratio, random_seed=args.seed,
        stratify_multilabel=True, overwrite=args.overwrite_split,
    )

    train_loader = make_train_loader(
        X, specialist_labels, split_indices["train"],
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        oversample=args.oversample,
        epoch_multiplier=args.oversample_epoch_multiplier,
    )
    val_loader  = make_eval_loader(X, specialist_labels, split_indices["val"],
                                   args.batch_size, args.num_workers, device.type == "cuda")
    test_loader = make_eval_loader(X, specialist_labels, split_indices["test"],
                                   args.batch_size, args.num_workers, device.type == "cuda")

    train_labels = specialist_labels[split_indices["train"]]
    pos_weight   = get_pos_weight(train_labels, device,
                                  power=args.pos_weight_power, max_value=args.pos_weight_max)
    print(f"Class pos weights: min={pos_weight.min():.2f}, max={pos_weight.max():.2f}")

    model = TTN102CombinedEnsemble(
        ttn=ttn,
        specialist_indices=specialist_indices,
        hidden_dims=args.hidden_dims,
        window_proj_dim=args.window_proj_dim,
        dropout=args.dropout,
        finetune_layers=args.finetune_ttn_layers,
    ).to(device)

    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params     = list(model.heads.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable:,} trainable / {total:,} total")
    print(f"  backbone (unfrozen): {sum(p.numel() for p in backbone_params):,} params @ lr={args.backbone_lr}")
    print(f"  heads:               {sum(p.numel() for p in head_params):,} params @ lr={args.learning_rate}")

    if args.check_only:
        xb, yb = next(iter(train_loader))
        logits = model(xb.to(device))
        print(f"Check-only OK: X={tuple(xb.shape)}, y={tuple(yb.shape)}, logits={tuple(logits.shape)}")
        return

    criterion = build_loss(args.loss_type, pos_weight=pos_weight, focal_gamma=args.focal_gamma)
    optimizer = Adam(
        [
            {"params": backbone_params, "lr": args.backbone_lr},
            {"params": head_params,     "lr": args.learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=args.lr_scheduler_factor,
                                  patience=args.lr_scheduler_patience,
                                  min_lr=args.lr_scheduler_min_lr)
    threshold_grid = build_threshold_grid(args.threshold_grid_step)
    early_stopping = EarlyStopping(patience=args.early_stopping_patience,
                                   min_delta=args.early_stopping_min_delta)

    best_state, best_thresholds, best_score = None, None, float("-inf")
    best_epoch, completed_epochs = 0, 0
    start_time = time.time()

    with log_path.open("w", newline="") as csv_file, details_path.open("w") as det_file:
        writer = csv.writer(csv_file)
        writer.writerow(["epoch", "train_loss", "train_f1_macro", "val_loss",
                         "val_f1_micro", "val_f1_macro", "score", "lr_head", "lr_backbone"])

        for epoch in range(1, args.epochs + 1):
            train_loss, train_m = train_epoch(model, train_loader, criterion, optimizer,
                                              device, args.grad_clip_norm)
            val_loss, val_labels, val_probs = evaluate(model, val_loader, criterion, device)
            thresholds  = tune_thresholds(val_labels, val_probs, args.threshold_mode,
                                          args.threshold_target_metric, threshold_grid)
            val_metrics = compute_metrics(val_labels, threshold_predictions(val_probs, thresholds))
            score       = select_early_stopping_score(val_metrics, args.early_stopping_metric,
                                                      args.early_stopping_blend_alpha)
            scheduler.step(score)

            lr_head     = float(optimizer.param_groups[1]["lr"])
            lr_backbone = float(optimizer.param_groups[0]["lr"])
            writer.writerow([epoch, float(train_loss), float(train_m["f1_macro"]),
                             float(val_loss), float(val_metrics["f1_micro"]),
                             float(val_metrics["f1_macro"]), float(score),
                             lr_head, lr_backbone])
            csv_file.flush()
            write_epoch_details(det_file, epoch=epoch, label_names=specialist_names,
                                thresholds=thresholds, train_metrics=train_m,
                                val_metrics=val_metrics, early_stopping_score=score)

            print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                  f"val_f1_micro={val_metrics['f1_micro']:.4f} | val_f1_macro={val_metrics['f1_macro']:.4f} | "
                  f"lr_head={lr_head:.2e} | lr_bb={lr_backbone:.2e}")

            if score > best_score:
                best_score      = float(score)
                best_epoch      = epoch
                best_state      = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_thresholds = thresholds.copy()
                torch.save(best_state, checkpoint_path)

            completed_epochs = epoch
            if epoch >= args.min_epochs_before_stopping and early_stopping(score):
                break

    model.load_state_dict(best_state)
    model.to(device)
    test_loss, test_labels, test_probs = evaluate(model, test_loader, criterion, device)
    test_metrics = compute_metrics(test_labels, threshold_predictions(test_probs, best_thresholds))

    write_threshold_artifact(threshold_path, label_names=specialist_names,
                             thresholds=best_thresholds, threshold_mode=args.threshold_mode,
                             threshold_target_metric=args.threshold_target_metric,
                             best_epoch=best_epoch)
    write_summary(summary_path, specialist_names=specialist_names, best_epoch=best_epoch,
                  best_score=best_score, completed_epochs=completed_epochs,
                  elapsed=time.time() - start_time, test_loss=test_loss,
                  test_metrics=test_metrics, args=args)

    print("\nTest metrics")
    print(f"  f1_micro:  {float(test_metrics['f1_micro']):.4f}")
    print(f"  f1_macro:  {float(test_metrics['f1_macro']):.4f}")
    for name, f1 in zip(specialist_names, test_metrics["per_class_f1"], strict=False):
        print(f"  {name:20s}: {float(f1):.4f}")


if __name__ == "__main__":
    main()
