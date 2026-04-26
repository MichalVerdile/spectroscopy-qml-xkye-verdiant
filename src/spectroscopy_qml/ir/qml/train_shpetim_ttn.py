"""Train TTN-guided Shpetim VQC specialist (frozen TTN 10.2 + VQC head)."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from spectroscopy_qml.ir.qml.model_shpetim_ttn import TTNShpetimEnsemble, load_ttn102
from spectroscopy_qml.ir.qml.quantum_model_shpetim import (
    TTN10_2_HARD_CLASS_INDICES,
    TTN10_2_HARD_CLASS_NAMES,
    N_SPECIALIST_CLASSES,
    select_specialist_labels,
)
from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment5.losses import build_loss
from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment5.train import (
    get_pos_weight,
    EarlyStopping,
)

DEFAULT_TTN_CHECKPOINT = Path(
    "src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment10_2"
    "/results/full_dataset_run_20260417_173715_percentile/ttn_ir_best.pt"
)
DEFAULT_TTN_CONFIG = Path(
    "src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment10_2"
    "/results/full_dataset_run_20260417_173715_percentile/run_config.json"
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train TTN-guided Shpetim VQC specialist.")
    p.add_argument("--spectra-cache", type=Path,
                   default=Path("data/cache/ir_spectra_len1800_snv_all.npz"))
    p.add_argument("--ttn-checkpoint", type=Path, default=DEFAULT_TTN_CHECKPOINT)
    p.add_argument("--ttn-config", type=Path, default=DEFAULT_TTN_CONFIG)
    p.add_argument("--split-path", type=Path, default=None)
    p.add_argument("--output-dir", type=Path,
                   default=Path("src/spectroscopy_qml/ir/qml/results/shpetim_ttn"))
    p.add_argument("--n-qubits", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=12)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--pos-weight-max", type=float, default=50.0)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--min-epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    p.add_argument("--check-only", action="store_true")
    return p


def tune_thresholds(y_true: np.ndarray, y_probs: np.ndarray) -> np.ndarray:
    grid = np.arange(0.05, 0.96, 0.05)
    thresholds = np.full(y_true.shape[1], 0.5, dtype=np.float32)
    for c in range(y_true.shape[1]):
        best_t, best_f1 = 0.5, -1.0
        for t in grid:
            f1 = f1_score(y_true[:, c], (y_probs[:, c] >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        thresholds[c] = best_t
    return thresholds


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * x_batch.size(0)
        all_preds.append((torch.sigmoid(logits).detach().cpu().numpy() >= 0.5).astype(np.int32))
        all_labels.append(y_batch.detach().cpu().numpy().astype(np.int32))
    n = sum(len(c) for c in all_labels)
    labels = np.vstack(all_labels)
    preds  = np.vstack(all_preds)
    return total_loss / n, float(f1_score(labels, preds, average="micro", zero_division=0))


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_probs, all_labels = 0.0, [], []
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        logits = model(x_batch)
        total_loss += criterion(logits, y_batch).item() * x_batch.size(0)
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(y_batch.cpu().numpy().astype(np.int32))
    n = sum(len(c) for c in all_labels)
    return total_loss / n, np.vstack(all_labels), np.vstack(all_probs).astype(np.float32)


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_config.json").write_text(
        json.dumps(vars(args), indent=2, default=str) + "\n"
    )

    print(f"Device: {device}")
    print(f"Specialist classes: {list(TTN10_2_HARD_CLASS_NAMES)}")

    print("Loading TTN 10.2 backbone...")
    ttn_config = json.loads(args.ttn_config.read_text())
    ttn = load_ttn102(args.ttn_checkpoint, ttn_config, device)

    print("Loading spectra...")
    cache  = np.load(args.spectra_cache)
    X      = cache["X"].astype(np.float32)
    y_full = cache["y"].astype(np.int32)
    print(f"Loaded X={X.shape}, y={y_full.shape}")

    spec_idx = list(TTN10_2_HARD_CLASS_INDICES)
    y_spec   = y_full[:, spec_idx]

    # Split — reuse existing split if provided
    if args.split_path and args.split_path.exists():
        split = np.load(args.split_path)
        train_idx, val_idx, test_idx = split["train_indices"], split["val_indices"], split["test_indices"]
        print(f"Loaded split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    else:
        rng = np.random.default_rng(args.seed)
        idx = rng.permutation(len(X))
        n_train = int(0.8 * len(X))
        n_val   = int(0.1 * len(X))
        train_idx = idx[:n_train]
        val_idx   = idx[n_train:n_train + n_val]
        test_idx  = idx[n_train + n_val:]
        print(f"Created split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    def make_loader(idx, shuffle):
        return DataLoader(
            TensorDataset(torch.from_numpy(X[idx]).float(),
                          torch.from_numpy(y_spec[idx]).float()),
            batch_size=args.batch_size, shuffle=shuffle,
        )

    train_loader = make_loader(train_idx, shuffle=True)
    val_loader   = make_loader(val_idx,   shuffle=False)
    test_loader  = make_loader(test_idx,  shuffle=False)

    pos_weight = get_pos_weight(y_spec[train_idx], device,
                                power=1.0, max_value=args.pos_weight_max)
    print(f"pos_weight: min={pos_weight.min():.1f} max={pos_weight.max():.1f}")

    model = TTNShpetimEnsemble(
        ttn=ttn,
        specialist_indices=tuple(spec_idx),
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable:,} trainable / {total:,} total (TTN frozen)")

    if args.check_only:
        xb, yb = next(iter(train_loader))
        out = model(xb.to(device))
        print(f"Check-only OK: x={tuple(xb.shape)} logits={tuple(out.shape)}")
        return

    criterion = build_loss("focal", pos_weight=pos_weight, focal_gamma=args.focal_gamma)
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6)
    stopper   = EarlyStopping(patience=args.patience, min_delta=1e-4)

    log_path  = args.output_dir / "training_log.csv"
    ckpt_path = args.output_dir / "best_model.pt"
    thresh_path = args.output_dir / "selected_thresholds.json"

    best_score, best_state, best_thresholds = float("-inf"), None, None
    thresholds = np.full(N_SPECIALIST_CLASSES, 0.5, dtype=np.float32)
    t0 = time.time()

    with log_path.open("w", newline="") as f:
        csv.writer(f).writerow(["epoch","train_loss","train_f1","val_loss",
                                 "val_f1_micro","val_f1_macro","score","lr"])

    for epoch in range(1, args.epochs + 1):
        t_ep = time.time()
        train_loss, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_labels, val_probs = evaluate(model, val_loader, criterion, device)

        thresholds  = tune_thresholds(val_labels, val_probs)
        val_preds   = (val_probs >= thresholds).astype(np.int32)
        val_micro   = float(f1_score(val_labels, val_preds, average="micro", zero_division=0))
        val_macro   = float(f1_score(val_labels, val_preds, average="macro", zero_division=0))
        score       = 0.5 * val_micro + 0.5 * val_macro
        lr          = float(optimizer.param_groups[0]["lr"])

        scheduler.step(score)
        print(f"Ep {epoch:03d}/{args.epochs} "
              f"loss={train_loss:.4f} tr_f1={train_f1:.4f} "
              f"val_mic={val_micro:.4f} val_mac={val_macro:.4f} "
              f"score={score:.4f} lr={lr:.1e} ({time.time()-t_ep:.0f}s)")

        with log_path.open("a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, train_f1, val_loss,
                                     val_micro, val_macro, score, lr])

        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_thresholds = thresholds.copy()
            torch.save(best_state, ckpt_path)
            print(f"  New best score={best_score:.4f}")

        if epoch >= args.min_epochs and stopper(score):
            print(f"Early stopping at epoch {epoch}.")
            break

    # Test evaluation
    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss, test_labels, test_probs = evaluate(model, test_loader, criterion, device)
    test_preds  = (test_probs >= best_thresholds).astype(np.int32)
    test_micro  = float(f1_score(test_labels, test_preds, average="micro", zero_division=0))
    test_macro  = float(f1_score(test_labels, test_preds, average="macro", zero_division=0))
    per_class   = f1_score(test_labels, test_preds, average=None, zero_division=0)

    print(f"\nTest: micro={test_micro:.4f}  macro={test_macro:.4f}  "
          f"elapsed={time.time()-t0:.0f}s")
    print("Per-class F1:")
    for name, f1 in zip(TTN10_2_HARD_CLASS_NAMES, per_class):
        print(f"  {name:20s}: {f1:.4f}")

    thresh_payload = {name: float(t) for name, t in zip(TTN10_2_HARD_CLASS_NAMES, best_thresholds)}
    thresh_path.write_text(json.dumps(thresh_payload, indent=2) + "\n")

    summary = {
        "test_f1_micro": test_micro, "test_f1_macro": test_macro,
        "best_val_score": best_score,
        "per_class_f1": {n: float(f) for n, f in zip(TTN10_2_HARD_CLASS_NAMES, per_class)},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
