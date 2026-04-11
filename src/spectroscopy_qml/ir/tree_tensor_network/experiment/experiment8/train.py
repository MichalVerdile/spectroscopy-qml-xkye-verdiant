"""
Experiment 8 — Training: Quantum Transfer Learning auf TTN Exp6 Backbone.

Der TTN-Exp6-Backbone (F1-micro ~0.87) wird eingefroren. Nur der klassische
Adapter (64→4), der Quantenschaltkreis (4 Qubits) und der Kopf (4→37) werden
trainiert.

Run vom Repo-Root:
    python src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment8/train.py

Argumente (alle optional, Defaults unten):
    --checkpoint   Pfad zur Exp6 .pt Datei
    --data-dir     Pfad zum data/raw Verzeichnis
    --max-files    Anzahl Parquet-Files (None = alle)
    --epochs       Trainingsepochen (Standard 50)
    --batch-size   Batch-Grösse (Standard 64)
    --lr           Lernrate (Standard 1e-3)
    --output-dir   Ausgabeverzeichnis

Outputs:
    results/training_log.csv
    results/summary.txt
    results/per_class_f1.png
    results/confusion_matrices.png
    models/qtl_best.pt
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment5.data_loader import (
    load_ir_data,
    load_or_create_split_indices,
    prepare_dataloaders_from_split_indices,
    FUNCTIONAL_GROUPS,
)
from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment8.model import (
    QuantumTransferModel,
    N_CLASSES,
)
from spectroscopy_qml.ir.qml.diagnostics import run_diagnostics

# ── Defaults ──────────────────────────────────────────────────────────────────

ROOT        = Path(__file__).resolve().parent
_EXP6_CKPT  = Path(
    "src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment6"
    "/results/full_dataset_run_20260409_194200_cpu/ttn_ir_best.pt"
)
_DATA_DIR   = Path("data/raw")
_OUTPUT_DIR = ROOT / "results"
_MODEL_DIR  = ROOT / "models"


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Experiment 8: Quantum Transfer Learning auf TTN Exp6 Backbone"
    )
    p.add_argument("--checkpoint",  type=Path, default=_EXP6_CKPT)
    p.add_argument("--data-dir",    type=Path, default=_DATA_DIR)
    p.add_argument("--output-dir",  type=Path, default=_OUTPUT_DIR)
    p.add_argument("--split-path",  type=Path, default=None)
    p.add_argument("--max-files",   type=int,  default=None,
                   help="None = alle Dateien (empfohlen: gleich wie Exp6)")
    p.add_argument("--epochs",      type=int,  default=50)
    p.add_argument("--batch-size",  type=int,  default=64)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--weight-decay",type=float, default=1e-6)
    p.add_argument("--patience",    type=int,  default=15)
    p.add_argument("--n-qubits",    type=int,  default=4)
    p.add_argument("--n-layers",    type=int,  default=3)
    p.add_argument("--pos-weight-max", type=float, default=50.0)
    p.add_argument("--apply-snv",   action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed",        type=int,  default=42)
    p.add_argument("--device",      choices=["auto","cpu","cuda","mps"], default="auto")
    p.add_argument("--num-workers", type=int,  default=0)
    return p


# ── Helpers ───────────────────────────────────────────────────────────────────

def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        if torch.cuda.is_available():    return torch.device("cuda")
        if torch.backends.mps.is_available(): return torch.device("mps")
        return torch.device("cpu")
    return torch.device(choice)


def compute_pos_weight(y_train: np.ndarray, device: torch.device,
                       pw_max: float) -> torch.Tensor:
    n   = y_train.shape[0]
    pos = y_train.sum(axis=0).clip(min=1)
    neg = n - pos
    w   = (neg / pos).clip(max=pw_max)
    return torch.FloatTensor(w).to(device)


def tune_thresholds_per_class(y_true: np.ndarray,
                               y_probs: np.ndarray) -> np.ndarray:
    grid = np.arange(0.05, 0.96, 0.05)
    thresholds = np.full(y_true.shape[1], 0.5)
    for c in range(y_true.shape[1]):
        best_t, best_f1 = 0.5, 0.0
        for t in grid:
            f1 = f1_score(y_true[:, c], (y_probs[:, c] >= t).astype(int),
                          zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[c] = best_t
    return thresholds


def train_epoch(model, loader, criterion, optimizer,
                device, thresholds) -> tuple[float, float]:
    model.train()
    total_loss, all_probs, all_labels = 0.0, [], []
    for spectra, labels in loader:
        spectra, labels = spectra.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(spectra)
        loss   = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * spectra.size(0)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.cpu().numpy())
    avg_loss  = total_loss / len(loader.dataset)
    all_probs  = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)
    f1 = f1_score(all_labels, (all_probs >= thresholds).astype(int),
                  average="micro", zero_division=0)
    return avg_loss, f1


@torch.no_grad()
def evaluate(model, loader, criterion, device,
             thresholds, return_probs: bool = False):
    model.eval()
    total_loss, all_probs, all_labels = 0.0, [], []
    for spectra, labels in loader:
        spectra, labels = spectra.to(device), labels.to(device)
        logits = model(spectra)
        loss   = criterion(logits, labels)
        total_loss += loss.item() * spectra.size(0)
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    avg_loss   = total_loss / len(loader.dataset)
    all_probs  = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)
    preds      = (all_probs >= thresholds).astype(int)
    f1_mic = f1_score(all_labels, preds, average="micro", zero_division=0)
    f1_mac = f1_score(all_labels, preds, average="macro", zero_division=0)
    if return_probs:
        return avg_loss, f1_mic, f1_mac, all_labels, all_probs
    return avg_loss, f1_mic, f1_mac, all_labels, preds


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = build_parser().parse_args()
    device = resolve_device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir   = args.output_dir
    model_dir = _MODEL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = model_dir / "qtl_best.pt"
    log_csv         = out_dir / "training_log.csv"
    summary_txt     = out_dir / "summary.txt"

    print("=" * 70)
    print("Experiment 8 — Quantum Transfer Learning")
    print(f"  Backbone:  {args.checkpoint}")
    print(f"  Qubits:    {args.n_qubits}  Layers: {args.n_layers}")
    print(f"  Device:    {device}")
    print("=" * 70)

    # ── Daten ─────────────────────────────────────────────────────────────────
    print(f"\nLade Daten (max_files={args.max_files}) …")
    X, y = load_ir_data(
        args.data_dir,
        target_length=1800,
        max_files=args.max_files,
        apply_snv=args.apply_snv,
    )
    print(f"  {X.shape[0]:,} Samples  |  {y.shape[1]} Klassen")

    split_path = args.split_path or (
        out_dir / f"data_split_seed{args.seed}"
        f"{'_all' if args.max_files is None else f'_files{args.max_files}'}.npz"
    )
    split_indices = load_or_create_split_indices(
        X, y,
        split_path=split_path,
        train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
        seed=args.seed,
    )
    train_loader, val_loader, test_loader = prepare_dataloaders_from_split_indices(
        X, y,
        split_indices=split_indices,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # ── Modell ────────────────────────────────────────────────────────────────
    print(f"\nBaue QuantumTransferModel …")
    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Exp6-Checkpoint nicht gefunden: {args.checkpoint}\n"
            "Bitte zuerst Experiment 6 trainieren oder --checkpoint anpassen."
        )
    model = QuantumTransferModel(
        checkpoint_path=args.checkpoint,
        device=device,
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
    ).to(device)
    model.param_summary()

    # ── Loss / Optimizer ──────────────────────────────────────────────────────
    train_labels = np.vstack([y.numpy() for _, y in train_loader])
    pos_weight   = compute_pos_weight(train_labels, device, args.pos_weight_max)
    print(f"\npos_weight  min={pos_weight.min():.1f}  max={pos_weight.max():.1f}"
          f"  mean={pos_weight.mean():.1f}  (cap={args.pos_weight_max})")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # Nur die trainierbaren Parameter: Adapter + qlayer + head
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                  patience=5, min_lr=1e-6)

    # ── CSV-Header ─────────────────────────────────────────────────────────────
    with open(log_csv, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "train_loss", "train_f1_micro",
            "val_loss", "val_f1_micro", "val_f1_macro", "blended_f1",
            "mean_threshold", "lr",
        ])

    # ── Trainingsloop ─────────────────────────────────────────────────────────
    print(f"\nTraining  epochs={args.epochs}  batch={args.batch_size}"
          f"  patience={args.patience}\n")

    thresholds  = np.full(N_CLASSES, 0.5)
    best_score  = -1.0
    best_state  = None
    best_thr    = thresholds.copy()
    no_improve  = 0
    t_start     = time.time()
    last_epoch  = 1

    for epoch in range(1, args.epochs + 1):
        last_epoch = epoch
        t_ep = time.time()

        train_loss, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device, thresholds)

        _, _, _, val_labels, val_probs = evaluate(
            model, val_loader, criterion, device, thresholds, return_probs=True)

        thresholds = tune_thresholds_per_class(val_labels, val_probs)
        val_preds  = (val_probs >= thresholds).astype(int)
        val_f1_mic = f1_score(val_labels, val_preds, average="micro", zero_division=0)
        val_f1_mac = f1_score(val_labels, val_preds, average="macro", zero_division=0)
        blended    = 0.5 * val_f1_mic + 0.5 * val_f1_mac

        val_loss, *_ = evaluate(model, val_loader, criterion, device, thresholds)
        scheduler.step(val_loss)
        lr_now = optimizer.param_groups[0]["lr"]
        ep_sec = time.time() - t_ep

        print(
            f"Ep {epoch:03d}/{args.epochs}  "
            f"loss={train_loss:.4f}  tr_f1={train_f1:.4f}  "
            f"val_mic={val_f1_mic:.4f}  val_mac={val_f1_mac:.4f}  "
            f"blend={blended:.4f}  lr={lr_now:.1e}  ({ep_sec:.1f}s)"
        )

        with open(log_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, train_loss, train_f1,
                val_loss, val_f1_mic, val_f1_mac, blended,
                float(thresholds.mean()), lr_now,
            ])

        if blended > best_score:
            best_score = blended
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_thr   = thresholds.copy()
            torch.save({
                "epoch":            epoch,
                "model_state_dict": best_state,
                "thresholds":       best_thr,
                "val_f1_micro":     val_f1_mic,
                "val_f1_macro":     val_f1_mac,
                "blended_f1":       blended,
                "n_qubits":         args.n_qubits,
                "n_layers":         args.n_layers,
                "checkpoint_src":   str(args.checkpoint),
            }, best_model_path)
            print(f"  ✓ New best  blended={best_score:.4f}")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"\nEarly stopping  (patience={args.patience})")
                break

    total_sec = time.time() - t_start
    print(f"\nFertig in {total_sec:.0f}s  ({total_sec/60:.1f} min)")

    # ── Test-Evaluation ───────────────────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)
        thresholds = best_thr

    test_loss, test_f1_mic, test_f1_mac, test_labels, test_preds = evaluate(
        model, test_loader, criterion, device, thresholds)

    print(f"\nTest  loss={test_loss:.4f}  "
          f"f1_micro={test_f1_mic:.4f}  f1_macro={test_f1_mac:.4f}")

    # ── Vergleichstabelle ─────────────────────────────────────────────────────
    benchmarks = [
        ("TTN Exp6 (Backbone)",      0.8697, 0.6240),
        ("QML Re-uploading (8 Q)",   0.3968, "—"),
        ("Exp8 QTL (4 Q, ours)",     test_f1_mic, test_f1_mac),
    ]
    print("\n" + "─" * 52)
    print(f"{'Modell':<30}  {'F1-micro':>10}  {'F1-macro':>10}")
    print("─" * 52)
    for name, mic, mac in benchmarks:
        mic_s = f"{mic:.4f}" if isinstance(mic, float) else mic
        mac_s = f"{mac:.4f}" if isinstance(mac, float) else mac
        print(f"{name:<30}  {mic_s:>10}  {mac_s:>10}")
    print("─" * 52)

    # ── Per-class Diagnostics ─────────────────────────────────────────────────
    run_diagnostics(
        test_labels, test_preds,
        out_dir=out_dir,
        model_name=f"Exp8 QTL ({args.n_qubits} Qubits, {args.n_layers} Layers)",
        benchmark_f1=0.89,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    with open(summary_txt, "w") as f:
        f.write("Experiment 8 — Quantum Transfer Learning\n")
        f.write("=" * 50 + "\n")
        f.write(f"backbone        : {args.checkpoint}\n")
        f.write(f"n_qubits        : {args.n_qubits}\n")
        f.write(f"n_layers        : {args.n_layers}\n")
        f.write(f"max_files       : {args.max_files}\n")
        f.write(f"samples         : {X.shape[0]}\n")
        f.write(f"batch_size      : {args.batch_size}\n")
        f.write(f"epochs_run      : {last_epoch}\n")
        f.write(f"best_blended    : {best_score:.4f}\n")
        f.write(f"test_f1_micro   : {test_f1_mic:.4f}\n")
        f.write(f"test_f1_macro   : {test_f1_mac:.4f}\n")
        f.write(f"training_time   : {total_sec:.0f}s\n")

    print(f"\nSummary → {summary_txt}")
    print(f"Log     → {log_csv}")
    print(f"Modell  → {best_model_path}")


if __name__ == "__main__":
    main()
