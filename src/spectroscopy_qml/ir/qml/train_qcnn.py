"""
Training script for QCNNIRClassifier (11-qubit amplitude-embedding QCNN + TTN).

Run from the repo root:
    python src/spectroscopy_qml/ir/qml/train_qcnn.py

Outputs (written next to this file):
    results_qcnn/training_log.csv
    results_qcnn/summary.txt
    models_qcnn/qcnn_best.pt

Notes on training speed
-----------------------
Each quantum circuit call on ``lightning.qubit`` processes a 2^11 = 2048-dimensional
state vector with adjoint differentiation over 160 quantum parameters.  Expect
~1–5 seconds per sample on CPU; reduce MAX_FILES or BATCH_SIZE if too slow.
Using a GPU (USE_GPU=True in model_qcnn.py) with ``lightning.gpu`` gives ~10–30×
speed-up if a CUDA device is available.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from spectroscopy_qml.ir.mps_encoder.data_loader import load_ir_data, prepare_dataloaders
from spectroscopy_qml.ir.qml.model_qcnn import QCNNIRClassifier, N_CLASSES, param_summary
from spectroscopy_qml.ir.qml.diagnostics import run_diagnostics

# ── Configuration ─────────────────────────────────────────────────────────────

# Data
DATA_DIR  = Path("data/raw")
MAX_FILES = 2             # start small — quantum circuits are slow per sample
APPLY_SNV = True
TARGET_LEN = 1800

# Training
BATCH_SIZE   = 16         # small batches — each fwd pass runs MAX_FILES×batch quantum calls
NUM_EPOCHS   = 50
LR           = 5e-3       # slightly higher LR for quantum params
WEIGHT_DECAY = 1e-6
PATIENCE     = 15         # early-stopping patience

# Cantor hierarchy mode
USE_CANTOR = False        # set True to use Haar-wavelet Cantor encoding

# Splits
TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
TEST_RATIO  = 0.1
SEED        = 42

# Output paths
ROOT        = Path(__file__).resolve().parent
MODEL_DIR   = ROOT / "models_qcnn"
RESULTS_DIR = ROOT / "results_qcnn"
BEST_MODEL  = MODEL_DIR / "qcnn_best.pt"
LOG_CSV     = RESULTS_DIR / "training_log.csv"
SUMMARY_TXT = RESULTS_DIR / "summary.txt"


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_pos_weight(y_train: np.ndarray, device: torch.device) -> torch.Tensor:
    pos = y_train.sum(axis=0).clip(min=1)
    neg = y_train.shape[0] - pos
    return torch.FloatTensor(neg / pos).to(device)


def tune_thresholds_per_class(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    grid: np.ndarray | None = None,
) -> np.ndarray:
    """
    Per-class grid search over threshold values to maximise per-class F1.
    Returns thresholds of shape (n_classes,).
    """
    if grid is None:
        grid = np.arange(0.05, 0.96, 0.05)
    n_classes = y_true.shape[1]
    thresholds = np.full(n_classes, 0.5)
    for c in range(n_classes):
        best_t, best_f1 = 0.5, 0.0
        for t in grid:
            preds = (y_probs[:, c] >= t).astype(int)
            f1 = f1_score(y_true[:, c], preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[c] = best_t
    return thresholds


def train_epoch(
    model: QCNNIRClassifier,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    thresholds: np.ndarray,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    all_probs, all_labels = [], []

    n_batches = len(loader)
    for batch_idx, (spectra, labels) in enumerate(loader):
        spectra, labels = spectra.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(spectra)
        loss   = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * spectra.size(0)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.cpu().numpy())

        if (batch_idx + 1) % max(1, n_batches // 5) == 0:
            print(
                f"    batch {batch_idx+1}/{n_batches}  "
                f"loss={loss.item():.4f}",
                end="\r",
            )

    avg_loss = total_loss / len(loader.dataset)
    all_probs  = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)
    f1 = f1_score(all_labels, (all_probs >= thresholds).astype(int),
                  average="micro", zero_division=0)
    return avg_loss, f1


@torch.no_grad()
def evaluate(
    model: QCNNIRClassifier,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    thresholds: np.ndarray,
    return_probs: bool = False,
) -> tuple:
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []

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
    f1_mic     = f1_score(all_labels, preds, average="micro",  zero_division=0)
    f1_mac     = f1_score(all_labels, preds, average="macro",  zero_division=0)

    if return_probs:
        return avg_loss, f1_mic, f1_mac, all_labels, all_probs
    return avg_loss, f1_mic, f1_mac, all_labels, preds


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print(f"QCNN + TTN Amplitude Embedding  (max_files={MAX_FILES})")
    print(f"use_cantor={USE_CANTOR}  batch={BATCH_SIZE}  epochs={NUM_EPOCHS}")
    print("=" * 70)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    print(f"Loading data  (max_files={MAX_FILES}) …")
    X, y = load_ir_data(
        DATA_DIR,
        target_length=TARGET_LEN,
        max_files=MAX_FILES,
        apply_snv=APPLY_SNV,
    )
    print(f"Dataset: {X.shape[0]:,} samples  |  {y.shape[1]} classes\n")

    train_loader, val_loader, test_loader = prepare_dataloaders(
        X, y,
        batch_size=BATCH_SIZE,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        random_seed=SEED,
        num_workers=0,
        pin_memory=False,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    print("Building QCNNIRClassifier …")
    model = QCNNIRClassifier(use_cantor=USE_CANTOR).to(device)
    param_summary(model)
    print()

    # ── Loss / optimiser ──────────────────────────────────────────────────────
    train_labels = np.vstack([y.numpy() for _, y in train_loader])
    pos_weight   = compute_pos_weight(train_labels, device)
    criterion    = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    # ── Output dirs ───────────────────────────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(LOG_CSV, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "train_loss", "train_f1_micro",
            "val_loss", "val_f1_micro", "val_f1_macro",
            "mean_threshold", "lr",
        ])

    # ── Training loop ─────────────────────────────────────────────────────────
    thresholds  = np.full(N_CLASSES, 0.5)
    best_val_f1 = -1.0
    best_state  = None
    best_thr    = thresholds.copy()
    no_improve  = 0
    t_start     = time.time()
    last_epoch  = 1

    for epoch in range(1, NUM_EPOCHS + 1):
        last_epoch = epoch
        t_ep = time.time()

        train_loss, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device, thresholds
        )

        val_loss, _, _, val_labels, val_probs = evaluate(
            model, val_loader, criterion, device, thresholds, return_probs=True
        )

        # Tune thresholds on this epoch's validation probs → use next epoch
        thresholds = tune_thresholds_per_class(val_labels, val_probs)
        val_f1_mic = f1_score(
            val_labels, (val_probs >= thresholds).astype(int),
            average="micro", zero_division=0,
        )
        val_f1_mac = f1_score(
            val_labels, (val_probs >= thresholds).astype(int),
            average="macro", zero_division=0,
        )
        blended_f1 = 0.5 * val_f1_mic + 0.5 * val_f1_mac

        scheduler.step(val_loss)
        lr_now = optimizer.param_groups[0]["lr"]
        ep_sec = time.time() - t_ep

        print(
            f"Ep {epoch:03d}/{NUM_EPOCHS}  "
            f"train_loss={train_loss:.4f}  train_f1={train_f1:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_f1_mic={val_f1_mic:.4f}  val_f1_mac={val_f1_mac:.4f}  "
            f"lr={lr_now:.1e}  ({ep_sec:.1f}s)"
        )

        with open(LOG_CSV, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, train_loss, train_f1,
                val_loss, val_f1_mic, val_f1_mac,
                float(thresholds.mean()), lr_now,
            ])

        if blended_f1 > best_val_f1:
            best_val_f1 = blended_f1
            best_state  = {k: v.clone() for k, v in model.state_dict().items()}
            best_thr    = thresholds.copy()
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": best_state,
                    "thresholds": best_thr,
                    "val_f1_micro": val_f1_mic,
                    "val_f1_macro": val_f1_mac,
                    "blended_f1": blended_f1,
                    "use_cantor": USE_CANTOR,
                },
                BEST_MODEL,
            )
            print(f"  ✓ New best  blended_f1={best_val_f1:.4f}")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}  (patience={PATIENCE})")
                break

    total_sec = time.time() - t_start
    print(f"\nTraining finished in {total_sec:.0f}s  ({total_sec/60:.1f} min)")

    # ── Test evaluation ───────────────────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)
        thresholds = best_thr

    test_loss, test_f1_mic, test_f1_mac, test_labels, test_preds = evaluate(
        model, test_loader, criterion, device, thresholds
    )
    print(f"\nTest  loss={test_loss:.4f}  f1_micro={test_f1_mic:.4f}  f1_macro={test_f1_mac:.4f}")

    # ── Per-class diagnostics (F1 table + confusion matrices) ─────────────────
    run_diagnostics(
        test_labels,
        test_preds,
        out_dir=RESULTS_DIR,
        model_name=f"QCNN+TTN (cantor={USE_CANTOR})",
        benchmark_f1=0.89,
    )

    # ── Comparison table ──────────────────────────────────────────────────────
    # Published / previously run benchmark values (fill in from your runs)
    benchmarks = {
        "CNN (Jung et al.)":   {"f1_micro": "?", "f1_macro": "?"},
        "MPS encoder":         {"f1_micro": "?", "f1_macro": "?"},
        "TTN (Experiment 5)":  {"f1_micro": "?", "f1_macro": "?"},
        "QML re-uploading":    {"f1_micro": "?", "f1_macro": "?"},
        "QCNN + TTN (ours)":   {"f1_micro": f"{test_f1_mic:.4f}", "f1_macro": f"{test_f1_mac:.4f}"},
    }
    print("\n" + "─" * 56)
    print(f"{'Model':<30}  {'F1-micro':>10}  {'F1-macro':>10}")
    print("─" * 56)
    for name, metrics in benchmarks.items():
        print(f"{name:<30}  {metrics['f1_micro']:>10}  {metrics['f1_macro']:>10}")
    print("─" * 56)

    # ── Summary ───────────────────────────────────────────────────────────────
    with open(SUMMARY_TXT, "w") as f:
        f.write("QCNNIRClassifier Training Summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"max_files      : {MAX_FILES}\n")
        f.write(f"samples        : {X.shape[0]}\n")
        f.write(f"batch_size     : {BATCH_SIZE}\n")
        f.write(f"epochs_run     : {last_epoch}\n")
        f.write(f"use_cantor     : {USE_CANTOR}\n")
        f.write(f"best_blended   : {best_val_f1:.4f}\n")
        f.write(f"test_f1_micro  : {test_f1_mic:.4f}\n")
        f.write(f"test_f1_macro  : {test_f1_mac:.4f}\n")
        f.write(f"training_time  : {total_sec:.0f}s\n")
        f.write(f"model_path     : {BEST_MODEL}\n")

    print(f"\nSummary  → {SUMMARY_TXT}")
    print(f"Log      → {LOG_CSV}")
    print(f"Model    → {BEST_MODEL}")

    # ── Bloch sphere visualisation (post-training) ────────────────────────────
    print("\nGenerating Bloch sphere visualisation …")
    try:
        from spectroscopy_qml.ir.qml.visualize_bloch import (
            compute_bloch_vectors,
            plot_bloch_sphere,
        )
        model.eval()
        bloch_vecs, bloch_labels = compute_bloch_vectors(
            model, test_loader, device, n_samples=min(200, len(test_loader.dataset))
        )
        bloch_fig_path = RESULTS_DIR / "bloch_sphere.png"
        plot_bloch_sphere(
            bloch_vecs,
            bloch_labels,
            save_path=bloch_fig_path,
            title=f"Bloch Sphere — QCNN root qubit (qubit {10})\n"
                  f"test F1-micro={test_f1_mic:.4f}",
        )
        print(f"Bloch sphere → {bloch_fig_path}")
    except Exception as exc:
        print(f"  (Bloch visualisation skipped: {exc})")


if __name__ == "__main__":
    main()
