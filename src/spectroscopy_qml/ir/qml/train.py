"""
Training script for QMLReuploadModel on IR spectra.

Fixes applied (vs. original):
  1. pos_weight clamped to max 50  (verhindert Dominanz ultra-seltener Klassen)
  2. Focal Loss  (alpha=0.25, gamma=2.0)  als Option  →  USE_FOCAL_LOSS = True
  3. Per-class threshold-Optimierung auf Validation-Set  (statt globalem Grid)
  4. Blended F1  (0.5*micro + 0.5*macro) als Early-Stopping-Metrik
  5. run_diagnostics() nach Test-Evaluation  →  per_class_f1.png + confusion_matrices.png

Run from the repo root:
    python src/spectroscopy_qml/ir/qml/train.py

Outputs:
    results/training_log.csv
    results/summary.txt
    results/per_class_f1.png
    results/confusion_matrices.png
    models/qml_best.pt
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from spectroscopy_qml.ir.mps_encoder.data_loader import load_ir_data, prepare_dataloaders
from spectroscopy_qml.ir.qml.model_qml_reupload import QMLReuploadModel, N_CLASSES
from spectroscopy_qml.ir.qml.diagnostics import run_diagnostics

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR    = Path("data/raw")
MAX_FILES   = 5
APPLY_SNV   = True
TARGET_LEN  = 1800

BATCH_SIZE   = 32
NUM_EPOCHS   = 50
LR           = 1e-3
WEIGHT_DECAY = 1e-6
PATIENCE     = 15           # erhöht: Quantum-Training konvergiert langsam

POS_WEIGHT_MAX = 50.0       # Fix 1: Clamp verhindert Recall-Bias bei Support=1
USE_FOCAL_LOSS = True       # Fix 2: Focal Loss unterdrückt leicht klassifizierbare Samples
FOCAL_ALPHA    = 0.25
FOCAL_GAMMA    = 2.0

TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
TEST_RATIO  = 0.1
SEED        = 42

ROOT        = Path(__file__).resolve().parent
MODEL_DIR   = ROOT / "models"
RESULTS_DIR = ROOT / "results"
BEST_MODEL  = MODEL_DIR / "qml_best.pt"
LOG_CSV     = RESULTS_DIR / "training_log.csv"
SUMMARY_TXT = RESULTS_DIR / "summary.txt"


# ── Loss ──────────────────────────────────────────────────────────────────────

def compute_pos_weight(y_train: np.ndarray, device: torch.device) -> torch.Tensor:
    """neg/pos per class, geclampt auf POS_WEIGHT_MAX."""
    n   = y_train.shape[0]
    pos = y_train.sum(axis=0).clip(min=1)
    neg = n - pos
    w   = (neg / pos).clip(max=POS_WEIGHT_MAX)
    return torch.FloatTensor(w).to(device)


class FocalBCELoss(nn.Module):
    """
    Focal loss für multi-label BCE.

    L = -alpha * (1 - p)^gamma * y*log(p)
        -(1-alpha) * p^gamma * (1-y)*log(1-p)

    Kombiniert mit pos_weight aus compute_pos_weight.
    """
    def __init__(
        self,
        pos_weight: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p   = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        # Focal modulation
        p_t        = p * targets + (1 - p) * (1 - targets)
        alpha_t    = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_w    = alpha_t * (1 - p_t) ** self.gamma
        return (focal_w * bce).mean()


# ── Per-class threshold tuning ────────────────────────────────────────────────

def tune_thresholds_per_class(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    grid: np.ndarray | None = None,
) -> np.ndarray:
    """Fix 3: Optimiert jeden Threshold separat für maximales per-class F1."""
    if grid is None:
        grid = np.arange(0.05, 0.96, 0.05)
    thresholds = np.full(y_true.shape[1], 0.5)
    for c in range(y_true.shape[1]):
        best_t, best_f1 = 0.5, 0.0
        for t in grid:
            preds = (y_probs[:, c] >= t).astype(int)
            f1 = f1_score(y_true[:, c], preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[c] = best_t
    return thresholds


# ── Training / evaluation ─────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device, thresholds):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

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
        all_preds.append((probs >= thresholds).astype(int))
        all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    f1 = f1_score(np.vstack(all_labels), np.vstack(all_preds),
                  average="micro", zero_division=0)
    return avg_loss, f1


@torch.no_grad()
def evaluate(model, loader, criterion, device, thresholds, return_probs=False):
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
    all_preds  = (all_probs >= thresholds).astype(int)
    f1_mic = f1_score(all_labels, all_preds, average="micro", zero_division=0)
    f1_mac = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    if return_probs:
        return avg_loss, f1_mic, f1_mac, all_labels, all_probs
    return avg_loss, f1_mic, f1_mac, all_labels, all_preds


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    loss_name = f"Focal(α={FOCAL_ALPHA},γ={FOCAL_GAMMA})" if USE_FOCAL_LOSS else "BCE"
    print("=" * 70)
    print(f"QML Re-uploading  |  files={MAX_FILES}  loss={loss_name}  "
          f"pw_max={POS_WEIGHT_MAX}")
    print("=" * 70)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    print(f"Loading data (max_files={MAX_FILES}) …")
    X, y = load_ir_data(DATA_DIR, target_length=TARGET_LEN,
                        max_files=MAX_FILES, apply_snv=APPLY_SNV)
    print(f"Dataset: {X.shape[0]:,} samples  |  {y.shape[1]} classes")

    train_loader, val_loader, test_loader = prepare_dataloaders(
        X, y,
        batch_size=BATCH_SIZE,
        train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO,
        random_seed=SEED, num_workers=0, pin_memory=False,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\nBuilding QMLReuploadModel …")
    model = QMLReuploadModel().to(device)
    total_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    q_p     = sum(p.numel() for p in model.qlayer.parameters() if p.requires_grad)
    print(f"  Total: {total_p:,}  Quantum: {q_p}  Classical: {total_p-q_p:,}")

    # ── Loss ──────────────────────────────────────────────────────────────────
    train_labels = np.vstack([y.numpy() for _, y in train_loader])
    pos_weight   = compute_pos_weight(train_labels, device)
    print(f"\npos_weight  min={pos_weight.min():.1f}  "
          f"max={pos_weight.max():.1f}  mean={pos_weight.mean():.1f}  "
          f"(clamped at {POS_WEIGHT_MAX})")

    if USE_FOCAL_LOSS:
        criterion = FocalBCELoss(pos_weight, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                  patience=5, min_lr=1e-6)

    # ── Output dirs ───────────────────────────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(LOG_CSV, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "train_loss", "train_f1_micro",
            "val_loss", "val_f1_micro", "val_f1_macro", "blended_f1",
            "mean_threshold", "lr",
        ])

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\nTraining  epochs={NUM_EPOCHS}  batch={BATCH_SIZE}  patience={PATIENCE}\n")

    thresholds  = np.full(N_CLASSES, 0.5)
    best_score  = -1.0
    best_state  = None
    best_thr    = thresholds.copy()
    no_improve  = 0
    t_start     = time.time()
    last_epoch  = 1

    for epoch in range(1, NUM_EPOCHS + 1):
        last_epoch = epoch
        t_ep = time.time()

        train_loss, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device, thresholds)

        _, _, _, val_labels, val_probs = evaluate(
            model, val_loader, criterion, device, thresholds, return_probs=True)

        # Fix 3: Per-class threshold-Optimierung auf aktuellen Val-Probs
        thresholds = tune_thresholds_per_class(val_labels, val_probs)
        val_preds  = (val_probs >= thresholds).astype(int)
        val_f1_mic = f1_score(val_labels, val_preds, average="micro", zero_division=0)
        val_f1_mac = f1_score(val_labels, val_preds, average="macro", zero_division=0)

        # Fix 4: Blended F1 als Early-Stopping-Metrik
        blended    = 0.5 * val_f1_mic + 0.5 * val_f1_mac

        val_loss, _, _, _, _ = evaluate(
            model, val_loader, criterion, device, thresholds)
        scheduler.step(val_loss)
        lr_now = optimizer.param_groups[0]["lr"]
        ep_sec = time.time() - t_ep

        print(
            f"Ep {epoch:03d}/{NUM_EPOCHS}  "
            f"loss={train_loss:.4f}  tr_f1={train_f1:.4f}  "
            f"val_mic={val_f1_mic:.4f}  val_mac={val_f1_mac:.4f}  "
            f"blend={blended:.4f}  lr={lr_now:.1e}  ({ep_sec:.1f}s)"
        )

        with open(LOG_CSV, "a", newline="") as f:
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
                "epoch": epoch,
                "model_state_dict": best_state,
                "thresholds": best_thr,
                "val_f1_micro": val_f1_mic,
                "val_f1_macro": val_f1_mac,
                "blended_f1":   blended,
            }, BEST_MODEL)
            print(f"  ✓ New best  blended={best_score:.4f}")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\nEarly stopping  (no improvement for {PATIENCE} epochs)")
                break

    total_sec = time.time() - t_start
    print(f"\nDone in {total_sec:.0f}s  ({total_sec/60:.1f} min)")

    # ── Test evaluation ───────────────────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)
        thresholds = best_thr

    test_loss, test_f1_mic, test_f1_mac, test_labels, test_preds = evaluate(
        model, test_loader, criterion, device, thresholds)
    print(f"\nTest  loss={test_loss:.4f}  "
          f"f1_micro={test_f1_mic:.4f}  f1_macro={test_f1_mac:.4f}")

    # Fix 5: Per-class Diagnostics
    run_diagnostics(
        test_labels, test_preds,
        out_dir=RESULTS_DIR,
        model_name=f"QML Re-uploading ({loss_name}, pw_max={POS_WEIGHT_MAX})",
        benchmark_f1=0.89,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    with open(SUMMARY_TXT, "w") as f:
        f.write("QML Re-uploading Training Summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"max_files      : {MAX_FILES}\n")
        f.write(f"samples        : {X.shape[0]}\n")
        f.write(f"batch_size     : {BATCH_SIZE}\n")
        f.write(f"epochs_run     : {last_epoch}\n")
        f.write(f"loss           : {loss_name}\n")
        f.write(f"pos_weight_max : {POS_WEIGHT_MAX}\n")
        f.write(f"best_blended   : {best_score:.4f}\n")
        f.write(f"test_f1_micro  : {test_f1_mic:.4f}\n")
        f.write(f"test_f1_macro  : {test_f1_mac:.4f}\n")
        f.write(f"test_loss      : {test_loss:.4f}\n")
        f.write(f"training_time  : {total_sec:.0f}s\n")

    print(f"\nSummary → {SUMMARY_TXT}")
    print(f"Log     → {LOG_CSV}")
    print(f"Model   → {BEST_MODEL}")


if __name__ == "__main__":
    main()
