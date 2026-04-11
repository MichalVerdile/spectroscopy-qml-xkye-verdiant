"""
Training-Script für den Cantor-Quanten IR-Klassifikator.

Trainingsstrategie
------------------
  Loss       : Focal-BCE + pos_weight (geclampt bei 50)
  Optimizer  : Adam
  LR-Scheduler: ReduceLROnPlateau (Faktor 0.5, Patience 5)
  Stopping   : Blended-F1 = 0.5×micro + 0.5×macro  (Patience 20)
  Thresholds : Per-Klasse Grid-Suche auf Val-Set nach jeder Epoche

Ausführung
----------
    python -m spectroscopy_qml.ir.quantum_ir.train

Ausgaben
--------
    results_cantor/training_log.csv
    results_cantor/summary.txt
    models_cantor/cantor_best.pt
    results_cantor/per_class_f1.png
    results_cantor/per_class_f1.csv
    results_cantor/confusion_matrices.png
    results_cantor/bloch_full_interactive.html
    results_cantor/bloch_full_matplotlib.png
    results_cantor/bloch_full_2d_projections.png
    results_cantor/bloch_full_per_class.png
"""

from __future__ import annotations

import argparse
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
from spectroscopy_qml.ir.qml.diagnostics import run_diagnostics
from spectroscopy_qml.ir.quantum_ir.model.classifier import CantorQuantumClassifier, N_CLASSES
from spectroscopy_qml.ir.quantum_ir.visualize.bloch_sphere import plot_all_levels

# ── Standard-Konfiguration ────────────────────────────────────────────────────
DATA_DIR         = Path("data/raw")
MAX_FILES        = 5
APPLY_SNV        = True
TARGET_LEN       = 1800

BATCH_SIZE       = 8          # Klein halten: 7 Quanten-Schaltkreise pro Forward-Pass
NUM_EPOCHS       = 100
LR               = 5e-3
WEIGHT_DECAY     = 1e-6
PATIENCE         = 20
LR_PATIENCE      = 8
LR_FACTOR        = 0.5
MIN_LR           = 1e-6

POS_WEIGHT_MAX   = 50.0
USE_FOCAL_LOSS   = True
FOCAL_ALPHA      = 0.25
FOCAL_GAMMA      = 2.0

SHARED_QUARTERS  = True       # Geteilte Gewichte für alle 4 Viertel-Encoder
USE_GPU          = False       # Lightning.qubit auf CPU (lightning.gpu falls verfügbar)

TRAIN_RATIO      = 0.8
VAL_RATIO        = 0.1
TEST_RATIO       = 0.1
SEED             = 42

ROOT             = Path(__file__).resolve().parent
MODEL_DIR        = ROOT / "models_cantor"
RESULTS_DIR      = ROOT / "results_cantor"
BEST_MODEL       = MODEL_DIR / "cantor_best.pt"
LOG_CSV          = RESULTS_DIR / "training_log.csv"
SUMMARY_TXT      = RESULTS_DIR / "summary.txt"


# ── Focal-BCE Loss ─────────────────────────────────────────────────────────────

def compute_pos_weight(y_train: np.ndarray, device: torch.device) -> torch.Tensor:
    """neg/pos per Klasse, geclampt auf POS_WEIGHT_MAX."""
    n   = y_train.shape[0]
    pos = y_train.sum(axis=0).clip(min=1)
    neg = n - pos
    w   = (neg / pos).clip(max=POS_WEIGHT_MAX)
    return torch.FloatTensor(w).to(device)


class FocalBCELoss(nn.Module):
    """
    Focal Loss für Multi-Label BCE.

    L = −α (1−p)^γ y·log(p)  −  (1−α) p^γ (1−y)·log(1−p)

    Kombiniert mit pos_weight für Ungleichgewicht-Korrektur.
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
        p     = torch.sigmoid(logits)
        bce   = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        p_t     = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_w = alpha_t * (1 - p_t) ** self.gamma
        return (focal_w * bce).mean()


# ── Per-Klasse Threshold-Tuning ────────────────────────────────────────────────

def tune_thresholds_per_class(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    grid: np.ndarray | None = None,
) -> np.ndarray:
    """Optimiert jeden Threshold separat für maximales per-class F1."""
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


# ── Training / Evaluation ─────────────────────────────────────────────────────

def train_epoch(
    model:      nn.Module,
    loader:     torch.utils.data.DataLoader,
    criterion:  nn.Module,
    optimizer:  torch.optim.Optimizer,
    device:     torch.device,
    thresholds: np.ndarray,
) -> tuple[float, float]:
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
def evaluate(
    model:       nn.Module,
    loader:      torch.utils.data.DataLoader,
    criterion:   nn.Module,
    device:      torch.device,
    thresholds:  np.ndarray,
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
    all_preds  = (all_probs >= thresholds).astype(int)

    f1_mic = f1_score(all_labels, all_preds, average="micro",  zero_division=0)
    f1_mac = f1_score(all_labels, all_preds, average="macro",  zero_division=0)

    if return_probs:
        return avg_loss, f1_mic, f1_mac, all_labels, all_probs
    return avg_loss, f1_mic, f1_mac, all_labels, all_preds


# ── Argument-Parser ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cantor-Quanten IR-Klassifikator Training"
    )
    p.add_argument("--data-dir",       default=DATA_DIR,    type=Path)
    p.add_argument("--max-files",      default=MAX_FILES,   type=int)
    p.add_argument("--batch-size",     default=BATCH_SIZE,  type=int)
    p.add_argument("--epochs",         default=NUM_EPOCHS,  type=int)
    p.add_argument("--lr",             default=LR,          type=float)
    p.add_argument("--patience",       default=PATIENCE,    type=int)
    p.add_argument("--pos-weight-max", default=POS_WEIGHT_MAX, type=float)
    p.add_argument("--shared-quarters", action="store_true",
                   default=SHARED_QUARTERS)
    p.add_argument("--separate-quarters", dest="shared_quarters",
                   action="store_false")
    p.add_argument("--seed",           default=SEED,        type=int)
    p.add_argument("--cache-path",     default=None,        type=Path,
                   help="Pfad zu einem .npz SNV-Cache (wird erstellt falls nicht vorhanden)")
    p.add_argument("--use-gpu",        action="store_true",
                   help="lightning.gpu für Quantenschaltkreise verwenden (CUDA erforderlich)")
    p.add_argument("--no-bloch",       action="store_true",
                   help="Bloch-Visualisierung nach Training überspringen")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    loss_name = (f"Focal(α={FOCAL_ALPHA},γ={FOCAL_GAMMA})"
                 if USE_FOCAL_LOSS else "BCE")
    quarters_mode = "geteilt" if args.shared_quarters else "separat"
    print("=" * 72)
    print(f"Cantor-Quanten IR-Klassifikator")
    print(f"  files={args.max_files}  batch={args.batch_size}  "
          f"loss={loss_name}  pw_max={args.pos_weight_max}")
    print(f"  Viertel-Encoder: {quarters_mode}")
    print("=" * 72)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── Daten ─────────────────────────────────────────────────────────────────
    print(f"Lade Daten (max_files={args.max_files}) …")
    X, y = load_ir_data(args.data_dir, target_length=TARGET_LEN,
                        max_files=args.max_files, apply_snv=APPLY_SNV,
                        cache_path=args.cache_path)
    print(f"Datensatz: {X.shape[0]:,} Samples  |  {y.shape[1]} Klassen")

    train_loader, val_loader, test_loader = prepare_dataloaders(
        X, y,
        batch_size=args.batch_size,
        train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO,
        random_seed=args.seed, num_workers=0, pin_memory=False,
    )

    # ── Modell ────────────────────────────────────────────────────────────────
    print("\nErstelle CantorQuantumClassifier …")
    use_gpu = args.use_gpu or USE_GPU
    model = CantorQuantumClassifier(
        use_gpu=use_gpu,
        shared_quarters=args.shared_quarters,
    ).to(device)
    model.param_summary()

    # ── Loss ──────────────────────────────────────────────────────────────────
    # Direkt aus dem originalen y-Array (vermeidet leeren DataLoader-Bug)
    n_train = int(len(X) * TRAIN_RATIO)
    train_labels = y[:n_train] if isinstance(y, np.ndarray) else y[:n_train].numpy()
    pos_weight   = compute_pos_weight(train_labels, device)
    print(f"\npos_weight  min={pos_weight.min():.1f}  "
          f"max={pos_weight.max():.1f}  mean={pos_weight.mean():.1f}  "
          f"(geclampt bei {args.pos_weight_max})")

    if USE_FOCAL_LOSS:
        criterion = FocalBCELoss(pos_weight, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min",
        factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=MIN_LR,
    )

    # ── Ausgabe-Verzeichnisse ─────────────────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(LOG_CSV, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "train_loss", "train_f1_micro",
            "val_loss", "val_f1_micro", "val_f1_macro", "blended_f1",
            "mean_threshold", "lr",
        ])

    # ── Trainings-Loop ────────────────────────────────────────────────────────
    print(f"\nTraining  epochs={args.epochs}  batch={args.batch_size}  "
          f"patience={args.patience}\n")
    print("Hinweis: Pro Epoche werden 7 Quantenschaltkreise pro Batch-Sample")
    print("         ausgeführt — rechne mit ~5–20 min/Epoche (CPU).\n")

    thresholds = np.full(N_CLASSES, 0.5)
    best_score = -1.0
    best_state = None
    best_thr   = thresholds.copy()
    no_improve = 0
    t_start    = time.time()
    last_epoch = 1

    for epoch in range(1, args.epochs + 1):
        last_epoch = epoch
        t_ep = time.time()

        train_loss, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device, thresholds)

        _, _, _, val_labels, val_probs = evaluate(
            model, val_loader, criterion, device, thresholds, return_probs=True)

        # Per-Klasse Threshold-Optimierung auf aktuellem Val-Set
        thresholds = tune_thresholds_per_class(val_labels, val_probs)
        val_preds  = (val_probs >= thresholds).astype(int)
        val_f1_mic = f1_score(val_labels, val_preds, average="micro", zero_division=0)
        val_f1_mac = f1_score(val_labels, val_preds, average="macro", zero_division=0)

        # Blended F1 als Early-Stopping-Metrik
        blended = 0.5 * val_f1_mic + 0.5 * val_f1_mac

        val_loss, _, _, _, _ = evaluate(
            model, val_loader, criterion, device, thresholds)
        scheduler.step(val_loss)
        lr_now = optimizer.param_groups[0]["lr"]
        ep_sec = time.time() - t_ep

        print(
            f"Ep {epoch:03d}/{args.epochs}  "
            f"loss={train_loss:.4f}  tr_f1={train_f1:.4f}  "
            f"val_mic={val_f1_mic:.4f}  val_mac={val_f1_mac:.4f}  "
            f"blend={blended:.4f}  lr={lr_now:.1e}  ({ep_sec:.0f}s)"
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
                "epoch":            epoch,
                "model_state_dict": best_state,
                "thresholds":       best_thr,
                "val_f1_micro":     val_f1_mic,
                "val_f1_macro":     val_f1_mac,
                "blended_f1":       blended,
                "args":             args,
            }, BEST_MODEL)
            print(f"  ✓ Neues Bestes  blended={best_score:.4f}")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"\nEarly Stopping  (keine Verbesserung für {args.patience} Epochen)")
                break

    total_sec = time.time() - t_start
    print(f"\nFertig in {total_sec:.0f}s ({total_sec/60:.1f} min)")

    # ── Test-Evaluation ───────────────────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)
        thresholds = best_thr

    test_loss, test_f1_mic, test_f1_mac, test_labels, test_preds = evaluate(
        model, test_loader, criterion, device, thresholds)

    print(f"\nTest  loss={test_loss:.4f}  "
          f"f1_micro={test_f1_mic:.4f}  f1_macro={test_f1_mac:.4f}")

    run_diagnostics(
        test_labels, test_preds,
        out_dir=RESULTS_DIR,
        model_name=f"Cantor-Quanten ({loss_name}, pw_max={args.pos_weight_max})",
        benchmark_f1=0.89,
    )

    # ── Bloch-Sphäre ──────────────────────────────────────────────────────────
    if not args.no_bloch:
        print("\nErstelle Bloch-Sphären-Visualisierungen …")
        plot_all_levels(
            model, test_loader, device,
            out_dir=RESULTS_DIR,
            n_samples=min(300, len(test_loader.dataset)),
        )

    # ── Zusammenfassung ───────────────────────────────────────────────────────
    with open(SUMMARY_TXT, "w") as f:
        f.write("Cantor-Quanten IR-Klassifikator — Trainingszusammenfassung\n")
        f.write("=" * 50 + "\n")
        f.write(f"max_files         : {args.max_files}\n")
        f.write(f"samples           : {X.shape[0]}\n")
        f.write(f"batch_size        : {args.batch_size}\n")
        f.write(f"epochs_run        : {last_epoch}\n")
        f.write(f"loss              : {loss_name}\n")
        f.write(f"pos_weight_max    : {args.pos_weight_max}\n")
        f.write(f"shared_quarters   : {args.shared_quarters}\n")
        f.write(f"best_blended      : {best_score:.4f}\n")
        f.write(f"test_f1_micro     : {test_f1_mic:.4f}\n")
        f.write(f"test_f1_macro     : {test_f1_mac:.4f}\n")
        f.write(f"test_loss         : {test_loss:.4f}\n")
        f.write(f"training_time     : {total_sec:.0f}s\n")

    print(f"\nZusammenfassung → {SUMMARY_TXT}")
    print(f"Log              → {LOG_CSV}")
    print(f"Modell           → {BEST_MODEL}")


if __name__ == "__main__":
    main()
