"""
Evaluation und Modellvergleich für den Cantor-Quanten IR-Klassifikator.

Vergleichstabelle am Ende zeigt:
  CNN (Jung et al.)  vs  MPS  vs  TTN Exp6  vs  QML Re-uploading  vs  Cantor-Quanten

Ausführung
----------
    python -m spectroscopy_qml.ir.quantum_ir.evaluate \\
        --checkpoint results_cantor/models_cantor/cantor_best.pt \\
        --data-dir data/raw

Ausgaben
--------
    results_cantor/eval_per_class_f1.csv
    results_cantor/eval_per_class_f1.png
    results_cantor/eval_confusion_matrices.png
    Vergleichstabelle auf stdout
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from spectroscopy_qml.ir.mps_encoder.data_loader import load_ir_data, prepare_dataloaders
from spectroscopy_qml.ir.qml.diagnostics import run_diagnostics
from spectroscopy_qml.ir.quantum_ir.model.classifier import CantorQuantumClassifier, N_CLASSES

# ── Bekannte Baseline-Ergebnisse ──────────────────────────────────────────────
# Quelle: Experiment-Runs in diesem Repository
BASELINES: list[dict] = [
    {
        "model":     "CNN (Jung et al.)",
        "f1_micro":  0.89,
        "f1_macro":  None,
        "notes":     "Ziel-Benchmark",
        "ref":       "—",
    },
    {
        "model":     "MPS (Exp1–4)",
        "f1_micro":  0.82,
        "f1_macro":  None,
        "notes":     "Matrix Product State",
        "ref":       "MPS-Experiment",
    },
    {
        "model":     "TTN Exp6 (chi=64)",
        "f1_micro":  0.8697,
        "f1_macro":  0.6240,
        "notes":     "Bestes klassisches Modell",
        "ref":       "Exp6 full_dataset_run",
    },
    {
        "model":     "QML Re-uploading",
        "f1_micro":  0.3968,
        "f1_macro":  None,
        "notes":     "8 Qubits, 5 Schichten (kein Focal Loss)",
        "ref":       "qml/train.py",
    },
    {
        "model":     "Quantum Transfer (Exp8)",
        "f1_micro":  None,
        "f1_macro":  None,
        "notes":     "TTN Exp6 Backbone + 4-Qubit-Kreis",
        "ref":       "Exp8 (noch nicht trainiert)",
    },
]


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_model(
    model:      nn.Module,
    loader:     torch.utils.data.DataLoader,
    criterion:  nn.Module,
    device:     torch.device,
    thresholds: np.ndarray,
    return_probs: bool = False,
) -> tuple:
    """
    Evaluiert das Modell auf einem DataLoader.

    Returns:
        (avg_loss, f1_micro, f1_macro, y_true, y_pred_or_probs)
    """
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


# ── Vergleichstabelle ─────────────────────────────────────────────────────────

def print_comparison_table(
    cantor_f1_micro: float,
    cantor_f1_macro: float,
    extra_rows: list[dict] | None = None,
) -> None:
    """
    Druckt eine formatierte Vergleichstabelle auf stdout.

    Args:
        cantor_f1_micro: Erreichtes F1_micro des Cantor-Quanten-Modells
        cantor_f1_macro: Erreichtes F1_macro des Cantor-Quanten-Modells
        extra_rows:      Optionale zusätzliche Zeilen (dicts mit 'model',
                         'f1_micro', 'f1_macro', 'notes')
    """
    # Cantor-Quanten-Ergebnis hinzufügen
    rows = BASELINES.copy()
    rows.append({
        "model":    "Cantor-Quanten (dieses Modell)",
        "f1_micro": cantor_f1_micro,
        "f1_macro": cantor_f1_macro,
        "notes":    "3-Level Cantor-Split + TTN-Merger",
        "ref":      "quantum_ir/train.py",
    })
    if extra_rows:
        rows.extend(extra_rows)

    # Spaltenbreiten
    w_model = max(len(r["model"]) for r in rows) + 2
    w_notes = max(len(r.get("notes", "")) for r in rows) + 2

    hdr_line = (
        f"{'Modell':<{w_model}}  "
        f"{'F1-Micro':>9}  "
        f"{'F1-Macro':>9}  "
        f"{'Ziel (89%)':>10}  "
        f"{'Anmerkungen':<{w_notes}}"
    )
    sep = "─" * len(hdr_line)

    print("\n" + sep)
    print("  MODELLVERGLEICH — IR-Spektren Klassifikation (37 Klassen)")
    print(sep)
    print(hdr_line)
    print(sep)

    target = 0.89
    for r in rows:
        mic  = r["f1_micro"]
        mac  = r["f1_macro"]
        name = r["model"]

        mic_str  = f"{mic:.4f}" if mic is not None else "  TBD  "
        mac_str  = f"{mac:.4f}" if mac is not None else "  TBD  "
        above    = "  ✓" if (mic is not None and mic >= target) else "  ✗"
        notes    = r.get("notes", "")

        # Highlight: bestes Ergebnis
        marker = " ★" if (mic is not None and
                          mic == max(r2["f1_micro"] for r2 in rows
                                     if r2["f1_micro"] is not None)) else "  "

        print(
            f"{name:<{w_model}}{marker}"
            f"{mic_str:>9}  "
            f"{mac_str:>9}  "
            f"{above:>10}  "
            f"{notes:<{w_notes}}"
        )

    print(sep)
    if cantor_f1_micro is not None:
        delta = cantor_f1_micro - 0.8697  # vs TTN Exp6
        sign  = "+" if delta >= 0 else ""
        print(f"  Cantor-Quanten vs TTN Exp6:  {sign}{delta:.4f} F1-Micro")
        delta_target = cantor_f1_micro - target
        sign = "+" if delta_target >= 0 else ""
        print(f"  Cantor-Quanten vs Ziel 89%:  {sign}{delta_target:.4f} F1-Micro")
    print(sep + "\n")


# ── Argument-Parser ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cantor-Quanten IR-Klassifikator Evaluation"
    )
    p.add_argument("--checkpoint", required=True, type=Path,
                   help="Pfad zur besten .pt-Datei (aus train.py)")
    p.add_argument("--data-dir",   default=Path("data/raw"), type=Path)
    p.add_argument("--max-files",  default=5,     type=int)
    p.add_argument("--batch-size", default=8,     type=int)
    p.add_argument("--out-dir",    default=None,  type=Path,
                   help="Ausgabeverzeichnis (Standard: neben Checkpoint)")
    p.add_argument("--seed",       default=42,    type=int)
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    out_dir = args.out_dir or args.checkpoint.parent.parent / "results_cantor"
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")
    print(f"Evaluation — Checkpoint: {args.checkpoint}")
    print(f"Device: {device}\n")

    # ── Checkpoint laden ──────────────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    thresholds = ckpt.get("thresholds", np.full(N_CLASSES, 0.5))
    train_args = ckpt.get("args", None)

    shared_quarters = getattr(train_args, "shared_quarters", True)

    model = CantorQuantumClassifier(shared_quarters=shared_quarters)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.param_summary()

    # ── Daten ─────────────────────────────────────────────────────────────────
    print(f"\nLade Daten (max_files={args.max_files}) …")
    X, y = load_ir_data(
        args.data_dir, target_length=1800,
        max_files=args.max_files, apply_snv=True,
    )
    print(f"Datensatz: {X.shape[0]:,} Samples  |  {y.shape[1]} Klassen")

    _, _, test_loader = prepare_dataloaders(
        X, y,
        batch_size=args.batch_size,
        train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
        random_seed=args.seed, num_workers=0, pin_memory=False,
    )

    # ── Dummy-Loss für evaluate_model ─────────────────────────────────────────
    n      = X.shape[0]
    pos    = y.numpy().sum(axis=0).clip(min=1)
    neg    = n - pos
    pw     = torch.FloatTensor((neg / pos).clip(max=50.0))
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    # ── Test-Evaluation ───────────────────────────────────────────────────────
    test_loss, test_f1_mic, test_f1_mac, test_labels, test_preds = evaluate_model(
        model, test_loader, criterion, device, thresholds)

    print(f"\nTest  loss={test_loss:.4f}  "
          f"f1_micro={test_f1_mic:.4f}  f1_macro={test_f1_mac:.4f}")

    # ── Detaillierte Diagnostics ──────────────────────────────────────────────
    run_diagnostics(
        test_labels, test_preds,
        out_dir=out_dir,
        model_name="Cantor-Quanten IR-Klassifikator",
        benchmark_f1=0.89,
    )

    # ── Vergleichstabelle ─────────────────────────────────────────────────────
    print_comparison_table(test_f1_mic, test_f1_mac)

    # ── Gespeicherte Metriken aus Checkpoint ──────────────────────────────────
    print("Checkpoint-Metriken (bestes Val-Ergebnis während Training):")
    print(f"  Epoche:      {ckpt.get('epoch', '?')}")
    print(f"  Val F1-Micro: {ckpt.get('val_f1_micro', '?'):.4f}")
    print(f"  Val F1-Macro: {ckpt.get('val_f1_macro', '?'):.4f}")
    print(f"  Blended F1:   {ckpt.get('blended_f1', '?'):.4f}")


if __name__ == "__main__":
    main()
