"""
Diagnostic metrics for multi-label IR classifiers.

Provides:
  - compute_full_metrics()       per-class F1 / precision / recall + micro/macro
  - print_per_class_table()      ASCII table sorted by F1, printed to stdout
  - plot_per_class_f1()          horizontal bar chart (all 37 classes)
  - plot_multilabel_confusion()  per-class 2×2 heatmap grid
  - save_per_class_csv()         CSV with all per-class statistics

Works with any multi-label classifier that outputs (N, 37) probability arrays —
not specific to QCNN.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    multilabel_confusion_matrix,
)

FG_NAMES: list[str] = [
    "Acid anhydride", "Acyl halide", "Alcohol", "Aldehyde", "Alkane",
    "Alkene", "Alkyne", "Amide", "Amine", "Arene", "Azo compound",
    "Carbamate", "Carboxylic acid", "Enamine", "Enol", "Ester", "Ether",
    "Haloalkane", "Hydrazine", "Hydrazone", "Imide", "Imine",
    "Isocyanate", "Isothiocyanate", "Ketone", "Nitrile", "Phenol",
    "Phosphine", "Sulfide", "Sulfonamide", "Sulfonate", "Sulfone",
    "Sulfonic acid", "Sulfoxide", "Thial", "Thioamide", "Thiol",
]


# ── Core metric computation ───────────────────────────────────────────────────

def compute_full_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str] = FG_NAMES,
) -> dict:
    """
    Compute per-class and aggregate metrics for multi-label classification.

    Args:
        y_true : (N, C) binary ground-truth matrix
        y_pred : (N, C) binary prediction matrix
        class_names: C class labels

    Returns dict with keys:
        f1_micro, f1_macro, precision_micro, precision_macro,
        recall_micro, recall_macro,
        per_class: list of dicts {name, f1, precision, recall, support, tp, fp, fn, tn}
        mcm: (C, 2, 2) multilabel confusion matrix  [[TN, FP], [FN, TP]]
    """
    n_classes = y_true.shape[1]

    f1_per    = f1_score(y_true, y_pred, average=None, zero_division=0)
    prec_per  = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec_per   = recall_score(y_true, y_pred, average=None, zero_division=0)
    mcm       = multilabel_confusion_matrix(y_true, y_pred)   # (C, 2, 2)

    per_class = []
    for c in range(n_classes):
        tn, fp, fn, tp = mcm[c].ravel()
        per_class.append({
            "name":      class_names[c] if c < len(class_names) else f"Class {c}",
            "f1":        float(f1_per[c]),
            "precision": float(prec_per[c]),
            "recall":    float(rec_per[c]),
            "support":   int(y_true[:, c].sum()),
            "tp":        int(tp),
            "fp":        int(fp),
            "fn":        int(fn),
            "tn":        int(tn),
        })

    return {
        "f1_micro":         float(f1_score(y_true, y_pred, average="micro",  zero_division=0)),
        "f1_macro":         float(f1_score(y_true, y_pred, average="macro",  zero_division=0)),
        "precision_micro":  float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "precision_macro":  float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_micro":     float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "recall_macro":     float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class":        per_class,
        "mcm":              mcm,
    }


# ── Console output ────────────────────────────────────────────────────────────

def print_per_class_table(metrics: dict, sort_by: str = "f1") -> None:
    """
    Print per-class F1 / precision / recall / support as an ASCII table,
    sorted descending by *sort_by* ("f1", "precision", "recall", "support").
    """
    per_class = sorted(metrics["per_class"], key=lambda d: d[sort_by], reverse=True)

    hdr = f"{'Class':<22}  {'F1':>6}  {'Prec':>6}  {'Rec':>6}  {'Supp':>5}  {'TP':>5}  {'FP':>5}  {'FN':>5}"
    sep = "─" * len(hdr)
    print(sep)
    print(hdr)
    print(sep)
    for d in per_class:
        bar = "█" * int(d["f1"] * 12)
        print(
            f"{d['name']:<22}  {d['f1']:>6.3f}  {d['precision']:>6.3f}  "
            f"{d['recall']:>6.3f}  {d['support']:>5d}  "
            f"{d['tp']:>5d}  {d['fp']:>5d}  {d['fn']:>5d}  {bar}"
        )
    print(sep)
    print(
        f"{'MICRO avg':<22}  {metrics['f1_micro']:>6.3f}  "
        f"{metrics['precision_micro']:>6.3f}  {metrics['recall_micro']:>6.3f}"
    )
    print(
        f"{'MACRO avg':<22}  {metrics['f1_macro']:>6.3f}  "
        f"{metrics['precision_macro']:>6.3f}  {metrics['recall_macro']:>6.3f}"
    )
    print(sep)


# ── Per-class F1 bar chart ────────────────────────────────────────────────────

def plot_per_class_f1(
    metrics: dict,
    save_path: Path | str | None = None,
    title: str = "Per-class F1",
    figsize: tuple[float, float] = (10, 9),
    benchmark_line: float | None = None,
) -> plt.Figure:
    """
    Horizontal bar chart of per-class F1 scores, sorted ascending.

    Args:
        benchmark_line: if given, draw a vertical dashed line at this F1 value
                        (e.g. 0.89 for the F1 > 89% target).
    """
    per_class = sorted(metrics["per_class"], key=lambda d: d["f1"])
    names  = [d["name"]      for d in per_class]
    f1s    = [d["f1"]        for d in per_class]
    precs  = [d["precision"] for d in per_class]
    recs   = [d["recall"]    for d in per_class]

    colours = ["#d7191c" if f < 0.5 else "#fdae61" if f < 0.75 else "#1a9641"
               for f in f1s]

    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(names))

    bars = ax.barh(y, f1s, color=colours, height=0.6, alpha=0.85, label="F1")
    ax.scatter(precs, y, marker="|", color="steelblue", s=80, zorder=5, label="Precision")
    ax.scatter(recs,  y, marker="D", color="darkorange", s=25, zorder=5, label="Recall")

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Score")
    ax.set_xlim(0, 1.05)
    ax.set_title(title, fontsize=11)
    ax.axvline(metrics["f1_micro"], color="gray",    lw=1.2, ls="--", alpha=0.7,
               label=f"F1-micro={metrics['f1_micro']:.3f}")
    ax.axvline(metrics["f1_macro"], color="dimgray", lw=1.2, ls=":",  alpha=0.7,
               label=f"F1-macro={metrics['f1_macro']:.3f}")
    if benchmark_line is not None:
        ax.axvline(benchmark_line, color="red", lw=1.5, ls="-.", alpha=0.6,
                   label=f"Target={benchmark_line:.2f}")

    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


# ── Multilabel confusion matrix grid ─────────────────────────────────────────

def plot_multilabel_confusion(
    metrics: dict,
    save_path: Path | str | None = None,
    title: str = "Per-class Confusion Matrices",
    ncols: int = 6,
    cell_size: float = 1.5,
) -> plt.Figure:
    """
    Grid of per-class 2×2 confusion matrices.

    Layout:
        [[TN  FP]
         [FN  TP]]

    Each cell is coloured by the normalised value (row-normalised by true count),
    and annotated with the raw count.  Classes with zero support are shown in grey.
    """
    per_class = metrics["per_class"]
    n_classes = len(per_class)
    nrows = int(np.ceil(n_classes / ncols))

    fig_w = ncols * cell_size * 2.2
    fig_h = nrows * cell_size * 2.4
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
    axes_flat = axes.flatten()

    for c, d in enumerate(per_class):
        ax = axes_flat[c]
        tn, fp, fn, tp = d["tn"], d["fp"], d["fn"], d["tp"]
        cm = np.array([[tn, fp], [fn, tp]], dtype=float)

        # Row-normalise (avoid divide-by-zero)
        row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
        cm_norm  = cm / row_sums

        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")

        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, f"{int(val)}", ha="center", va="center",
                    fontsize=6.5, color="black" if cm_norm[i, j] < 0.6 else "white")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"], fontsize=5.5)
        ax.set_yticklabels(["True 0", "True 1"], fontsize=5.5)

        colour = "black" if d["f1"] >= 0.5 else "#cc0000"
        ax.set_title(f"{d['name']}\nF1={d['f1']:.2f}  n={d['support']}",
                     fontsize=6, pad=2, color=colour)

    # Hide unused axes
    for c in range(n_classes, len(axes_flat)):
        axes_flat[c].set_visible(False)

    fig.suptitle(title, fontsize=12, y=1.01)
    fig.tight_layout(pad=0.4, h_pad=0.8, w_pad=0.3)

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


# ── CSV export ────────────────────────────────────────────────────────────────

def save_per_class_csv(metrics: dict, save_path: Path | str) -> None:
    """Save per-class statistics to a CSV file."""
    import csv
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["class", "f1", "precision", "recall", "support", "tp", "fp", "fn", "tn"]
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for d in metrics["per_class"]:
            writer.writerow({
                "class":     d["name"],
                "f1":        f"{d['f1']:.4f}",
                "precision": f"{d['precision']:.4f}",
                "recall":    f"{d['recall']:.4f}",
                "support":   d["support"],
                "tp":        d["tp"],
                "fp":        d["fp"],
                "fn":        d["fn"],
                "tn":        d["tn"],
            })
    print(f"Saved: {save_path}")


# ── All-in-one diagnostics ────────────────────────────────────────────────────

def run_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_dir: Path | str,
    model_name: str = "model",
    class_names: Sequence[str] = FG_NAMES,
    benchmark_f1: float | None = 0.89,
) -> dict:
    """
    Compute and save all diagnostics in one call.

    Creates:
      {out_dir}/per_class_f1.csv
      {out_dir}/per_class_f1.png
      {out_dir}/confusion_matrices.png

    Returns the full metrics dict.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = compute_full_metrics(y_true, y_pred, class_names)

    print(f"\n{'='*60}")
    print(f"Diagnostics — {model_name}")
    print_per_class_table(metrics)

    save_per_class_csv(metrics, out_dir / "per_class_f1.csv")

    plot_per_class_f1(
        metrics,
        save_path=out_dir / "per_class_f1.png",
        title=f"Per-class F1 — {model_name}",
        benchmark_line=benchmark_f1,
    )

    plot_multilabel_confusion(
        metrics,
        save_path=out_dir / "confusion_matrices.png",
        title=f"Per-class Confusion Matrices — {model_name}",
    )

    plt.close("all")
    return metrics
