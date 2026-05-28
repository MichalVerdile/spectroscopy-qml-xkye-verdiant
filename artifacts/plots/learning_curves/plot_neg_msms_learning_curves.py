#!/usr/bin/env python3
"""Plot negative-mode MS/MS learning-curve comparisons from training logs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
TITLE = "MS/MS-"
SOURCES = {
    "MPS": ("epoch_csv", ROOT / "artifacts/Error_Analyse/MPS/MSMS-/training_log.csv"),
    "TTN": ("epoch_csv", ROOT / "artifacts/Error_Analyse/TTN/MSMS-/training_log.csv"),
    "CNN": ("cnn_csv", ROOT / "benchmark/cnn/models/neg_msms/original/training_logs.csv"),
    "XGBoost": ("xgb_csv", ROOT / "benchmark/xgb/models/neg_msms/training_logs.csv"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/plots/learning_curves/neg_msms_learning_curves.png",
        help="Output PNG path.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_epoch_log(path: Path, split: str) -> tuple[list[float], list[float], list[float]]:
    rows = read_csv(path)
    x = [float(row["epoch"]) for row in rows if row.get("epoch")]
    micro = [float(row[f"{split}_f1_micro"]) for row in rows if row.get("epoch")]
    macro = [float(row[f"{split}_f1_macro"]) for row in rows if row.get("epoch")]
    return x, micro, macro


def read_xgb_log(path: Path, split: str) -> tuple[list[float], list[float], list[float]]:
    rows = [row for row in read_csv(path) if row.get("event") == "learning_curve" and row.get("step")]
    x = [float(row["step"]) for row in rows]
    micro = [float(row[f"{split}_f1_micro"]) for row in rows]
    macro = [float(row[f"{split}_f1_macro"]) for row in rows]
    return x, micro, macro


def read_cnn_log(path: Path, split: str) -> tuple[list[float], list[float], list[float]]:
    rows = [row for row in read_csv(path) if row.get("event") == "epoch_finished" and row.get("epoch")]
    x = [float(row["epoch"]) for row in rows]
    micro = [float(row[f"{split}_f1_micro"]) for row in rows]
    macro = [float(row[f"{split}_f1_macro"]) for row in rows]
    return x, micro, macro


READERS = {
    "epoch_csv": read_epoch_log,
    "cnn_csv": read_cnn_log,
    "xgb_csv": read_xgb_log,
}



def load_series(split: str) -> tuple[dict[str, tuple[list[float], list[float], list[float]]], list[str]]:
    loaded: dict[str, tuple[list[float], list[float], list[float]]] = {}
    skipped: list[str] = []
    for label, (reader_name, path) in SOURCES.items():
        if not path.exists():
            skipped.append(f"{label}: missing {path}")
            continue
        x, micro, macro = READERS[reader_name](path, split)
        if not x:
            skipped.append(f"{label}: no plottable rows in {path}")
            continue
        loaded[label] = (x, micro, macro)
    return loaded, skipped


def build_plot(output: Path) -> list[str]:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(18, 14), constrained_layout=True)
    specs = [
        ("train", "micro", axes[0, 0], f"{TITLE} Train F1 Micro"),
        ("train", "macro", axes[0, 1], f"{TITLE} Train F1 Macro"),
        ("val", "micro", axes[1, 0], f"{TITLE} Val F1 Micro"),
        ("val", "macro", axes[1, 1], f"{TITLE} Val F1 Macro"),
    ]

    cache: dict[str, dict[str, tuple[list[float], list[float], list[float]]]] = {}
    skipped_all: list[str] = []
    for split, metric, axis, title in specs:
        if split not in cache:
            cache[split], skipped = load_series(split)
            skipped_all.extend(skipped)
        for label, (x, micro, macro) in cache[split].items():
            y = micro if metric == "micro" else macro
            axis.plot(x, y, linewidth=2.5, label=label)
        axis.set_title(title, fontsize=20)
        axis.set_xlabel("Epoch / Step", fontsize=14)
        axis.set_ylabel(f"F1 {metric.capitalize()}", fontsize=14)
        axis.set_xlim(left=0)
        axis.set_ylim(0, 1)
        axis.grid(True, alpha=0.3)
        axis.tick_params(labelsize=12)
        axis.legend(fontsize=12)

    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Saved plot to: {output}")
    return sorted(set(skipped_all))


def print_summary() -> None:
    print(f"Dataset: {TITLE}")
    for split in ("train", "val"):
        series, skipped = load_series(split)
        print(f"\n--- {split} ---")
        for label, (x, micro, macro) in series.items():
            print(
                f"{label:8s} points={len(x):3d} "
                f"x=[{x[0]:.0f},{x[-1]:.0f}] "
                f"{split}_f1_micro=[{micro[0]:.4f},{micro[-1]:.4f}] "
                f"{split}_f1_macro=[{macro[0]:.4f},{macro[-1]:.4f}]"
            )
        for item in skipped:
            print(f"skipped: {item}")


def main() -> None:
    args = parse_args()
    skipped = build_plot(args.output)
    print_summary()
    if skipped:
        print("\nSkipped sources:")
        for item in skipped:
            print(f"- {item}")


if __name__ == "__main__":
    main()
