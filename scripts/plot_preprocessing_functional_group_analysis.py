#!/usr/bin/env python3
"""Aggregate five-seed per-class metrics and create one bar chart per modality."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


MODELS = ("CNN", "MPS", "TTN", "XGBoost")
MODEL_LABELS = {"cnn": "CNN", "xgboost": "XGBoost", "mps": "MPS", "ttn": "TTN"}
MODALITY_LABELS = {
    "cnmr": "¹³C-NMR",
    "hnmr": "¹H-NMR",
    "ir": "IR",
    "msms_neg": "MS/MS(−)",
    "msms_pos": "MS/MS(+)",
}
COLORS = {"CNN": "#2CA02C", "MPS": "#1F77B4", "TTN": "#FF7F0E", "XGBoost": "#D62728"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--tensor-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--analysis-label", default="Preprocessed")
    parser.add_argument(
        "--subtitle",
        default="FIR + Gaussian smoothing + sqrt transformation + SNV normalization",
    )
    parser.add_argument("--allow-incomplete-xgboost", action="store_true")
    parser.add_argument("--exclude-xgboost", action="store_true")
    parser.add_argument(
        "--order-from-summary",
        type=Path,
        help="Use the functional-group ordering derived from another summary CSV.",
    )
    return parser.parse_args()


def load_records(directory: Path) -> list[dict]:
    records = []
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text())
        groups = payload["functional_groups"]
        arrays = {
            "support": payload["support_per_class"],
            "f1": payload["f1_per_class"],
            "precision": payload["precision_per_class"],
            "recall": payload["recall_per_class"],
            "tp": payload["true_positives_per_class"],
            "fp": payload["false_positives_per_class"],
            "fn": payload["false_negatives_per_class"],
        }
        if any(len(values) != len(groups) for values in arrays.values()):
            raise ValueError(f"Inconsistent per-class arrays in {path}")
        model = MODEL_LABELS[payload["model"]]
        for index, group in enumerate(groups):
            records.append(
                {
                    "modality": payload["modality"],
                    "model": model,
                    "seed": int(payload["seed"]),
                    "functional_group": group,
                    **{name: values[index] for name, values in arrays.items()},
                }
            )
    return records


def build_figure(
    summary: pd.DataFrame,
    modality: str,
    analysis_label: str,
    subtitle: str,
    models: tuple[str, ...],
    reference_summary: pd.DataFrame | None = None,
) -> plt.Figure:
    block = summary[summary["modality"] == modality].copy()
    order_source = (
        reference_summary[reference_summary["modality"] == modality]
        if reference_summary is not None
        else block
    )
    order = (
        order_source.groupby("functional_group")["f1_mean"]
        .mean()
        .sort_values(ascending=True)
        .index.tolist()
    )
    y = np.arange(len(order), dtype=float)
    bar_height = 0.76 / len(models)
    offsets = (np.arange(len(models)) - (len(models) - 1) / 2) * bar_height
    fig, ax = plt.subplots(figsize=(14, 18))
    for model, offset in zip(models, offsets, strict=True):
        indexed = block[block["model"] == model].set_index("functional_group").reindex(order)
        ax.barh(
            y + offset,
            indexed["f1_mean"],
            height=bar_height,
            xerr=indexed["f1_sd"].fillna(0),
            color=COLORS[model],
            label=model,
            alpha=0.9,
            error_kw={"elinewidth": 0.7, "capsize": 1.5, "capthick": 0.7},
        )

    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=9)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Test F1 per functional group (mean ± sample SD across available seeds)")
    ax.set_title(
        f"{MODALITY_LABELS[modality]} — {analysis_label} functional-group recognition\n"
        f"{subtitle}",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", ncol=2, frameon=False)
    ax.text(
        0,
        -0.045,
        "Mean F1 = 0 means no correct positive detections for that model/group in any available seed. "
        "Error bars show seed-to-seed variability; no bar is shown when only one seed exists.",
        transform=ax.transAxes,
        fontsize=9,
        color="#444444",
    )
    fig.subplots_adjust(left=0.28, right=0.98, top=0.94, bottom=0.07)
    return fig


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(args.baseline_dir) + load_records(args.tensor_dir)
    detailed = pd.DataFrame(records)
    models = tuple(model for model in MODELS if not (args.exclude_xgboost and model == "XGBoost"))
    detailed = detailed[detailed["model"].isin(models)].copy()
    counts = detailed.groupby(["modality", "model"])["seed"].nunique()
    expected_pairs = {(modality, model) for modality in MODALITY_LABELS for model in models}
    if set(counts.index) != expected_pairs:
        raise ValueError("At least one modality/model combination is missing")
    incomplete = counts[counts != 5]
    allowed = args.allow_incomplete_xgboost and all(
        model == "XGBoost" and count >= 1
        for (_, model), count in incomplete.items()
    )
    if len(incomplete) and not allowed:
        raise ValueError(f"Incomplete seed coverage:\n{incomplete}")

    summary = (
        detailed.groupby(["modality", "model", "functional_group"], as_index=False)
        .agg(
            f1_mean=("f1", "mean"),
            f1_sd=("f1", "std"),
            precision_mean=("precision", "mean"),
            precision_sd=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_sd=("recall", "std"),
            support_mean=("support", "mean"),
            n_seeds=("seed", "nunique"),
            seeds_detected=("f1", lambda values: int((values > 0).sum())),
            tp_mean=("tp", "mean"),
            fp_mean=("fp", "mean"),
            fn_mean=("fn", "mean"),
        )
    )
    recognition = (
        summary.assign(
            detected_all_seeds=summary["seeds_detected"] == summary["n_seeds"],
            not_detected_any_seed=summary["seeds_detected"] == 0,
            strong_f1=summary["f1_mean"] >= 0.5,
        )
        .groupby(["modality", "model"], as_index=False)
        .agg(
            mean_class_f1=("f1_mean", "mean"),
            n_seeds=("n_seeds", "max"),
            detected_all_seeds=("detected_all_seeds", "sum"),
            detected_at_least_one_seed=("seeds_detected", lambda values: int((values > 0).sum())),
            not_detected_any_seed=("not_detected_any_seed", "sum"),
            groups_with_mean_f1_at_least_0_5=("strong_f1", "sum"),
        )
    )

    detailed.to_csv(args.output_dir / "per_class_metrics_all_seeds.csv", index=False)
    summary.to_csv(args.output_dir / "per_class_summary_mean_sd.csv", index=False)
    recognition.to_csv(args.output_dir / "recognition_summary.csv", index=False)

    reference_summary = (
        pd.read_csv(args.order_from_summary) if args.order_from_summary else None
    )

    pdf_path = args.output_dir / "functional_group_error_analysis_all_spectra.pdf"
    with PdfPages(pdf_path) as pdf:
        for modality in MODALITY_LABELS:
            figure = build_figure(
                summary,
                modality,
                args.analysis_label,
                args.subtitle,
                models,
                reference_summary,
            )
            figure.savefig(
                args.output_dir / f"functional_group_f1_{modality}.png",
                dpi=300,
                bbox_inches="tight",
            )
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)
    print(f"Wrote figures, tables and {pdf_path}")


if __name__ == "__main__":
    main()
