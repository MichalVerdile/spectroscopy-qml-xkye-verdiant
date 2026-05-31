from __future__ import annotations

import csv
import json
import os
import statistics
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MPL_CACHE_DIR = ROOT / ".cache" / "matplotlib"
XDG_CACHE_DIR = ROOT / ".cache"
XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ARTIFACT_ROOT = ROOT / "artifacts" / "Error_Analyse"
PDF_EXPORT_ROOT = ROOT / "artifacts" / "Error_Analyse_PDF_Overleaf"
AUDIT_SUMMARY_PATH = PDF_EXPORT_ROOT / "audit_summary.md"

BASE_COLOR = "#55C1BE"
LOW_COLOR = "#BFE8E4"
SEVERE_COLOR = "#B00020"


@dataclass(frozen=True)
class DatasetConfig:
    model_key: str
    modality_key: str
    model_display: str
    modality_display: str

    @property
    def artifact_dir(self) -> Path:
        return ARTIFACT_ROOT / self.model_key / self.modality_key

    @property
    def export_prefix(self) -> str:
        return f"{self.model_key}_{self.modality_key}"


DATASETS = [
    DatasetConfig("CNN", "IR", "CNN", "IR"),
    DatasetConfig("CNN", "H-NMR", "CNN", "H-NMR"),
    DatasetConfig("CNN", "C-NMR", "CNN", "C-NMR"),
    DatasetConfig("CNN", "MSMS+", "CNN", "MSMS+"),
    DatasetConfig("CNN", "MSMS-", "CNN", "MSMS-"),
    DatasetConfig("MPS", "IR", "MPS", "IR"),
    DatasetConfig("MPS", "H-NMR", "MPS", "H-NMR"),
    DatasetConfig("MPS", "C-NMR", "MPS", "C-NMR"),
    DatasetConfig("MPS", "MSMS+", "MPS", "MSMS+"),
    DatasetConfig("MPS", "MSMS-", "MPS", "MSMS-"),
    DatasetConfig("TTN", "IR", "TTN", "IR"),
    DatasetConfig("TTN", "H-NMR", "TTN", "H-NMR"),
    DatasetConfig("TTN", "C-NMR", "TTN", "C-NMR"),
    DatasetConfig("TTN", "MSMS+", "TTN", "MSMS+"),
    DatasetConfig("TTN", "MSMS-", "TTN", "MSMS-"),
    DatasetConfig("XGB", "IR", "XGBoost", "IR"),
    DatasetConfig("XGB", "H_NMR", "XGBoost", "H-NMR"),
    DatasetConfig("XGB", "C_NMR", "XGBoost", "C-NMR"),
    DatasetConfig("XGB", "MSMS+", "XGBoost", "MSMS+"),
    DatasetConfig("XGB", "MSMS-", "XGBoost", "MSMS-"),
]


def fmt_decimal(value: float, digits: int = 4) -> str:
    quant = Decimal("1").scaleb(-digits)
    return str(Decimal(str(value)).quantize(quant, rounding=ROUND_HALF_UP))


def fmt_percent(value: float, digits: int = 4) -> str:
    return f"{fmt_decimal(value, digits)}%"


def split_display(split_name: str) -> str:
    if split_name == "test":
        return "Test"
    if split_name == "validation":
        return "Validation"
    raise ValueError(f"Unsupported split: {split_name}")


def load_rows(csv_path: Path) -> list[dict[str, object]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            rows.append(
                {
                    "label_name": row["label_name"],
                    "error_rate": float(row["error_rate"]) * 100.0,
                }
            )
    rows.sort(key=lambda item: float(item["error_rate"]), reverse=True)
    return rows


def load_metrics(metrics_path: Path) -> dict[str, float]:
    with metrics_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def bar_colors(values: list[float]) -> list[str]:
    colors = []
    for value in values:
        if value > 50:
            colors.append(SEVERE_COLOR)
        elif value > 20:
            colors.append(BASE_COLOR)
        else:
            colors.append(LOW_COLOR)
    return colors


def build_titles(dataset: DatasetConfig, split_name: str) -> tuple[str, str]:
    split_label = split_display(split_name)
    plot_title = f"{dataset.model_display} {dataset.modality_display} {split_label} Set"
    summary_title = f"{dataset.model_display.upper()} {dataset.modality_display.upper()} {split_label.upper()} SET ERROR ANALYSIS"
    return plot_title, summary_title


def render_detailed_figure(
    rows: list[dict[str, object]],
    metrics: dict[str, float],
    dataset: DatasetConfig,
    split_name: str,
    output_png: Path,
    output_pdf: Path,
) -> dict[str, float]:
    values = [float(row["error_rate"]) for row in rows]
    labels = [str(row["label_name"]) for row in rows]
    plot_title, summary_title = build_titles(dataset, split_name)

    mean_value = statistics.fmean(values)
    median_value = statistics.median(values)
    std_value = statistics.stdev(values) if len(values) > 1 else 0.0
    max_idx = max(range(len(values)), key=values.__getitem__)
    min_idx = min(range(len(values)), key=values.__getitem__)

    fig = plt.figure(figsize=(16, 12))

    ax1 = plt.subplot(2, 2, 1)
    ax1.barh(range(len(rows)), values, color=bar_colors(values), edgecolor="black", linewidth=0.5)
    ax1.set_yticks(range(len(rows)))
    ax1.set_yticklabels(labels, fontsize=8)
    ax1.set_xlabel("Missed-positive error rate: FN / positives (%)")
    ax1.set_title(f"{plot_title} - All 37 Functional Groups")
    ax1.set_xlim(0, max(105.0, max(values) + 5.0))
    ax1.grid(axis="x", alpha=0.3)
    for idx, value in enumerate(values):
        if value >= 1:
            ax1.text(value + 1, idx, fmt_percent(value), va="center", fontsize=7)
    ax1.invert_yaxis()

    ax2 = plt.subplot(2, 2, 2)
    ax2.hist(values, bins=18, color=BASE_COLOR, alpha=0.75, edgecolor="black")
    ax2.axvline(mean_value, color="black", linestyle="--", label=f"Mean {fmt_percent(mean_value)}")
    ax2.axvline(median_value, color="black", linestyle=":", label=f"Median {fmt_percent(median_value)}")
    ax2.set_xlabel("Error rate (%)")
    ax2.set_ylabel("# Functional Groups")
    ax2.set_title("Error-Rate Distribution")
    ax2.legend()

    ax3 = plt.subplot(2, 2, 3)
    top_rows = rows[:10]
    top_values = [float(row["error_rate"]) for row in top_rows]
    top_labels = [str(row["label_name"]) for row in top_rows]
    ax3.bar(range(len(top_rows)), top_values, color=BASE_COLOR, edgecolor="black")
    ax3.set_xticks(range(len(top_rows)))
    ax3.set_xticklabels(top_labels, rotation=45, ha="right")
    ax3.set_ylabel("Error rate (%)")
    ax3.set_title("Top 10 Problem Groups")
    ax3.set_ylim(0, max(105.0, max(top_values) + 8.0))
    ax3.grid(axis="y", alpha=0.3)
    for idx, value in enumerate(top_values):
        ax3.text(idx, value + 1.5, fmt_percent(value), ha="center", fontsize=9)

    ax4 = plt.subplot(2, 2, 4)
    ax4.axis("off")
    summary = (
        f"{summary_title}\n\n"
        f"Global metrics:\n"
        f"  F1 micro:          {fmt_decimal(metrics['f1_micro'])}\n"
        f"  F1 macro:          {fmt_decimal(metrics['f1_macro'])}\n"
        f"  Precision micro:   {fmt_decimal(metrics['precision_micro'])}\n"
        f"  Recall micro:      {fmt_decimal(metrics['recall_micro'])}\n"
        f"  Hamming accuracy:  {fmt_decimal(metrics['hamming_accuracy'])}\n\n"
        f"Error-rate statistics:\n"
        f"  Mean:              {fmt_percent(mean_value)}\n"
        f"  Median:            {fmt_percent(median_value)}\n"
        f"  Std:               {fmt_percent(std_value)}\n"
        f"  Max:               {fmt_percent(values[max_idx])} ({labels[max_idx]})\n"
        f"  Min:               {fmt_percent(values[min_idx])} ({labels[min_idx]})\n\n"
        f"Problem classes:\n"
        f"  Severe (>50%):     {sum(value > 50 for value in values)} groups\n"
        f"  High (20-50%):     {sum(20 < value <= 50 for value in values)} groups\n"
        f"  Moderate (5-20%):  {sum(5 < value <= 20 for value in values)} groups\n"
        f"  Low (<=5%):        {sum(value <= 5 for value in values)} groups\n"
    )
    ax4.text(
        0.02,
        0.98,
        summary,
        transform=ax4.transAxes,
        va="top",
        family="monospace",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "#F4F4F4", "alpha": 0.95},
    )

    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)

    return {
        "mean": mean_value,
        "median": median_value,
        "std": std_value,
        "f1_micro": float(metrics["f1_micro"]),
        "f1_macro": float(metrics["f1_macro"]),
    }


def render_top_problem_figure(
    rows: list[dict[str, object]],
    dataset: DatasetConfig,
    split_name: str,
    output_png: Path,
    output_pdf: Path,
) -> None:
    plot_title, _ = build_titles(dataset, split_name)
    top_rows = rows[:15]
    values = [float(row["error_rate"]) for row in top_rows]
    labels = [str(row["label_name"]) for row in top_rows]

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.barh(range(len(top_rows)), values, color=BASE_COLOR, edgecolor="black")
    ax.set_yticks(range(len(top_rows)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Error rate: FN / positives (%)")
    ax.set_title(f"{plot_title} - Top 15 Problem Groups")
    ax.set_xlim(0, max(105.0, max(values) + 5.0))
    ax.grid(axis="x", alpha=0.3)
    for idx, value in enumerate(values):
        ax.text(value + 1, idx, fmt_percent(value), va="center", fontsize=9)
    ax.invert_yaxis()
    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def export_pdf_path(dataset: DatasetConfig, png_name: str) -> Path:
    stem = Path(png_name).stem
    return PDF_EXPORT_ROOT / f"{dataset.export_prefix}_{stem}.pdf"


def write_audit_summary(lines: list[str]) -> None:
    PDF_EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
    AUDIT_SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    summary_lines = [
        "# Error-Analysis Audit",
        "",
        "The figures in `artifacts/Error_Analyse` were regenerated from the existing CSV and JSON metrics files.",
        "This corrects mislabeled figure titles and recreates matching PDF exports for Overleaf.",
        "",
        "| Dataset | Split | F1 Macro | Mean Error | Median Error |",
        "| --- | --- | ---: | ---: | ---: |",
    ]

    for dataset in DATASETS:
        for split_name in ("test", "validation"):
            csv_path = dataset.artifact_dir / f"{split_name}_error_analysis.csv"
            metrics_path = dataset.artifact_dir / f"{split_name}_error_analysis_metrics.json"
            detailed_png = dataset.artifact_dir / f"{split_name}_error_analysis_detailed.png"
            top_png = dataset.artifact_dir / f"{split_name}_top_problem_groups.png"

            if not csv_path.exists() or not metrics_path.exists():
                raise FileNotFoundError(f"Missing required artifact for {dataset.export_prefix} {split_name}")

            rows = load_rows(csv_path)
            metrics = load_metrics(metrics_path)

            stats = render_detailed_figure(
                rows,
                metrics,
                dataset,
                split_name,
                detailed_png,
                export_pdf_path(dataset, detailed_png.name),
            )
            render_top_problem_figure(
                rows,
                dataset,
                split_name,
                top_png,
                export_pdf_path(dataset, top_png.name),
            )

            summary_lines.append(
                f"| {dataset.export_prefix} | {split_name} | {fmt_decimal(stats['f1_macro'])} | {fmt_percent(stats['mean'])} | {fmt_percent(stats['median'])} |"
            )

    write_audit_summary(summary_lines)


if __name__ == "__main__":
    main()
