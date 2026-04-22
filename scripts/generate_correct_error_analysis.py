#!/usr/bin/env python3
"""Generate verified CNN vs TTN 10.2 error-analysis CSVs and PNGs.

The previous root-level plotting scripts used hard-coded error tables and an
incorrect functional-group index mapping. This script computes the CNN metrics
from the saved CNN result pickle and the TTN metrics from the real experiment
10.2 checkpoint, split artifact, and selected thresholds.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment5.data_loader import (  # noqa: E402
    FUNCTIONAL_GROUPS,
    load_ir_data,
)
from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment10_2.model import (  # noqa: E402
    TTNIRClassifier10_2,
)

LABEL_NAMES = list(FUNCTIONAL_GROUPS.keys())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate corrected CNN/TTN 10.2 error-analysis figures."
    )
    parser.add_argument(
        "--cnn-pickle",
        type=Path,
        default=Path("benchmark/cnn/models/ir/original/results.pickle"),
    )
    parser.add_argument(
        "--ttn-run-dir",
        type=Path,
        default=Path(
            "src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment10_2/"
            "results/full_dataset_run_20260417_173715_percentile"
        ),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"], default="auto")
    parser.add_argument(
        "--skip-ttn-inference",
        action="store_true",
        help="Only regenerate plots from existing CSV files in output-dir.",
    )
    return parser


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def load_cnn_predictions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if "pred" in payload and "tgt" in payload:
        pred = np.asarray(payload["pred"])
        target = np.asarray(payload["tgt"])
    elif "test_predictions" in payload and "test_targets" in payload:
        pred = np.asarray(payload["test_predictions"])
        target = np.asarray(payload["test_targets"])
    else:
        raise KeyError(f"Unexpected CNN results format: {list(payload.keys())}")
    pred = (pred >= 0.5).astype(np.int32) if pred.dtype.kind == "f" else pred.astype(np.int32)
    return pred, target.astype(np.int32)


def per_label_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    rows = []
    for idx, name in enumerate(LABEL_NAMES):
        yt = y_true[:, idx].astype(np.int32)
        yp = y_pred[:, idx].astype(np.int32)
        tp = int(((yt == 1) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        tn = int(((yt == 0) & (yp == 0)).sum())
        positives = int(yt.sum())
        total = int(len(yt))
        rows.append(
            {
                "label_idx": idx,
                "label_name": name,
                "positive_samples": positives,
                "tp": tp,
                "fn": fn,
                "fp": fp,
                "tn": tn,
                "f1": f1_score(yt, yp, zero_division=0),
                "precision": precision_score(yt, yp, zero_division=0),
                "recall": recall_score(yt, yp, zero_division=0),
                # Primary "error rate" used in the report: missed positives.
                "error_rate": fn / positives if positives > 0 else 0.0,
                # Secondary binary label error over all samples.
                "binary_error_rate": (fp + fn) / total if total > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("error_rate", ascending=False).reset_index(drop=True)


def global_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_micro": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "recall_micro": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "hamming_accuracy": float((y_true == y_pred).mean()),
    }


def load_thresholds(run_dir: Path) -> np.ndarray:
    data = json.loads((run_dir / "selected_thresholds.json").read_text())
    raw = data["thresholds"]
    if isinstance(raw, dict):
        return np.asarray([raw[name] for name in LABEL_NAMES], dtype=np.float32)
    return np.asarray(raw, dtype=np.float32)


def build_ttn_model(run_dir: Path, device: torch.device) -> TTNIRClassifier10_2:
    config = json.loads((run_dir / "run_config.json").read_text())
    model = TTNIRClassifier10_2(
        num_labels=int(config.get("num_labels", len(LABEL_NAMES))),
        chi=int(config.get("chi", 64)),
        input_dim=int(config.get("input_dim", 1800)),
        segment_window_size=int(config.get("segment_window_size", 48)),
        segment_stride=int(config.get("segment_stride", 43)),
        segment_mode=config.get("segment_mode", "overlap"),
        segment_offset=config.get("segment_offset"),
        segment_state_normalize=bool(config.get("segment_state_normalize", True)),
        merge_mode=config.get("merge_mode", "relaxed"),
        merge_residual_weight=float(config.get("merge_residual_weight", 0.1)),
        merge_renormalize_output=bool(config.get("merge_renormalize_output", True)),
        lorentz_gamma=float(config.get("lorentz_gamma", 3.0)),
        lorentz_kernel_half_width=int(config.get("lorentz_kernel_half_width", 15)),
        lorentz_norm_mode=config.get("lorentz_norm_mode", "percentile"),
    )
    state = torch.load(run_dir / "ttn_ir_best.pt", map_location=device)
    if any(str(key).startswith("_orig_mod.") for key in state):
        state = {key.removeprefix("_orig_mod."): value for key, value in state.items()}
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def compute_ttn_predictions(
    run_dir: Path,
    data_dir: Path,
    cache_path: Path | None,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    config = json.loads((run_dir / "run_config.json").read_text())
    split_path = Path(config["split_path"]) if config.get("split_path") else run_dir / "data_split_seed42_all.npz"
    thresholds = load_thresholds(run_dir)
    model = build_ttn_model(run_dir, device)

    X, y = load_ir_data(
        data_dir=data_dir,
        target_length=int(config.get("input_dim", 1800)),
        max_files=config.get("max_files"),
        apply_snv=bool(config.get("apply_snv", True)),
        cache_path=cache_path,
        overwrite_cache=False,
    )
    split = np.load(split_path)
    test_idx = split["test_indices"].astype(np.int64)
    y_true = y[test_idx].astype(np.int32)
    y_pred = np.zeros_like(y_true, dtype=np.int32)

    print(f"TTN test inference: {len(test_idx)} samples on {device}")
    with torch.no_grad():
        for batch_no, start in enumerate(range(0, len(test_idx), batch_size), start=1):
            end = min(start + batch_size, len(test_idx))
            spectra = torch.from_numpy(X[test_idx[start:end]]).float().to(device)
            logits = model(spectra)
            probs = torch.sigmoid(logits.float()).cpu().numpy()
            y_pred[start:end] = (probs >= thresholds).astype(np.int32)
            if batch_no % 10 == 0 or end == len(test_idx):
                print(f"  batch {batch_no:04d}  {end}/{len(test_idx)}")
    return y_true, y_pred


def plot_individual(
    df: pd.DataFrame,
    metrics: dict[str, float],
    title: str,
    color: str,
    out_path: Path,
) -> None:
    values = df["error_rate"] * 100.0
    fig = plt.figure(figsize=(16, 12))

    ax1 = plt.subplot(2, 2, 1)
    colors = ["#B00020" if x > 50 else color if x > 20 else "#BFE8E4" for x in values]
    ax1.barh(range(len(df)), values, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_yticks(range(len(df)))
    ax1.set_yticklabels(df["label_name"], fontsize=8)
    ax1.set_xlabel("Missed-positive error rate: FN / positives (%)")
    ax1.set_title(f"{title} - All 37 Functional Groups")
    ax1.set_xlim(0, max(105, values.max() + 5))
    ax1.grid(axis="x", alpha=0.3)
    for i, value in enumerate(values):
        if value >= 1:
            ax1.text(value + 1, i, f"{value:.1f}%", va="center", fontsize=7)
    ax1.invert_yaxis()

    ax2 = plt.subplot(2, 2, 2)
    ax2.hist(values, bins=18, color=color, alpha=0.75, edgecolor="black")
    ax2.axvline(values.mean(), color="black", linestyle="--", label=f"Mean {values.mean():.1f}%")
    ax2.axvline(values.median(), color="black", linestyle=":", label=f"Median {values.median():.1f}%")
    ax2.set_xlabel("Error rate (%)")
    ax2.set_ylabel("# Functional Groups")
    ax2.set_title("Error-Rate Distribution")
    ax2.legend()

    ax3 = plt.subplot(2, 2, 3)
    top = df.head(10)
    top_values = top["error_rate"] * 100.0
    ax3.bar(range(len(top)), top_values, color=color, edgecolor="black")
    ax3.set_xticks(range(len(top)))
    ax3.set_xticklabels(top["label_name"], rotation=45, ha="right")
    ax3.set_ylabel("Error rate (%)")
    ax3.set_title("Top 10 Problem Groups")
    ax3.set_ylim(0, max(105, top_values.max() + 8))
    ax3.grid(axis="y", alpha=0.3)
    for i, value in enumerate(top_values):
        ax3.text(i, value + 1.5, f"{value:.1f}%", ha="center", fontsize=9)

    ax4 = plt.subplot(2, 2, 4)
    ax4.axis("off")
    max_row = df.iloc[0]
    min_row = df.iloc[-1]
    summary = (
        f"{title.upper()} ERROR ANALYSIS\n\n"
        f"Global metrics:\n"
        f"  F1 micro:          {metrics['f1_micro']:.4f}\n"
        f"  F1 macro:          {metrics['f1_macro']:.4f}\n"
        f"  Precision micro:   {metrics['precision_micro']:.4f}\n"
        f"  Recall micro:      {metrics['recall_micro']:.4f}\n"
        f"  Hamming accuracy:  {metrics['hamming_accuracy']:.4f}\n\n"
        f"Error-rate statistics:\n"
        f"  Mean:              {values.mean():.2f}%\n"
        f"  Median:            {values.median():.2f}%\n"
        f"  Std:               {values.std(ddof=1):.2f}%\n"
        f"  Max:               {values.max():.1f}% ({max_row['label_name']})\n"
        f"  Min:               {values.min():.1f}% ({min_row['label_name']})\n\n"
        f"Problem classes:\n"
        f"  Severe (>50%):     {(values > 50).sum()} groups\n"
        f"  High (20-50%):     {((values > 20) & (values <= 50)).sum()} groups\n"
        f"  Moderate (5-20%):  {((values > 5) & (values <= 20)).sum()} groups\n"
        f"  Low (<=5%):        {(values <= 5).sum()} groups\n"
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
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(cnn_df: pd.DataFrame, ttn_df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    comparison = cnn_df.merge(
        ttn_df,
        on=["label_idx", "label_name"],
        suffixes=("_cnn", "_ttn"),
    )
    comparison["error_diff"] = comparison["error_rate_cnn"] - comparison["error_rate_ttn"]
    comparison["abs_error_diff"] = comparison["error_diff"].abs()
    comparison = comparison.sort_values("abs_error_diff", ascending=False).reset_index(drop=True)

    fig = plt.figure(figsize=(18, 14))
    top = comparison.head(15)
    x = np.arange(len(top))
    width = 0.38

    ax1 = plt.subplot(2, 3, 1)
    ax1.bar(x - width / 2, top["error_rate_cnn"] * 100, width, label="CNN", color="#FF6B6B")
    ax1.bar(x + width / 2, top["error_rate_ttn"] * 100, width, label="TTN 10.2", color="#4ECDC4")
    ax1.set_xticks(x)
    ax1.set_xticklabels(top["label_name"], rotation=45, ha="right")
    ax1.set_ylabel("Error rate (%)")
    ax1.set_title("Top 15 Absolute Error-Rate Differences")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    ax2 = plt.subplot(2, 3, 2)
    ax2.scatter(
        comparison["error_rate_cnn"] * 100,
        comparison["error_rate_ttn"] * 100,
        s=np.clip(comparison["positive_samples_cnn"] / 30, 20, 400),
        alpha=0.65,
        edgecolors="black",
    )
    ax2.plot([0, 100], [0, 100], "k--", linewidth=1)
    ax2.set_xlabel("CNN error rate (%)")
    ax2.set_ylabel("TTN 10.2 error rate (%)")
    ax2.set_title("CNN vs TTN Error Rates")
    ax2.grid(alpha=0.3)
    for _, row in comparison.iterrows():
        if row["error_rate_cnn"] > 0.7 or row["error_rate_ttn"] > 0.7:
            ax2.annotate(row["label_name"], (row["error_rate_cnn"] * 100, row["error_rate_ttn"] * 100), fontsize=8)

    ax3 = plt.subplot(2, 3, 3)
    by_diff = comparison.sort_values("error_diff", ascending=True)
    colors = ["#4ECDC4" if x < 0 else "#FF6B6B" for x in by_diff["error_diff"]]
    ax3.barh(range(len(by_diff)), by_diff["error_diff"] * 100, color=colors)
    ax3.set_yticks(range(len(by_diff)))
    ax3.set_yticklabels(by_diff["label_name"], fontsize=7)
    ax3.axvline(0, color="black", linewidth=1)
    ax3.set_xlabel("Error difference CNN - TTN (%)")
    ax3.set_title("Positive = TTN lower miss rate")
    ax3.grid(axis="x", alpha=0.3)

    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(comparison["error_rate_cnn"] * 100, bins=18, color="#FF6B6B", alpha=0.75, edgecolor="black")
    ax4.set_title("CNN Error Distribution")
    ax4.set_xlabel("Error rate (%)")

    ax5 = plt.subplot(2, 3, 5)
    ax5.hist(comparison["error_rate_ttn"] * 100, bins=18, color="#4ECDC4", alpha=0.75, edgecolor="black")
    ax5.set_title("TTN 10.2 Error Distribution")
    ax5.set_xlabel("Error rate (%)")

    ax6 = plt.subplot(2, 3, 6)
    ax6.axis("off")
    cnn_better = int((comparison["error_rate_cnn"] < comparison["error_rate_ttn"]).sum())
    ttn_better = int((comparison["error_rate_ttn"] < comparison["error_rate_cnn"]).sum())
    equal = int((comparison["error_rate_ttn"] == comparison["error_rate_cnn"]).sum())
    overlap = int(((comparison["error_rate_cnn"] > 0.2) & (comparison["error_rate_ttn"] > 0.2)).sum())
    summary = (
        "CNN VS TTN 10.2 SUMMARY\n\n"
        f"CNN mean error:       {comparison['error_rate_cnn'].mean() * 100:.2f}%\n"
        f"CNN median error:     {comparison['error_rate_cnn'].median() * 100:.2f}%\n"
        f"CNN max error:        {comparison['error_rate_cnn'].max() * 100:.1f}%\n\n"
        f"TTN mean error:       {comparison['error_rate_ttn'].mean() * 100:.2f}%\n"
        f"TTN median error:     {comparison['error_rate_ttn'].median() * 100:.2f}%\n"
        f"TTN max error:        {comparison['error_rate_ttn'].max() * 100:.1f}%\n\n"
        f"Classes where CNN lower error: {cnn_better}\n"
        f"Classes where TTN lower error: {ttn_better}\n"
        f"Equal error:                  {equal}\n"
        f"Both >20% error:              {overlap}\n"
    )
    ax6.text(
        0.02,
        0.98,
        summary,
        transform=ax6.transAxes,
        va="top",
        family="monospace",
        fontsize=10.5,
        bbox={"boxstyle": "round", "facecolor": "#EEF4FF", "alpha": 0.95},
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return comparison


def plot_top_problems(cnn_df: pd.DataFrame, ttn_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    for ax, df, title, color in [
        (axes[0], cnn_df.head(15), "CNN Baseline - Top 15 Problem Groups", "#FF6B6B"),
        (axes[1], ttn_df.head(15), "TTN 10.2 - Top 15 Problem Groups", "#4ECDC4"),
    ]:
        values = df["error_rate"] * 100.0
        ax.barh(range(len(df)), values, color=color, edgecolor="black")
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df["label_name"], fontsize=10)
        ax.set_xlabel("Error rate: FN / positives (%)")
        ax.set_title(title)
        ax.set_xlim(0, max(105, values.max() + 5))
        ax.grid(axis="x", alpha=0.3)
        for i, value in enumerate(values):
            ax.text(value + 1, i, f"{value:.1f}%", va="center", fontsize=9)
        ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_ttn_inference:
        cnn_df = pd.read_csv(args.output_dir / "CNN_error_analysis.csv")
        ttn_df = pd.read_csv(args.output_dir / "TTN_error_analysis.csv")
        cnn_metrics = json.loads((args.output_dir / "CNN_error_analysis_metrics.json").read_text())
        ttn_metrics = json.loads((args.output_dir / "TTN_error_analysis_metrics.json").read_text())
    else:
        print("Computing CNN metrics from saved baseline predictions...")
        cnn_pred, cnn_true = load_cnn_predictions(args.cnn_pickle)
        cnn_df = per_label_metrics(cnn_true, cnn_pred)
        cnn_metrics = global_metrics(cnn_true, cnn_pred)

        print("Computing TTN 10.2 metrics from checkpoint and real test split...")
        device = resolve_device(args.device)
        ttn_true, ttn_pred = compute_ttn_predictions(
            run_dir=args.ttn_run_dir,
            data_dir=args.data_dir,
            cache_path=args.cache_path,
            batch_size=args.batch_size,
            device=device,
        )
        ttn_df = per_label_metrics(ttn_true, ttn_pred)
        ttn_metrics = global_metrics(ttn_true, ttn_pred)

        cnn_df.to_csv(args.output_dir / "CNN_error_analysis.csv", index=False)
        ttn_df.to_csv(args.output_dir / "TTN_error_analysis.csv", index=False)
        (args.output_dir / "CNN_error_analysis_metrics.json").write_text(json.dumps(cnn_metrics, indent=2) + "\n")
        (args.output_dir / "TTN_error_analysis_metrics.json").write_text(json.dumps(ttn_metrics, indent=2) + "\n")

    print("Writing corrected PNGs...")
    plot_individual(
        cnn_df,
        cnn_metrics,
        "CNN Baseline",
        "#FF6B6B",
        args.output_dir / "CNN_error_analysis_detailed.png",
    )
    plot_individual(
        ttn_df,
        ttn_metrics,
        "TTN 10.2",
        "#4ECDC4",
        args.output_dir / "TTN_error_analysis_detailed.png",
    )
    comparison = plot_comparison(
        cnn_df,
        ttn_df,
        args.output_dir / "error_analysis_comparison_CNN_vs_TTN.png",
    )
    comparison.to_csv(args.output_dir / "CNN_vs_TTN_error_comparison.csv", index=False)
    plot_top_problems(cnn_df, ttn_df, args.output_dir / "top_problems_CNN_vs_TTN.png")

    print("\nDone.")
    print(f"Output dir: {args.output_dir}")
    print(f"CNN f1_micro={cnn_metrics['f1_micro']:.6f}, f1_macro={cnn_metrics['f1_macro']:.6f}")
    print(f"TTN f1_micro={ttn_metrics['f1_micro']:.6f}, f1_macro={ttn_metrics['f1_macro']:.6f}")


if __name__ == "__main__":
    main()
