"""Evaluation script for TTN C-NMR Experiment 10.2."""

from __future__ import annotations

from pathlib import Path
import json
import sys

import click
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = Path(__file__).resolve().parents[4]
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.spectroscopy_qml.msms_neg.tree_tensor_network.helpers.data_loader import (  # noqa: E402
    FUNCTIONAL_GROUPS,
    IRSpectraDataset,
    load_cnmr_data,
)
from src.spectroscopy_qml.msms_neg.tree_tensor_network.helpers.train_helpers import (  # noqa: E402
    count_available_data_files,
    resolve_device,
    resolve_used_file_count,
)
from src.spectroscopy_qml.msms_neg.tree_tensor_network.helpers.evaluation_helpers import (  # noqa: E402
    resolve_cache_path,
)
from src.spectroscopy_qml.msms_neg.tree_tensor_network.model import (  # noqa: E402
    TTNCnmrClassifier10_2,
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_run_dir(output_dir: Path) -> Path:
    if (output_dir / "run_config.json").exists():
        return output_dir

    child_run_dirs = [
        child for child in output_dir.iterdir()
        if child.is_dir() and (child / "run_config.json").exists()
    ]
    if len(child_run_dirs) == 1:
        return child_run_dirs[0]
    if len(child_run_dirs) == 0:
        raise FileNotFoundError(
            f"No run_config.json found in {output_dir} or its direct child directories."
        )
    raise ValueError(
        f"Multiple run directories found under {output_dir}. Please pass a specific --output_dir."
    )


def _resolve_specialist_indices(run_config: dict) -> list[int] | None:
    raw_value = run_config.get("specialist_indices")
    if raw_value in (None, "", False):
        return None
    return [int(index.strip()) for index in str(raw_value).split(",") if index.strip()]


def _build_model_from_run_config(run_config: dict) -> TTNCnmrClassifier10_2:
    specialist_indices = _resolve_specialist_indices(run_config)
    effective_num_labels = len(specialist_indices) if specialist_indices is not None else int(run_config["num_labels"])
    return TTNCnmrClassifier10_2(
        num_labels=effective_num_labels,
        chi=int(run_config["chi"]),
        input_dim=int(run_config["input_dim"]),
        segment_window_size=int(run_config["segment_window_size"]),
        segment_stride=int(run_config["segment_stride"]),
        segment_mode=str(run_config["segment_mode"]),
        segment_offset=run_config.get("segment_offset"),
        segment_state_normalize=bool(run_config["segment_state_normalize"]),
        merge_mode=str(run_config["merge_mode"]),
        merge_residual_weight=float(run_config["merge_residual_weight"]),
        merge_renormalize_output=bool(run_config["merge_renormalize_output"]),
        lorentz_gamma=float(run_config["lorentz_gamma"]),
        lorentz_kernel_half_width=int(run_config["lorentz_kernel_half_width"]),
        lorentz_norm_mode=str(run_config["lorentz_norm_mode"]),
    )


def _resolve_cache_path_from_run_config(run_config: dict) -> Path:
    args = type("Args", (), run_config.copy())()
    return resolve_cache_path(args)


def _resolve_split_path(run_config: dict, run_dir: Path, data_dir: Path) -> Path:
    configured_split_path = run_config.get("split_path")
    if configured_split_path:
        return Path(configured_split_path)

    total_data_files = count_available_data_files(data_dir)
    used_data_files = resolve_used_file_count(total_data_files, run_config.get("max_files"))
    split_suffix = "all" if run_config.get("max_files") is None else f"files{used_data_files}"
    return run_dir / f"data_split_seed{int(run_config['seed'])}_{split_suffix}.npz"


def _strip_compile_prefix_if_needed(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    if not all(key.startswith("_orig_mod.") for key in state_dict):
        return state_dict
    return {key.removeprefix("_orig_mod."): value for key, value in state_dict.items()}


def _create_eval_dataloader(
    X_split: np.ndarray,
    y_split: np.ndarray,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    dataset = IRSpectraDataset(X_split, y_split)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
    )


def _build_eval_loaders(
    X: np.ndarray,
    y: np.ndarray,
    split_path: Path,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader]:
    payload = np.load(split_path, allow_pickle=False)
    val_indices = payload["val_indices"].astype(np.int64, copy=False)
    test_indices = payload["test_indices"].astype(np.int64, copy=False)
    val_loader = _create_eval_dataloader(X[val_indices], y[val_indices], batch_size, num_workers, pin_memory)
    test_loader = _create_eval_dataloader(X[test_indices], y[test_indices], batch_size, num_workers, pin_memory)
    return val_loader, test_loader


def compute_multilabel_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute global multilabel classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_samples": f1_score(y_true, y_pred, average="samples", zero_division=0),
        "precision_micro": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_micro": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def compute_global_metrics_with_hamming(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute global metrics including multilabel Hamming accuracy."""
    metrics = compute_multilabel_metrics(y_true, y_pred)
    metrics["hamming_accuracy"] = float((y_true == y_pred).mean())
    return metrics


def bootstrap_confidence_intervals(
    results: dict,
    n_bootstrap: int = 2000,
    ci_level: float = 0.95,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Estimate percentile bootstrap confidence intervals over test samples.

    The bootstrap resamples complete samples/rows from y_true and y_pred.

    This estimates uncertainty caused by the finite test set.
    It does not include uncertainty from retraining, hyperparameter search,
    checkpoint selection, or threshold tuning.
    """
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be > 0")

    if not 0.0 < ci_level < 1.0:
        raise ValueError("ci_level must be between 0 and 1")

    y_true = results["y_true"]
    y_pred = results["y_pred"]

    n_samples = y_true.shape[0]
    if n_samples < 2:
        raise ValueError("Bootstrap confidence intervals require at least 2 samples")

    rng = np.random.default_rng(random_seed)

    point_metrics = compute_global_metrics_with_hamming(y_true, y_pred)
    metric_names = list(point_metrics.keys())

    bootstrap_values = {
        metric_name: np.empty(n_bootstrap, dtype=float)
        for metric_name in metric_names
    }

    for bootstrap_idx in range(n_bootstrap):
        sample_indices = rng.integers(0, n_samples, size=n_samples)

        sampled_y_true = y_true[sample_indices]
        sampled_y_pred = y_pred[sample_indices]

        sampled_metrics = compute_global_metrics_with_hamming(
            sampled_y_true,
            sampled_y_pred,
        )

        for metric_name in metric_names:
            bootstrap_values[metric_name][bootstrap_idx] = sampled_metrics[metric_name]

    alpha = 1.0 - ci_level
    lower_percentile = 100.0 * alpha / 2.0
    upper_percentile = 100.0 * (1.0 - alpha / 2.0)

    rows = []
    for metric_name in metric_names:
        values = bootstrap_values[metric_name]

        rows.append(
            {
                "metric": metric_name,
                "point_estimate": float(point_metrics[metric_name]),
                "bootstrap_mean": float(values.mean()),
                "bootstrap_std": float(values.std(ddof=1)),
                "ci_level": float(ci_level),
                "ci_lower": float(np.percentile(values, lower_percentile)),
                "ci_upper": float(np.percentile(values, upper_percentile)),
                "n_bootstrap": int(n_bootstrap),
            }
        )

    return pd.DataFrame(rows)


def save_bootstrap_ci_artifacts(
    results: dict,
    output_dir: Path,
    split_name: str,
    n_bootstrap: int,
    ci_level: float,
    random_seed: int,
) -> tuple[Path, Path, pd.DataFrame]:
    """Save bootstrap confidence intervals as CSV and JSON."""
    ci_df = bootstrap_confidence_intervals(
        results,
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
        random_seed=random_seed,
    )

    ci_csv_path = output_dir / f"{split_name}_bootstrap_confidence_intervals.csv"
    ci_json_path = output_dir / f"{split_name}_bootstrap_confidence_intervals.json"

    ci_df.to_csv(ci_csv_path, index=False)

    ci_json_path.write_text(
        json.dumps(ci_df.to_dict(orient="records"), indent=2) + "\n",
        encoding="utf-8",
    )

    return ci_csv_path, ci_json_path, ci_df


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    thresholds: np.ndarray | None = None,
) -> dict:
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for spectra, labels in dataloader:
            spectra = spectra.to(device)
            labels = labels.to(device)

            logits = model(spectra)
            probs_np = torch.sigmoid(logits).cpu().numpy()
            thresh = thresholds if thresholds is not None else 0.5
            preds = (probs_np >= thresh).astype(float)

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds)
            all_probs.append(probs_np)

    y_true = np.vstack(all_labels)
    y_pred = np.vstack(all_preds)
    y_prob = np.vstack(all_probs)

    metrics = compute_multilabel_metrics(y_true, y_pred)

    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    true_positives = np.logical_and(y_true == 1, y_pred == 1).sum(axis=0)
    false_positives = np.logical_and(y_true == 0, y_pred == 1).sum(axis=0)
    false_negatives = np.logical_and(y_true == 1, y_pred == 0).sum(axis=0)
    true_negatives = np.logical_and(y_true == 0, y_pred == 0).sum(axis=0)
    specificity_per_class = np.divide(
        true_negatives,
        true_negatives + false_positives,
        out=np.zeros_like(true_negatives, dtype=float),
        where=(true_negatives + false_positives) != 0,
    )
    false_positive_indices = [
        np.flatnonzero((y_true[:, index] == 0) & (y_pred[:, index] == 1))
        for index in range(y_true.shape[1])
    ]
    false_negative_indices = [
        np.flatnonzero((y_true[:, index] == 1) & (y_pred[:, index] == 0))
        for index in range(y_true.shape[1])
    ]

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "metrics": metrics,
        "f1_per_class": f1_per_class,
        "precision_per_class": precision_per_class,
        "recall_per_class": recall_per_class,
        "specificity_per_class": specificity_per_class,
        "true_positives_per_class": true_positives,
        "false_positives_per_class": false_positives,
        "false_negatives_per_class": false_negatives,
        "true_negatives_per_class": true_negatives,
        "false_positive_indices_per_class": false_positive_indices,
        "false_negative_indices_per_class": false_negative_indices,
    }


def build_error_analysis_dataframe(results: dict, functional_groups: list[str]) -> pd.DataFrame:
    rows = []
    y_true = results["y_true"]
    y_pred = results["y_pred"]

    for index, name in enumerate(functional_groups):
        yt = y_true[:, index].astype(np.int32)
        yp = y_pred[:, index].astype(np.int32)
        positives = int(yt.sum())
        total = int(len(yt))
        tp = int(results["true_positives_per_class"][index])
        fp = int(results["false_positives_per_class"][index])
        fn = int(results["false_negatives_per_class"][index])
        tn = int(results["true_negatives_per_class"][index])
        rows.append(
            {
                "label_idx": index,
                "label_name": name,
                "positive_samples": positives,
                "tp": tp,
                "fn": fn,
                "fp": fp,
                "tn": tn,
                "f1": float(results["f1_per_class"][index]),
                "precision": float(results["precision_per_class"][index]),
                "recall": float(results["recall_per_class"][index]),
                "specificity": float(results["specificity_per_class"][index]),
                "error_rate": fn / positives if positives > 0 else 0.0,
                "binary_error_rate": (fp + fn) / total if total > 0 else 0.0,
            }
        )

    return pd.DataFrame(rows).sort_values("error_rate", ascending=False).reset_index(drop=True)


def build_global_error_metrics(results: dict) -> dict[str, float]:
    return compute_global_metrics_with_hamming(
        results["y_true"],
        results["y_pred"],
    )


def plot_error_analysis_detail(
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
    ax1.set_title(f"{title} - All Functional Groups")
    ax1.set_xlim(0, max(105, float(values.max()) + 5))
    ax1.grid(axis="x", alpha=0.3)
    for i, value in enumerate(values):
        if value >= 1:
            ax1.text(value + 1, i, f"{value:.1f}%", va="center", fontsize=7)
    ax1.invert_yaxis()

    ax2 = plt.subplot(2, 2, 2)
    ax2.hist(values, bins=18, color=color, alpha=0.75, edgecolor="black")
    ax2.axvline(values.mean(), color="black", linestyle="--", label=f"Mean {values.mean():.1f}%")
    ax2.axvline(
        values.median(),
        color="black",
        linestyle=":",
        label=f"Median {values.median():.1f}%",
    )
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
    ax3.set_ylim(0, max(105, float(top_values.max()) + 8))
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


def plot_top_problem_groups(df: pd.DataFrame, title: str, color: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 9))
    top = df.head(15)
    values = top["error_rate"] * 100.0
    ax.barh(range(len(top)), values, color=color, edgecolor="black")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["label_name"], fontsize=10)
    ax.set_xlabel("Error rate: FN / positives (%)")
    ax.set_title(title)
    ax.set_xlim(0, max(105, float(values.max()) + 5))
    ax.grid(axis="x", alpha=0.3)
    for i, value in enumerate(values):
        ax.text(value + 1, i, f"{value:.1f}%", va="center", fontsize=9)
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_error_analysis_artifacts(
    results: dict,
    functional_groups: list[str],
    output_dir: Path,
    split_name: str,
    title: str,
    color: str,
) -> tuple[Path, Path, Path]:
    df = build_error_analysis_dataframe(results, functional_groups)
    metrics = build_global_error_metrics(results)

    csv_path = output_dir / f"{split_name}_error_analysis.csv"
    metrics_path = output_dir / f"{split_name}_error_analysis_metrics.json"
    detailed_plot_path = output_dir / f"{split_name}_error_analysis_detailed.png"
    top_plot_path = output_dir / f"{split_name}_top_problem_groups.png"

    df.to_csv(csv_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    plot_error_analysis_detail(df, metrics, title, color, detailed_plot_path)
    plot_top_problem_groups(df, f"{title} - Top 15 Problem Groups", color, top_plot_path)

    return csv_path, metrics_path, detailed_plot_path


def print_detailed_results(results: dict, functional_groups: list[str], output_file=None) -> None:
    def print_line(text: str) -> None:
        print(text)
        if output_file:
            output_file.write(text + "\n")

    metrics = results["metrics"]

    print_line("=" * 80)
    print_line("Overall Metrics")
    print_line("=" * 80)
    print_line(f"Accuracy:          {metrics['accuracy']:.4f}")
    print_line(f"F1 Micro:          {metrics['f1_micro']:.4f}")
    print_line(f"F1 Macro:          {metrics['f1_macro']:.4f}")
    print_line(f"F1 Weighted:       {metrics['f1_weighted']:.4f}")
    print_line(f"F1 Samples:        {metrics['f1_samples']:.4f}")
    print_line(f"Precision Micro:   {metrics['precision_micro']:.4f}")
    print_line(f"Precision Macro:   {metrics['precision_macro']:.4f}")
    print_line(f"Precision Weighted: {metrics['precision_weighted']:.4f}")
    print_line(f"Recall Micro:      {metrics['recall_micro']:.4f}")
    print_line(f"Recall Macro:      {metrics['recall_macro']:.4f}")
    print_line(f"Recall Weighted:   {metrics['recall_weighted']:.4f}")

    print_line("\n" + "=" * 80)
    print_line("Per-Class Metrics")
    print_line("=" * 80)
    print_line(
        f"{'Functional Group':<25} {'F1':>8} {'Prec':>8} {'Recall':>8} {'Spec':>8} {'Support':>8} {'TP':>6} {'FP':>6} {'FN':>6} {'TN':>6}"
    )
    print_line("-" * 120)

    for i, fg_name in enumerate(functional_groups):
        support = int(results["y_true"][:, i].sum())
        print_line(
            f"{fg_name:<25} "
            f"{results['f1_per_class'][i]:>8.4f} "
            f"{results['precision_per_class'][i]:>8.4f} "
            f"{results['recall_per_class'][i]:>8.4f} "
            f"{results['specificity_per_class'][i]:>8.4f} "
            f"{support:>8} "
            f"{results['true_positives_per_class'][i]:>6} "
            f"{results['false_positives_per_class'][i]:>6} "
            f"{results['false_negatives_per_class'][i]:>6} "
            f"{results['true_negatives_per_class'][i]:>6}"
        )

    total_errors_per_class = (
        results["false_positives_per_class"] + results["false_negatives_per_class"]
    )
    worst_indices = np.argsort(total_errors_per_class)[::-1][:5]

    print_line("\n" + "=" * 80)
    print_line("Error Analysis")
    print_line("=" * 80)
    for index in worst_indices:
        false_positive_indices = results["false_positive_indices_per_class"][index][:10].tolist()
        false_negative_indices = results["false_negative_indices_per_class"][index][:10].tolist()
        print_line(
            f"{functional_groups[index]}: "
            f"errors={int(total_errors_per_class[index])}, "
            f"FP={int(results['false_positives_per_class'][index])}, "
            f"FN={int(results['false_negatives_per_class'][index])}, "
            f"example FP indices={false_positive_indices}, "
            f"example FN indices={false_negative_indices}"
        )

    print_line("\n" + "=" * 80)
    print_line("Prediction Statistics")
    print_line("=" * 80)

    y_true = results["y_true"]
    y_pred = results["y_pred"]

    print_line(f"Total samples: {len(y_true)}")
    print_line(f"Average labels per sample (true): {y_true.sum(axis=1).mean():.2f}")
    print_line(f"Average labels per sample (pred): {y_pred.sum(axis=1).mean():.2f}")
    print_line(
        f"Samples with exact match: {(y_true == y_pred).all(axis=1).sum()} "
        f"({(y_true == y_pred).all(axis=1).mean()*100:.1f}%)"
    )


def _count_model_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


@click.command()
@click.option(
    "--model_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to model checkpoint (default: use ttn_cnmr_best.pt in the resolved run directory)",
)
@click.option(
    "--data_dir",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to data directory (default: use data_dir from run_config.json)",
)
@click.option(
    "--output_dir",
    type=click.Path(path_type=Path),
    default=Path("src/spectroscopy_qml/msms_neg/tree_tensor_network/results_600"),
    help="Run directory or parent results directory for experiment 10.2",
)
@click.option(
    "--run_config_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to run_config.json (default: resolve from output_dir)",
)
@click.option(
    "--thresholds_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to selected_thresholds.json (default: resolve from run directory)",
)
@click.option(
    "--split_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to saved split artifact (default: resolve from run_config.json)",
)
@click.option(
    "--device",
    type=click.Choice(["auto", "cpu", "cuda", "mps"]),
    default="auto",
    show_default=True,
    help="Device to use for evaluation",
)
@click.option(
    "--bootstrap-iterations",
    type=int,
    default=2000,
    show_default=True,
    help="Number of bootstrap resamples for test-set confidence intervals. Use 0 to disable.",
)
@click.option(
    "--bootstrap-ci",
    type=float,
    default=0.95,
    show_default=True,
    help="Confidence level for percentile bootstrap intervals.",
)
@click.option(
    "--bootstrap-seed",
    type=int,
    default=None,
    help="Random seed for bootstrap resampling. Defaults to seed from run_config.json.",
)
def main(
    model_path: Path | None,
    data_dir: Path | None,
    output_dir: Path,
    run_config_path: Path | None,
    thresholds_path: Path | None,
    split_path: Path | None,
    device: str,
    bootstrap_iterations: int,
    bootstrap_ci: float,
    bootstrap_seed: int | None,
) -> None:
    print("=" * 80)
    print("TTN C-NMR Experiment 10.2 Evaluation")
    print("=" * 80)

    run_dir = run_config_path.parent if run_config_path is not None else _resolve_run_dir(output_dir)
    run_config_path = run_config_path or (run_dir / "run_config.json")
    thresholds_path = thresholds_path or (run_dir / "selected_thresholds.json")
    model_path = model_path or (run_dir / "ttn_cnmr_best.pt")
    results_dir = run_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    run_config = _load_json(run_config_path)
    data_dir = data_dir or Path(run_config["data_dir"])
    if not data_dir.exists():
        project_root = Path(__file__).parents[6]
        data_dir = project_root / "data" / "raw"
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    eval_device = resolve_device(device)
    print(f"Using device: {eval_device}")

    specialist_indices = _resolve_specialist_indices(run_config)
    all_functional_groups = list(FUNCTIONAL_GROUPS.keys())
    functional_groups = (
        [all_functional_groups[index] for index in specialist_indices]
        if specialist_indices is not None
        else all_functional_groups
    )

    print(f"\nLoading model from: {model_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    checkpoint = torch.load(model_path, map_location=eval_device)

    model = _build_model_from_run_config(run_config)
    state_dict = checkpoint if isinstance(checkpoint, dict) else checkpoint.state_dict()
    state_dict = _strip_compile_prefix_if_needed(state_dict)
    model.load_state_dict(state_dict)
    model = model.to(eval_device)

    print(f"Model loaded with {_count_model_parameters(model):,} parameters")
    print(f"Run config: {run_config_path}")

    threshold_payload = _load_json(thresholds_path)
    thresholds = np.asarray(
        [threshold_payload["thresholds"][name] for name in functional_groups],
        dtype=np.float32,
    )
    print(f"Thresholds: mean={thresholds.mean():.3f}, std={thresholds.std():.3f}")

    print(f"\nLoading data from: {data_dir}")
    X, y = load_cnmr_data(
        data_dir=data_dir,
        target_length=int(run_config["input_dim"]),
        max_files=run_config.get("max_files"),
        apply_snv=bool(run_config.get("apply_snv", True)),
        cache_path=_resolve_cache_path_from_run_config(run_config),
        overwrite_cache=False,
    )
    if specialist_indices is not None:
        y = y[:, specialist_indices]

    split_path = split_path or _resolve_split_path(run_config, run_dir, data_dir)
    if not split_path.exists():
        raise FileNotFoundError(f"Split artifact not found: {split_path}")

    val_loader, test_loader = _build_eval_loaders(
        X,
        y,
        split_path=split_path,
        batch_size=int(run_config["batch_size"]),
        num_workers=int(run_config["num_workers"]),
        pin_memory=eval_device.type == "cuda",
    )
    print(f"Using fixed split artifact: {split_path}")

    print("\n" + "=" * 80)
    print("Evaluating on Test Set")
    print("=" * 80)

    test_results = evaluate_model(model, test_loader, eval_device, thresholds=thresholds)

    eval_output_path = results_dir / "evaluation_results.txt"
    with eval_output_path.open("w", encoding="utf-8") as handle:
        print_detailed_results(test_results, functional_groups, handle)

    test_csv_path, test_metrics_path, test_plot_path = save_error_analysis_artifacts(
        test_results,
        functional_groups,
        results_dir,
        split_name="test",
        title="TTN Experiment 10.2 Test Set",
        color="#4ECDC4",
    )

    print(f"\nDetailed results saved to: {eval_output_path}")
    print(f"Test error-analysis CSV saved to: {test_csv_path}")
    print(f"Test error-analysis metrics saved to: {test_metrics_path}")
    print(f"Test error-analysis plot saved to: {test_plot_path}")

    test_ci_csv_path = None
    test_ci_json_path = None
    test_ci_df = None

    if bootstrap_iterations > 0:
        print("\n" + "=" * 80)
        print("Computing Bootstrap Confidence Intervals on Test Set")
        print("=" * 80)

        effective_bootstrap_seed = (
            bootstrap_seed
            if bootstrap_seed is not None
            else int(run_config.get("seed", 42))
        )

        test_ci_csv_path, test_ci_json_path, test_ci_df = save_bootstrap_ci_artifacts(
            test_results,
            results_dir,
            split_name="test",
            n_bootstrap=bootstrap_iterations,
            ci_level=bootstrap_ci,
            random_seed=effective_bootstrap_seed,
        )

        print(f"Bootstrap iterations: {bootstrap_iterations}")
        print(f"Confidence level: {bootstrap_ci:.3f}")
        print(f"Bootstrap seed: {effective_bootstrap_seed}")
        print(f"Test bootstrap CI CSV saved to: {test_ci_csv_path}")
        print(f"Test bootstrap CI JSON saved to: {test_ci_json_path}")

        print("\nBootstrap confidence intervals:")
        print(
            test_ci_df[
                ["metric", "point_estimate", "ci_lower", "ci_upper", "bootstrap_std"]
            ].to_string(index=False)
        )
    else:
        print("\nBootstrap confidence intervals disabled.")

    print("\n" + "=" * 80)
    print("Evaluating on Validation Set")
    print("=" * 80)

    val_results = evaluate_model(model, val_loader, eval_device, thresholds=thresholds)

    val_output_path = results_dir / "validation_results.txt"
    with val_output_path.open("w", encoding="utf-8") as handle:
        print_detailed_results(val_results, functional_groups, handle)

    val_csv_path, val_metrics_path, val_plot_path = save_error_analysis_artifacts(
        val_results,
        functional_groups,
        results_dir,
        split_name="validation",
        title="TTN Experiment 10.2 Validation Set",
        color="#FF6B6B",
    )

    print(f"\nValidation results saved to: {val_output_path}")
    print(f"Validation error-analysis CSV saved to: {val_csv_path}")
    print(f"Validation error-analysis metrics saved to: {val_metrics_path}")
    print(f"Validation error-analysis plot saved to: {val_plot_path}")

    summary_path = results_dir / "summary_evaluation.txt"
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("=" * 80 + "\n")
        handle.write("TTN C-NMR Experiment 10.2 - Evaluation Summary\n")
        handle.write("=" * 80 + "\n\n")

        handle.write("Model Information:\n")
        handle.write(f"  Model path: {model_path}\n")
        handle.write(f"  Run config: {run_config_path}\n")
        handle.write(f"  Split artifact: {split_path}\n")
        handle.write(f"  Parameters: {_count_model_parameters(model):,}\n")
        handle.write(f"  Thresholds: mean={thresholds.mean():.3f}, std={thresholds.std():.3f}\n\n")

        handle.write("Test Set Performance:\n")
        for metric, value in test_results["metrics"].items():
            handle.write(f"  {metric}: {value:.4f}\n")
        handle.write(f"  hamming_accuracy: {build_global_error_metrics(test_results)['hamming_accuracy']:.4f}\n")

        if test_ci_df is not None:
            handle.write("\nTest Set Bootstrap Confidence Intervals:\n")
            handle.write(
                f"  Method: percentile bootstrap over test samples, "
                f"n_bootstrap={bootstrap_iterations}, ci_level={bootstrap_ci:.3f}\n"
            )

            for _, row in test_ci_df.iterrows():
                handle.write(
                    f"  {row['metric']}: "
                    f"point={row['point_estimate']:.4f}, "
                    f"CI=[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}], "
                    f"bootstrap_std={row['bootstrap_std']:.4f}\n"
                )

        handle.write("\nValidation Set Performance:\n")
        for metric, value in val_results["metrics"].items():
            handle.write(f"  {metric}: {value:.4f}\n")
        handle.write(f"  hamming_accuracy: {build_global_error_metrics(val_results)['hamming_accuracy']:.4f}\n")

        handle.write("\nTop 5 Best Performing Functional Groups (by F1 score):\n")
        top_indices = np.argsort(test_results["f1_per_class"])[::-1][:5]
        for index in top_indices:
            fg_name = functional_groups[index]
            f1_value = test_results["f1_per_class"][index]
            support = int(test_results["y_true"][:, index].sum())
            handle.write(f"  {fg_name}: F1={f1_value:.4f} (support={support})\n")

        handle.write("\nBottom 5 Worst Performing Functional Groups (by F1 score):\n")
        bottom_indices = np.argsort(test_results["f1_per_class"])[:5]
        for index in bottom_indices:
            fg_name = functional_groups[index]
            f1_value = test_results["f1_per_class"][index]
            support = int(test_results["y_true"][:, index].sum())
            handle.write(f"  {fg_name}: F1={f1_value:.4f} (support={support})\n")

        total_errors_per_class = (
            test_results["false_positives_per_class"] + test_results["false_negatives_per_class"]
        )
        worst_error_indices = np.argsort(total_errors_per_class)[::-1][:5]

        handle.write("\nTop 5 Functional Groups by Total Errors:\n")
        for index in worst_error_indices:
            fg_name = functional_groups[index]
            fp_examples = test_results["false_positive_indices_per_class"][index][:10].tolist()
            fn_examples = test_results["false_negative_indices_per_class"][index][:10].tolist()
            handle.write(
                f"  {fg_name}: "
                f"errors={int(total_errors_per_class[index])}, "
                f"TP={int(test_results['true_positives_per_class'][index])}, "
                f"FP={int(test_results['false_positives_per_class'][index])}, "
                f"FN={int(test_results['false_negatives_per_class'][index])}, "
                f"TN={int(test_results['true_negatives_per_class'][index])}, "
                f"specificity={test_results['specificity_per_class'][index]:.4f}, "
                f"example_fp_indices={fp_examples}, "
                f"example_fn_indices={fn_examples}\n"
            )

        handle.write("\nGenerated Error Analysis Artifacts:\n")
        handle.write(f"  Test CSV: {test_csv_path}\n")
        handle.write(f"  Test metrics JSON: {test_metrics_path}\n")
        handle.write(f"  Test plot: {test_plot_path}\n")

        if test_ci_csv_path is not None:
            handle.write(f"  Test bootstrap CI CSV: {test_ci_csv_path}\n")
            handle.write(f"  Test bootstrap CI JSON: {test_ci_json_path}\n")

        handle.write(f"  Validation CSV: {val_csv_path}\n")
        handle.write(f"  Validation metrics JSON: {val_metrics_path}\n")
        handle.write(f"  Validation plot: {val_plot_path}\n")

    print(f"\nSummary saved to: {summary_path}")
    print("\n" + "=" * 80)
    print("Evaluation Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()