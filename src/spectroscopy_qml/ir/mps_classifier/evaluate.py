"""
Evaluation script for MPS Functional Group Classifier.

Includes:
- evaluation on reconstructed test split
- evaluation on checkpoint validation fold
- per-class error analysis
- bootstrap confidence intervals over the test set
"""

from pathlib import Path
from dataclasses import asdict, is_dataclass
import json

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
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader

from spectroscopy_qml.ir.mps_classifier.config import (
    DATA_CONFIG,
    MODEL_CONFIG,
    PATH_CONFIG,
    TRAINING_CONFIG,
)
from spectroscopy_qml.ir.mps_classifier.data_loader import (
    FUNCTIONAL_GROUPS,
    IRSpectraDataset,
    load_ir_data,
)
from spectroscopy_qml.ir.mps_classifier.model import MPSFunctionalGroupClassifier


def _get_model_config_kwargs(model_config) -> dict[str, object]:
    """Normalize saved model config into classifier constructor kwargs."""
    if is_dataclass(model_config):
        return asdict(model_config)
    if isinstance(model_config, dict):
        return model_config.copy()
    raise TypeError(f"Unsupported model config type in checkpoint: {type(model_config)!r}")


def _create_eval_dataloader(X_split: np.ndarray, y_split: np.ndarray) -> DataLoader:
    """Create a deterministic dataloader matching the training-time evaluation settings."""
    dataset = IRSpectraDataset(X_split, y_split)
    return DataLoader(
        dataset,
        batch_size=TRAINING_CONFIG.batch_size,
        shuffle=False,
        num_workers=TRAINING_CONFIG.num_workers,
        pin_memory=TRAINING_CONFIG.pin_memory,
        persistent_workers=True if TRAINING_CONFIG.num_workers > 0 else False,
    )


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


def _reconstruct_checkpoint_splits(
    X: np.ndarray,
    y: np.ndarray,
    checkpoint: dict,
) -> tuple[DataLoader | None, DataLoader]:
    """Rebuild the original test split and selected validation fold used during training.

    This works for existing checkpoints as long as dataset loading order and config values
    are unchanged relative to the training run.
    """
    dataset_indices = np.arange(len(X))
    trainval_indices, test_indices = train_test_split(
        dataset_indices,
        test_size=TRAINING_CONFIG.test_ratio,
        random_state=TRAINING_CONFIG.random_seed,
        shuffle=True,
    )

    test_loader = _create_eval_dataloader(X[test_indices], y[test_indices])

    best_fold = int(checkpoint.get("best_fold", 1))
    num_folds = int(checkpoint.get("cv_num_folds", TRAINING_CONFIG.num_folds))
    if num_folds < 1:
        raise ValueError(f"Checkpoint contains invalid cv_num_folds={num_folds}")

    relative_trainval_indices = np.arange(len(trainval_indices))
    if num_folds == 1:
        val_fraction = TRAINING_CONFIG.val_ratio / (
            TRAINING_CONFIG.train_ratio + TRAINING_CONFIG.val_ratio
        )
        _, val_relative_indices = train_test_split(
            relative_trainval_indices,
            test_size=val_fraction,
            random_state=TRAINING_CONFIG.random_seed,
            shuffle=True,
        )
    else:
        if best_fold < 1 or best_fold > num_folds:
            raise ValueError(
                f"Checkpoint best_fold={best_fold} is outside the expected range 1..{num_folds}"
            )
        fold_splits = list(
            KFold(
                n_splits=num_folds,
                shuffle=True,
                random_state=TRAINING_CONFIG.random_seed,
            ).split(relative_trainval_indices)
        )
        _, val_relative_indices = fold_splits[best_fold - 1]

    val_indices = trainval_indices[val_relative_indices]
    val_loader = _create_eval_dataloader(X[val_indices], y[val_indices])
    return val_loader, test_loader


def evaluate_model(
    model: nn.Module,
    dataloader,
    device: torch.device,
    thresholds: np.ndarray | None = None,
) -> dict:
    """
    Evaluate model on a dataset.

    Args:
        model: Trained model
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
        thresholds: Per-class decision thresholds. Defaults to 0.5.

    Returns:
        Dictionary containing predictions, labels, probabilities, and metrics.
    """
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
    """Build a per-label error-analysis table matching the standalone analysis script."""
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
    """Return overall metrics in the same structure used by the external analysis script."""
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
    """Create the detailed 2x2 error-analysis figure used in the standalone script."""
    values = df["error_rate"] * 100.0
    fig = plt.figure(figsize=(16, 12))

    ax1 = plt.subplot(2, 2, 1)
    colors = ["#B00020" if x > 50 else color if x > 20 else "#BFE8E4" for x in values]
    ax1.barh(range(len(df)), values, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_yticks(range(len(df)))
    ax1.set_yticklabels(df["label_name"], fontsize=8)
    ax1.set_xlabel("Missed-positive error rate: FN / positives (%)")
    ax1.set_title(f"{title} - All 37 Functional Groups")
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
    """Create the top-problem bar chart companion figure."""
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
) -> tuple[Path, Path, Path, Path]:
    """Persist CSV, JSON, and PNG error-analysis artifacts for one evaluation split."""
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

    return csv_path, metrics_path, detailed_plot_path, top_plot_path


def print_detailed_results(results: dict, functional_groups: list, output_file=None):
    """Print detailed evaluation results."""

    def print_line(text):
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
        f"{'Functional Group':<25} {'F1':>8} {'Prec':>8} {'Recall':>8} "
        f"{'Spec':>8} {'Support':>8} {'TP':>6} {'FP':>6} {'FN':>6} {'TN':>6}"
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
        f"({(y_true == y_pred).all(axis=1).mean() * 100:.1f}%)"
    )


@click.command()
@click.option(
    "--model_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to model checkpoint. Default: use best_model_path from config.",
)
@click.option(
    "--data_dir",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to data directory. Default: use data_dir from config.",
)
@click.option(
    "--output_dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory for results. Default: use results_dir from config.",
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
    help="Random seed for bootstrap resampling. Defaults to TRAINING_CONFIG.random_seed.",
)
def main(
    model_path,
    data_dir,
    output_dir,
    bootstrap_iterations,
    bootstrap_ci,
    bootstrap_seed,
):
    """Evaluate trained MPS model."""

    print("=" * 80)
    print("MPS Functional Group Classifier Evaluation")
    print("=" * 80)

    model_path = model_path or Path(PATH_CONFIG.best_model_path)
    data_dir = data_dir or Path(PATH_CONFIG.data_dir)

    if not data_dir.exists():
        project_root = Path(__file__).parents[4]
        data_dir = project_root / "data" / "raw"

    output_dir = output_dir or Path(PATH_CONFIG.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: CUDA ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    print(f"\nLoading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model_config = checkpoint.get("config", MODEL_CONFIG)
    model = MPSFunctionalGroupClassifier(**_get_model_config_kwargs(model_config))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    print(f"Model loaded (trained for {checkpoint['epoch']} epochs)")
    print(f"Validation loss: {checkpoint['val_loss']:.4f}")

    thresholds = checkpoint.get("thresholds", None)
    if thresholds is None:
        thresholds = np.full(model.num_classes, 0.5, dtype=float)
    elif torch.is_tensor(thresholds):
        thresholds = thresholds.detach().cpu().numpy().astype(float)
    else:
        thresholds = np.asarray(thresholds, dtype=float)

    if thresholds.shape != (model.num_classes,):
        raise ValueError(
            f"Expected thresholds with shape ({model.num_classes},), "
            f"but got {thresholds.shape}"
        )

    print(f"Thresholds: mean={thresholds.mean():.3f}, std={thresholds.std():.3f}")

    print(f"\nLoading data from: {data_dir}")
    X, y = load_ir_data(
        data_dir,
        target_length=DATA_CONFIG.target_length,
        max_files=DATA_CONFIG.max_files,
        apply_savgol=DATA_CONFIG.apply_savgol,
        savgol_window_length=DATA_CONFIG.savgol_window_length,
        savgol_polyorder=DATA_CONFIG.savgol_polyorder,
        apply_snv=DATA_CONFIG.apply_snv,
    )

    val_loader, test_loader = _reconstruct_checkpoint_splits(X, y, checkpoint)

    print(
        "Reconstructed training-time splits from checkpoint: "
        f"best_fold={checkpoint.get('best_fold', 'unknown')}, "
        f"cv_num_folds={checkpoint.get('cv_num_folds', TRAINING_CONFIG.num_folds)}"
    )

    functional_groups = list(FUNCTIONAL_GROUPS.keys())

    print("\n" + "=" * 80)
    print("Evaluating on Test Set")
    print("=" * 80)

    test_results = evaluate_model(model, test_loader, device, thresholds=thresholds)

    eval_output_path = output_dir / "evaluation_results.txt"
    with open(eval_output_path, "w", encoding="utf-8") as f:
        print_detailed_results(test_results, functional_groups, f)

    (
        test_csv_path,
        test_metrics_path,
        test_plot_path,
        test_top_plot_path,
    ) = save_error_analysis_artifacts(
        test_results,
        functional_groups,
        output_dir,
        split_name="test",
        title="MPS Test Set",
        color="#4ECDC4",
    )

    print(f"\nDetailed results saved to: {eval_output_path}")
    print(f"Test error-analysis CSV saved to: {test_csv_path}")
    print(f"Test error-analysis metrics saved to: {test_metrics_path}")
    print(f"Test error-analysis plot saved to: {test_plot_path}")
    print(f"Test top-problem-groups plot saved to: {test_top_plot_path}")

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
            else TRAINING_CONFIG.random_seed
        )

        test_ci_csv_path, test_ci_json_path, test_ci_df = save_bootstrap_ci_artifacts(
            test_results,
            output_dir,
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
    print("Evaluating on Checkpoint Validation Fold")
    print("=" * 80)

    val_results = evaluate_model(model, val_loader, device, thresholds=thresholds)

    val_output_path = output_dir / "validation_results.txt"
    with open(val_output_path, "w", encoding="utf-8") as f:
        print_detailed_results(val_results, functional_groups, f)

    (
        val_csv_path,
        val_metrics_path,
        val_plot_path,
        val_top_plot_path,
    ) = save_error_analysis_artifacts(
        val_results,
        functional_groups,
        output_dir,
        split_name="validation",
        title="MPS Validation Set",
        color="#FF6B6B",
    )

    print(f"\nValidation results saved to: {val_output_path}")
    print(f"Validation error-analysis CSV saved to: {val_csv_path}")
    print(f"Validation error-analysis metrics saved to: {val_metrics_path}")
    print(f"Validation error-analysis plot saved to: {val_plot_path}")
    print(f"Validation top-problem-groups plot saved to: {val_top_plot_path}")

    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("MPS Functional Group Classifier - Evaluation Summary\n")
        f.write("=" * 80 + "\n\n")

        f.write("Model Information:\n")
        f.write(f"  Model path: {model_path}\n")
        f.write(f"  Trained epochs: {checkpoint['epoch']}\n")
        f.write(f"  Parameters: {model.get_num_parameters():,}\n")
        f.write(f"  Thresholds: mean={thresholds.mean():.3f}, std={thresholds.std():.3f}\n\n")

        f.write("Test Set Performance:\n")
        for metric, value in test_results["metrics"].items():
            f.write(f"  {metric}: {value:.4f}\n")
        f.write(
            f"  hamming_accuracy: "
            f"{build_global_error_metrics(test_results)['hamming_accuracy']:.4f}\n"
        )

        if test_ci_df is not None:
            f.write("\nTest Set Bootstrap Confidence Intervals:\n")
            f.write(
                f"  Method: percentile bootstrap over test samples, "
                f"n_bootstrap={bootstrap_iterations}, ci_level={bootstrap_ci:.3f}\n"
            )

            for _, row in test_ci_df.iterrows():
                f.write(
                    f"  {row['metric']}: "
                    f"point={row['point_estimate']:.4f}, "
                    f"CI=[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}], "
                    f"bootstrap_std={row['bootstrap_std']:.4f}\n"
                )

        f.write("\nValidation Set Performance:\n")
        for metric, value in val_results["metrics"].items():
            f.write(f"  {metric}: {value:.4f}\n")
        f.write(
            f"  hamming_accuracy: "
            f"{build_global_error_metrics(val_results)['hamming_accuracy']:.4f}\n"
        )

        f.write("\nTop 5 Best Performing Functional Groups on Test Set (by F1 score):\n")
        top_indices = np.argsort(test_results["f1_per_class"])[::-1][:5]
        for idx in top_indices:
            fg_name = functional_groups[idx]
            f1 = test_results["f1_per_class"][idx]
            support = int(test_results["y_true"][:, idx].sum())
            f.write(f"  {fg_name}: F1={f1:.4f} (support={support})\n")

        f.write("\nBottom 5 Worst Performing Functional Groups on Test Set (by F1 score):\n")
        bottom_indices = np.argsort(test_results["f1_per_class"])[:5]
        for idx in bottom_indices:
            fg_name = functional_groups[idx]
            f1 = test_results["f1_per_class"][idx]
            support = int(test_results["y_true"][:, idx].sum())
            f.write(f"  {fg_name}: F1={f1:.4f} (support={support})\n")

        total_errors_per_class = (
            test_results["false_positives_per_class"]
            + test_results["false_negatives_per_class"]
        )
        worst_error_indices = np.argsort(total_errors_per_class)[::-1][:5]

        f.write("\nTop 5 Functional Groups by Total Errors on Test Set:\n")
        for idx in worst_error_indices:
            fg_name = functional_groups[idx]
            fp_examples = test_results["false_positive_indices_per_class"][idx][:10].tolist()
            fn_examples = test_results["false_negative_indices_per_class"][idx][:10].tolist()

            f.write(
                f"  {fg_name}: "
                f"errors={int(total_errors_per_class[idx])}, "
                f"TP={int(test_results['true_positives_per_class'][idx])}, "
                f"FP={int(test_results['false_positives_per_class'][idx])}, "
                f"FN={int(test_results['false_negatives_per_class'][idx])}, "
                f"TN={int(test_results['true_negatives_per_class'][idx])}, "
                f"specificity={test_results['specificity_per_class'][idx]:.4f}, "
                f"example_fp_indices={fp_examples}, "
                f"example_fn_indices={fn_examples}\n"
            )

        f.write("\nGenerated Error Analysis Artifacts:\n")
        f.write(f"  Test results TXT: {eval_output_path}\n")
        f.write(f"  Test CSV: {test_csv_path}\n")
        f.write(f"  Test metrics JSON: {test_metrics_path}\n")
        f.write(f"  Test detailed plot: {test_plot_path}\n")
        f.write(f"  Test top-problem-groups plot: {test_top_plot_path}\n")

        if test_ci_csv_path is not None:
            f.write(f"  Test bootstrap CI CSV: {test_ci_csv_path}\n")
            f.write(f"  Test bootstrap CI JSON: {test_ci_json_path}\n")

        f.write(f"  Validation results TXT: {val_output_path}\n")
        f.write(f"  Validation CSV: {val_csv_path}\n")
        f.write(f"  Validation metrics JSON: {val_metrics_path}\n")
        f.write(f"  Validation detailed plot: {val_plot_path}\n")
        f.write(f"  Validation top-problem-groups plot: {val_top_plot_path}\n")

    print(f"\nSummary saved to: {summary_path}")
    print("\n" + "=" * 80)
    print("Evaluation Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()