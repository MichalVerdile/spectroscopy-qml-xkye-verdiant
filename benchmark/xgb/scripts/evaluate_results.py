import argparse
import json
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def load_results(model_dir):
    """
    Loads result pickle files.

    Supports both old format:
        {"pred": ..., "tgt": ...}

    and new format:
        {
            "val_pred": ..., "val_tgt": ...,
            "test_pred": ..., "test_tgt": ...
        }
    """
    results = {}

    pickle_files = list(Path(model_dir).glob("*.pickle"))

    for pickle_file in pickle_files:
        try:
            with open(pickle_file, "rb") as f:
                data = pickle.load(f)

            if not isinstance(data, dict):
                continue

            # Old format
            if "pred" in data and "tgt" in data:
                results["test"] = {
                    "predictions": data["pred"],
                    "targets": data["tgt"],
                }

            # New validation format
            if "val_pred" in data and "val_tgt" in data:
                results["val"] = {
                    "predictions": data["val_pred"],
                    "targets": data["val_tgt"],
                }

            # New test format
            if "test_pred" in data and "test_tgt" in data:
                results["test"] = {
                    "predictions": data["test_pred"],
                    "targets": data["test_tgt"],
                }

        except Exception as e:
            print(f"Warning: Could not load {pickle_file.name}: {e}")

    return results


def prepare_predictions_and_targets(predictions, targets):
    """
    Convert predictions and targets to integer multi-label matrices.

    If predictions are probabilities in [0, 1], they are thresholded at 0.5.
    """
    pred = np.asarray(predictions)
    tgt = np.asarray(targets)

    if np.issubdtype(pred.dtype, np.floating):
        if pred.min() >= 0 and pred.max() <= 1:
            pred = (pred >= 0.5).astype(int)

    pred = pred.astype(int)
    tgt = tgt.astype(int)

    return pred, tgt


def compute_global_metrics_with_hamming(y_true, y_pred):
    """
    Compute global multi-label metrics.

    This mirrors the bootstrap metric structure from the MPS evaluation script
    and also keeps your original names such as subset_accuracy.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "subset_accuracy": accuracy_score(y_true, y_pred),
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_micro": precision_score(
            y_true, y_pred, average="micro", zero_division=0
        ),
        "precision_macro": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "precision_weighted": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "recall_micro": recall_score(
            y_true, y_pred, average="micro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "recall_weighted": recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "hamming_accuracy": float((y_true == y_pred).mean()),
    }

    # Only valid for multilabel indicator targets.
    if y_true.ndim == 2:
        metrics["f1_samples"] = f1_score(
            y_true, y_pred, average="samples", zero_division=0
        )

    return metrics


def calculate_metrics(predictions, targets):
    """Calculate metrics for multi-label classification."""
    pred, tgt = prepare_predictions_and_targets(predictions, targets)
    metrics = compute_global_metrics_with_hamming(tgt, pred)
    return metrics, pred, tgt


def bootstrap_confidence_intervals(
    results,
    n_bootstrap=2000,
    ci_level=0.95,
    random_seed=42,
):
    """
    Estimate percentile bootstrap confidence intervals over samples.

    The bootstrap resamples complete rows from y_true and y_pred.

    This estimates uncertainty caused by the finite evaluation set.
    It does not include uncertainty from retraining, hyperparameter search,
    checkpoint selection, or threshold tuning.
    """
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be > 0")

    if not 0.0 < ci_level < 1.0:
        raise ValueError("ci_level must be between 0 and 1")

    y_true = results["tgt"]
    y_pred = results["pred"]

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


def safe_filename(name):
    """Create a filesystem-safe name for result artifacts."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def save_bootstrap_ci_artifacts(
    result_name,
    results,
    output_dir,
    n_bootstrap,
    ci_level,
    random_seed,
):
    """Save bootstrap confidence intervals as CSV and JSON."""
    ci_df = bootstrap_confidence_intervals(
        results,
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
        random_seed=random_seed,
    )

    safe_name = safe_filename(result_name)

    ci_csv_path = output_dir / f"{safe_name}_bootstrap_confidence_intervals.csv"
    ci_json_path = output_dir / f"{safe_name}_bootstrap_confidence_intervals.json"

    ci_df.to_csv(ci_csv_path, index=False)

    ci_json_path.write_text(
        json.dumps(ci_df.to_dict(orient="records"), indent=2) + "\n",
        encoding="utf-8",
    )

    return ci_csv_path, ci_json_path, ci_df


def print_summary(model_name, split_name, metrics):
    print(f"\n{'=' * 80}")
    print(f"{model_name} [{split_name}]")
    print(f"{'=' * 80}")
    print(f"  F1 Score (micro):     {metrics['f1_micro']:.4f}")
    print(f"  F1 Score (macro):     {metrics['f1_macro']:.4f}")

    if "f1_weighted" in metrics:
        print(f"  F1 Score (weighted):  {metrics['f1_weighted']:.4f}")

    if "f1_samples" in metrics:
        print(f"  F1 Score (samples):   {metrics['f1_samples']:.4f}")

    print(f"  Precision (micro):    {metrics['precision_micro']:.4f}")
    print(f"  Recall (micro):       {metrics['recall_micro']:.4f}")
    print(f"  Precision (macro):    {metrics['precision_macro']:.4f}")
    print(f"  Recall (macro):       {metrics['recall_macro']:.4f}")
    print(f"  Subset Accuracy:      {metrics['subset_accuracy']:.4f}")
    print(f"  Hamming Accuracy:     {metrics['hamming_accuracy']:.4f}")


def print_bootstrap_summary(ci_df):
    """Print a compact bootstrap CI table."""
    columns = [
        "metric",
        "point_estimate",
        "ci_lower",
        "ci_upper",
        "bootstrap_std",
    ]

    print("\nBootstrap confidence intervals:")
    print(ci_df[columns].to_string(index=False))


def plot_metrics_comparison(results_dict, output_dir):
    model_names = []
    f1_micros = []
    f1_macros = []
    subset_accuracies = []
    hamming_accuracies = []

    for result_name, data in results_dict.items():
        metrics = data["metrics"]

        model_names.append(result_name)
        f1_micros.append(metrics["f1_micro"])
        f1_macros.append(metrics["f1_macro"])
        subset_accuracies.append(metrics["subset_accuracy"])
        hamming_accuracies.append(metrics["hamming_accuracy"])

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Multi-Label Classification Performance Comparison", fontsize=16)

    x = np.arange(len(model_names))
    width = 0.6

    axes[0, 0].bar(x, f1_micros, width, color="steelblue", alpha=0.8)
    axes[0, 0].set_ylabel("Score")
    axes[0, 0].set_title("F1 Score (Micro) - PRIMARY METRIC")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(model_names, rotation=45, ha="right")
    axes[0, 0].set_ylim([0, 1])
    axes[0, 0].grid(axis="y", alpha=0.3)

    for i, v in enumerate(f1_micros):
        axes[0, 0].text(
            i,
            v + 0.02,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    axes[0, 1].bar(x, f1_macros, width, color="coral", alpha=0.8)
    axes[0, 1].set_ylabel("Score")
    axes[0, 1].set_title("F1 Score (Macro)")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(model_names, rotation=45, ha="right")
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].grid(axis="y", alpha=0.3)

    for i, v in enumerate(f1_macros):
        axes[0, 1].text(
            i,
            v + 0.02,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    axes[1, 0].bar(x, subset_accuracies, width, color="lightgreen", alpha=0.8)
    axes[1, 0].set_ylabel("Score")
    axes[1, 0].set_title("Subset Accuracy (Exact Match)")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(model_names, rotation=45, ha="right")
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].grid(axis="y", alpha=0.3)

    for i, v in enumerate(subset_accuracies):
        axes[1, 0].text(
            i,
            v + 0.02,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    axes[1, 1].bar(x, hamming_accuracies, width, color="plum", alpha=0.8)
    axes[1, 1].set_ylabel("Score")
    axes[1, 1].set_title("Hamming Accuracy")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(model_names, rotation=45, ha="right")
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].grid(axis="y", alpha=0.3)

    for i, v in enumerate(hamming_accuracies):
        axes[1, 1].text(
            i,
            v + 0.02,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()

    output_path = output_dir / "metrics_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved metrics comparison to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate model results")
    parser.add_argument(
        "--models_dir",
        type=str,
        default="./benchmark/cnn/models",
        help="Directory containing model subdirectories",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./benchmark/cnn/results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        "--bootstrap_iterations",
        dest="bootstrap_iterations",
        type=int,
        default=2000,
        help="Number of bootstrap resamples. Use 0 to disable.",
    )
    parser.add_argument(
        "--bootstrap-ci",
        "--bootstrap_ci",
        dest="bootstrap_ci",
        type=float,
        default=0.95,
        help="Confidence level for percentile bootstrap intervals.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        "--bootstrap_seed",
        dest="bootstrap_seed",
        type=int,
        default=42,
        help="Random seed for bootstrap resampling.",
    )

    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("Model Evaluation Summary")
    print("=" * 80)

    results_dict = {}

    for model_path in sorted(models_dir.glob("*")):
        if not model_path.is_dir():
            continue

        model_name = model_path.name
        loaded_results = load_results(model_path)

        if not loaded_results:
            print(f"\nNo results found for model: {model_name}")
            continue

        for split_name, split_results in loaded_results.items():
            predictions = split_results["predictions"]
            targets = split_results["targets"]

            raw_pred = np.asarray(predictions)
            raw_tgt = np.asarray(targets)

            print(
                model_name,
                split_name,
                raw_pred.shape,
                raw_tgt.shape,
                raw_pred.dtype,
                "pred min/max:",
                raw_pred.min(),
                raw_pred.max(),
                "unique pred sample:",
                np.unique(raw_pred)[:10],
            )

            metrics, pred, tgt = calculate_metrics(predictions, targets)

            result_name = f"{model_name}_{split_name}"

            results_dict[result_name] = {
                "metrics": metrics,
                "pred": pred,
                "tgt": tgt,
                "bootstrap_ci": None,
                "bootstrap_ci_csv_path": None,
                "bootstrap_ci_json_path": None,
            }

            print_summary(model_name, split_name, metrics)

            if args.bootstrap_iterations > 0:
                print("\n" + "=" * 80)
                print(f"Computing Bootstrap Confidence Intervals: {result_name}")
                print("=" * 80)

                try:
                    ci_csv_path, ci_json_path, ci_df = save_bootstrap_ci_artifacts(
                        result_name=result_name,
                        results=results_dict[result_name],
                        output_dir=output_dir,
                        n_bootstrap=args.bootstrap_iterations,
                        ci_level=args.bootstrap_ci,
                        random_seed=args.bootstrap_seed,
                    )

                    results_dict[result_name]["bootstrap_ci"] = ci_df
                    results_dict[result_name]["bootstrap_ci_csv_path"] = ci_csv_path
                    results_dict[result_name]["bootstrap_ci_json_path"] = ci_json_path

                    print(f"Bootstrap iterations: {args.bootstrap_iterations}")
                    print(f"Confidence level: {args.bootstrap_ci:.3f}")
                    print(f"Bootstrap seed: {args.bootstrap_seed}")
                    print(f"Bootstrap CI CSV saved to: {ci_csv_path}")
                    print(f"Bootstrap CI JSON saved to: {ci_json_path}")

                    print_bootstrap_summary(ci_df)

                except ValueError as e:
                    print(f"Warning: Bootstrap skipped for {result_name}: {e}")

    if not results_dict:
        print("\nNo model results found in", models_dir)
        print("Make sure training has completed and results are saved.")
        return

    print(f"\n{'=' * 80}")
    print("Generating Visualizations")
    print("=" * 80 + "\n")

    plot_metrics_comparison(results_dict, output_dir)

    summary_file = output_dir / "summary.txt"

    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("Model Evaluation Summary\n")
        f.write("=" * 80 + "\n\n")

        for result_name, data in results_dict.items():
            metrics = data["metrics"]

            f.write(f"Result: {result_name}\n")
            f.write(f"  Accuracy:           {metrics['accuracy']:.4f}\n")
            f.write(f"  F1 Score (micro):   {metrics['f1_micro']:.4f}\n")
            f.write(f"  F1 Score (macro):   {metrics['f1_macro']:.4f}\n")

            if "f1_weighted" in metrics:
                f.write(f"  F1 Score (weighted): {metrics['f1_weighted']:.4f}\n")

            if "f1_samples" in metrics:
                f.write(f"  F1 Score (samples): {metrics['f1_samples']:.4f}\n")

            f.write(f"  Precision (micro):  {metrics['precision_micro']:.4f}\n")
            f.write(f"  Recall (micro):     {metrics['recall_micro']:.4f}\n")
            f.write(f"  Precision (macro):  {metrics['precision_macro']:.4f}\n")
            f.write(f"  Recall (macro):     {metrics['recall_macro']:.4f}\n")

            if "precision_weighted" in metrics:
                f.write(
                    f"  Precision (weighted): {metrics['precision_weighted']:.4f}\n"
                )

            if "recall_weighted" in metrics:
                f.write(f"  Recall (weighted): {metrics['recall_weighted']:.4f}\n")

            f.write(f"  Subset Accuracy:    {metrics['subset_accuracy']:.4f}\n")
            f.write(f"  Hamming Accuracy:   {metrics['hamming_accuracy']:.4f}\n")

            ci_df = data.get("bootstrap_ci")

            if ci_df is not None:
                f.write("\n")
                f.write("  Bootstrap Confidence Intervals:\n")
                f.write(
                    f"    Method: percentile bootstrap over evaluation samples, "
                    f"n_bootstrap={args.bootstrap_iterations}, "
                    f"ci_level={args.bootstrap_ci:.3f}, "
                    f"seed={args.bootstrap_seed}\n"
                )

                for _, row in ci_df.iterrows():
                    f.write(
                        f"    {row['metric']}: "
                        f"point={row['point_estimate']:.4f}, "
                        f"CI=[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}], "
                        f"bootstrap_std={row['bootstrap_std']:.4f}\n"
                    )

                f.write(
                    f"    Bootstrap CI CSV: {data['bootstrap_ci_csv_path']}\n"
                )
                f.write(
                    f"    Bootstrap CI JSON: {data['bootstrap_ci_json_path']}\n"
                )

            f.write("\n")

    print(f"Saved summary to {summary_file}")

    print(f"\n{'=' * 80}")
    print("Evaluation Complete")
    print(f"Results saved to: {output_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()