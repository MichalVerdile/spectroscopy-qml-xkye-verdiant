import argparse
import csv
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def load_results(model_dir):
    results = {}

    pickle_files = list(Path(model_dir).glob("*.pickle"))
    print(f"  Found {len(pickle_files)} pickle file(s) in {model_dir}")

    for pickle_file in pickle_files:
        try:
            with open(pickle_file, "rb") as f:
                data = pickle.load(f)
                print(f"  Loaded {pickle_file.name}: type={type(data)}")

                if isinstance(data, dict):
                    print(f"    Keys: {list(data.keys())}")

                    if "pred" in data and "tgt" in data:
                        results["predictions"] = data["pred"]
                        results["targets"] = data["tgt"]
                        print("    ✓ Found pred and tgt (original format)")

                    elif "test_predictions" in data and "test_targets" in data:
                        results["predictions"] = data["test_predictions"]
                        results["targets"] = data["test_targets"]
                        print("    ✓ Found test_predictions and test_targets (k-fold format)")

                    else:
                        print(
                            "    ✗ Missing required keys "
                            "(need 'pred'/'tgt' or 'test_predictions'/'test_targets')"
                        )
                else:
                    print("    ✗ Data is not a dictionary")

        except Exception as e:
            print(f"Warning: Could not load {pickle_file.name}: {e}")

    return results


def binarize_predictions(predictions, threshold=0.5):
    """Convert model outputs to binary multi-label predictions."""
    pred = np.array(predictions)

    if np.issubdtype(pred.dtype, np.floating):
        if pred.min() >= 0 and pred.max() <= 1:
            pred = (pred >= threshold).astype(int)

    return pred.astype(int)


def compute_multilabel_metrics(pred, tgt):
    """Calculate metrics for binary multi-label classification arrays."""
    return {
        "f1_micro": f1_score(tgt, pred, average="micro", zero_division=0),
        "f1_macro": f1_score(tgt, pred, average="macro", zero_division=0),
        "precision_micro": precision_score(tgt, pred, average="micro", zero_division=0),
        "precision_macro": precision_score(tgt, pred, average="macro", zero_division=0),
        "recall_micro": recall_score(tgt, pred, average="micro", zero_division=0),
        "recall_macro": recall_score(tgt, pred, average="macro", zero_division=0),
        "subset_accuracy": accuracy_score(tgt, pred),
        "hamming_accuracy": float((tgt == pred).mean()),
    }


def calculate_metrics(predictions, targets, threshold=0.5):
    """Calculate metrics for multi-label classification."""
    pred = binarize_predictions(predictions, threshold=threshold)
    tgt = np.array(targets).astype(int)

    metrics = compute_multilabel_metrics(pred, tgt)

    return metrics, pred, tgt


def bootstrap_confidence_intervals(
    pred,
    tgt,
    n_bootstrap=2000,
    ci_level=0.95,
    random_seed=42,
):
    """Estimate percentile bootstrap confidence intervals over test samples.

    The bootstrap resamples complete rows from pred/tgt.

    This estimates uncertainty due to the finite evaluation set.
    It does not include retraining uncertainty, hyperparameter uncertainty,
    or threshold-selection uncertainty.
    """
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be > 0")

    if not 0.0 < ci_level < 1.0:
        raise ValueError("ci_level must be between 0 and 1")

    pred = np.asarray(pred).astype(int)
    tgt = np.asarray(tgt).astype(int)

    if pred.shape != tgt.shape:
        raise ValueError(f"Prediction and target shapes differ: {pred.shape} vs {tgt.shape}")

    n_samples = tgt.shape[0]
    if n_samples < 2:
        raise ValueError("Bootstrap confidence intervals require at least 2 samples")

    rng = np.random.default_rng(random_seed)

    point_metrics = compute_multilabel_metrics(pred, tgt)
    metric_names = list(point_metrics.keys())

    bootstrap_values = {
        metric_name: np.empty(n_bootstrap, dtype=float)
        for metric_name in metric_names
    }

    for bootstrap_idx in range(n_bootstrap):
        sample_indices = rng.integers(0, n_samples, size=n_samples)

        sampled_pred = pred[sample_indices]
        sampled_tgt = tgt[sample_indices]

        sampled_metrics = compute_multilabel_metrics(sampled_pred, sampled_tgt)

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

    return rows


def save_bootstrap_ci_artifacts(
    results_dict,
    output_dir,
    model_type,
    n_bootstrap,
    ci_level,
    random_seed,
):
    """Compute and save bootstrap confidence intervals for all evaluated models."""
    all_rows = []

    for model_name, data in results_dict.items():
        print(f"\nComputing bootstrap confidence intervals for {model_name}")

        model_seed = random_seed + len(all_rows)

        ci_rows = bootstrap_confidence_intervals(
            data["pred"],
            data["tgt"],
            n_bootstrap=n_bootstrap,
            ci_level=ci_level,
            random_seed=model_seed,
        )

        data["bootstrap_ci"] = ci_rows

        for row in ci_rows:
            all_rows.append(
                {
                    "model": model_name,
                    **row,
                }
            )

        print_bootstrap_summary(model_name, ci_rows)

    csv_path = output_dir / f"bootstrap_confidence_intervals_{model_type}.csv"
    json_path = output_dir / f"bootstrap_confidence_intervals_{model_type}.json"

    fieldnames = [
        "model",
        "metric",
        "point_estimate",
        "bootstrap_mean",
        "bootstrap_std",
        "ci_level",
        "ci_lower",
        "ci_upper",
        "n_bootstrap",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(all_rows, handle, indent=2)

    print(f"\nSaved bootstrap confidence intervals CSV to {csv_path}")
    print(f"Saved bootstrap confidence intervals JSON to {json_path}")

    return csv_path, json_path


def print_bootstrap_summary(model_name, ci_rows):
    print(f"\nBootstrap confidence intervals for {model_name}:")
    print(
        f"{'Metric':<20} "
        f"{'Point':>10} "
        f"{'CI Lower':>10} "
        f"{'CI Upper':>10} "
        f"{'Std':>10}"
    )
    print("-" * 65)

    for row in ci_rows:
        print(
            f"{row['metric']:<20} "
            f"{row['point_estimate']:>10.4f} "
            f"{row['ci_lower']:>10.4f} "
            f"{row['ci_upper']:>10.4f} "
            f"{row['bootstrap_std']:>10.4f}"
        )


def print_summary(model_name, metrics):
    print(f"\n{'=' * 80}")
    print(f"{model_name}")
    print(f"{'=' * 80}")
    print(f"  F1 Score (micro):    {metrics['f1_micro']:.4f}")
    print(f"  F1 Score (macro):    {metrics['f1_macro']:.4f}")
    print(f"  Precision (micro):   {metrics['precision_micro']:.4f}")
    print(f"  Precision (macro):   {metrics['precision_macro']:.4f}")
    print(f"  Recall (micro):      {metrics['recall_micro']:.4f}")
    print(f"  Recall (macro):      {metrics['recall_macro']:.4f}")
    print(f"  Subset Accuracy:     {metrics['subset_accuracy']:.4f}")
    print(f"  Hamming Accuracy:    {metrics['hamming_accuracy']:.4f}")


def plot_metrics_comparison(results_dict, output_dir, model_type):
    model_names = []
    f1_micros = []
    f1_macros = []
    subset_accuracies = []
    hamming_accuracies = []

    for model_name, data in results_dict.items():
        metrics = data["metrics"]
        model_names.append(model_name)
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
        axes[0, 0].text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    axes[0, 1].bar(x, f1_macros, width, color="coral", alpha=0.8)
    axes[0, 1].set_ylabel("Score")
    axes[0, 1].set_title("F1 Score (Macro)")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(model_names, rotation=45, ha="right")
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].grid(axis="y", alpha=0.3)

    for i, v in enumerate(f1_macros):
        axes[0, 1].text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    axes[1, 0].bar(x, subset_accuracies, width, color="lightgreen", alpha=0.8)
    axes[1, 0].set_ylabel("Score")
    axes[1, 0].set_title("Subset Accuracy (Exact Match)")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(model_names, rotation=45, ha="right")
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].grid(axis="y", alpha=0.3)

    for i, v in enumerate(subset_accuracies):
        axes[1, 0].text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    axes[1, 1].bar(x, hamming_accuracies, width, color="plum", alpha=0.8)
    axes[1, 1].set_ylabel("Score")
    axes[1, 1].set_title("Hamming Accuracy")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(model_names, rotation=45, ha="right")
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].grid(axis="y", alpha=0.3)

    for i, v in enumerate(hamming_accuracies):
        axes[1, 1].text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()

    output_plot = output_dir / f"metrics_comparison_{model_type}.png"
    plt.savefig(output_plot, dpi=300, bbox_inches="tight")
    print(f"Saved metrics comparison to {output_plot}")

    plt.close()


def save_per_class_f1(results_dict, output_dir, model_type):
    """Save per-class F1 scores for each model into a file with class names."""

    output_file = output_dir / f"per_class_f1_{model_type}.txt"

    fallback_class_names = [
        "Acid anhydride",
        "Acyl halide",
        "Alcohol",
        "Aldehyde",
        "Alkane",
        "Alkene",
        "Alkyne",
        "Amide",
        "Amine",
        "Arene",
        "Azo compound",
        "Carbamate",
        "Carboxylic acid",
        "Enamine",
        "Enol",
        "Ester",
        "Ether",
        "Haloalkane",
        "Hydrazine",
        "Hydrazone",
        "Imide",
        "Imine",
        "Isocyanate",
        "Isothiocyanate",
        "Ketone",
        "Nitrile",
        "Phenol",
        "Phosphine",
        "Sulfide",
        "Sulfonamide",
        "Sulfonate",
        "Sulfone",
        "Sulfonic acid",
        "Sulfoxide",
        "Thial",
        "Thioamide",
        "Thiol",
    ]

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"Per-Class F1 Scores - {model_type.upper()}\n")
        f.write("=" * 80 + "\n\n")

        for model_name, data in results_dict.items():
            pred = data["pred"]
            tgt = data["tgt"]

            class_names = data.get("class_names", fallback_class_names)
            f1_per_class = f1_score(tgt, pred, average=None, zero_division=0)

            f.write(f"Model: {model_name}\n")

            for i, score in enumerate(f1_per_class):
                name = class_names[i] if i < len(class_names) else f"class_{i}"
                f.write(f"  {name:<20} : {score:.4f}\n")

            f.write("\n")

    print(f"Saved per-class F1 scores to {output_file}")


def write_summary(
    results_dict,
    output_dir,
    model_type,
    bootstrap_ci_csv_path=None,
    bootstrap_ci_json_path=None,
):
    summary_file = output_dir / f"summary_{model_type}.txt"

    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"CNN Model Evaluation Summary - {model_type.upper()}\n")
        f.write("=" * 80 + "\n\n")

        for model_name, data in results_dict.items():
            metrics = data["metrics"]

            f.write(f"Model: {model_name}\n")
            f.write(f"  F1 Score (micro):    {metrics['f1_micro']:.4f}\n")
            f.write(f"  F1 Score (macro):    {metrics['f1_macro']:.4f}\n")
            f.write(f"  Precision (micro):   {metrics['precision_micro']:.4f}\n")
            f.write(f"  Precision (macro):   {metrics['precision_macro']:.4f}\n")
            f.write(f"  Recall (micro):      {metrics['recall_micro']:.4f}\n")
            f.write(f"  Recall (macro):      {metrics['recall_macro']:.4f}\n")
            f.write(f"  Subset Accuracy:     {metrics['subset_accuracy']:.4f}\n")
            f.write(f"  Hamming Accuracy:    {metrics['hamming_accuracy']:.4f}\n")

            if "bootstrap_ci" in data:
                f.write("\n")
                f.write("  Bootstrap Confidence Intervals:\n")
                f.write("  Method: percentile bootstrap over evaluation samples\n")

                for row in data["bootstrap_ci"]:
                    f.write(
                        f"    {row['metric']}: "
                        f"point={row['point_estimate']:.4f}, "
                        f"CI=[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}], "
                        f"bootstrap_std={row['bootstrap_std']:.4f}, "
                        f"n_bootstrap={row['n_bootstrap']}\n"
                    )

            f.write("\n")

        f.write("Generated Artifacts:\n")
        f.write(f"  Metrics comparison plot: {output_dir / f'metrics_comparison_{model_type}.png'}\n")
        f.write(f"  Per-class F1 file: {output_dir / f'per_class_f1_{model_type}.txt'}\n")

        if bootstrap_ci_csv_path is not None:
            f.write(f"  Bootstrap CI CSV: {bootstrap_ci_csv_path}\n")
        if bootstrap_ci_json_path is not None:
            f.write(f"  Bootstrap CI JSON: {bootstrap_ci_json_path}\n")

    print(f"Saved summary to {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate CNN model results")

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
        "--model_type",
        type=str,
        choices=["k_fold", "original"],
        required=True,
        help="Type of model to evaluate: 'k_fold' or 'original'",
    )

    parser.add_argument(
        "--prediction_threshold",
        type=float,
        default=0.5,
        help="Threshold for converting probability predictions to binary labels",
    )

    parser.add_argument(
        "--bootstrap_iterations",
        type=int,
        default=2000,
        help="Number of bootstrap resamples. Use 0 to disable bootstrap confidence intervals.",
    )

    parser.add_argument(
        "--bootstrap_ci",
        type=float,
        default=0.95,
        help="Confidence level for percentile bootstrap intervals",
    )

    parser.add_argument(
        "--bootstrap_seed",
        type=int,
        default=42,
        help="Random seed for bootstrap resampling",
    )

    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"CNN Model Evaluation Summary - {args.model_type.upper()}")
    print("=" * 80)

    results_dict = {}

    for model_path in sorted(models_dir.glob("*")):
        if model_path.is_dir():
            model_name = model_path.name
            model_results_path = model_path / args.model_type

            if not model_results_path.exists():
                print(f"Warning: {model_results_path} does not exist, skipping {model_name}")
                continue

            results = load_results(model_results_path)

            if "predictions" in results and "targets" in results:
                raw_pred = np.array(results["predictions"])
                raw_tgt = np.array(results["targets"])

                print(
                    model_name,
                    raw_pred.shape,
                    raw_tgt.shape,
                    raw_pred.dtype,
                    "pred min/max:",
                    raw_pred.min(),
                    raw_pred.max(),
                    "unique pred (sample):",
                    np.unique(raw_pred)[:10],
                )

                metrics, pred, tgt = calculate_metrics(
                    results["predictions"],
                    results["targets"],
                    threshold=args.prediction_threshold,
                )

                results_dict[model_name] = {
                    "metrics": metrics,
                    "pred": pred,
                    "tgt": tgt,
                }

                print_summary(model_name, metrics)

            else:
                print(f"Warning: No valid predictions/targets found in {model_results_path}")

    if not results_dict:
        print("\nNo model results found in", models_dir)
        print("Make sure training has completed and results are saved.")
        return

    bootstrap_ci_csv_path = None
    bootstrap_ci_json_path = None

    if args.bootstrap_iterations > 0:
        print(f"\n{'=' * 80}")
        print("Computing Bootstrap Confidence Intervals")
        print("=" * 80)

        bootstrap_ci_csv_path, bootstrap_ci_json_path = save_bootstrap_ci_artifacts(
            results_dict=results_dict,
            output_dir=output_dir,
            model_type=args.model_type,
            n_bootstrap=args.bootstrap_iterations,
            ci_level=args.bootstrap_ci,
            random_seed=args.bootstrap_seed,
        )
    else:
        print("\nBootstrap confidence intervals disabled.")

    print(f"\n{'=' * 80}")
    print("Generating Visualizations")
    print("=" * 80 + "\n")

    plot_metrics_comparison(results_dict, output_dir, args.model_type)
    save_per_class_f1(results_dict, output_dir, args.model_type)

    write_summary(
        results_dict=results_dict,
        output_dir=output_dir,
        model_type=args.model_type,
        bootstrap_ci_csv_path=bootstrap_ci_csv_path,
        bootstrap_ci_json_path=bootstrap_ci_json_path,
    )

    print(f"\n{'=' * 80}")
    print("Evaluation Complete")
    print(f"Results saved to: {output_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()