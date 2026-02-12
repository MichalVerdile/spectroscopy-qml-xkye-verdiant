import argparse
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

    for pickle_file in pickle_files:
        try:
            with open(pickle_file, "rb") as f:
                data = pickle.load(f)

                if isinstance(data, dict) and "pred" in data and "tgt" in data:
                    results["predictions"] = data["pred"]
                    results["targets"] = data["tgt"]
        except Exception as e:
            print(f"Warning: Could not load {pickle_file.name}: {e}")

    return results


def calculate_metrics(predictions, targets):
    """Calculate metrics for multi-label classification."""
    metrics = {}

    pred = np.array(predictions)
    tgt = np.array(targets)

    if pred.dtype in [np.float64, np.float32]:
        if pred.min() >= 0 and pred.max() <= 1:
            pred = (pred >= 0.5).astype(int)

    pred = pred.astype(int)
    tgt = tgt.astype(int)

    # Multi-label metrics
    metrics["f1_micro"] = f1_score(tgt, pred, average="micro")
    metrics["f1_macro"] = f1_score(tgt, pred, average="macro")
    metrics["subset_accuracy"] = accuracy_score(tgt, pred)
    metrics["hamming_accuracy"] = (tgt == pred).mean()

    metrics["precision_macro"] = precision_score(tgt, pred, average="macro")
    metrics["recall_macro"] = recall_score(tgt, pred, average="macro")

    return metrics, pred, tgt


def print_summary(model_name, metrics):
    print(f"\n{'=' * 80}")
    print(f"{model_name}")
    print(f"{'=' * 80}")
    print(f"  F1 Score (micro):    {metrics['f1_micro']:.4f}")
    print(f"  F1 Score (macro):    {metrics['f1_macro']:.4f}")
    print(f"  Subset Accuracy:     {metrics['subset_accuracy']:.4f}")
    print(f"  Hamming Accuracy:    {metrics['hamming_accuracy']:.4f}")
    print(f"  Precision (macro):   {metrics['precision_macro']:.4f}")
    print(f"  Recall (macro):      {metrics['recall_macro']:.4f}")


def plot_metrics_comparison(results_dict, output_dir):
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
    plt.savefig(output_dir / "metrics_comparison.png", dpi=300, bbox_inches="tight")
    print(f"Saved metrics comparison to {output_dir / 'metrics_comparison.png'}")
    plt.close()


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

    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("CNN Model Evaluation Summary")
    print("=" * 80)

    results_dict = {}
    for model_path in sorted(models_dir.glob("*")):
        if model_path.is_dir():
            model_name = model_path.name
            results = load_results(model_path)
            pred = np.array(results["predictions"])
            tgt = np.array(results["targets"])

            print(
                model_name,
                pred.shape,
                tgt.shape,
                pred.dtype,
                "pred min/max:",
                pred.min(),
                pred.max(),
                "unique pred (sample):",
                np.unique(pred)[:10],
            )

            if "predictions" in results and "targets" in results:
                metrics, pred, tgt = calculate_metrics(results["predictions"], results["targets"])

                results_dict[model_name] = {
                    "metrics": metrics,
                    "pred": pred,
                    "tgt": tgt,
                }

                print_summary(model_name, metrics)

    if not results_dict:
        print("\nNo model results found in", models_dir)
        print("Make sure training has completed and results are saved.")
        return

    print(f"\n{'=' * 80}")
    print("Generating Visualizations")
    print("=" * 80 + "\n")

    plot_metrics_comparison(results_dict, output_dir)

    summary_file = output_dir / "summary.txt"
    with open(summary_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("CNN Model Evaluation Summary\n")
        f.write("=" * 80 + "\n\n")

        for model_name, data in results_dict.items():
            metrics = data["metrics"]
            f.write(f"Model: {model_name}\n")
            f.write(f"  F1 Score (micro):    {metrics['f1_micro']:.4f}\n")
            f.write(f"  F1 Score (macro):    {metrics['f1_macro']:.4f}\n")
            f.write(f"  Subset Accuracy:     {metrics['subset_accuracy']:.4f}\n")
            f.write(f"  Hamming Accuracy:    {metrics['hamming_accuracy']:.4f}\n")
            f.write(f"  Precision (macro):   {metrics['precision_macro']:.4f}\n")
            f.write(f"  Recall (macro):      {metrics['recall_macro']:.4f}\n\n")

    print(f"Saved summary to {summary_file}")

    print(f"\n{'=' * 80}")
    print("Evaluation Complete")
    print(f"Results saved to: {output_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
