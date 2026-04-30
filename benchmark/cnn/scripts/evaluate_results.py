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

    metrics["f1_micro"] = f1_score(tgt, pred, average="micro")
    metrics["f1_macro"] = f1_score(tgt, pred, average="macro")

    metrics["precision_micro"] = precision_score(tgt, pred, average="micro")
    metrics["precision_macro"] = precision_score(tgt, pred, average="macro")

    metrics["recall_micro"] = recall_score(tgt, pred, average="micro")
    metrics["recall_macro"] = recall_score(tgt, pred, average="macro")

    metrics["subset_accuracy"] = accuracy_score(tgt, pred)
    metrics["hamming_accuracy"] = (tgt == pred).mean()

    return metrics, pred, tgt


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

    # Fallback class names (from training code)
    fallback_class_names = [
        "Acid anhydride", "Acyl halide", "Alcohol", "Aldehyde", "Alkane",
        "Alkene", "Alkyne", "Amide", "Amine", "Arene", "Azo compound",
        "Carbamate", "Carboxylic acid", "Enamine", "Enol", "Ester", "Ether",
        "Haloalkane", "Hydrazine", "Hydrazone", "Imide", "Imine",
        "Isocyanate", "Isothiocyanate", "Ketone", "Nitrile", "Phenol",
        "Phosphine", "Sulfide", "Sulfonamide", "Sulfonate", "Sulfone",
        "Sulfonic acid", "Sulfoxide", "Thial", "Thioamide", "Thiol"
    ]

    with open(output_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(f"Per-Class F1 Scores - {model_type.upper()}\n")
        f.write("=" * 80 + "\n\n")

        for model_name, data in results_dict.items():
            pred = data["pred"]
            tgt = data["tgt"]

            # Try to get class names from data, else fallback
            class_names = data.get("class_names", fallback_class_names)

            # Compute per-class F1
            f1_per_class = f1_score(tgt, pred, average=None)

            f.write(f"Model: {model_name}\n")

            for i, score in enumerate(f1_per_class):
                name = class_names[i] if i < len(class_names) else f"class_{i}"
                f.write(f"  {name:<20} : {score:.4f}\n")

            f.write("\n")

    print(f"Saved per-class F1 scores to {output_file}")


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

                metrics, pred, tgt = calculate_metrics(
                    results["predictions"],
                    results["targets"],
                )

                results_dict[model_name] = {
                    "metrics": metrics,
                    "pred": pred,
                    "tgt": tgt,
                }

                print_summary(model_name, metrics)
                save_per_class_f1(results_dict, output_dir, args.model_type)

            else:
                print(f"Warning: No valid predictions/targets found in {model_results_path}")

    if not results_dict:
        print("\nNo model results found in", models_dir)
        print("Make sure training has completed and results are saved.")
        return

    print(f"\n{'=' * 80}")
    print("Generating Visualizations")
    print("=" * 80 + "\n")

    plot_metrics_comparison(results_dict, output_dir, args.model_type)

    summary_file = output_dir / f"summary_{args.model_type}.txt"

    with open(summary_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(f"CNN Model Evaluation Summary - {args.model_type.upper()}\n")
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
            f.write(f"  Hamming Accuracy:    {metrics['hamming_accuracy']:.4f}\n\n")

    print(f"Saved summary to {summary_file}")

    print(f"\n{'=' * 80}")
    print("Evaluation Complete")
    print(f"Results saved to: {output_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()