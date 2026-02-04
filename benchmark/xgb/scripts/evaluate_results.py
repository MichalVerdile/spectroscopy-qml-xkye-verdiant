import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
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
    metrics = {}

    pred = np.array(predictions)
    tgt = np.array(targets)

    if len(pred.shape) > 1 and pred.shape[1] > 1:
        pred_classes = np.argmax(pred, axis=1)
        tgt_classes = np.argmax(tgt, axis=1) if len(tgt.shape) > 1 else tgt
    else:
        pred_classes = pred
        tgt_classes = tgt

    metrics["accuracy"] = accuracy_score(tgt_classes, pred_classes)
    metrics["f1_macro"] = f1_score(tgt_classes, pred_classes, average="macro", zero_division=0)
    metrics["f1_micro"] = f1_score(tgt_classes, pred_classes, average="micro", zero_division=0)
    metrics["precision"] = precision_score(
        tgt_classes, pred_classes, average="macro", zero_division=0
    )
    metrics["recall"] = recall_score(tgt_classes, pred_classes, average="macro", zero_division=0)

    return metrics, pred_classes, tgt_classes


def print_summary(model_name, metrics):
    print(f"\n{'=' * 80}")
    print(f"{model_name}")
    print(f"{'=' * 80}")
    print(f"  Accuracy:       {metrics['accuracy']:.4f}")
    print(f"  F1 Score (macro): {metrics['f1_macro']:.4f}")
    print(f"  F1 Score (micro): {metrics['f1_micro']:.4f}")
    print(f"  Precision:      {metrics['precision']:.4f}")
    print(f"  Recall:         {metrics['recall']:.4f}")


def plot_confusion_matrices(results_dict, output_dir):
    n_models = len(results_dict)
    if n_models == 0:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("CNN Model Confusion Matrices", fontsize=16)
    axes = axes.flatten()

    for idx, (model_name, data) in enumerate(results_dict.items()):
        if idx >= 6:
            break

        pred_classes = data["pred_classes"]
        tgt_classes = data["tgt_classes"]

        cm = confusion_matrix(tgt_classes, pred_classes)

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx], cbar=True)
        axes[idx].set_title(f"{model_name}")
        axes[idx].set_ylabel("True Label")
        axes[idx].set_xlabel("Predicted Label")

    for idx in range(n_models, 6):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrices.png", dpi=300, bbox_inches="tight")
    print(f"Saved confusion matrices to {output_dir / 'confusion_matrices.png'}")
    plt.close()


def plot_metrics_comparison(results_dict, output_dir):
    model_names = []
    accuracies = []
    f1_scores = []
    precisions = []
    recalls = []

    for model_name, data in results_dict.items():
        metrics = data["metrics"]
        model_names.append(model_name)
        accuracies.append(metrics["accuracy"])
        f1_scores.append(metrics["f1_macro"])
        precisions.append(metrics["precision"])
        recalls.append(metrics["recall"])

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("CNN Model Performance Comparison", fontsize=16)

    x = np.arange(len(model_names))
    width = 0.6

    axes[0, 0].bar(x, accuracies, width, color="steelblue", alpha=0.8)
    axes[0, 0].set_ylabel("Score")
    axes[0, 0].set_title("Accuracy")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(model_names, rotation=45, ha="right")
    axes[0, 0].set_ylim([0, 1])
    axes[0, 0].grid(axis="y", alpha=0.3)
    for i, v in enumerate(accuracies):
        axes[0, 0].text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    axes[0, 1].bar(x, f1_scores, width, color="coral", alpha=0.8)
    axes[0, 1].set_ylabel("Score")
    axes[0, 1].set_title("F1 Score (Macro)")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(model_names, rotation=45, ha="right")
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].grid(axis="y", alpha=0.3)
    for i, v in enumerate(f1_scores):
        axes[0, 1].text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    axes[1, 0].bar(x, precisions, width, color="lightgreen", alpha=0.8)
    axes[1, 0].set_ylabel("Score")
    axes[1, 0].set_title("Precision")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(model_names, rotation=45, ha="right")
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].grid(axis="y", alpha=0.3)
    for i, v in enumerate(precisions):
        axes[1, 0].text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    axes[1, 1].bar(x, recalls, width, color="plum", alpha=0.8)
    axes[1, 1].set_ylabel("Score")
    axes[1, 1].set_title("Recall")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(model_names, rotation=45, ha="right")
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].grid(axis="y", alpha=0.3)
    for i, v in enumerate(recalls):
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

            if "predictions" in results and "targets" in results:
                metrics, pred_classes, tgt_classes = calculate_metrics(
                    results["predictions"], results["targets"]
                )

                results_dict[model_name] = {
                    "metrics": metrics,
                    "pred_classes": pred_classes,
                    "tgt_classes": tgt_classes,
                }

                print_summary(model_name, metrics)

    if not results_dict:
        print("\nNo model results found in", models_dir)
        print("Make sure training has completed and results are saved.")
        return

    # Create visualizations
    print(f"\n{'=' * 80}")
    print("Generating Visualizations")
    print("=" * 80 + "\n")

    plot_confusion_matrices(results_dict, output_dir)
    plot_metrics_comparison(results_dict, output_dir)

    # Save summary to text file
    summary_file = output_dir / "summary.txt"
    with open(summary_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("CNN Model Evaluation Summary\n")
        f.write("=" * 80 + "\n\n")

        for model_name, data in results_dict.items():
            metrics = data["metrics"]
            f.write(f"Model: {model_name}\n")
            f.write(f"  Accuracy:       {metrics['accuracy']:.4f}\n")
            f.write(f"  F1 Score (macro): {metrics['f1_macro']:.4f}\n")
            f.write(f"  Precision:      {metrics['precision']:.4f}\n")
            f.write(f"  Recall:         {metrics['recall']:.4f}\n\n")

    print(f"Saved summary to {summary_file}")

    print(f"\n{'=' * 80}")
    print("Evaluation Complete")
    print(f"Results saved to: {output_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
