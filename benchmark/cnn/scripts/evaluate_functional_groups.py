"""
Evaluate CNN models per functional group.
Shows detailed metrics including false positives/negatives for each functional group.
"""

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Functional groups in the same order as the model output
FUNCTIONAL_GROUPS = [
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


def load_results(model_dir):
    """Load predictions and targets from results.pickle file."""
    results_file = Path(model_dir) / "results.pickle"
    
    if not results_file.exists():
        return None
    
    try:
        with open(results_file, "rb") as f:
            data = pickle.load(f)
            
            if isinstance(data, dict) and "pred" in data and "tgt" in data:
                pred = np.array(data["pred"])
                tgt = np.array(data["tgt"])
                
                # Convert to binary if needed
                if pred.dtype in [np.float64, np.float32]:
                    if pred.min() >= 0 and pred.max() <= 1:
                        pred = (pred >= 0.5).astype(int)
                
                return {
                    "predictions": pred.astype(int),
                    "targets": tgt.astype(int),
                    "f1_score": data.get("f1_score", None),
                }
    except Exception as e:
        print(f"Error loading {results_file}: {e}")
        return None
    
    return None


def calculate_per_class_metrics(predictions, targets):
    """Calculate detailed metrics for each functional group."""
    num_classes = predictions.shape[1]
    metrics = []
    
    for i in range(num_classes):
        pred_i = predictions[:, i]
        tgt_i = targets[:, i]
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(tgt_i, pred_i).ravel()
        
        # Calculate metrics
        precision = precision_score(tgt_i, pred_i, zero_division=0)
        recall = recall_score(tgt_i, pred_i, zero_division=0)
        f1 = f1_score(tgt_i, pred_i, zero_division=0)
        accuracy = accuracy_score(tgt_i, pred_i)
        
        # Support (number of true instances)
        support = np.sum(tgt_i)
        
        metrics.append({
            "functional_group": FUNCTIONAL_GROUPS[i] if i < len(FUNCTIONAL_GROUPS) else f"FG_{i}",
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": accuracy,
            "support": int(support),
        })
    
    return pd.DataFrame(metrics)


def calculate_overall_metrics(predictions, targets):
    """Calculate overall multi-label metrics."""
    metrics = {
        "f1_micro": f1_score(targets, predictions, average="micro"),
        "f1_macro": f1_score(targets, predictions, average="macro"),
        "f1_weighted": f1_score(targets, predictions, average="weighted"),
        "precision_micro": precision_score(targets, predictions, average="micro"),
        "precision_macro": precision_score(targets, predictions, average="macro"),
        "recall_micro": recall_score(targets, predictions, average="micro"),
        "recall_macro": recall_score(targets, predictions, average="macro"),
        "subset_accuracy": accuracy_score(targets, predictions),
        "hamming_accuracy": (targets == predictions).mean(),
    }
    return metrics


def print_detailed_report(model_name, df_metrics, overall_metrics):
    """Print detailed evaluation report."""
    print(f"\n{'=' * 100}")
    print(f"Model: {model_name}")
    print(f"{'=' * 100}")
    
    # Overall metrics
    print("\n--- Overall Metrics ---")
    print(f"F1 Score (micro):      {overall_metrics['f1_micro']:.4f}")
    print(f"F1 Score (macro):      {overall_metrics['f1_macro']:.4f}")
    print(f"F1 Score (weighted):   {overall_metrics['f1_weighted']:.4f}")
    print(f"Precision (micro):     {overall_metrics['precision_micro']:.4f}")
    print(f"Precision (macro):     {overall_metrics['precision_macro']:.4f}")
    print(f"Recall (micro):        {overall_metrics['recall_micro']:.4f}")
    print(f"Recall (macro):        {overall_metrics['recall_macro']:.4f}")
    print(f"Subset Accuracy:       {overall_metrics['subset_accuracy']:.4f}")
    print(f"Hamming Accuracy:      {overall_metrics['hamming_accuracy']:.4f}")
    
    # Per-class metrics
    print("\n--- Per-Functional-Group Metrics ---")
    print(f"{'Functional Group':<25} {'Support':>8} {'TP':>6} {'FP':>6} {'TN':>8} {'FN':>6} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Accuracy':>10}")
    print("-" * 150)
    
    for _, row in df_metrics.iterrows():
        print(f"{row['functional_group']:<25} "
              f"{row['support']:>8} "
              f"{row['true_positives']:>6} "
              f"{row['false_positives']:>6} "
              f"{row['true_negatives']:>8} "
              f"{row['false_negatives']:>6} "
              f"{row['precision']:>10.4f} "
              f"{row['recall']:>10.4f} "
              f"{row['f1_score']:>10.4f} "
              f"{row['accuracy']:>10.4f}")
    
    # Summary statistics
    print("\n--- Summary Statistics ---")
    print(f"Mean F1 Score:      {df_metrics['f1_score'].mean():.4f}")
    print(f"Std F1 Score:       {df_metrics['f1_score'].std():.4f}")
    print(f"Mean Precision:     {df_metrics['precision'].mean():.4f}")
    print(f"Mean Recall:        {df_metrics['recall'].mean():.4f}")
    
    # Worst performing functional groups
    print("\n--- Top 5 Worst Performing Functional Groups (by F1) ---")
    worst_fgs = df_metrics.nsmallest(5, 'f1_score')[['functional_group', 'f1_score', 'precision', 'recall', 'support']]
    for _, row in worst_fgs.iterrows():
        print(f"  {row['functional_group']:<25} F1: {row['f1_score']:.4f}, "
              f"Precision: {row['precision']:.4f}, Recall: {row['recall']:.4f}, "
              f"Support: {row['support']}")
    
    # Best performing functional groups
    print("\n--- Top 5 Best Performing Functional Groups (by F1) ---")
    best_fgs = df_metrics.nlargest(5, 'f1_score')[['functional_group', 'f1_score', 'precision', 'recall', 'support']]
    for _, row in best_fgs.iterrows():
        print(f"  {row['functional_group']:<25} F1: {row['f1_score']:.4f}, "
              f"Precision: {row['precision']:.4f}, Recall: {row['recall']:.4f}, "
              f"Support: {row['support']}")
    
    # Most false positives
    print("\n--- Top 5 Functional Groups with Most False Positives ---")
    most_fp = df_metrics.nlargest(5, 'false_positives')[['functional_group', 'false_positives', 'precision', 'support']]
    for _, row in most_fp.iterrows():
        print(f"  {row['functional_group']:<25} FP: {row['false_positives']:>6}, "
              f"Precision: {row['precision']:.4f}, Support: {row['support']}")
    
    # Most false negatives
    print("\n--- Top 5 Functional Groups with Most False Negatives ---")
    most_fn = df_metrics.nlargest(5, 'false_negatives')[['functional_group', 'false_negatives', 'recall', 'support']]
    for _, row in most_fn.iterrows():
        print(f"  {row['functional_group']:<25} FN: {row['false_negatives']:>6}, "
              f"Recall: {row['recall']:.4f}, Support: {row['support']}")


def plot_per_class_metrics(df_metrics, model_name, output_dir):
    """Create visualizations for per-class metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle(f"Per-Functional-Group Metrics: {model_name}", fontsize=16, fontweight='bold')
    
    # Sort by F1 score for better visualization
    df_sorted = df_metrics.sort_values('f1_score', ascending=True)
    
    # F1 Score
    ax1 = axes[0, 0]
    colors = ['red' if f1 < 0.5 else 'orange' if f1 < 0.7 else 'green' for f1 in df_sorted['f1_score']]
    ax1.barh(df_sorted['functional_group'], df_sorted['f1_score'], color=colors, alpha=0.7)
    ax1.set_xlabel('F1 Score', fontweight='bold')
    ax1.set_title('F1 Score by Functional Group')
    ax1.set_xlim([0, 1])
    ax1.grid(axis='x', alpha=0.3)
    ax1.axvline(x=0.7, color='gray', linestyle='--', alpha=0.5, label='Good threshold')
    
    # Precision vs Recall
    ax2 = axes[0, 1]
    scatter = ax2.scatter(df_metrics['recall'], df_metrics['precision'], 
                         s=df_metrics['support']/10, alpha=0.6, 
                         c=df_metrics['f1_score'], cmap='RdYlGn', vmin=0, vmax=1)
    ax2.set_xlabel('Recall', fontweight='bold')
    ax2.set_ylabel('Precision', fontweight='bold')
    ax2.set_title('Precision vs Recall (bubble size = support)')
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    ax2.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label='F1 Score')
    
    # False Positives and False Negatives
    ax3 = axes[1, 0]
    df_sorted_errors = df_metrics.sort_values('false_positives', ascending=True)
    x = np.arange(len(df_sorted_errors))
    width = 0.35
    ax3.barh(x - width/2, df_sorted_errors['false_positives'], width, label='False Positives', color='red', alpha=0.7)
    ax3.barh(x + width/2, df_sorted_errors['false_negatives'], width, label='False Negatives', color='orange', alpha=0.7)
    ax3.set_yticks(x)
    ax3.set_yticklabels(df_sorted_errors['functional_group'])
    ax3.set_xlabel('Count', fontweight='bold')
    ax3.set_title('False Positives vs False Negatives')
    ax3.legend()
    ax3.grid(axis='x', alpha=0.3)
    
    # Support distribution
    ax4 = axes[1, 1]
    df_sorted_support = df_metrics.sort_values('support', ascending=True)
    ax4.barh(df_sorted_support['functional_group'], df_sorted_support['support'], 
             color='steelblue', alpha=0.7)
    ax4.set_xlabel('Support (# of positive instances)', fontweight='bold')
    ax4.set_title('Class Distribution')
    ax4.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / f"{model_name}_per_class_metrics.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {output_file}")
    plt.close()


def save_detailed_report(model_name, df_metrics, overall_metrics, output_dir):
    """Save detailed report to CSV and text files."""
    # Save per-class metrics to CSV
    csv_file = output_dir / f"{model_name}_per_class_metrics.csv"
    df_metrics.to_csv(csv_file, index=False)
    print(f"Saved per-class metrics to {csv_file}")
    
    # Save overall metrics to text file
    txt_file = output_dir / f"{model_name}_evaluation_report.txt"
    with open(txt_file, 'w') as f:
        f.write(f"{'=' * 100}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"{'=' * 100}\n\n")
        
        f.write("--- Overall Metrics ---\n")
        for key, value in overall_metrics.items():
            f.write(f"{key:<25} {value:.4f}\n")
        
        f.write("\n--- Per-Functional-Group Metrics ---\n")
        f.write(f"{'Functional Group':<25} {'Support':>8} {'TP':>6} {'FP':>6} {'TN':>8} {'FN':>6} "
                f"{'Precision':>10} {'Recall':>10} {'F1':>10} {'Accuracy':>10}\n")
        f.write("-" * 150 + "\n")
        
        for _, row in df_metrics.iterrows():
            f.write(f"{row['functional_group']:<25} "
                   f"{row['support']:>8} "
                   f"{row['true_positives']:>6} "
                   f"{row['false_positives']:>6} "
                   f"{row['true_negatives']:>8} "
                   f"{row['false_negatives']:>6} "
                   f"{row['precision']:>10.4f} "
                   f"{row['recall']:>10.4f} "
                   f"{row['f1_score']:>10.4f} "
                   f"{row['accuracy']:>10.4f}\n")
        
        f.write("\n--- Summary Statistics ---\n")
        f.write(f"Mean F1 Score:      {df_metrics['f1_score'].mean():.4f}\n")
        f.write(f"Std F1 Score:       {df_metrics['f1_score'].std():.4f}\n")
        f.write(f"Mean Precision:     {df_metrics['precision'].mean():.4f}\n")
        f.write(f"Mean Recall:        {df_metrics['recall'].mean():.4f}\n")
    
    print(f"Saved evaluation report to {txt_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CNN models with per-functional-group analysis"
    )
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
        "--model",
        type=str,
        default=None,
        help="Specific model to evaluate (e.g., 'cnmr', 'ir'). If not specified, evaluates all models.",
    )
    
    args = parser.parse_args()
    
    models_dir = Path(args.models_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 100)
    print("CNN Model Per-Functional-Group Evaluation")
    print("=" * 100)
    
    # Determine which models to evaluate
    if args.model:
        model_dirs = [models_dir / args.model]
    else:
        model_dirs = sorted([d for d in models_dir.glob("*") if d.is_dir()])
    
    all_results = {}
    
    for model_dir in model_dirs:
        if not model_dir.is_dir():
            continue
            
        model_name = model_dir.name
        print(f"\nEvaluating model: {model_name}")
        
        results = load_results(model_dir)
        
        if results is None:
            print(f"  No results.pickle found for {model_name}")
            continue
        
        predictions = results["predictions"]
        targets = results["targets"]
        
        print(f"  Loaded {predictions.shape[0]} samples with {predictions.shape[1]} functional groups")
        
        # Calculate metrics
        df_metrics = calculate_per_class_metrics(predictions, targets)
        overall_metrics = calculate_overall_metrics(predictions, targets)
        
        # Store results
        all_results[model_name] = {
            "df_metrics": df_metrics,
            "overall_metrics": overall_metrics,
        }
        
        # Print detailed report
        print_detailed_report(model_name, df_metrics, overall_metrics)
        
        # Save reports
        save_detailed_report(model_name, df_metrics, overall_metrics, output_dir)
        
        # Create visualizations
        plot_per_class_metrics(df_metrics, model_name, output_dir)
    
    if not all_results:
        print("\nNo model results found.")
        return
    
    # Create comparison across models
    print(f"\n{'=' * 100}")
    print("Cross-Model Comparison")
    print(f"{'=' * 100}\n")
    
    comparison_data = []
    for model_name, data in all_results.items():
        metrics = data["overall_metrics"]
        comparison_data.append({
            "Model": model_name,
            "F1 (micro)": metrics["f1_micro"],
            "F1 (macro)": metrics["f1_macro"],
            "Precision (macro)": metrics["precision_macro"],
            "Recall (macro)": metrics["recall_macro"],
            "Subset Accuracy": metrics["subset_accuracy"],
            "Hamming Accuracy": metrics["hamming_accuracy"],
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    print(df_comparison.to_string(index=False))
    
    # Save comparison
    comparison_file = output_dir / "model_comparison.csv"
    df_comparison.to_csv(comparison_file, index=False)
    print(f"\nSaved model comparison to {comparison_file}")
    
    print(f"\n{'=' * 100}")
    print("Evaluation Complete")
    print(f"Results saved to: {output_dir}")
    print(f"{'=' * 100}\n")


if __name__ == "__main__":
    main()
