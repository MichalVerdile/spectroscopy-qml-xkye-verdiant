from pathlib import Path

import click
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    multilabel_confusion_matrix,
    precision_score,
    recall_score,
)

from spectroscopy_qml.ir.mps_encoder_chemical.config import (
    DATA_CONFIG,
    MODEL_CONFIG,
    PATH_CONFIG,
    TRAINING_CONFIG,
)
from spectroscopy_qml.ir.mps_encoder_chemical.data_loader import (
    FUNCTIONAL_GROUPS,
    load_ir_data,
    prepare_dataloaders,
)
from spectroscopy_qml.ir.mps_encoder_chemical.model import MPSFunctionalGroupClassifier


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
        thresholds: Per-class decision thresholds (shape: num_classes). Defaults to 0.5.

    Returns:
        Dictionary containing predictions, labels, and metrics
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for spectra, labels in dataloader:
            spectra = spectra.to(device)
            labels = labels.to(device)

            # Forward pass
            logits = model(spectra)
            probs_np = torch.sigmoid(logits).cpu().numpy()
            thresh = thresholds if thresholds is not None else 0.5
            preds = (probs_np >= thresh).astype(float)

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds)
            all_probs.append(probs_np)

    # Concatenate results
    y_true = np.vstack(all_labels)
    y_pred = np.vstack(all_preds)
    y_prob = np.vstack(all_probs)

    # Compute metrics
    metrics = {
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

    # Per-class metrics
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "metrics": metrics,
        "f1_per_class": f1_per_class,
        "precision_per_class": precision_per_class,
        "recall_per_class": recall_per_class,
    }


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
    print_line(f"{'Functional Group':<25} {'F1':>8} {'Prec':>8} {'Recall':>8} {'Support':>8}")
    print_line("-" * 80)

    for i, fg_name in enumerate(functional_groups):
        support = int(results["y_true"][:, i].sum())
        print_line(
            f"{fg_name:<25} "
            f"{results['f1_per_class'][i]:>8.4f} "
            f"{results['precision_per_class'][i]:>8.4f} "
            f"{results['recall_per_class'][i]:>8.4f} "
            f"{support:>8}"
        )

    # Statistics on predictions
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


@click.command()
@click.option(
    "--model_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to model checkpoint (default: use best_model_path from config)",
)
@click.option(
    "--data_dir",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to data directory (default: use data_dir from config)",
)
@click.option(
    "--output_dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory for results (default: use results_dir from config)",
)
def main(model_path, data_dir, output_dir):
    """Evaluate trained MPS model."""

    print("=" * 80)
    print("MPS Functional Group Classifier Evaluation")
    print("=" * 80)

    # Set paths
    model_path = model_path or Path(PATH_CONFIG.best_model_path)
    data_dir = data_dir or Path(PATH_CONFIG.data_dir)
    if not data_dir.exists():
        project_root = Path(__file__).parents[4]
        data_dir = project_root / "data" / "raw"
    output_dir = output_dir or Path(PATH_CONFIG.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: CUDA ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    # Load model
    print(f"\nLoading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model = MPSFunctionalGroupClassifier(
        input_dim=MODEL_CONFIG.input_dim,
        num_sites=MODEL_CONFIG.num_sites,
        physical_dim=MODEL_CONFIG.physical_dim,
        bond_dim=MODEL_CONFIG.bond_dim,
        num_classes=MODEL_CONFIG.num_classes,
        dropout_rate=MODEL_CONFIG.dropout_rate,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    print(f"Model loaded (trained for {checkpoint['epoch']} epochs)")
    print(f"Validation loss: {checkpoint['val_loss']:.4f}")

    # Load per-class thresholds tuned on the validation set during training
    thresholds = checkpoint.get("thresholds", np.full(MODEL_CONFIG.num_classes, 0.5))
    print(f"Thresholds: mean={thresholds.mean():.3f}, std={thresholds.std():.3f}")

    # Load data
    print(f"\nLoading data from: {data_dir}")
    X, y = load_ir_data(
        data_dir,
        target_length=DATA_CONFIG.target_length,
        max_files=DATA_CONFIG.max_files,
        apply_snv=DATA_CONFIG.apply_snv,
    )

    # Prepare dataloaders
    train_loader, val_loader, test_loader = prepare_dataloaders(
        X,
        y,
        batch_size=TRAINING_CONFIG.batch_size,
        train_ratio=TRAINING_CONFIG.train_ratio,
        val_ratio=TRAINING_CONFIG.val_ratio,
        test_ratio=TRAINING_CONFIG.test_ratio,
        random_seed=TRAINING_CONFIG.random_seed,
    )

    # Get functional group names
    functional_groups = list(FUNCTIONAL_GROUPS.keys())

    # Evaluate on test set
    print("\n" + "=" * 80)
    print("Evaluating on Test Set")
    print("=" * 80)

    test_results = evaluate_model(model, test_loader, device, thresholds=thresholds)

    # Save detailed results
    eval_output_path = output_dir / "evaluation_results.txt"
    with open(eval_output_path, "w") as f:
        print_detailed_results(test_results, functional_groups, f)

    print(f"\nDetailed results saved to: {eval_output_path}")

    # Also evaluate on validation set
    print("\n" + "=" * 80)
    print("Evaluating on Validation Set")
    print("=" * 80)

    val_results = evaluate_model(model, val_loader, device, thresholds=thresholds)

    val_output_path = output_dir / "validation_results.txt"
    with open(val_output_path, "w") as f:
        print_detailed_results(val_results, functional_groups, f)

    print(f"\nValidation results saved to: {val_output_path}")

    # Update summary with evaluation results
    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w") as f:
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

        f.write("\nValidation Set Performance:\n")
        for metric, value in val_results["metrics"].items():
            f.write(f"  {metric}: {value:.4f}\n")

        f.write("\nTop 5 Best Performing Functional Groups (by F1 score):\n")
        top_indices = np.argsort(test_results["f1_per_class"])[::-1][:5]
        for idx in top_indices:
            fg_name = functional_groups[idx]
            f1 = test_results["f1_per_class"][idx]
            support = int(test_results["y_true"][:, idx].sum())
            f.write(f"  {fg_name}: F1={f1:.4f} (support={support})\n")

        f.write("\nBottom 5 Worst Performing Functional Groups (by F1 score):\n")
        bottom_indices = np.argsort(test_results["f1_per_class"])[:5]
        for idx in bottom_indices:
            fg_name = functional_groups[idx]
            f1 = test_results["f1_per_class"][idx]
            support = int(test_results["y_true"][:, idx].sum())
            f.write(f"  {fg_name}: F1={f1:.4f} (support={support})\n")

    print(f"\nSummary saved to: {summary_path}")
    print("\n" + "=" * 80)
    print("Evaluation Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
