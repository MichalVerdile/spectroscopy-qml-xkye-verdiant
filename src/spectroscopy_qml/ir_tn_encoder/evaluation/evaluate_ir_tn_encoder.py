"""
Evaluation script for trained IR TN encoder models.

Loads trained models and evaluates them on the test set,
producing comprehensive results and visualizations.

Usage:
    python evaluate_ir_tn_encoder.py --model_path models/model.pt --data_dir data/raw
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from spectroscopy_qml.ir_tn_encoder.utils.cnn_classifier import FunctionalGroupClassifier
from spectroscopy_qml.ir_tn_encoder.utils.mlp_encoder import MLPEncoder
from spectroscopy_qml.ir_tn_encoder.utils.mps_encoder import MPSEncoder
from spectroscopy_qml.ir_tn_encoder.utils.preprocess_ir_data import (
    IRFunctionalGroupDataset,
    create_data_splits,
)
from spectroscopy_qml.ir_tn_encoder.utils.train import (
    collect_predictions,
    compute_f1_scores,
    count_parameters,
    evaluate,
    find_best_threshold,
)


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""

    # Model
    model_path: str | None = None
    model_type: str = "mps"  # "mps", "mps_simple", "mlp"

    # Model architecture
    embedding_dim: int = 128
    num_sites: int = 30
    physical_dim: int = 8
    bond_dim: int = 16

    # Data
    data_dir: str | None = None
    target_length: int = 600
    normalization: str = "zscore"
    max_chunks: int | None = None
    batch_size: int = 64

    # Evaluation
    threshold_metric: str = "f1_micro"  # "f1_micro" or "f1_macro"

    # Output
    output_dir: str | None = None
    seed: int = 42

    # Hardware
    device: str = "auto"

    def __post_init__(self) -> None:
        # Set default paths relative to script location
        script_dir = Path(__file__).resolve().parent

        if self.data_dir is None:
            self.data_dir = str(script_dir.parent.parent.parent.parent / "data" / "raw")

        if self.output_dir is None:
            self.output_dir = str(script_dir.parent / "results")

        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"


@dataclass
class EvaluationResult:
    """Result of model evaluation."""

    model_name: str
    model_path: str | None
    num_parameters: int

    # Validation metrics
    val_f1_micro: float
    val_f1_macro: float
    best_threshold: float

    # Test metrics
    test_loss: float
    test_f1_micro: float
    test_f1_macro: float

    # Per-class F1 scores (optional)
    per_class_f1: dict[str, float] | None = None

    # Configuration
    config: dict[str, Any] | None = None

    # Metadata
    timestamp: str | None = None


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    config: EvaluationConfig,
    num_classes: int,
) -> tuple[nn.Module, str]:
    """
    Load a trained model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint (.pt file)
        config: Evaluation configuration
        num_classes: Number of output classes

    Returns:
        Loaded model and model name
    """
    checkpoint_path = Path(checkpoint_path)

    # Create encoder based on config
    encoder: nn.Module
    if config.model_type == "mlp":
        encoder = MLPEncoder(
            input_length=config.target_length,
            embedding_dim=config.embedding_dim,
        )
        model_name = "MLP"
    else:  # mps
        encoder = MPSEncoder(
            input_length=config.target_length,
            num_sites=config.num_sites,
            physical_dim=config.physical_dim,
            bond_dim=config.bond_dim,
            embedding_dim=config.embedding_dim,
        )
        model_name = f"MPS (D={config.bond_dim})"

    # Create full model
    model = FunctionalGroupClassifier(
        encoder=encoder,
        num_classes=num_classes,
        embedding_dim=config.embedding_dim,
    )

    # Load checkpoint
    print(f"Loading model from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=config.device)
    model.load_state_dict(state_dict)
    model.to(config.device)
    model.eval()

    print(f"Loaded {model_name} with {count_parameters(model):,} parameters")

    return model, model_name


def compute_per_class_f1(
    predictions: np.ndarray,
    targets: np.ndarray,
    class_names: list[str],
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Compute per-class F1 scores.

    Args:
        predictions: Model predictions (probabilities) of shape (n_samples, n_classes)
        targets: Ground truth labels of shape (n_samples, n_classes)
        class_names: List of class names
        threshold: Classification threshold

    Returns:
        Dictionary mapping class names to F1 scores
    """
    from sklearn.metrics import f1_score

    pred_binary = (predictions > threshold).astype(int)
    target_int = targets.astype(int)

    per_class_scores: dict[str, float] = {}

    for i, class_name in enumerate(class_names):
        if targets[:, i].sum() == 0:
            # No positive samples for this class
            per_class_scores[class_name] = 0.0
        else:
            f1 = f1_score(
                target_int[:, i],
                pred_binary[:, i],
                average="binary",
                zero_division=0,
            )
            per_class_scores[class_name] = float(f1)

    return per_class_scores


def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    test_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    config: EvaluationConfig,
    class_mask: torch.Tensor | None = None,
    class_names: list[str] | None = None,
) -> tuple[float, float, float, float, float, float, dict[str, float] | None]:
    """
    Evaluate a model on validation and test sets.

    Returns:
        Tuple of (val_f1_micro, val_f1_macro, best_threshold, test_loss, test_f1_micro, test_f1_macro, per_class_f1)
    """
    device = config.device
    criterion = nn.BCEWithLogitsLoss()

    # Collect validation predictions for threshold tuning
    val_predictions, val_targets = collect_predictions(
        model, val_loader, device, class_mask=class_mask
    )

    # Find best threshold on validation set
    best_threshold = find_best_threshold(
        val_predictions,
        val_targets,
        metric=config.threshold_metric,
    )

    # Compute validation metrics
    val_f1_micro, val_f1_macro = compute_f1_scores(
        val_predictions,
        val_targets,
        threshold=best_threshold,
    )

    print("\nValidation Results:")
    print(f"  Best Threshold: {best_threshold:.3f}")
    print(f"  Val F1 Micro: {val_f1_micro:.4f}")
    print(f"  Val F1 Macro: {val_f1_macro:.4f}")

    # Evaluate on test set
    test_loss, test_f1_micro, test_f1_macro = evaluate(
        model,
        test_loader,
        criterion,
        device,
        class_mask=class_mask,
        threshold=best_threshold,
    )

    print("\nTest Results:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test F1 Micro: {test_f1_micro:.4f}")
    print(f"  Test F1 Macro: {test_f1_macro:.4f}")

    # Compute per-class F1 scores if class names provided
    per_class_f1 = None
    if class_names is not None:
        test_predictions, test_targets = collect_predictions(
            model, test_loader, device, class_mask=class_mask
        )
        per_class_f1 = compute_per_class_f1(
            test_predictions,
            test_targets,
            class_names,
            threshold=best_threshold,
        )

        print("\nPer-Class F1 Scores (Top 10):")
        sorted_classes = sorted(per_class_f1.items(), key=lambda x: x[1], reverse=True)
        for class_name, f1 in sorted_classes[:10]:
            print(f"  {class_name:<25}: {f1:.4f}")

    return (
        val_f1_micro,
        val_f1_macro,
        best_threshold,
        test_loss,
        test_f1_micro,
        test_f1_macro,
        per_class_f1,
    )


def save_evaluation_results(result: EvaluationResult, output_dir: str | Path) -> None:
    """Save evaluation results to JSON file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"evaluation_{timestamp}.json"

    # Convert to dict
    result_dict = asdict(result)
    result_dict["timestamp"] = timestamp

    with open(output_file, "w") as f:
        json.dump(result_dict, f, indent=2)

    print(f"\nResults saved to {output_file}")

    # Also save a human-readable summary
    summary_file = output_dir / f"evaluation_{timestamp}_summary.txt"
    with open(summary_file, "w") as f:
        f.write("Model Evaluation Summary\n")
        f.write(f"{'=' * 80}\n\n")
        f.write(f"Model: {result.model_name}\n")
        f.write(f"Model Path: {result.model_path}\n")
        f.write(f"Parameters: {result.num_parameters:,}\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        f.write("Validation Metrics:\n")
        f.write(f"  F1 Micro: {result.val_f1_micro:.4f}\n")
        f.write(f"  F1 Macro: {result.val_f1_macro:.4f}\n")
        f.write(f"  Best Threshold: {result.best_threshold:.3f}\n\n")
        f.write("Test Metrics:\n")
        f.write(f"  Loss: {result.test_loss:.4f}\n")
        f.write(f"  F1 Micro: {result.test_f1_micro:.4f}\n")
        f.write(f"  F1 Macro: {result.test_f1_macro:.4f}\n\n")

        if result.per_class_f1:
            f.write("Per-Class F1 Scores:\n")
            sorted_classes = sorted(result.per_class_f1.items(), key=lambda x: x[1], reverse=True)
            for class_name, f1 in sorted_classes:
                f.write(f"  {class_name:<30}: {f1:.4f}\n")

    print(f"Summary saved to {summary_file}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate trained IR TN encoder model")

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="mps",
        choices=["mps", "mps_simple", "mlp"],
        help="Type of encoder model",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to data directory (default: auto-detect from script location)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for evaluation results (default: auto-detect from script location)",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=128,
        help="Embedding dimension",
    )
    parser.add_argument(
        "--num_sites",
        type=int,
        default=30,
        help="Number of sites for MPS encoder",
    )
    parser.add_argument(
        "--physical_dim",
        type=int,
        default=8,
        help="Physical dimension for MPS encoder",
    )
    parser.add_argument(
        "--bond_dim",
        type=int,
        default=16,
        help="Bond dimension for MPS encoder",
    )
    parser.add_argument(
        "--target_length",
        type=int,
        default=600,
        help="Resample spectra to this length",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--threshold_metric",
        type=str,
        default="f1_micro",
        choices=["f1_micro", "f1_macro"],
        help="Validation metric used to tune decision threshold",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    # Create config
    config = EvaluationConfig(
        model_path=args.model_path,
        model_type=args.model_type,
        embedding_dim=args.embedding_dim,
        num_sites=args.num_sites,
        physical_dim=args.physical_dim,
        bond_dim=args.bond_dim,
        data_dir=args.data_dir,
        target_length=args.target_length,
        batch_size=args.batch_size,
        threshold_metric=args.threshold_metric,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    print("=" * 80)
    print("IR TN Encoder Model Evaluation")
    print("=" * 80)
    print(f"Model path: {config.model_path}")
    print(f"Model type: {config.model_type}")
    print(f"Device: {config.device}")
    print()

    # Set seed
    torch.manual_seed(config.seed)

    # Load dataset
    print("Loading dataset...")
    dataset = IRFunctionalGroupDataset(
        data_dir=config.data_dir,
        target_length=config.target_length,
        normalization=config.normalization,
        max_chunks=config.max_chunks,
    )

    # Create splits (use same seed as training)
    train_set, val_set, test_set = create_data_splits(dataset, seed=config.seed)

    # Data loaders
    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Class weights and mask
    class_mask = dataset.get_active_class_mask()
    class_names = list(dataset.functional_groups.keys())

    print(f"Dataset: {len(dataset)} samples")
    print(f"Val: {len(val_set)}, Test: {len(test_set)}")
    print(f"Active classes: {int(class_mask.sum().item())}/{dataset.num_classes}")
    print()

    # Load model
    model, model_name = load_model_from_checkpoint(
        config.model_path,
        config,
        dataset.num_classes,
    )

    # Evaluate
    (
        val_f1_micro,
        val_f1_macro,
        best_threshold,
        test_loss,
        test_f1_micro,
        test_f1_macro,
        per_class_f1,
    ) = evaluate_model(
        model,
        val_loader,
        test_loader,
        config,
        class_mask=class_mask,
        class_names=class_names,
    )

    # Create result
    result = EvaluationResult(
        model_name=model_name,
        model_path=str(config.model_path),
        num_parameters=count_parameters(model),
        val_f1_micro=val_f1_micro,
        val_f1_macro=val_f1_macro,
        best_threshold=best_threshold,
        test_loss=test_loss,
        test_f1_micro=test_f1_micro,
        test_f1_macro=test_f1_macro,
        per_class_f1=per_class_f1,
        config={
            "model_type": config.model_type,
            "embedding_dim": config.embedding_dim,
            "num_sites": config.num_sites,
            "physical_dim": config.physical_dim,
            "bond_dim": config.bond_dim,
            "target_length": config.target_length,
            "threshold_metric": config.threshold_metric,
        },
    )

    # Save results
    save_evaluation_results(result, config.output_dir)

    print("\n" + "=" * 80)
    print("Evaluation completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
