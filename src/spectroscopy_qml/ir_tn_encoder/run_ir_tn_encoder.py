# We provide two MPS implementations:
# MPSEncoder (main tensor-core version) and MPSEncoderSimple (clearer reference version).
# The simple version is used for validation and debugging to ensure correctness
# and stability of the main implementation.

"""
Main experiment runner for TN evaluation.

Compares Tensor Network (MPS) encoder against CNN baseline
for functional group prediction from IR spectra.

Usage:
    python run_experiment.py --data_dir data/raw --output_dir results/tn_evaluation

Features:
    - Bond dimension sweep for MPS encoder
    - Comparison with CNN baseline
    - Parameter count reporting
    - Results export to JSON
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader

from .utils.cnn_classifier import FunctionalGroupClassifier
from .utils.mlp_encoder import MLPEncoder
from .utils.mps_encoder import MPSEncoder
from .utils.preprocess_ir_data import (
    IRFunctionalGroupDataset,
    create_data_splits,
)
from .utils.train import (
    TrainingConfig,
    TrainingMetrics,
    collect_predictions,
    compute_f1_scores,
    count_parameters,
    evaluate,
    find_best_threshold,
    train_model,
)


@dataclass
class ExperimentConfig:
    """Configuration for the full experiment."""

    # Data
    data_dir: str = "../../../data/raw"
    max_chunks: int | None = None  # Limit chunks for testing
    target_length: int = 512
    normalization: Literal["zscore"] = "zscore"

    # Model
    embedding_dim: int = 128

    # MPS specific
    num_sites: int = 32
    physical_dim: int = 8
    bond_dims: list[int] = None  # type: ignore[assignment]  # Set in __post_init__

    # CNN baseline source
    benchmark_cnn_results_path: str = "../../../benchmark/cnn/models/ir/results.pickle"

    # Training
    batch_size: int = 64
    learning_rate: float = 1e-3
    mps_learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    num_epochs: int = 100
    patience: int = 15
    grad_clip_norm: float | None = 1.0
    pos_weight_cap: float = 20.0
    threshold_metric: str = "f1_micro"  # "f1_micro" or "f1_macro"

    # Output
    output_dir: str = "results"
    models_dir: str = "models"
    seed: int = 42

    def __post_init__(self) -> None:
        if self.bond_dims is None:
            self.bond_dims = [4, 8, 16, 32]


@dataclass
class ExperimentResult:
    """Result of a single model experiment."""

    model_name: str
    num_parameters: int
    best_val_f1_micro: float
    best_val_f1_macro: float
    test_f1_micro: float
    test_f1_macro: float
    test_loss: float
    best_epoch: int
    training_time_seconds: float
    config: dict[str, object]


def load_benchmark_cnn_result(results_path: str | Path) -> ExperimentResult:
    """
    Load benchmark CNN predictions/targets and convert to ExperimentResult.

    Expected pickle schema:
        {"pred": np.ndarray, "tgt": np.ndarray}
    """
    results_path = Path(results_path)
    with open(results_path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict) or "pred" not in data or "tgt" not in data:
        raise ValueError(
            f"Unexpected benchmark CNN results format in {results_path}. "
            "Expected dict with keys 'pred' and 'tgt'."
        )

    pred = np.asarray(data["pred"])
    tgt = np.asarray(data["tgt"])
    test_f1_micro, test_f1_macro = compute_f1_scores(pred, tgt)

    return ExperimentResult(
        model_name="CNN (benchmark)",
        num_parameters=0,  # Not tracked in stored benchmark results.
        best_val_f1_micro=0.0,  # Validation metrics not available in pickle.
        best_val_f1_macro=0.0,
        test_f1_micro=test_f1_micro,
        test_f1_macro=test_f1_macro,
        test_loss=-1.0,  # Loss not available in stored benchmark results.
        best_epoch=-1,
        training_time_seconds=0.0,  # Training was done outside this run.
        config={"source": "benchmark", "results_path": str(results_path)},
    )


def run_single_experiment(
    encoder_name: str,
    encoder: torch.nn.Module,
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    test_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    num_classes: int,
    embedding_dim: int,
    training_config: TrainingConfig,
    models_dir: str | Path,
    pos_weight: torch.Tensor | None = None,
    class_mask: torch.Tensor | None = None,
    threshold_metric: str = "f1_micro",
) -> ExperimentResult:
    """
    Run a single experiment with given encoder.

    Returns:
        ExperimentResult with all metrics
    """
    import time

    # Create full model
    model = FunctionalGroupClassifier(
        encoder=encoder,
        num_classes=num_classes,
        embedding_dim=embedding_dim,
    )

    num_params = count_parameters(model)
    print(f"\n{'=' * 80}")
    print(f"Training: {encoder_name}")
    print(f"Parameters: {num_params:,}")
    print(f"{'=' * 80}")

    # Train
    start_time = time.time()
    metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        pos_weight=pos_weight,
        class_mask=class_mask,
    )
    training_time = time.time() - start_time

    # Test evaluation
    model.to(training_config.device)
    criterion = torch.nn.BCEWithLogitsLoss()
    val_predictions, val_targets = collect_predictions(
        model, val_loader, training_config.device, class_mask=class_mask
    )
    best_threshold = find_best_threshold(
        val_predictions,
        val_targets,
        metric=threshold_metric,
    )
    test_loss, test_f1_micro, test_f1_macro = evaluate(
        model,
        test_loader,
        criterion,
        training_config.device,
        class_mask=class_mask,
        threshold=best_threshold,
    )

    print(f"\nTest Results for {encoder_name}:")
    print(f"  Best Threshold (val): {best_threshold:.2f}")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test F1 Micro: {test_f1_micro:.4f}")
    print(f"  Test F1 Macro: {test_f1_macro:.4f}")

    # Save trained model
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Create filename-safe model name
    safe_model_name = (
        encoder_name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = models_dir / f"{safe_model_name}_{timestamp}.pt"

    torch.save(model.state_dict(), model_path)
    print(f"  Model saved to: {model_path}")

    # Get best val F1 macro
    best_val_f1_macro = metrics.val_f1_macro[metrics.best_epoch] if metrics.val_f1_macro else 0.0

    return ExperimentResult(
        model_name=encoder_name,
        num_parameters=num_params,
        best_val_f1_micro=metrics.best_val_f1_micro,
        best_val_f1_macro=best_val_f1_macro,
        test_f1_micro=test_f1_micro,
        test_f1_macro=test_f1_macro,
        test_loss=test_loss,
        best_epoch=metrics.best_epoch,
        training_time_seconds=training_time,
        config={"threshold": best_threshold, "threshold_metric": threshold_metric},
    )


def run_full_experiment(config: ExperimentConfig) -> list[ExperimentResult]:
    """
    Run the full experiment comparing MPS vs baselines.

    Returns:
        List of ExperimentResult for all models
    """
    print("=" * 80)
    print("TN Encoder Evaluation Experiment")
    print("=" * 80)
    print(f"Config: {config}")
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

    # Create splits
    train_set, val_set, test_set = create_data_splits(dataset, seed=config.seed)

    # Data loaders
    # Use multiple workers for parallel data loading to avoid CPU bottleneck
    num_workers = 4 if config.batch_size >= 512 else 2
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
    )

    # Class weights
    pos_weight = dataset.get_label_weights()
    class_mask = dataset.get_active_class_mask()
    active_classes = int(class_mask.sum().item())
    print(f"Active classes for training/metrics: {active_classes}/{dataset.num_classes}")
    if config.pos_weight_cap > 0:
        pos_weight = torch.clamp(pos_weight, max=config.pos_weight_cap)
        print(f"Capped positive class weights at {config.pos_weight_cap}")

    # Training config (baseline models)
    baseline_training_config = TrainingConfig(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        batch_size=config.batch_size,
        num_epochs=config.num_epochs,
        patience=config.patience,
        grad_clip_norm=config.grad_clip_norm,
        save_best=False,  # Don't save individual models
    )
    # Training config (MPS models benefit from lower LR)
    mps_training_config = TrainingConfig(
        learning_rate=config.mps_learning_rate,
        weight_decay=config.weight_decay,
        batch_size=config.batch_size,
        num_epochs=config.num_epochs,
        patience=config.patience,
        grad_clip_norm=config.grad_clip_norm,
        save_best=False,  # Don't save individual models
    )

    num_classes = dataset.num_classes
    results: list[ExperimentResult] = []

    # ===== Baseline: CNN (benchmark only) =====
    print("\n" + "=" * 80)
    print("Using benchmark CNN results")
    print("=" * 80)
    print(f"Loading: {config.benchmark_cnn_results_path}")
    cnn_result = load_benchmark_cnn_result(config.benchmark_cnn_results_path)
    print(f"  Test F1 Micro: {cnn_result.test_f1_micro:.4f}")
    print(f"  Test F1 Macro: {cnn_result.test_f1_macro:.4f}")
    results.append(cnn_result)

    # ===== Baseline: MLP =====
    mlp_encoder = MLPEncoder(
        input_length=config.target_length,
        embedding_dim=config.embedding_dim,
    )
    mlp_result = run_single_experiment(
        encoder_name="MLP",
        encoder=mlp_encoder,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_classes=num_classes,
        embedding_dim=config.embedding_dim,
        training_config=baseline_training_config,
        models_dir=config.models_dir,
        pos_weight=pos_weight,
        class_mask=class_mask,
        threshold_metric=config.threshold_metric,
    )
    results.append(mlp_result)

    # ===== MPS: Bond dimension sweep =====
    for bond_dim in config.bond_dims:
        mps_encoder = MPSEncoder(
            input_length=config.target_length,
            num_sites=config.num_sites,
            physical_dim=config.physical_dim,
            bond_dim=bond_dim,
            embedding_dim=config.embedding_dim,
        )
        mps_result = run_single_experiment(
            encoder_name=f"MPS (D={bond_dim})",
            encoder=mps_encoder,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            num_classes=num_classes,
            embedding_dim=config.embedding_dim,
            training_config=mps_training_config,
            models_dir=config.models_dir,
            pos_weight=pos_weight,
            class_mask=class_mask,
            threshold_metric=config.threshold_metric,
        )
        mps_result.config = {
            "num_sites": config.num_sites,
            "physical_dim": config.physical_dim,
            "bond_dim": bond_dim,
        }
        results.append(mps_result)

    return results


def print_results_summary(results: list[ExperimentResult]) -> None:
    """Print a formatted summary of all results."""
    print("\n" + "=" * 100)
    print("RESULTS SUMMARY")
    print("=" * 100)

    # Header
    print(
        f"{'Model':<25} {'Params':>12} {'Val F1μ':>10} {'Val F1M':>10} "
        f"{'Test F1μ':>10} {'Test F1M':>10} {'Time (s)':>10}"
    )
    print("-" * 100)

    # Sort by test F1 micro
    sorted_results = sorted(results, key=lambda x: x.test_f1_micro, reverse=True)

    for r in sorted_results:
        print(
            f"{r.model_name:<25} {r.num_parameters:>12,} {r.best_val_f1_micro:>10.4f} "
            f"{r.best_val_f1_macro:>10.4f} {r.test_f1_micro:>10.4f} {r.test_f1_macro:>10.4f} "
            f"{r.training_time_seconds:>10.1f}"
        )

    print("-" * 100)

    # Best model
    best = sorted_results[0]
    print(f"\nBest model: {best.model_name} with Test F1 Micro = {best.test_f1_micro:.4f}")


def save_results(results: list[ExperimentResult], output_dir: str | Path) -> None:
    """Save results to JSON file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"results_{timestamp}.json"

    # Convert to dict
    results_dict = {
        "timestamp": timestamp,
        "results": [asdict(r) for r in results],
    }

    with open(output_file, "w") as f:
        json.dump(results_dict, f, indent=2)

    print(f"\nResults saved to {output_file}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run TN encoder evaluation experiment")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../../../data/raw",
        help="Path to data directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="models",
        help="Output directory for trained models",
    )
    parser.add_argument(
        "--max_chunks",
        type=int,
        default=None,
        help="Maximum number of data chunks to load (for testing)",
    )
    parser.add_argument(
        "--target_length",
        type=int,
        default=512,
        help="Resample spectra to this length",
    )
    parser.add_argument(
        "--normalization",
        type=str,
        default="zscore",
        choices=["zscore"],
        help="Spectrum normalization method",
    )
    parser.add_argument(
        "--num_sites",
        type=int,
        default=32,
        help="Number of sites for MPS encoder",
    )
    parser.add_argument(
        "--physical_dim",
        type=int,
        default=8,
        help="Physical dimension for MPS encoder",
    )
    parser.add_argument(
        "--bond_dims",
        type=int,
        nargs="+",
        default=[4, 8, 16, 32],
        help="Bond dimensions for MPS sweep",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8192,
        help="Batch size for training",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Maximum number of training epochs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.011,
        help="Learning rate for baseline models",
    )
    parser.add_argument(
        "--mps_learning_rate",
        type=float,
        default=0.011,
        help="Learning rate for MPS models",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience",
    )
    parser.add_argument(
        "--grad_clip_norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm (set <=0 to disable)",
    )
    parser.add_argument(
        "--pos_weight_cap",
        type=float,
        default=20.0,
        help="Maximum positive class weight for BCE loss (set <=0 to disable capping)",
    )
    parser.add_argument(
        "--threshold_metric",
        type=str,
        default="f1_micro",
        choices=["f1_micro", "f1_macro"],
        help="Validation metric used to tune decision threshold",
    )
    parser.add_argument(
        "--benchmark_cnn_results_path",
        type=str,
        default="../../../benchmark/cnn/models/ir/results.pickle",
        help="Path to benchmark CNN results pickle",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    config = ExperimentConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        models_dir=args.models_dir,
        max_chunks=args.max_chunks,
        target_length=args.target_length,
        normalization=args.normalization,
        num_sites=args.num_sites,
        physical_dim=args.physical_dim,
        bond_dims=args.bond_dims,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        mps_learning_rate=args.mps_learning_rate,
        patience=args.patience,
        grad_clip_norm=float(args.grad_clip_norm) if args.grad_clip_norm > 0 else None,
        pos_weight_cap=args.pos_weight_cap,
        threshold_metric=args.threshold_metric,
        benchmark_cnn_results_path=args.benchmark_cnn_results_path,
        seed=args.seed,
    )

    # Run experiment
    results = run_full_experiment(config)

    # Print summary
    print_results_summary(results)

    # Save results
    save_results(results, config.output_dir)


if __name__ == "__main__":
    main()
