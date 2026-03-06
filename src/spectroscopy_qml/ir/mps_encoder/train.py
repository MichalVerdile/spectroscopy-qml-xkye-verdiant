"""
Training script for MPS Functional Group Classifier.
"""

import csv
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from spectroscopy_qml.ir.mps_encoder.config import (
    DATA_CONFIG,
    MODEL_CONFIG,
    PATH_CONFIG,
    TRAINING_CONFIG,
)
from spectroscopy_qml.ir.mps_encoder.data_loader import load_ir_data, prepare_dataloaders
from spectroscopy_qml.ir.mps_encoder.model import MPSFunctionalGroupClassifier


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""

    def __init__(self, patience: int = 15, min_delta: float = 1e-4, verbose: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

        return self.early_stop


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Compute evaluation metrics.

    Args:
        y_true: Ground truth labels (n_samples, n_classes)
        y_pred: Predicted labels (n_samples, n_classes)

    Returns:
        Dictionary of metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_micro": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_micro": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }
    return metrics


def train_epoch(
    model: nn.Module, dataloader, criterion, optimizer, device: torch.device
) -> tuple[float, dict[str, float]]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    all_labels = []
    all_preds = []

    for spectra, labels in dataloader:
        spectra = spectra.to(device)
        labels = labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        logits = model(spectra)
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * spectra.size(0)

        # Collect predictions for metrics
        preds = (torch.sigmoid(logits) > 0.5).float()
        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)

    # Compute metrics
    all_labels = np.vstack(all_labels)
    all_preds = np.vstack(all_preds)
    metrics = compute_metrics(all_labels, all_preds)

    return avg_loss, metrics


def validate(
    model: nn.Module, dataloader, criterion, device: torch.device
) -> tuple[float, dict[str, float]]:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for spectra, labels in dataloader:
            spectra = spectra.to(device)
            labels = labels.to(device)

            # Forward pass
            logits = model(spectra)
            loss = criterion(logits, labels)

            total_loss += loss.item() * spectra.size(0)

            # Collect predictions for metrics
            preds = (torch.sigmoid(logits) > 0.5).float()
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)

    # Compute metrics
    all_labels = np.vstack(all_labels)
    all_preds = np.vstack(all_preds)
    metrics = compute_metrics(all_labels, all_preds)

    return avg_loss, metrics


def train_model():
    """Main training function."""
    print("=" * 80)
    print("MPS Functional Group Classifier Training")
    print("=" * 80)

    # Set random seeds for reproducibility
    torch.manual_seed(TRAINING_CONFIG.random_seed)
    np.random.seed(TRAINING_CONFIG.random_seed)

    # Determine device
    if torch.cuda.is_available() and TRAINING_CONFIG.device == "cuda":
        device = torch.device("cuda")
        print(f"Using device: CUDA ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    # Create output directories
    model_dir = Path(PATH_CONFIG.model_dir)
    results_dir = Path(PATH_CONFIG.results_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n" + "=" * 80)
    print("Loading Data")
    print("=" * 80)

    data_dir = Path(PATH_CONFIG.data_dir)
    if not data_dir.exists():
        # Fall back to raw data if processed doesn't exist
        project_root = Path(__file__).parents[4]
        data_dir = project_root / "data" / "raw"
        print(f"Processed data not found, using raw data from: {data_dir}")

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

    # Initialize model
    print("\n" + "=" * 80)
    print("Model Architecture")
    print("=" * 80)

    model = MPSFunctionalGroupClassifier(
        input_dim=MODEL_CONFIG.input_dim,
        num_sites=MODEL_CONFIG.num_sites,
        physical_dim=MODEL_CONFIG.physical_dim,
        bond_dim=MODEL_CONFIG.bond_dim,
        num_classes=MODEL_CONFIG.num_classes,
        dropout_rate=MODEL_CONFIG.dropout_rate,
    )
    model = model.to(device)

    print(f"Model parameters: {model.get_num_parameters():,}")
    print("\nModel configuration:")
    print(f"  Input dimension: {MODEL_CONFIG.input_dim}")
    print(f"  Number of sites: {MODEL_CONFIG.num_sites}")
    print(f"  Site dimension: {MODEL_CONFIG.input_dim // MODEL_CONFIG.num_sites}")
    print(f"  Physical dimension: {MODEL_CONFIG.physical_dim}")
    print(f"  Bond dimension: {MODEL_CONFIG.bond_dim}")
    print(f"  Number of classes: {MODEL_CONFIG.num_classes}")

    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(
        model.parameters(),
        lr=TRAINING_CONFIG.learning_rate,
        weight_decay=TRAINING_CONFIG.weight_decay,
    )

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=TRAINING_CONFIG.lr_scheduler_factor,
        patience=TRAINING_CONFIG.lr_scheduler_patience,
        min_lr=TRAINING_CONFIG.lr_scheduler_min_lr,
        verbose=True,
    )

    # Early stopping
    early_stopping = EarlyStopping(
        patience=TRAINING_CONFIG.patience, min_delta=TRAINING_CONFIG.min_delta, verbose=True
    )

    # Training loop
    print("\n" + "=" * 80)
    print("Training")
    print("=" * 80)

    best_val_loss = float("inf")
    training_log = []

    # Create CSV log file
    log_file = Path(PATH_CONFIG.training_log_path)
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_f1_micro",
                "train_f1_macro",
                "val_loss",
                "val_f1_micro",
                "val_f1_macro",
                "lr",
            ]
        )

    start_time = time.time()

    for epoch in range(TRAINING_CONFIG.num_epochs):
        epoch_start = time.time()

        # Train
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, device)

        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # Log progress
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch+1}/{TRAINING_CONFIG.num_epochs} ({epoch_time:.1f}s)")
        print(
            f"  Train Loss: {train_loss:.4f} | F1 Micro: {train_metrics['f1_micro']:.4f} | F1 Macro: {train_metrics['f1_macro']:.4f}"
        )
        print(
            f"  Val Loss:   {val_loss:.4f} | F1 Micro: {val_metrics['f1_micro']:.4f} | F1 Macro: {val_metrics['f1_macro']:.4f}"
        )
        print(f"  LR: {current_lr:.2e}")

        # Save training log
        log_entry = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_f1_micro": train_metrics["f1_micro"],
            "train_f1_macro": train_metrics["f1_macro"],
            "val_loss": val_loss,
            "val_f1_micro": val_metrics["f1_micro"],
            "val_f1_macro": val_metrics["f1_macro"],
            "lr": current_lr,
        }
        training_log.append(log_entry)

        # Append to CSV
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch + 1,
                    train_loss,
                    train_metrics["f1_micro"],
                    train_metrics["f1_macro"],
                    val_loss,
                    val_metrics["f1_micro"],
                    val_metrics["f1_macro"],
                    current_lr,
                ]
            )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_metrics": val_metrics,
                    "config": MODEL_CONFIG,
                },
                PATH_CONFIG.best_model_path,
            )
            print(f"  ✓ Best model saved (val_loss: {val_loss:.4f})")

        # Early stopping check
        if early_stopping(val_loss):
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.1f} minutes")

    # Evaluate on test set
    print("\n" + "=" * 80)
    print("Final Evaluation on Test Set")
    print("=" * 80)

    # Load best model
    checkpoint = torch.load(PATH_CONFIG.best_model_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_metrics = validate(model, test_loader, criterion, device)

    print("\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  F1 Micro: {test_metrics['f1_micro']:.4f}")
    print(f"  F1 Macro: {test_metrics['f1_macro']:.4f}")
    print(f"  F1 Weighted: {test_metrics['f1_weighted']:.4f}")
    print(f"  Precision Micro: {test_metrics['precision_micro']:.4f}")
    print(f"  Precision Macro: {test_metrics['precision_macro']:.4f}")
    print(f"  Recall Micro: {test_metrics['recall_micro']:.4f}")
    print(f"  Recall Macro: {test_metrics['recall_macro']:.4f}")

    # Save summary
    summary_path = Path(PATH_CONFIG.summary_path)
    with open(summary_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("MPS Functional Group Classifier - Training Summary\n")
        f.write("=" * 80 + "\n\n")

        f.write("Model Configuration:\n")
        f.write(f"  Input dimension: {MODEL_CONFIG.input_dim}\n")
        f.write(f"  Number of sites: {MODEL_CONFIG.num_sites}\n")
        f.write(f"  Site dimension: {MODEL_CONFIG.input_dim // MODEL_CONFIG.num_sites}\n")
        f.write(f"  Physical dimension: {MODEL_CONFIG.physical_dim}\n")
        f.write(f"  Bond dimension: {MODEL_CONFIG.bond_dim}\n")
        f.write(f"  Number of classes: {MODEL_CONFIG.num_classes}\n")
        f.write(f"  Dropout rate: {MODEL_CONFIG.dropout_rate}\n")
        f.write(f"  Total parameters: {model.get_num_parameters():,}\n\n")

        f.write("Training Configuration:\n")
        f.write(f"  Batch size: {TRAINING_CONFIG.batch_size}\n")
        f.write(f"  Epochs trained: {len(training_log)}\n")
        f.write(f"  Learning rate: {TRAINING_CONFIG.learning_rate}\n")
        f.write(f"  Weight decay: {TRAINING_CONFIG.weight_decay}\n")
        f.write(f"  Training time: {total_time/60:.1f} minutes\n\n")

        f.write("Best Validation Results:\n")
        f.write(f"  Epoch: {checkpoint['epoch']}\n")
        f.write(f"  Loss: {checkpoint['val_loss']:.4f}\n")
        f.write(f"  F1 Micro: {checkpoint['val_metrics']['f1_micro']:.4f}\n")
        f.write(f"  F1 Macro: {checkpoint['val_metrics']['f1_macro']:.4f}\n\n")

        f.write("Test Set Results:\n")
        f.write(f"  Loss: {test_loss:.4f}\n")
        f.write(f"  Accuracy: {test_metrics['accuracy']:.4f}\n")
        f.write(f"  F1 Micro: {test_metrics['f1_micro']:.4f}\n")
        f.write(f"  F1 Macro: {test_metrics['f1_macro']:.4f}\n")
        f.write(f"  F1 Weighted: {test_metrics['f1_weighted']:.4f}\n")
        f.write(f"  Precision Micro: {test_metrics['precision_micro']:.4f}\n")
        f.write(f"  Precision Macro: {test_metrics['precision_macro']:.4f}\n")
        f.write(f"  Recall Micro: {test_metrics['recall_micro']:.4f}\n")
        f.write(f"  Recall Macro: {test_metrics['recall_macro']:.4f}\n")

    print(f"\nSummary saved to: {summary_path}")
    print(f"Best model saved to: {PATH_CONFIG.best_model_path}")
    print(f"Training log saved to: {PATH_CONFIG.training_log_path}")

    # Note about .keras format
    print("\n" + "=" * 80)
    print("Note: Model saved as .pt (PyTorch format)")
    print("For TensorFlow/Keras compatibility, consider using ONNX export")
    print("=" * 80)


if __name__ == "__main__":
    train_model()
