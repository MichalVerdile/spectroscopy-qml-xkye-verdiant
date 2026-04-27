import csv
import gc
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from spectroscopy_qml.ir.mps_encoder_without_feature_map.config import (
    DATA_CONFIG,
    MODEL_CONFIG,
    PATH_CONFIG,
    TRAINING_CONFIG,
)
from spectroscopy_qml.ir.mps_encoder_without_feature_map.data_loader import load_ir_data, prepare_dataloaders
from spectroscopy_qml.ir.mps_encoder_without_feature_map.model import MPSFunctionalGroupClassifier


class EarlyStopping:
    """Early stopping to stop training when validation metric doesn't improve."""

    def __init__(
        self, patience: int = 15, min_delta: float = 1e-4, mode: str = "max", verbose: bool = True
    ):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: "min" for loss (lower is better) or "max" for metrics (higher is better)
            verbose: Print progress messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def _is_improvement(self, score: float) -> bool:
        """Check if score is an improvement over best score."""
        if self.mode == "min":
            return score < self.best_score - self.min_delta  # type: ignore[operator]
        else:  # mode == "max"
            return score > self.best_score + self.min_delta  # type: ignore[operator]


def _resolve_training_device(requested_device: str) -> torch.device:
    """Resolve device and fail fast on incompatible CUDA runtime/build."""
    if requested_device != "cuda":
        print("Using device: CPU")
        return torch.device("cpu")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False. "
            "Check NVIDIA driver, CUDA runtime, and PyTorch CUDA build."
        )

    try:
        # Smoke test a real CUDA kernel to catch 'no kernel image' early.
        _ = (torch.tensor([1.0], device="cuda") * 2.0).item()
        torch.cuda.synchronize()
    except Exception as exc:
        torch_ver = torch.__version__
        torch_cuda = torch.version.cuda
        dev_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        raise RuntimeError(
            "CUDA initialization failed for this PyTorch build. "
            f"GPU='{dev_name}', capability={capability}, torch={torch_ver}, "
            f"torch_cuda={torch_cuda}. Original error: {exc}\n"
            "Likely cause: incompatible PyTorch wheel for this GPU architecture (sm_120).\n"
            "Install a PyTorch build that explicitly supports Blackwell sm_120 "
            "(typically the latest stable or nightly with CUDA 12.8+)."
        ) from exc

    print(f"Using device: CUDA ({torch.cuda.get_device_name(0)})")
    return torch.device("cuda")


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


def compute_pos_weight(y_train: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Compute pos_weight for BCEWithLogitsLoss to handle class imbalance.

    pos_weight[i] = (num_negatives[i] / num_positives[i])
    This upweights the loss for rare positive classes.

    Args:
        y_train: Training labels (n_samples, n_classes)
        device: Device to create tensor on

    Returns:
        Tensor of pos_weight values (n_classes,)
    """
    num_samples = y_train.shape[0]
    num_positives = y_train.sum(axis=0)  # Sum along samples
    num_negatives = num_samples - num_positives

    # Avoid division by zero for classes with no positives
    # Use 1.0 as default weight if no positives
    pos_weight = np.where(num_positives > 0, num_negatives / num_positives, 1.0)

    return torch.FloatTensor(pos_weight).to(device)


def tune_thresholds(
    y_true: np.ndarray, y_probs: np.ndarray, metric: str = "f1_micro"
) -> np.ndarray:
    """
    Find optimal per-class thresholds that maximize the given metric.

    Uses a simple grid search over thresholds for each class independently.
    For micro F1, we optimize all thresholds jointly using a coarse grid.

    Args:
        y_true: Ground truth labels (n_samples, n_classes)
        y_probs: Predicted probabilities (n_samples, n_classes)
        metric: Metric to optimize ("f1_micro" or "f1_macro")

    Returns:
        Array of optimal thresholds (n_classes,)
    """
    n_classes = y_true.shape[1]

    if metric == "f1_micro":
        # For micro F1, try global thresholds (same for all classes)
        # This is more efficient and often works well for micro averaging
        best_score = 0.0
        best_threshold = 0.5

        for threshold in np.arange(0.1, 0.9, 0.01):
            y_pred = (y_probs >= threshold).astype(int)
            score = f1_score(y_true, y_pred, average="micro", zero_division=0)
            if score > best_score:
                best_score = score
                best_threshold = threshold

        # Use same threshold for all classes (micro F1 optimization)
        thresholds = np.full(n_classes, best_threshold)

    else:
        # For macro F1 or per-class optimization, tune each class independently
        thresholds = np.zeros(n_classes)

        for i in range(n_classes):
            best_score = 0.0
            best_threshold = 0.5

            for threshold in np.arange(0.1, 0.9, 0.05):
                y_pred_i = (y_probs[:, i] >= threshold).astype(int)
                score = f1_score(y_true[:, i], y_pred_i, zero_division=0)
                if score > best_score:
                    best_score = score
                    best_threshold = threshold

            thresholds[i] = best_threshold

    return thresholds


def train_epoch(
    model: nn.Module,
    dataloader,
    criterion,
    optimizer,
    device: torch.device,
    thresholds: np.ndarray,
    scaler=None,
    use_amp: bool = False,
) -> tuple[float, dict[str, float]]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    all_labels = []
    all_preds = []

    for spectra, labels in dataloader:
        spectra = spectra.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Mixed precision training
        if use_amp and scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = model(spectra)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training
            logits = model(spectra)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * spectra.size(0)

        # Collect predictions for metrics using tuned thresholds
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        preds = (probs >= thresholds).astype(float)
        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds)

    avg_loss = total_loss / len(dataloader.dataset)

    # Compute metrics
    all_labels = np.vstack(all_labels)
    all_preds = np.vstack(all_preds)
    metrics = compute_metrics(all_labels, all_preds)

    return avg_loss, metrics


def validate(
    model: nn.Module,
    dataloader,
    criterion,
    device: torch.device,
    thresholds: np.ndarray | None = None,
    use_amp: bool = False,
    return_probs: bool = False,
) -> tuple[float, dict[str, float]] | tuple[float, dict[str, float], np.ndarray, np.ndarray]:
    """Validate the model.

    Args:
        model: Model to validate
        dataloader: Validation dataloader
        criterion: Loss criterion
        device: Device to run on
        thresholds: Per-class thresholds for prediction (if None, use 0.5 for all)
        use_amp: Use automatic mixed precision
        return_probs: If True, also return labels and probabilities for threshold tuning

    Returns:
        avg_loss, metrics (and optionally all_labels, all_probs)
    """
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs = []

    # Use default threshold of 0.5 if not provided
    if thresholds is None:
        thresholds = np.full(model.num_classes, 0.5)

    with torch.no_grad():
        for spectra, labels in dataloader:
            spectra = spectra.to(device)
            labels = labels.to(device)

            # Mixed precision inference
            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits = model(spectra)
                    loss = criterion(logits, labels)
            else:
                logits = model(spectra)
                loss = criterion(logits, labels)

            total_loss += loss.item() * spectra.size(0)

            # Collect probabilities and labels
            probs = torch.sigmoid(logits).cpu().numpy()
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs)

    avg_loss = total_loss / len(dataloader.dataset)

    # Concatenate all batches
    all_labels = np.vstack(all_labels)
    all_probs = np.vstack(all_probs)

    # Apply thresholds to get predictions
    all_preds = (all_probs >= thresholds).astype(float)
    metrics = compute_metrics(all_labels, all_preds)

    if return_probs:
        return avg_loss, metrics, all_labels, all_probs
    else:
        return avg_loss, metrics


def train_model(X: np.ndarray | None = None, y: np.ndarray | None = None):
    """Main training function."""
    print("=" * 80)
    print("MPS Functional Group Classifier Training")
    print("=" * 80)

    # Set random seeds for reproducibility
    torch.manual_seed(TRAINING_CONFIG.random_seed)
    np.random.seed(TRAINING_CONFIG.random_seed)

    # Determine device and validate CUDA kernel compatibility upfront.
    device = _resolve_training_device(TRAINING_CONFIG.device)

    # Create output directories
    model_dir = Path(PATH_CONFIG.model_dir)
    results_dir = Path(PATH_CONFIG.results_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load data once unless preloaded arrays are provided
    print("\n" + "=" * 80)
    print("Loading Data")
    print("=" * 80)

    if X is None or y is None:
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
    else:
        print("Using preloaded dataset passed to train_model()")

    # Prepare dataloaders
    train_loader, val_loader, test_loader = prepare_dataloaders(
        X,
        y,
        batch_size=TRAINING_CONFIG.batch_size,
        train_ratio=TRAINING_CONFIG.train_ratio,
        val_ratio=TRAINING_CONFIG.val_ratio,
        test_ratio=TRAINING_CONFIG.test_ratio,
        random_seed=TRAINING_CONFIG.random_seed,
        num_workers=TRAINING_CONFIG.num_workers,
        pin_memory=TRAINING_CONFIG.pin_memory,
    )

    print("\nDataLoader optimization:")
    print(f"  Batch size: {TRAINING_CONFIG.batch_size}")
    print(f"  Num workers: {TRAINING_CONFIG.num_workers}")
    print(f"  Pin memory: {TRAINING_CONFIG.pin_memory}")
    print(f"  Mixed precision (AMP): {TRAINING_CONFIG.use_amp}")

    # Initialize model
    print("\n" + "=" * 80)
    print("Model Architecture")
    print("=" * 80)

    model = MPSFunctionalGroupClassifier(
        input_dim=MODEL_CONFIG.input_dim,
        num_sites=MODEL_CONFIG.num_sites,
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
    print(f"  MPS physical dimension: {model.physical_dim}")
    print(f"  Bond dimension: {MODEL_CONFIG.bond_dim}")
    print(f"  Number of classes: {MODEL_CONFIG.num_classes}")

    # Compute class weights from training set for imbalance handling
    print("\n" + "=" * 80)
    print("Class Imbalance Handling")
    print("=" * 80)

    # Get training labels to compute pos_weight
    train_labels = []
    for _, labels in train_loader:
        train_labels.append(labels.numpy())
    train_labels = np.vstack(train_labels)

    pos_weight = compute_pos_weight(train_labels, device)
    print("\nClass imbalance weights (pos_weight):")
    print(
        f"  Min: {pos_weight.min():.2f}, Max: {pos_weight.max():.2f}, Mean: {pos_weight.mean():.2f}"
    )
    print(
        f"  Classes with high imbalance (weight > 10): {(pos_weight > 10).sum()}/{len(pos_weight)}"
    )

    # Loss function with class weights and optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
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
    )

    # Early stopping - monitor validation micro F1 (higher is better)
    early_stopping = EarlyStopping(
        patience=TRAINING_CONFIG.patience,
        min_delta=TRAINING_CONFIG.min_delta,
        mode="max",  # Maximize micro F1
        verbose=True,
    )

    # Mixed precision scaler
    scaler = None
    if TRAINING_CONFIG.use_amp and device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()
        print("\nUsing Automatic Mixed Precision (AMP) for faster training")

    # Training loop
    print("\n" + "=" * 80)
    print("Training")
    print("=" * 80)

    best_val_f1 = 0.0  # Track best validation micro F1
    best_thresholds = np.full(MODEL_CONFIG.num_classes, 0.5)  # Initialize with 0.5
    best_val_loss = float("inf")  # Store best validation loss
    best_val_metrics = None  # Store best validation metrics
    best_epoch = 0  # Track best epoch
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
                "threshold_mean",
                "threshold_std",
                "lr",
            ]
        )

    start_time = time.time()

    for epoch in range(TRAINING_CONFIG.num_epochs):
        epoch_start = time.time()

        # Train with current thresholds (for metrics only; loss uses logits)
        train_loss, train_metrics = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            best_thresholds,
            scaler,
            TRAINING_CONFIG.use_amp,
        )

        # Validate and get probabilities for threshold tuning
        val_loss, val_metrics_default, val_labels, val_probs = validate(
            model,
            val_loader,
            criterion,
            device,
            thresholds=None,
            use_amp=TRAINING_CONFIG.use_amp,
            return_probs=True,
        )

        # Tune thresholds on validation set to maximize micro F1
        tuned_thresholds = tune_thresholds(val_labels, val_probs, metric="f1_micro")

        # Recompute validation metrics with tuned thresholds
        val_preds_tuned = (val_probs >= tuned_thresholds).astype(float)
        val_metrics = compute_metrics(val_labels, val_preds_tuned)

        # Update learning rate based on validation loss
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
        print(f"  Thresholds: mean={tuned_thresholds.mean():.3f}, std={tuned_thresholds.std():.3f}")
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
            "threshold_mean": tuned_thresholds.mean(),
            "threshold_std": tuned_thresholds.std(),
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
                    tuned_thresholds.mean(),
                    tuned_thresholds.std(),
                    current_lr,
                ]
            )

        # Track best model based on validation micro F1 (not loss)
        if val_metrics["f1_micro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_micro"]
            best_thresholds = tuned_thresholds  # Update best thresholds
            best_val_loss = val_loss
            best_val_metrics = val_metrics.copy()
            best_epoch = epoch + 1
            # Save directly to disk – no in-memory copy kept to save RAM
            torch.save(
                {
                    "epoch": best_epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "val_metrics": best_val_metrics,
                    "thresholds": best_thresholds,
                    "config": MODEL_CONFIG,
                },
                PATH_CONFIG.best_model_path,
            )
            print(f"  ✓ New best model saved to disk (val_f1_micro: {val_metrics['f1_micro']:.4f})")

        # Early stopping check - monitor validation micro F1
        if early_stopping(val_metrics["f1_micro"]):
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.1f} minutes")

    print(f"\n✓ Best model already saved to disk from epoch {best_epoch}")
    print(f"  Validation F1 Micro: {best_val_metrics['f1_micro']:.4f}")
    print(f"  Validation F1 Macro: {best_val_metrics['f1_macro']:.4f}")

    # Evaluate on test set
    print("\n" + "=" * 80)
    print("Final Evaluation on Test Set")
    print("=" * 80)

    # Load best model (weights_only=False needed for PyTorch 2.6+ compatibility with custom classes)
    checkpoint = torch.load(PATH_CONFIG.best_model_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load best thresholds from checkpoint
    best_thresholds = checkpoint.get("thresholds", np.full(MODEL_CONFIG.num_classes, 0.5))
    print(f"\nUsing tuned thresholds from best model (mean: {best_thresholds.mean():.3f})")

    test_loss, test_metrics = validate(
        model,
        test_loader,
        criterion,
        device,
        thresholds=best_thresholds,
        use_amp=TRAINING_CONFIG.use_amp,
    )

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
        f.write(f"  MPS physical dimension: {model.physical_dim}\n")
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
        f.write(f"  F1 Macro: {checkpoint['val_metrics']['f1_macro']:.4f}\n")
        f.write(f"  Threshold mean: {checkpoint['thresholds'].mean():.3f}\n")
        f.write(f"  Threshold std: {checkpoint['thresholds'].std():.3f}\n\n")

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

    # ------------------------------------------------------------------ #
    # Explicit memory cleanup so grid search runs don't accumulate RAM /  #
    # GPU memory across consecutive training runs.                        #
    # ------------------------------------------------------------------ #
    return_val = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_metrics": best_val_metrics,
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "best_model_path": str(PATH_CONFIG.best_model_path),
        "summary_path": str(summary_path),
        "training_log_path": str(log_file),
        "num_epochs_trained": len(training_log),
    }

    del model, optimizer, criterion, scheduler
    del train_loader, val_loader, test_loader
    if scaler is not None:
        del scaler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return return_val


if __name__ == "__main__":
    train_model()
