"""
Training utilities for the TN evaluation experiment.

Includes:
- Training loop with validation
- Metrics computation (F1 micro/macro)
- Early stopping
- Model checkpoint saving
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Optimization
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    optimizer: str = "adamw"  # "adamw" or "adam"
    batch_size: int = 64
    num_epochs: int = 100
    grad_clip_norm: float | None = None

    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4

    # Checkpointing
    save_dir: str | Path | None = None
    save_best: bool = True

    # Hardware
    device: str = "auto"

    # Logging
    log_interval: int = 10

    def __post_init__(self) -> None:
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        if self.save_dir is not None:
            self.save_dir = Path(self.save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class TrainingMetrics:
    """Metrics tracked during training."""

    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    train_f1_micro: list[float] = field(default_factory=list)
    train_f1_macro: list[float] = field(default_factory=list)
    val_f1_micro: list[float] = field(default_factory=list)
    val_f1_macro: list[float] = field(default_factory=list)
    best_val_f1_micro: float = 0.0
    best_epoch: int = 0


class EarlyStopping:
    """Early stopping handler."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = "max"):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for metrics like F1
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value: float | None = None
        self.should_stop = False

    def __call__(self, value: float) -> bool:
        """
        Check if training should stop.

        Args:
            value: Current metric value

        Returns:
            True if training should stop
        """
        if self.best_value is None:
            self.best_value = value
            return False

        if self.mode == "max":
            improved = value > self.best_value + self.min_delta
        else:
            improved = value < self.best_value - self.min_delta

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


def compute_f1_scores(
    predictions: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
    ignore_empty_classes: bool = True,
) -> tuple[float, float]:
    """
    Compute F1 micro and macro scores.

    Args:
        predictions: Model predictions (probabilities)
        targets: Ground truth labels
        threshold: Classification threshold

    Returns:
        Tuple of (f1_micro, f1_macro)
    """
    if ignore_empty_classes and targets.ndim == 2 and targets.shape[1] > 0:
        active = targets.sum(axis=0) > 0
        if np.any(active):
            predictions = predictions[:, active]
            targets = targets[:, active]

    # Binarize predictions
    pred_binary = (predictions > threshold).astype(int)
    target_int = targets.astype(int)

    # Handle edge cases
    if pred_binary.sum() == 0 and target_int.sum() == 0:
        return 1.0, 1.0
    if pred_binary.sum() == 0 or target_int.sum() == 0:
        return 0.0, 0.0

    f1_micro = f1_score(target_int, pred_binary, average="micro", zero_division=0)
    f1_macro = f1_score(target_int, pred_binary, average="macro", zero_division=0)

    return float(f1_micro), float(f1_macro)


def collect_predictions(
    model: nn.Module,
    dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: str,
    class_mask: torch.Tensor | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Collect probability predictions and targets for threshold tuning/evaluation.
    """
    with torch.no_grad():
        model.eval()
        all_predictions: list[np.ndarray] = []
        all_targets: list[np.ndarray] = []

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)
            if class_mask is not None:
                logits = logits[:, class_mask]
                batch_y = batch_y[:, class_mask]

            probs = torch.sigmoid(logits)
            all_predictions.append(probs.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())

        return np.concatenate(all_predictions, axis=0), np.concatenate(all_targets, axis=0)


def find_best_threshold(
    predictions: np.ndarray,
    targets: np.ndarray,
    metric: str = "f1_micro",
    thresholds: np.ndarray | None = None,
) -> float:
    """
    Find a global probability threshold that maximizes validation F1.
    """
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 17)

    best_threshold = 0.5
    best_score = -1.0

    for threshold in thresholds:
        f1_micro, f1_macro = compute_f1_scores(predictions, targets, threshold=float(threshold))
        score = f1_micro if metric == "f1_micro" else f1_macro
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    grad_clip_norm: float | None = None,
    class_mask: torch.Tensor | None = None,
) -> tuple[float, float, float]:
    """
    Train for one epoch.

    Returns:
        Tuple of (average_loss, f1_micro, f1_macro)
    """
    model.train()
    total_loss = 0.0
    all_predictions: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        if class_mask is not None:
            logits_for_loss = logits[:, class_mask]
            batch_y_for_loss = batch_y[:, class_mask]
        else:
            logits_for_loss = logits
            batch_y_for_loss = batch_y

        loss = criterion(logits_for_loss, batch_y_for_loss)
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)

        # Collect predictions for metrics
        with torch.no_grad():
            probs = torch.sigmoid(logits_for_loss)
            all_predictions.append(probs.cpu().numpy())
            all_targets.append(batch_y_for_loss.cpu().numpy())

    # Compute metrics
    all_predictions_arr = np.concatenate(all_predictions, axis=0)
    all_targets_arr = np.concatenate(all_targets, axis=0)
    f1_micro, f1_macro = compute_f1_scores(all_predictions_arr, all_targets_arr)

    avg_loss = total_loss / float(len(dataloader.dataset))

    return avg_loss, f1_micro, f1_macro


def evaluate(
    model: nn.Module,
    dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    device: str,
    class_mask: torch.Tensor | None = None,
    threshold: float = 0.5,
) -> tuple[float, float, float]:
    """
    Evaluate model on a dataset.

    Returns:
        Tuple of (average_loss, f1_micro, f1_macro)
    """
    with torch.no_grad():
        model.eval()
        total_loss = 0.0
        all_predictions: list[np.ndarray] = []
        all_targets: list[np.ndarray] = []

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)
            if class_mask is not None:
                logits_for_loss = logits[:, class_mask]
                batch_y_for_loss = batch_y[:, class_mask]
            else:
                logits_for_loss = logits
                batch_y_for_loss = batch_y

            loss = criterion(logits_for_loss, batch_y_for_loss)

            total_loss += loss.item() * batch_x.size(0)

            probs = torch.sigmoid(logits_for_loss)
            all_predictions.append(probs.cpu().numpy())
            all_targets.append(batch_y_for_loss.cpu().numpy())

        # Compute metrics
        all_predictions_arr = np.concatenate(all_predictions, axis=0)
        all_targets_arr = np.concatenate(all_targets, axis=0)
        f1_micro, f1_macro = compute_f1_scores(
            all_predictions_arr, all_targets_arr, threshold=threshold
        )

        avg_loss = total_loss / float(len(dataloader.dataset))

        return avg_loss, f1_micro, f1_macro


def train_model(
    model: nn.Module,
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    config: TrainingConfig,
    pos_weight: torch.Tensor | None = None,
    class_mask: torch.Tensor | None = None,
) -> TrainingMetrics:
    """
    Full training loop with validation and early stopping.

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        pos_weight: Optional positive class weights for BCEWithLogitsLoss

    Returns:
        Training metrics
    """
    device = config.device
    model = model.to(device)
    class_mask_device = class_mask.to(device) if class_mask is not None else None

    # Loss function
    if pos_weight is not None:
        if class_mask_device is not None:
            pos_weight = pos_weight[class_mask]
        pos_weight = pos_weight.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer
    optimizer: torch.optim.Optimizer
    if config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.patience,
        min_delta=config.min_delta,
        mode="max",
    )

    # Metrics tracking
    metrics = TrainingMetrics()
    best_model_state: dict[str, Any] | None = None

    print(f"Training on {device}")
    print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")
    print("-" * 80)

    for epoch in range(config.num_epochs):
        # Train
        train_loss, train_f1_micro, train_f1_macro = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            grad_clip_norm=config.grad_clip_norm,
            class_mask=class_mask_device,
        )

        # Validate
        val_loss, val_f1_micro, val_f1_macro = evaluate(
            model, val_loader, criterion, device, class_mask=class_mask_device
        )

        # Record metrics
        metrics.train_losses.append(train_loss)
        metrics.val_losses.append(val_loss)
        metrics.train_f1_micro.append(train_f1_micro)
        metrics.train_f1_macro.append(train_f1_macro)
        metrics.val_f1_micro.append(val_f1_micro)
        metrics.val_f1_macro.append(val_f1_macro)

        # Update best model
        if val_f1_micro > metrics.best_val_f1_micro:
            metrics.best_val_f1_micro = val_f1_micro
            metrics.best_epoch = epoch
            if config.save_best:
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Logging
        lr = optimizer.param_groups[0]["lr"]
        is_best = "★" if val_f1_micro > metrics.best_val_f1_micro else " "
        print(
            f"Epoch {epoch + 1:3d}/{config.num_epochs} {is_best} | "
            f"LR: {lr:.2e} | "
            f"Train Loss: {train_loss:.4f} F1: {train_f1_micro:.4f}/{train_f1_macro:.4f} | "
            f"Val Loss: {val_loss:.4f} F1: {val_f1_micro:.4f}/{val_f1_macro:.4f}"
        )

        # Learning rate scheduling
        scheduler.step(val_f1_micro)

        # Early stopping
        if early_stopping(val_f1_micro):
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Save best model
    if config.save_dir is not None and best_model_state is not None:
        save_path = Path(config.save_dir) / "best_model.pt"
        torch.save(best_model_state, save_path)
        print(f"Saved best model to {save_path}")

    print("-" * 80)
    print(
        f"Best validation F1 (micro): {metrics.best_val_f1_micro:.4f} at epoch {metrics.best_epoch + 1}"
    )

    return metrics


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
