"""
Training script for MPS Functional Group Classifier.
"""

import csv
import gc
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold, train_test_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from spectroscopy_qml.hnmr.mps_classifier_hnmr.config import (
    DATA_CONFIG,
    MODEL_CONFIG,
    PATH_CONFIG,
    TRAINING_CONFIG,
)
from spectroscopy_qml.hnmr.mps_classifier_hnmr.data_loader import IRSpectraDataset, load_ir_data
from spectroscopy_qml.hnmr.mps_classifier_hnmr.model import MPSFunctionalGroupClassifier


def _get_model_config_kwargs(model_config) -> dict[str, object]:
    """Normalize saved model config into classifier constructor kwargs."""
    if is_dataclass(model_config):
        return asdict(model_config)
    if isinstance(model_config, dict):
        return model_config.copy()
    raise TypeError(f"Unsupported model config type in checkpoint: {type(model_config)!r}")


def _build_model_from_checkpoint(checkpoint: dict, device: torch.device) -> nn.Module:
    """Recreate the classifier using the architecture stored in the checkpoint."""
    model_config = checkpoint.get("config", MODEL_CONFIG)
    model = MPSFunctionalGroupClassifier(**_get_model_config_kwargs(model_config))
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device)


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
    num_positives = y_train.sum(axis=0, dtype=np.float32)
    num_negatives = num_samples - num_positives

    # Avoid division by zero for classes with no positives without triggering a warning.
    pos_weight = np.ones_like(num_positives, dtype=np.float32)
    np.divide(
        num_negatives,
        num_positives,
        out=pos_weight,
        where=num_positives > 0,
    )

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
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training
            logits = model(spectra)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
            if use_amp and device.type == "cuda":
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


def _create_model(device: torch.device) -> MPSFunctionalGroupClassifier:
    """Create a classifier instance from the active model config."""
    model = MPSFunctionalGroupClassifier(
        input_dim=MODEL_CONFIG.input_dim,
        num_sites=MODEL_CONFIG.num_sites,
        physical_dim=MODEL_CONFIG.physical_dim,
        bond_dim=MODEL_CONFIG.bond_dim,
        num_classes=MODEL_CONFIG.num_classes,
        dropout_rate=MODEL_CONFIG.dropout_rate,
        classifier_head=MODEL_CONFIG.classifier_head,
        num_sites_2=MODEL_CONFIG.num_sites_2,
        physical_dim_2=MODEL_CONFIG.physical_dim_2,
        bond_dim_2=MODEL_CONFIG.bond_dim_2,
    )
    return model.to(device)


def _create_dataloader(dataset: IRSpectraDataset, shuffle: bool) -> DataLoader:
    """Create a dataloader using the configured performance settings."""
    return DataLoader(
        dataset,
        batch_size=TRAINING_CONFIG.batch_size,
        shuffle=shuffle,
        num_workers=TRAINING_CONFIG.num_workers,
        pin_memory=TRAINING_CONFIG.pin_memory,
        persistent_workers=True if TRAINING_CONFIG.num_workers > 0 else False,
    )


def _aggregate_fold_metrics(fold_results: list[dict]) -> tuple[dict[str, float], dict[str, float]]:
    """Compute mean and std for validation metrics across folds."""
    metric_names = fold_results[0]["val_metrics"].keys()
    metric_means = {
        metric: float(np.mean([result["val_metrics"][metric] for result in fold_results]))
        for metric in metric_names
    }
    metric_stds = {
        metric: float(np.std([result["val_metrics"][metric] for result in fold_results]))
        for metric in metric_names
    }
    return metric_means, metric_stds


def _build_fold_splits(
    X_trainval: np.ndarray,
    y_trainval: np.ndarray,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], str]:
    """Build either a single holdout split or K-fold splits from config."""
    if TRAINING_CONFIG.num_folds < 1:
        raise ValueError(f"num_folds must be >= 1, got {TRAINING_CONFIG.num_folds}")

    indices = np.arange(len(X_trainval))
    if TRAINING_CONFIG.num_folds == 1:
        val_fraction = TRAINING_CONFIG.val_ratio / (TRAINING_CONFIG.train_ratio + TRAINING_CONFIG.val_ratio)
        train_indices, val_indices = train_test_split(
            indices,
            test_size=val_fraction,
            random_state=TRAINING_CONFIG.random_seed,
            shuffle=True,
        )
        return [(train_indices, val_indices)], "single holdout split"

    if TRAINING_CONFIG.num_folds > len(X_trainval):
        raise ValueError(
            f"num_folds ({TRAINING_CONFIG.num_folds}) cannot exceed the number of train/val samples "
            f"({len(X_trainval)})"
        )

    kfold = KFold(
        n_splits=TRAINING_CONFIG.num_folds,
        shuffle=True,
        random_state=TRAINING_CONFIG.random_seed,
    )
    return list(kfold.split(X_trainval, y_trainval)), f"{TRAINING_CONFIG.num_folds}-fold CV"


def _resolve_parallel_fold_devices(base_device: torch.device, n_folds: int) -> list[torch.device]:
    """Resolve device assignments for concurrent fold training."""
    requested_workers = min(TRAINING_CONFIG.parallel_fold_workers, n_folds)
    if requested_workers <= 1:
        return []

    if base_device.type == "cuda":
        configured_devices = [
            torch.device(device_name) for device_name in TRAINING_CONFIG.parallel_fold_cuda_devices
        ]
        if configured_devices:
            if len(configured_devices) < requested_workers:
                print(
                    "Parallel fold training requested, but fewer CUDA devices were configured "
                    "than workers. Falling back to sequential execution."
                )
                return []
            return configured_devices[:requested_workers]

        available_gpus = torch.cuda.device_count()
        if available_gpus < requested_workers:
            print(
                "Parallel fold training requested, but not enough CUDA devices are available "
                f"({available_gpus} found for {requested_workers} requested workers). "
                "Falling back to sequential execution."
            )
            return []

        return [torch.device(f"cuda:{index}") for index in range(requested_workers)]

    return [torch.device("cpu") for _ in range(requested_workers)]


def _write_training_log(log_file: Path, training_log: list[dict]) -> None:
    """Persist epoch logs after all folds complete."""
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "fold",
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
        for entry in sorted(training_log, key=lambda item: (item["fold"], item["epoch"])):
            writer.writerow(
                [
                    entry["fold"],
                    entry["epoch"],
                    entry["train_loss"],
                    entry["train_f1_micro"],
                    entry["train_f1_macro"],
                    entry["val_loss"],
                    entry["val_f1_micro"],
                    entry["val_f1_macro"],
                    entry["threshold_mean"],
                    entry["threshold_std"],
                    entry["lr"],
                ]
            )


def _train_single_fold(
    fold_idx: int,
    n_folds: int,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    X_trainval: np.ndarray,
    y_trainval: np.ndarray,
    device: torch.device,
    use_amp: bool,
) -> dict:
    """Train one fold and return its best checkpoint payload plus epoch logs."""
    prefix = f"[Fold {fold_idx}/{n_folds}]"

    def log(message: str) -> None:
        print(f"{prefix} {message}")

    fold_start = time.time()
    log("Starting")

    X_train_fold = X_trainval[train_indices]
    y_train_fold = y_trainval[train_indices]
    X_val_fold = X_trainval[val_indices]
    y_val_fold = y_trainval[val_indices]

    fold_pos_weight = compute_pos_weight(y_train_fold, device)
    log("Class imbalance weights (pos_weight):")
    log(
        f"  Min: {fold_pos_weight.min():.2f}, Max: {fold_pos_weight.max():.2f}, Mean: {fold_pos_weight.mean():.2f}"
    )
    log(
        "  Classes with high imbalance (weight > 10): "
        f"{(fold_pos_weight > 10).sum()}/{len(fold_pos_weight)}"
    )

    model = _create_model(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=fold_pos_weight)
    optimizer = Adam(
        model.parameters(),
        lr=TRAINING_CONFIG.learning_rate,
        weight_decay=TRAINING_CONFIG.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=TRAINING_CONFIG.lr_scheduler_factor,
        patience=TRAINING_CONFIG.lr_scheduler_patience,
        min_lr=TRAINING_CONFIG.lr_scheduler_min_lr,
    )
    early_stopping = EarlyStopping(
        patience=TRAINING_CONFIG.patience,
        min_delta=TRAINING_CONFIG.min_delta,
        mode="max",
        verbose=True,
    )
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    train_dataset = IRSpectraDataset(X_train_fold, y_train_fold)
    val_dataset = IRSpectraDataset(X_val_fold, y_val_fold)
    fold_train_loader = _create_dataloader(train_dataset, shuffle=True)
    fold_val_loader = _create_dataloader(val_dataset, shuffle=False)

    fold_best_val_f1 = -1.0
    fold_best_state = None
    fold_best_thresholds = np.full(MODEL_CONFIG.num_classes, 0.5, dtype=np.float32)
    fold_best_val_loss = float("inf")
    fold_best_val_metrics = None
    fold_best_train_loss = 0.0
    fold_best_train_metrics = None
    fold_best_epoch = 0
    fold_epochs_trained = 0
    fold_training_log = []

    try:
        for epoch in range(TRAINING_CONFIG.num_epochs):
            epoch_start = time.time()

            fold_train_loss, fold_train_metrics = train_epoch(
                model,
                fold_train_loader,
                criterion,
                optimizer,
                device,
                fold_best_thresholds,
                scaler,
                use_amp,
            )

            fold_val_loss, fold_val_metrics, fold_val_labels, fold_val_probs = validate(
                model,
                fold_val_loader,
                criterion,
                device,
                thresholds=fold_best_thresholds,
                use_amp=use_amp,
                return_probs=True,
            )

            logged_thresholds = fold_best_thresholds
            logged_val_metrics = fold_val_metrics

            if fold_val_metrics["f1_micro"] > fold_best_val_f1:
                tuned_thresholds = tune_thresholds(
                    fold_val_labels,
                    fold_val_probs,
                    metric="f1_micro",
                )
                tuned_val_preds = (fold_val_probs >= tuned_thresholds).astype(float)
                tuned_val_metrics = compute_metrics(fold_val_labels, tuned_val_preds)
                logged_thresholds = tuned_thresholds
                logged_val_metrics = tuned_val_metrics

                if tuned_val_metrics["f1_micro"] > fold_best_val_f1:
                    fold_best_val_f1 = tuned_val_metrics["f1_micro"]
                    fold_best_state = {
                        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                    }
                    fold_best_thresholds = tuned_thresholds.copy()
                    fold_best_val_loss = fold_val_loss
                    fold_best_val_metrics = tuned_val_metrics.copy()
                    fold_best_train_loss = fold_train_loss
                    fold_best_train_metrics = fold_train_metrics.copy()
                    fold_best_epoch = epoch + 1
                    log(
                        "New fold best checkpoint "
                        f"(epoch {fold_best_epoch}, val_f1_micro: {fold_best_val_f1:.4f})"
                    )

            scheduler.step(fold_val_loss)
            current_lr = optimizer.param_groups[0]["lr"]
            epoch_time = time.time() - epoch_start
            fold_epochs_trained = epoch + 1

            log(
                f"Epoch {epoch + 1}/{TRAINING_CONFIG.num_epochs} ({epoch_time:.1f}s) | "
                f"Train Loss: {fold_train_loss:.4f} F1µ: {fold_train_metrics['f1_micro']:.4f} | "
                f"Val Loss: {fold_val_loss:.4f} F1µ: {logged_val_metrics['f1_micro']:.4f}"
            )
            log(
                f"  Thresholds: mean={logged_thresholds.mean():.3f}, std={logged_thresholds.std():.3f} | "
                f"LR: {current_lr:.2e}"
            )

            fold_training_log.append(
                {
                    "fold": fold_idx,
                    "epoch": epoch + 1,
                    "train_loss": fold_train_loss,
                    "train_f1_micro": fold_train_metrics["f1_micro"],
                    "train_f1_macro": fold_train_metrics["f1_macro"],
                    "val_loss": fold_val_loss,
                    "val_f1_micro": logged_val_metrics["f1_micro"],
                    "val_f1_macro": logged_val_metrics["f1_macro"],
                    "threshold_mean": float(logged_thresholds.mean()),
                    "threshold_std": float(logged_thresholds.std()),
                    "lr": current_lr,
                }
            )

            if early_stopping(logged_val_metrics["f1_micro"]):
                log(f"Early stopping triggered at epoch {epoch + 1}")
                break

        if fold_best_state is None or fold_best_val_metrics is None or fold_best_train_metrics is None:
            raise RuntimeError(f"Fold {fold_idx} did not produce a valid checkpoint.")

        fold_time = time.time() - fold_start
        log(
            f"Summary: best_epoch={fold_best_epoch}, val_loss={fold_best_val_loss:.4f}, "
            f"val_f1_micro={fold_best_val_metrics['f1_micro']:.4f}, "
            f"training_time={fold_time / 60:.1f} minutes"
        )

        return {
            "fold": fold_idx,
            "best_epoch": fold_best_epoch,
            "epochs_trained": fold_epochs_trained,
            "train_loss": fold_best_train_loss,
            "train_metrics": fold_best_train_metrics,
            "val_loss": fold_best_val_loss,
            "val_metrics": fold_best_val_metrics,
            "threshold_mean": float(fold_best_thresholds.mean()),
            "threshold_std": float(fold_best_thresholds.std()),
            "training_time_minutes": fold_time / 60,
            "best_state": fold_best_state,
            "best_thresholds": fold_best_thresholds,
            "pos_weight": fold_pos_weight.detach().cpu(),
            "training_log": fold_training_log,
        }
    finally:
        del model, optimizer, criterion, scheduler
        del fold_train_loader, fold_val_loader, train_dataset, val_dataset
        del fold_pos_weight
        if scaler is not None:
            del scaler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


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
            apply_savgol=DATA_CONFIG.apply_savgol,
            savgol_window_length=DATA_CONFIG.savgol_window_length,
            savgol_polyorder=DATA_CONFIG.savgol_polyorder,
            apply_snv=DATA_CONFIG.apply_snv,
        )
    else:
        print("Using preloaded dataset passed to train_model()")

    # Split off test set first, then use 5-Fold CV on the remaining data
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TRAINING_CONFIG.test_ratio,
        random_state=TRAINING_CONFIG.random_seed, shuffle=True,
    )

    fold_splits, validation_mode = _build_fold_splits(X_trainval, y_trainval)
    n_folds = len(fold_splits)

    # Create fixed test dataloader
    test_dataset = IRSpectraDataset(X_test, y_test)
    test_loader = DataLoader(
        test_dataset, batch_size=TRAINING_CONFIG.batch_size, shuffle=False,
        num_workers=TRAINING_CONFIG.num_workers, pin_memory=TRAINING_CONFIG.pin_memory,
        persistent_workers=True if TRAINING_CONFIG.num_workers > 0 else False,
    )

    print(f"\nData split:")
    print(f"  Train+Val (K-Fold): {len(X_trainval)} samples")
    print(f"  Test:               {len(X_test)} samples ({TRAINING_CONFIG.test_ratio:.0%})")
    print(f"  Validation mode:    {validation_mode}")
    print(f"  Fold splits:        {n_folds}")

    print("\nDataLoader optimization:")
    print(f"  Batch size: {TRAINING_CONFIG.batch_size}")
    print(f"  Num workers: {TRAINING_CONFIG.num_workers}")
    print(f"  Pin memory: {TRAINING_CONFIG.pin_memory}")
    print(f"  Mixed precision (AMP): {TRAINING_CONFIG.use_amp}")
    print(f"  Cross-validation mode: {validation_mode}")

    # Initialize one model instance for architecture reporting.
    print("\n" + "=" * 80)
    print("Model Architecture")
    print("=" * 80)

    model = _create_model(device)

    print(f"Model parameters: {model.get_num_parameters():,}")
    print("\nModel configuration:")
    print(f"  Input dimension: {MODEL_CONFIG.input_dim}")
    print(f"  Classifier head: {MODEL_CONFIG.classifier_head}")
    print(f"  Number of classes: {MODEL_CONFIG.num_classes}")
    print(f"  Number of sites (MPS 1): {MODEL_CONFIG.num_sites}")
    print(f"  Site dimension (MPS 1): {MODEL_CONFIG.input_dim // MODEL_CONFIG.num_sites}")
    print(f"  Physical dimension (MPS 1): {MODEL_CONFIG.physical_dim}")
    print(f"  Bond dimension (MPS 1): {MODEL_CONFIG.bond_dim}")
    print(f"  Number of sites (MPS 2): {MODEL_CONFIG.num_sites_2}")
    print(f"  Site dimension (MPS 2): {MODEL_CONFIG.input_dim // MODEL_CONFIG.num_sites_2}")
    print(f"  Physical dimension (MPS 2): {MODEL_CONFIG.physical_dim_2}")
    print(f"  Bond dimension (MPS 2): {MODEL_CONFIG.bond_dim_2}")
    del model

    # Class weights are computed per fold to avoid leaking validation-label statistics.
    print("\n" + "=" * 80)
    print("Class Imbalance Handling")
    print("=" * 80)
    print("  pos_weight is recomputed separately inside each training fold.")

    use_amp = TRAINING_CONFIG.use_amp and device.type == "cuda"
    if use_amp:
        print("\nUsing Automatic Mixed Precision (AMP) for faster training")

    # Training loop
    print("\n" + "=" * 80)
    print("Training")
    print("=" * 80)

    best_val_f1 = -1.0
    best_val_loss = float("inf")
    best_val_metrics = None
    best_epoch = 0
    best_fold = 0
    training_log = []
    fold_results = []

    log_file = Path(PATH_CONFIG.training_log_path)
    start_time = time.time()

    parallel_devices = _resolve_parallel_fold_devices(device, n_folds)
    if parallel_devices:
        print(
            "Running folds in parallel on devices: "
            + ", ".join(str(fold_device) for fold_device in parallel_devices)
        )
        with ThreadPoolExecutor(max_workers=len(parallel_devices)) as executor:
            futures = []
            for fold_idx, (train_indices, val_indices) in enumerate(fold_splits, start=1):
                fold_device = parallel_devices[(fold_idx - 1) % len(parallel_devices)]
                futures.append(
                    executor.submit(
                        _train_single_fold,
                        fold_idx,
                        n_folds,
                        train_indices,
                        val_indices,
                        X_trainval,
                        y_trainval,
                        fold_device,
                        use_amp and fold_device.type == "cuda",
                    )
                )

            for future in as_completed(futures):
                fold_result = future.result()
                training_log.extend(fold_result.pop("training_log"))
                fold_results.append(fold_result)
    else:
        print("Running folds sequentially")
        for fold_idx, (train_indices, val_indices) in enumerate(fold_splits, start=1):
            fold_result = _train_single_fold(
                fold_idx,
                n_folds,
                train_indices,
                val_indices,
                X_trainval,
                y_trainval,
                device,
                use_amp,
            )
            training_log.extend(fold_result.pop("training_log"))
            fold_results.append(fold_result)

    fold_results.sort(key=lambda result: result["fold"])
    _write_training_log(log_file, training_log)

    best_fold_result = max(fold_results, key=lambda result: result["val_metrics"]["f1_micro"])
    best_val_f1 = best_fold_result["val_metrics"]["f1_micro"]
    best_val_loss = best_fold_result["val_loss"]
    best_val_metrics = best_fold_result["val_metrics"].copy()
    best_epoch = best_fold_result["best_epoch"]
    best_fold = best_fold_result["fold"]
    torch.save(
        {
            "epoch": best_epoch,
            "best_fold": best_fold,
            "model_state_dict": best_fold_result["best_state"],
            "val_loss": best_val_loss,
            "val_metrics": best_val_metrics,
            "thresholds": best_fold_result["best_thresholds"],
            "pos_weight": best_fold_result["pos_weight"],
            "cv_num_folds": n_folds,
            "config": MODEL_CONFIG,
        },
        PATH_CONFIG.best_model_path,
    )
    print(
        "\n✓ Best model saved to disk "
        f"(fold {best_fold}, epoch {best_epoch}, val_f1_micro: {best_val_f1:.4f})"
    )

    for fold_result in fold_results:
        if fold_result is not best_fold_result:
            del fold_result["best_state"]

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.1f} minutes")

    cv_val_metrics, cv_val_metric_stds = _aggregate_fold_metrics(fold_results)
    cv_val_loss_mean = float(np.mean([result["val_loss"] for result in fold_results]))
    cv_val_loss_std = float(np.std([result["val_loss"] for result in fold_results]))

    print("\nCross-validation summary:")
    print(f"  Val Loss: {cv_val_loss_mean:.4f} ± {cv_val_loss_std:.4f}")
    print(f"  Val F1 Micro: {cv_val_metrics['f1_micro']:.4f} ± {cv_val_metric_stds['f1_micro']:.4f}")
    print(f"  Val F1 Macro: {cv_val_metrics['f1_macro']:.4f} ± {cv_val_metric_stds['f1_macro']:.4f}")

    print(f"\n✓ Best model already saved to disk from fold {best_fold}, epoch {best_epoch}")
    print(f"  Validation F1 Micro: {best_val_metrics['f1_micro']:.4f}")
    print(f"  Validation F1 Macro: {best_val_metrics['f1_macro']:.4f}")

    # Evaluate on test set
    print("\n" + "=" * 80)
    print("Final Evaluation on Test Set")
    print("=" * 80)

    # Load best model (weights_only=False needed for PyTorch 2.6+ compatibility with custom classes)
    checkpoint = torch.load(PATH_CONFIG.best_model_path, weights_only=False)
    model = _build_model_from_checkpoint(checkpoint, device)

    # Load best thresholds from checkpoint
    num_checkpoint_classes = model.num_classes
    best_thresholds = checkpoint.get("thresholds", np.full(num_checkpoint_classes, 0.5))
    checkpoint_pos_weight = checkpoint.get("pos_weight")
    if checkpoint_pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.as_tensor(checkpoint_pos_weight, dtype=torch.float32, device=device)
        )
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=compute_pos_weight(y_trainval, device))
    print(f"\nUsing tuned thresholds from best model (mean: {best_thresholds.mean():.3f})")

    test_loss, test_metrics = validate(
        model,
        test_loader,
        criterion,
        device,
        thresholds=best_thresholds,
        use_amp=use_amp,
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
        f.write(f"  Classifier head: {MODEL_CONFIG.classifier_head}\n")
        f.write(f"  Number of classes: {MODEL_CONFIG.num_classes}\n")
        f.write(f"  Dropout rate: {MODEL_CONFIG.dropout_rate}\n")
        f.write(f"  Total parameters: {model.get_num_parameters():,}\n\n")
        f.write(f"  Number of sites (MPS 1): {MODEL_CONFIG.num_sites}\n")
        f.write(f"  Site dimension (MPS 1): {MODEL_CONFIG.input_dim // MODEL_CONFIG.num_sites}\n")
        f.write(f"  Physical dimension (MPS 1): {MODEL_CONFIG.physical_dim}\n")
        f.write(f"  Bond dimension (MPS 1): {MODEL_CONFIG.bond_dim}\n")
        f.write(f"  Number of sites (MPS 2): {MODEL_CONFIG.num_sites_2}\n")
        f.write(f"  Site dimension (MPS 2): {MODEL_CONFIG.input_dim // MODEL_CONFIG.num_sites_2}\n")
        f.write(f"  Physical dimension (MPS 2): {MODEL_CONFIG.physical_dim_2}\n")
        f.write(f"  Bond dimension (MPS 2): {MODEL_CONFIG.bond_dim_2}\n\n")

        f.write("Training Configuration:\n")
        f.write(f"  Batch size: {TRAINING_CONFIG.batch_size}\n")
        f.write(f"  Folds trained: {n_folds}\n")
        f.write(f"  Total epochs trained across folds: {sum(result['epochs_trained'] for result in fold_results)}\n")
        f.write(f"  Learning rate: {TRAINING_CONFIG.learning_rate}\n")
        f.write(f"  Weight decay: {TRAINING_CONFIG.weight_decay}\n")
        f.write(f"  Training time: {total_time/60:.1f} minutes\n\n")

        f.write("Cross-Validation Results (mean ± std over folds):\n")
        f.write(f"  Val Loss: {cv_val_loss_mean:.4f} ± {cv_val_loss_std:.4f}\n")
        for metric_name, metric_value in cv_val_metrics.items():
            f.write(
                f"  {metric_name}: {metric_value:.4f} ± {cv_val_metric_stds[metric_name]:.4f}\n"
            )
        f.write("\n")

        f.write("Best Fold Checkpoint:\n")
        f.write(f"  Fold: {checkpoint['best_fold']}\n")
        f.write(f"  Epoch: {checkpoint['epoch']}\n")
        f.write(f"  Loss: {checkpoint['val_loss']:.4f}\n")
        f.write(f"  F1 Micro: {checkpoint['val_metrics']['f1_micro']:.4f}\n")
        f.write(f"  F1 Macro: {checkpoint['val_metrics']['f1_macro']:.4f}\n")
        f.write(f"  Threshold mean: {checkpoint['thresholds'].mean():.3f}\n")
        f.write(f"  Threshold std: {checkpoint['thresholds'].std():.3f}\n\n")

        f.write("Per-Fold Best Validation Results:\n")
        for result in fold_results:
            f.write(
                f"  Fold {result['fold']}: epoch={result['best_epoch']}, "
                f"val_loss={result['val_loss']:.4f}, "
                f"f1_micro={result['val_metrics']['f1_micro']:.4f}, "
                f"f1_macro={result['val_metrics']['f1_macro']:.4f}\n"
            )
        f.write("\n")

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
        "best_fold": best_fold,
        "best_val_loss": best_val_loss,
        "best_val_metrics": best_val_metrics,
        "cv_val_loss_mean": cv_val_loss_mean,
        "cv_val_loss_std": cv_val_loss_std,
        "cv_val_metrics": cv_val_metrics,
        "cv_val_metrics_std": cv_val_metric_stds,
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "best_model_path": str(PATH_CONFIG.best_model_path),
        "summary_path": str(summary_path),
        "training_log_path": str(log_file),
        "num_epochs_trained": sum(result["epochs_trained"] for result in fold_results),
    }

    del model, criterion
    del test_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return return_val


if __name__ == "__main__":
    train_model()
