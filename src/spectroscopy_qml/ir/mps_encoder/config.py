"""
Configuration for MPS Functional Group Classifier training and evaluation.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    input_dim: int = 1800
    num_sites: int = 36
    physical_dim: int = 8
    bond_dim: int = 16
    num_classes: int = 37
    dropout_rate: float = 0.2


@dataclass
class DataConfig:
    """Data preprocessing configuration."""

    apply_snv: bool = False  # Apply Standard Normal Variate normalization
    target_length: int = 1800  # Target spectrum length after interpolation
    max_files: int | None = None  # Maximum number of files to load (for testing)


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    batch_size: int = 64
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 15  # Early stopping patience
    min_delta: float = 1e-4  # Minimum change for early stopping

    # Learning rate scheduler
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 5
    lr_scheduler_min_lr: float = 1e-6

    # Data split
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    random_seed: int = 42

    # Device
    device: str = "cuda"  # Will be set to "cpu" if CUDA not available


@dataclass
class PathConfig:
    """File paths for data and outputs."""

    # Data paths
    data_dir: str = "data/raw"

    # Output paths
    model_dir: str = "src/spectroscopy_qml/ir/mps_encoder/models"
    results_dir: str = "src/spectroscopy_qml/ir/mps_encoder/results"

    # Model checkpoint
    best_model_path: str = "src/spectroscopy_qml/ir/mps_encoder/models/mps_model_best.pt"

    # Results
    summary_path: str = "src/spectroscopy_qml/ir/mps_encoder/results/summary.txt"
    training_log_path: str = "src/spectroscopy_qml/ir/mps_encoder/results/training_log.csv"


# Global configurations
MODEL_CONFIG = ModelConfig()
DATA_CONFIG = DataConfig()
TRAINING_CONFIG = TrainingConfig()
PATH_CONFIG = PathConfig()
