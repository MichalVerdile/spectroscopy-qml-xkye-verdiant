"""
Configuration for MPS Functional Group Classifier training and evaluation.
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    input_dim: int = 1800
    num_sites: int = 9
    physical_dim: int = 150
    bond_dim: int = 128
    num_classes: int = 37
    dropout_rate: float = 0.05


@dataclass
class DataConfig:
    """Data preprocessing configuration."""

    modality: str = "ir"  # "ir", "nmr", "msms"
    input_column: str | None = None
    normalization_method: str | None = None  # "snv", "quantile", "pqn", "none"

    # Legacy IR flag kept for backward compatibility
    apply_snv: bool = True
    target_length: int = 1800
    max_files: int | None = 1


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    batch_size: int = 256
    num_epochs: int = 200
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    patience: int = 20
    min_delta: float = 1e-4

    # Data loading optimization
    num_workers: int = 4
    pin_memory: bool = True

    # Mixed precision training for speed
    use_amp: bool = True

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
    model_dir: str = "src/spectroscopy_qml/mps_encoder/models"
    results_dir: str = "src/spectroscopy_qml/mps_encoder/results"

    # Model checkpoint
    best_model_path: str = "src/spectroscopy_qml/mps_encoder/models/mps_model_best.pt"

    # Results
    summary_path: str = "src/spectroscopy_qml/mps_encoder/results/summary.txt"
    training_log_path: str = "src/spectroscopy_qml/mps_encoder/results/training_log.csv"


# Global configurations
MODEL_CONFIG = ModelConfig()
DATA_CONFIG = DataConfig()
TRAINING_CONFIG = TrainingConfig()
PATH_CONFIG = PathConfig()
