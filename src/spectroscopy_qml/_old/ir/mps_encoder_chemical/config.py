from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    input_dim: int = 1800
    num_sites: int = 5
    physical_dim: int = 450
    bond_dim: int = 128
    num_classes: int = 37
    dropout_rate: float = 0.0


@dataclass
class DataConfig:
    """Data preprocessing configuration."""

    apply_snv: bool = True
    target_length: int = 1800
    max_files: int | None = None


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    batch_size: int = 2048
    num_epochs: int = 250
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    patience: int = 25
    min_delta: float = 1e-4

    # Data loading optimization
    num_workers: int = 8
    pin_memory: bool = True

    # Mixed precision training for speed
    use_amp: bool = True

    # Learning rate scheduler
    lr_scheduler_factor: float = 0.7
    lr_scheduler_patience: int = 20
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
    model_dir: str = "src/spectroscopy_qml/ir/mps_encoder_chemical/models"
    results_dir: str = "src/spectroscopy_qml/ir/mps_encoder_chemical/results"

    # Model checkpoint
    best_model_path: str = "src/spectroscopy_qml/ir/mps_encoder_chemical/models/mps_model_best.pt"

    # Results
    summary_path: str = "src/spectroscopy_qml/ir/mps_encoder_chemical/results/summary.txt"
    training_log_path: str = "src/spectroscopy_qml/ir/mps_encoder_chemical/results/training_log.csv"


@dataclass
class GridSearchConfig:
    """Exhaustive grid search configuration.

    Every value list below is used in a full Cartesian product, so all
    combinations are executed.
    """

    # Model architecture search space
    num_sites_values: tuple[int, ...] = (25, 36, 45)
    physical_dim_values: tuple[int, ...] = (360, 450, 600)
    bond_dim_values: tuple[int, ...] = (256, 512, 768)
    dropout_rate_values: tuple[float, ...] = (0.0,)

    # Training hyperparameter search space
    batch_size_values: tuple[int, ...] = (2048,)
    learning_rate_values: tuple[float, ...] = (1e-3,)
    weight_decay_values: tuple[float, ...] = (1e-6,)
    num_epochs_values: tuple[int, ...] = (1,)
    patience_values: tuple[int, ...] = (20,)
    lr_scheduler_factor_values: tuple[float, ...] = (0.7,)
    lr_scheduler_patience_values: tuple[int, ...] = (3,)

    # Selection criterion for best configuration
    optimize_metric: str = "val_f1_micro"
    maximize_metric: bool = True

    # Output
    grid_search_results_csv: str = (
        "src/spectroscopy_qml/ir/mps_encoder_chemical/results/grid_search/grid_search_results.csv"
    )
    grid_search_summary_path: str = (
        "src/spectroscopy_qml/ir/mps_encoder_chemical/results/grid_search/grid_search_summary.txt"
    )


# Global configurations
MODEL_CONFIG = ModelConfig()
DATA_CONFIG = DataConfig()
TRAINING_CONFIG = TrainingConfig()
PATH_CONFIG = PathConfig()
GRID_SEARCH_CONFIG = GridSearchConfig()
