from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    input_dim: int = 1800
    num_sites: int = 5
    physical_dim: int = 450
    bond_dim: int = 128
    num_classes: int = 37
    dropout_rate: float = 0.49

    # Classifier head: "cnn" (MPS+CNN hybrid) or "mps" (pure MPS classifier)
    classifier_head: str = "mps"

    # Second MPS encoder (fine-grained, e.g. 1800 sites)
    num_sites_2: int = 10
    physical_dim_2: int = 450
    bond_dim_2: int = 128

    # Quantum segment branch (third branch)
    use_quantum_branch: bool = True
    quantum_backend: str = "pennylane"          # "pennylane" or "qiskit"
    quantum_num_qubits: int = 3                  # number of qubits per circuit
    quantum_num_layers: int = 3                  # number of circuit layers
    quantum_segment_length: int = 9             # must equal quantum_num_qubits * quantum_num_layers
    quantum_segment_stride: int = 2             # < segment_length for overlap
    quantum_measure_correlations: bool = True   # also measure <ZiZj> correlations


@dataclass
class DataConfig:
    """Data preprocessing configuration."""

    apply_snv: bool = True
    apply_savgol: bool = True
    savgol_window_length: int = 25
    savgol_polyorder: int = 4
    target_length: int = 1800
    max_files: int | None = None


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    batch_size: int = 2048
    num_epochs: int = 200
    learning_rate: float = 2.5e-4
    weight_decay: float = 1e-6
    patience: int = 10
    min_delta: float = 1e-4

    # Data loading optimization
    num_workers: int = 8
    pin_memory: bool = True

    # Mixed precision training for speed
    use_amp: bool = True

    # Cross-validation execution
    parallel_fold_workers: int = 5
    parallel_fold_cuda_devices: tuple[str, ...] = ()

    # Learning rate scheduler
    lr_scheduler_factor: float = 0.7
    lr_scheduler_patience: int = 10
    lr_scheduler_min_lr: float = 1e-6

    # Data split
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    num_folds: int = 1
    random_seed: int = 42

    # Quantum branch loss weight (competitive loss coefficient)
    quantum_loss_weight: float = 0.5

    # Device
    device: str = "cuda"  # Will be set to "cpu" if CUDA not available


@dataclass
class PathConfig:
    """File paths for data and outputs."""

    # Data paths
    data_dir: str = "data/raw"

    # Output paths
    model_dir: str = "src/spectroscopy_qml/ir/quantum_mps_classifier/models1"
    results_dir: str = "src/spectroscopy_qml/ir/quantum_mps_classifier/results1"

    # Model checkpoint
    best_model_path: str = "src/spectroscopy_qml/ir/quantum_mps_classifier/models1/mps_model_best.pt"

    # Results
    summary_path: str = "src/spectroscopy_qml/ir/quantum_mps_classifier/results1/summary.txt"
    training_log_path: str = "src/spectroscopy_qml/ir/quantum_mps_classifier/results1/training_log.csv"


@dataclass
class ThresholdConfig:
    """Threshold tuning configuration.

    mode:
        - "global"    – a single threshold is swept and applied to all classes.
        - "per_class" – an independent threshold is selected for each class.

    target_metric:
        Metric maximised during the sweep on the validation set.
        - "f1_micro"     – global micro-F1  (only valid with mode="global")
        - "f1_macro"     – global macro-F1  (valid for both modes)
        - "per_class_f1" – per-class F1     (only valid with mode="per_class")

    grid_step:
        Step size for the threshold candidate grid in [0, 1].

    thresholds_filename:
        Name of the JSON file written to the run/model directory that stores
        the selected thresholds (mirrors ``selected_thresholds.json`` in
        experiment10_2).
    """

    mode: str = "per_class"                # "global" | "per_class"
    target_metric: str = "per_class_f1"    # "f1_micro" | "f1_macro" | "per_class_f1"
    grid_step: float = 0.02
    thresholds_filename: str = "selected_thresholds.json"


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
        "src/spectroscopy_qml/ir/quantum_mps_classifier/results/grid_search/grid_search_results.csv"
    )
    grid_search_summary_path: str = (
        "src/spectroscopy_qml/ir/quantum_mps_classifier/results/grid_search/grid_search_summary.txt"
    )


# Global configurations
MODEL_CONFIG = ModelConfig()
DATA_CONFIG = DataConfig()
TRAINING_CONFIG = TrainingConfig()
PATH_CONFIG = PathConfig()
THRESHOLD_CONFIG = ThresholdConfig()
GRID_SEARCH_CONFIG = GridSearchConfig()
