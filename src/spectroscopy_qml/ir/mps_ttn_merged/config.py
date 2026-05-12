from dataclasses import dataclass


@dataclass
class FusionConfig:
    """Compatibility container for legacy merged-model settings."""

    input_dim: int = 1800
    num_classes: int = 37
    fusion_weight_init: float = 0.5
    fusion_weight_trainable: bool = True


@dataclass
class DataConfig:
    """Shared dataset scope for both branches."""

    data_dir: str = "data/raw"
    input_dim: int = 1800
    num_classes: int = 37
    max_files: int | None = None


@dataclass
class SplitConfig:
    """Shared split settings used by both branches."""

    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    num_workers: int = 8
    device: str = "auto"
    seed: int = 42


@dataclass
class MPSDataConfig:
    """MPS preprocessing defaults matching the standalone trainer."""

    apply_snv: bool = True
    apply_savgol: bool = True
    savgol_window_length: int = 25
    savgol_polyorder: int = 4


@dataclass
class TTNDataConfig:
    """TTN preprocessing defaults matching experiment 10.2."""

    apply_snv: bool = True
    apply_savgol: bool = False
    savgol_window_length: int = 25
    savgol_polyorder: int = 4


@dataclass
class MPSBranchConfig:
    """Architecture configuration for the MPS branch."""

    num_sites: int = 5
    physical_dim: int = 450
    bond_dim: int = 128
    dropout_rate: float = 0.49
    classifier_head: str = "mps"
    num_sites_2: int = 10
    physical_dim_2: int = 450
    bond_dim_2: int = 128


@dataclass
class TTNBranchConfig:
    """Architecture configuration for the TTN experiment 10.2 branch."""

    chi: int = 64
    segment_window_size: int = 48
    segment_stride: int = 43
    segment_mode: str = "overlap"
    segment_offset: int | None = None
    segment_state_normalize: bool = True
    merge_mode: str = "relaxed"
    merge_residual_weight: float = 0.1
    merge_renormalize_output: bool = True
    lorentz_gamma: float = 3.0
    lorentz_kernel_half_width: int = 15
    lorentz_norm_mode: str = "percentile"


@dataclass
class MPSTrainingConfig:
    """Training defaults that follow the standalone MPS trainer."""

    batch_size: int = 2048
    epochs: int = 200
    learning_rate: float = 2.5e-4
    weight_decay: float = 1e-6
    lr_scheduler_factor: float = 0.7
    lr_scheduler_patience: int = 10
    lr_scheduler_min_lr: float = 1e-6
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    grad_clip_norm: float = 1.0
    amp: bool = True
    compile: bool = False
    threshold_mode: str = "per_class"
    threshold_target_metric: str = "per_class_f1"
    threshold_grid_step: float = 0.02


@dataclass
class TTNTrainingConfig:
    """Training defaults that follow the standalone TTN experiment 10.2 trainer."""

    batch_size: int = 1024
    epochs: int = 200
    learning_rate: float = 3e-4
    weight_decay: float = 1e-6
    lr_scheduler_factor: float = 0.9
    lr_scheduler_patience: int = 5
    lr_scheduler_min_lr: float = 1e-6
    threshold_mode: str = "per_class"
    threshold_target_metric: str = "per_class_f1"
    threshold_grid_step: float = 0.02
    early_stopping_metric: str = "blended_f1"
    early_stopping_blend_alpha: float = 0.5
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-4
    min_epochs_before_stopping: int = 30
    grad_clip_norm: float = 1.0
    amp: bool = True
    compile: bool = True
    pos_weight_power: float = 0.5
    pos_weight_max: float | None = None
    loss_type: str = "bce"
    focal_gamma: float = 2.0


@dataclass
class PathConfig:
    """Output paths for the orchestration run."""

    output_dir: str = "src/spectroscopy_qml/ir/mps_ttn_merged/results"
    selector_model_path: str = "src/spectroscopy_qml/ir/mps_ttn_merged/results/mps_ttn_selector_best.pt"
    mps_model_path: str = "src/spectroscopy_qml/ir/mps_ttn_merged/results/mps_best.pt"
    ttn_model_path: str = "src/spectroscopy_qml/ir/mps_ttn_merged/results/ttn_best.pt"
    summary_path: str = "src/spectroscopy_qml/ir/mps_ttn_merged/results/summary.txt"
    selector_threshold_path: str = "src/spectroscopy_qml/ir/mps_ttn_merged/results/selected_thresholds.json"
    prediction_artifact_path: str = "src/spectroscopy_qml/ir/mps_ttn_merged/results/branch_predictions.npz"
    mps_log_path: str = "src/spectroscopy_qml/ir/mps_ttn_merged/results/mps_training_log.csv"
    ttn_log_path: str = "src/spectroscopy_qml/ir/mps_ttn_merged/results/ttn_training_log.csv"


FUSION_CONFIG = FusionConfig()
DATA_CONFIG = DataConfig()
SPLIT_CONFIG = SplitConfig()
MPS_DATA_CONFIG = MPSDataConfig()
TTN_DATA_CONFIG = TTNDataConfig()
MPS_CONFIG = MPSBranchConfig()
TTN_CONFIG = TTNBranchConfig()
MPS_TRAINING_CONFIG = MPSTrainingConfig()
TTN_TRAINING_CONFIG = TTNTrainingConfig()
PATH_CONFIG = PathConfig()