from dataclasses import dataclass

from spectroscopy_qml.ir.mps_classifier.config import PATH_CONFIG as MPS_PATH_CONFIG


@dataclass
class HybridPathConfig:
    """Filesystem locations for the frozen-MPS + QCNN specialist experiment."""

    data_dir: str = MPS_PATH_CONFIG.data_dir
    mps_checkpoint_path: str = MPS_PATH_CONFIG.best_model_path
    model_dir: str = "src/spectroscopy_qml/ir/mps_qcnn_hybrid/models"
    results_dir: str = "src/spectroscopy_qml/ir/mps_qcnn_hybrid/results"
    latent_cache_path: str = "src/spectroscopy_qml/ir/mps_qcnn_hybrid/results/latent_cache.npz"
    specialist_checkpoint_path: str = (
        "src/spectroscopy_qml/ir/mps_qcnn_hybrid/models/qcnn_specialist_best.pt"
    )
    metrics_path: str = "src/spectroscopy_qml/ir/mps_qcnn_hybrid/results/metrics_comparison.json"
    per_class_metrics_path: str = (
        "src/spectroscopy_qml/ir/mps_qcnn_hybrid/results/per_class_metrics.csv"
    )
    specialist_delta_metrics_path: str = (
        "src/spectroscopy_qml/ir/mps_qcnn_hybrid/results/specialist_label_deltas.csv"
    )
    specialist_labels_path: str = (
        "src/spectroscopy_qml/ir/mps_qcnn_hybrid/results/selected_specialist_labels.csv"
    )
    summary_path: str = "src/spectroscopy_qml/ir/mps_qcnn_hybrid/results/summary.txt"


@dataclass
class SpecialistSelectionConfig:
    """Controls how rare or hard labels are assigned to the specialist head."""

    max_labels: int = 12
    min_train_positives: int = 20
    max_train_prevalence: float = 0.05
    max_val_f1: float = 0.70
    combined_score_weight_prevalence: float = 0.5
    combined_score_weight_val_f1: float = 0.5
    use_frequency_rule: bool = True
    use_validation_f1_rule: bool = True


@dataclass
class QCNNConfig:
    """Architecture of the lightweight QCNN specialist head."""

    compressed_dim: int = 8
    num_qubits: int = 4
    circuit_layers: int = 2
    hidden_dim: int = 64
    dropout_rate: float = 0.1


@dataclass
class HybridTrainingConfig:
    """Training settings for the QCNN specialist only."""

    random_seed: int = 42
    extractor_device: str = "cuda"
    specialist_device: str = "cpu"
    latent_batch_size: int = 1024
    specialist_batch_size: int = 256
    num_epochs: int = 60
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 12
    min_delta: float = 1e-4
    loss_type: str = "weighted_bce"
    focal_gamma: float = 2.0
    global_threshold: float = 0.5
    merge_strategies: tuple[str, ...] = ("replace", "blend")
    blend_alpha: float = 0.5
    num_workers: int = 0


HYBRID_PATH_CONFIG = HybridPathConfig()
SELECTION_CONFIG = SpecialistSelectionConfig()
QCNN_CONFIG = QCNNConfig()
TRAINING_CONFIG = HybridTrainingConfig()
