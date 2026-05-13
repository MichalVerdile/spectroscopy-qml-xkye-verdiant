from __future__ import annotations

import json
import random
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import KFold, train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from spectroscopy_qml.ir.mps_classifier.config import (
    DATA_CONFIG as MPS_DATA_CONFIG,
    MODEL_CONFIG as MPS_MODEL_CONFIG,
    TRAINING_CONFIG as MPS_TRAINING_CONFIG,
)
from spectroscopy_qml.ir.mps_classifier.data_loader import FUNCTIONAL_GROUPS, load_ir_data
from spectroscopy_qml.ir.mps_qcnn_hybrid.config import (
    HYBRID_PATH_CONFIG,
    QCNN_CONFIG,
    SELECTION_CONFIG,
    TRAINING_CONFIG,
    HybridPathConfig,
    HybridTrainingConfig,
    QCNNConfig,
    SpecialistSelectionConfig,
)
from spectroscopy_qml.ir.mps_qcnn_hybrid.frozen_mps_model import MPSFunctionalGroupClassifier
from spectroscopy_qml.ir.mps_qcnn_hybrid.model import LatentQCNNHead

LABEL_NAMES = list(FUNCTIONAL_GROUPS.keys())


class EarlyStopping:
    """Early stop on validation specialist-label macro-F1 at fixed threshold."""

    def __init__(self, patience: int, min_delta: float):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score: float | None = None
        self.counter = 0

    def step(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


class FocalLossWithLogits(nn.Module):
    """Multi-label focal loss with optional positive-class weighting."""

    def __init__(self, gamma: float = 2.0, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.clone().detach())
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
            pos_weight=self.pos_weight,
        )
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        return ((1.0 - pt) ** self.gamma * bce).mean()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _get_model_config_kwargs(model_config) -> dict[str, object]:
    if is_dataclass(model_config):
        return asdict(model_config)
    if isinstance(model_config, dict):
        return model_config.copy()
    raise TypeError(f"Unsupported model config type: {type(model_config)!r}")


def _resolve_device(requested_device: str) -> torch.device:
    if requested_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_frozen_mps_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[MPSFunctionalGroupClassifier, dict]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Trained MPS checkpoint not found at {checkpoint_path}. "
            "Point the hybrid experiment to an existing trained checkpoint first."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint.get("config", MPS_MODEL_CONFIG)
    model = MPSFunctionalGroupClassifier(**_get_model_config_kwargs(model_config))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint


def _reconstruct_split_indices(num_samples: int, checkpoint: dict) -> dict[str, np.ndarray]:
    dataset_indices = np.arange(num_samples)
    trainval_indices, test_indices = train_test_split(
        dataset_indices,
        test_size=MPS_TRAINING_CONFIG.test_ratio,
        random_state=MPS_TRAINING_CONFIG.random_seed,
        shuffle=True,
    )

    best_fold = int(checkpoint.get("best_fold", 1))
    num_folds = int(checkpoint.get("cv_num_folds", MPS_TRAINING_CONFIG.num_folds))
    relative_trainval_indices = np.arange(len(trainval_indices))

    if num_folds == 1:
        val_fraction = MPS_TRAINING_CONFIG.val_ratio / (
            MPS_TRAINING_CONFIG.train_ratio + MPS_TRAINING_CONFIG.val_ratio
        )
        train_relative_indices, val_relative_indices = train_test_split(
            relative_trainval_indices,
            test_size=val_fraction,
            random_state=MPS_TRAINING_CONFIG.random_seed,
            shuffle=True,
        )
    else:
        fold_splits = list(
            KFold(
                n_splits=num_folds,
                shuffle=True,
                random_state=MPS_TRAINING_CONFIG.random_seed,
            ).split(relative_trainval_indices)
        )
        train_relative_indices, val_relative_indices = fold_splits[best_fold - 1]

    return {
        "train": trainval_indices[train_relative_indices],
        "val": trainval_indices[val_relative_indices],
        "test": test_indices,
    }


def _encode_split(
    model: MPSFunctionalGroupClassifier,
    X_split: np.ndarray,
    y_split: np.ndarray,
    device: torch.device,
    batch_size: int,
    num_workers: int = 0,
) -> dict[str, np.ndarray]:
    dataset = TensorDataset(
        torch.tensor(X_split, dtype=torch.float32),
        torch.tensor(y_split, dtype=torch.float32),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    latents: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    with torch.no_grad():
        for spectra, target in dataloader:
            spectra = spectra.to(device)
            latent = model.encode_latent(spectra)
            split_logits = model.classify_from_latent(latent)
            latents.append(latent.cpu().numpy().astype(np.float32))
            logits.append(split_logits.cpu().numpy().astype(np.float32))
            labels.append(target.numpy().astype(np.float32))

    latent_array = np.vstack(latents)
    logits_array = np.vstack(logits)
    label_array = np.vstack(labels)
    probabilities = 1.0 / (1.0 + np.exp(-logits_array))

    return {
        "latents": latent_array,
        "labels": label_array,
        "logits": logits_array,
        "probs": probabilities.astype(np.float32),
    }


def _save_latent_cache(
    cache_path: Path,
    split_cache: dict[str, dict[str, np.ndarray]],
    global_threshold: float,
    split_indices: dict[str, np.ndarray],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "global_threshold": np.asarray(global_threshold, dtype=np.float32),
    }
    for split_name, values in split_cache.items():
        for key, array in values.items():
            payload[f"{split_name}_{key}"] = array
        payload[f"{split_name}_indices"] = split_indices[split_name].astype(np.int64)
    np.savez_compressed(cache_path, **payload)


def _per_class_f1(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return f1_score(y_true, y_pred, average=None, zero_division=0)


def _compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    global_threshold: float,
    specialist_indices: list[int],
) -> dict[str, object]:
    y_pred = (y_prob >= global_threshold).astype(np.float32)
    per_class_f1 = _per_class_f1(y_true, y_pred)

    metrics: dict[str, object] = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "per_class_f1": per_class_f1,
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "y_pred": y_pred,
    }

    if specialist_indices:
        specialist_true = y_true[:, specialist_indices]
        specialist_pred = y_pred[:, specialist_indices]
        metrics["specialist_f1_macro"] = float(
            f1_score(specialist_true, specialist_pred, average="macro", zero_division=0)
        )
        metrics["specialist_f1_micro"] = float(
            f1_score(specialist_true, specialist_pred, average="micro", zero_division=0)
        )
        metrics["specialist_per_class_f1"] = per_class_f1[specialist_indices]
    else:
        metrics["specialist_f1_macro"] = 0.0
        metrics["specialist_f1_micro"] = 0.0
        metrics["specialist_per_class_f1"] = np.array([], dtype=np.float32)

    return metrics


def _select_specialist_labels(
    y_train: np.ndarray,
    y_val: np.ndarray,
    baseline_val_prob: np.ndarray,
    global_threshold: float,
    selection_config: SpecialistSelectionConfig,
) -> tuple[list[int], pd.DataFrame]:
    val_pred = (baseline_val_prob >= global_threshold).astype(np.float32)
    num_labels = y_train.shape[1]
    label_names = LABEL_NAMES[:num_labels]
    train_positive = y_train.sum(axis=0)
    val_positive = y_val.sum(axis=0)
    train_prevalence = y_train.mean(axis=0)
    val_f1 = _per_class_f1(y_val, val_pred)

    rows = []
    candidate_indices = []

    for label_idx, label_name in enumerate(label_names):
        positive_ok = train_positive[label_idx] >= selection_config.min_train_positives
        rare_rule = (
            selection_config.use_frequency_rule
            and train_prevalence[label_idx] <= selection_config.max_train_prevalence
        )
        hard_rule = (
            selection_config.use_validation_f1_rule
            and val_positive[label_idx] > 0
            and val_f1[label_idx] <= selection_config.max_val_f1
        )
        is_candidate = positive_ok and (rare_rule or hard_rule)
        if is_candidate:
            candidate_indices.append(label_idx)

        rows.append(
            {
                "label_idx": label_idx,
                "label_name": label_name,
                "train_positives": int(train_positive[label_idx]),
                "val_positives": int(val_positive[label_idx]),
                "train_prevalence": float(train_prevalence[label_idx]),
                "baseline_val_f1": float(val_f1[label_idx]),
                "selected": False,
                "selection_scope": "train+validation_only",
                "is_candidate": bool(is_candidate),
                "rare_rule": bool(rare_rule),
                "hard_rule": bool(hard_rule),
            }
        )

    report = pd.DataFrame(rows)
    prevalence_weight = selection_config.combined_score_weight_prevalence
    val_f1_weight = selection_config.combined_score_weight_val_f1
    report["combined_selection_score"] = (
        prevalence_weight * report["train_prevalence"]
        + val_f1_weight * report["baseline_val_f1"]
    )

    selected = sorted(
        candidate_indices,
        key=lambda idx: (
            float(report.loc[report["label_idx"] == idx, "combined_selection_score"].item()),
            val_f1[idx],
            train_prevalence[idx],
            idx,
        ),
    )[: selection_config.max_labels]

    if not selected:
        fallback_candidates = [
            idx for idx in range(y_train.shape[1]) if train_positive[idx] >= selection_config.min_train_positives
        ]
        selected = sorted(
            fallback_candidates,
            key=lambda idx: (
                float(report.loc[report["label_idx"] == idx, "combined_selection_score"].item()),
                val_f1[idx],
                train_prevalence[idx],
                idx,
            ),
        )[: selection_config.max_labels]

    report["selected"] = report["label_idx"].isin(selected)
    return selected, report.sort_values(
        ["selected", "combined_selection_score", "baseline_val_f1", "train_prevalence"],
        ascending=[False, True, True, True],
    )


def _compute_pos_weight(y_train: np.ndarray, device: torch.device) -> torch.Tensor:
    positives = y_train.sum(axis=0, dtype=np.float32)
    negatives = y_train.shape[0] - positives
    pos_weight = np.ones_like(positives, dtype=np.float32)
    np.divide(negatives, positives, out=pos_weight, where=positives > 0)
    return torch.tensor(pos_weight, dtype=torch.float32, device=device)


def _build_specialist_loss(
    training_config: HybridTrainingConfig,
    pos_weight: torch.Tensor,
) -> nn.Module:
    if training_config.loss_type == "focal":
        return FocalLossWithLogits(gamma=training_config.focal_gamma, pos_weight=pos_weight)
    if training_config.loss_type == "weighted_bce":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    raise ValueError(f"Unsupported loss_type={training_config.loss_type!r}")


def _make_latent_loader(
    latents: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(latents, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.float32),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def _evaluate_specialist(
    model: LatentQCNNHead,
    dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> tuple[float, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    probs: list[np.ndarray] = []

    with torch.no_grad():
        for latent, labels in dataloader:
            latent = latent.to(device)
            labels = labels.to(device)
            logits = model(latent)
            if criterion is not None:
                batch_loss = criterion(logits, labels)
                total_loss += float(batch_loss.item()) * latent.size(0)
                total_count += latent.size(0)
            probs.append(torch.sigmoid(logits).cpu().numpy().astype(np.float32))

    mean_loss = total_loss / max(total_count, 1) if criterion is not None else 0.0
    return mean_loss, np.vstack(probs)


def _merge_probabilities(
    baseline_prob: np.ndarray,
    specialist_prob: np.ndarray,
    specialist_indices: list[int],
    strategy: str,
    blend_alpha: float,
) -> np.ndarray:
    merged = baseline_prob.copy()
    if not specialist_indices:
        return merged

    if strategy == "replace":
        merged[:, specialist_indices] = specialist_prob
        return merged
    if strategy == "blend":
        merged[:, specialist_indices] = (
            blend_alpha * specialist_prob
            + (1.0 - blend_alpha) * merged[:, specialist_indices]
        )
        return merged

    raise ValueError(f"Unsupported merge_strategy={strategy!r}")


def _train_specialist_head(
    train_latents: np.ndarray,
    train_labels: np.ndarray,
    val_latents: np.ndarray,
    val_labels: np.ndarray,
    qcnn_config: QCNNConfig,
    training_config: HybridTrainingConfig,
    checkpoint_path: Path,
) -> tuple[LatentQCNNHead, dict[str, float]]:
    device = _resolve_device(training_config.specialist_device)
    latent_dim = train_latents.shape[1]
    num_labels = train_labels.shape[1]

    model = LatentQCNNHead(
        latent_dim=latent_dim,
        num_labels=num_labels,
        compressed_dim=qcnn_config.compressed_dim,
        num_qubits=qcnn_config.num_qubits,
        circuit_layers=qcnn_config.circuit_layers,
        hidden_dim=qcnn_config.hidden_dim,
        dropout_rate=qcnn_config.dropout_rate,
    ).to(device)

    pos_weight = _compute_pos_weight(train_labels, device)
    criterion = _build_specialist_loss(training_config, pos_weight)
    optimizer = Adam(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    train_loader = _make_latent_loader(
        train_latents,
        train_labels,
        batch_size=training_config.specialist_batch_size,
        shuffle=True,
        num_workers=training_config.num_workers,
    )
    val_loader = _make_latent_loader(
        val_latents,
        val_labels,
        batch_size=training_config.specialist_batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
    )

    early_stopping = EarlyStopping(
        patience=training_config.patience,
        min_delta=training_config.min_delta,
    )

    best_state = None
    best_metrics = {"specialist_f1_macro": -1.0, "macro_f1": -1.0, "micro_f1": -1.0}

    for epoch in range(training_config.num_epochs):
        model.train()
        for latent, labels in train_loader:
            latent = latent.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(latent)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        val_loss, val_prob = _evaluate_specialist(model, val_loader, device, criterion)
        val_metrics = _compute_metrics(
            val_labels,
            val_prob,
            training_config.global_threshold,
            list(range(num_labels)),
        )
        specialist_f1 = float(val_metrics["specialist_f1_macro"])

        if specialist_f1 > best_metrics["specialist_f1_macro"] + training_config.min_delta:
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            best_metrics = {
                "specialist_f1_macro": specialist_f1,
                "specialist_f1_micro": float(val_metrics["specialist_f1_micro"]),
                "macro_f1": float(val_metrics["macro_f1"]),
                "micro_f1": float(val_metrics["micro_f1"]),
                "val_loss": float(val_loss),
                "epoch": float(epoch + 1),
            }

        if early_stopping.step(specialist_f1):
            break

    if best_state is None:
        raise RuntimeError("QCNN specialist training did not produce a valid checkpoint.")

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "metrics": best_metrics,
            "qcnn_config": asdict(qcnn_config),
            "training_config": asdict(training_config),
        },
        checkpoint_path,
    )

    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    return model, best_metrics


def _build_per_class_comparison(
    y_train: np.ndarray,
    split_results: dict[str, dict[str, dict[str, object]]],
    specialist_indices: list[int],
) -> pd.DataFrame:
    rows = []
    train_prevalence = y_train.mean(axis=0)

    for label_idx, label_name in enumerate(LABEL_NAMES):
        row = {
            "label_idx": label_idx,
            "label_name": label_name,
            "train_prevalence": float(train_prevalence[label_idx]),
            "is_specialist": label_idx in specialist_indices,
        }
        for split_name, result_pair in split_results.items():
            baseline_f1 = float(result_pair["baseline"]["per_class_f1"][label_idx])
            row[f"{split_name}_baseline_f1"] = baseline_f1
            for variant_name, metrics in result_pair.items():
                if variant_name == "baseline":
                    continue
                hybrid_f1 = float(metrics["per_class_f1"][label_idx])
                row[f"{split_name}_{variant_name}_f1"] = hybrid_f1
                row[f"{split_name}_{variant_name}_delta_f1"] = hybrid_f1 - baseline_f1
        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["is_specialist", "label_idx"],
        ascending=[False, True],
    )


def _build_specialist_delta_table(
    per_class_comparison: pd.DataFrame,
    specialist_indices: list[int],
) -> pd.DataFrame:
    specialist_df = per_class_comparison[per_class_comparison["label_idx"].isin(specialist_indices)].copy()
    delta_columns = [
        column for column in specialist_df.columns
        if column.endswith("_delta_f1")
    ]
    ordered_columns = ["label_idx", "label_name", "train_prevalence"] + delta_columns
    if delta_columns:
        return specialist_df.loc[:, ordered_columns].sort_values(
            by=delta_columns + ["label_idx"],
            ascending=[False] * len(delta_columns) + [True],
        )
    return specialist_df.loc[:, ordered_columns].sort_values(by=["label_idx"], ascending=[True])


def _write_summary(
    summary_path: Path,
    checkpoint_path: Path,
    specialist_indices: list[int],
    selection_report: pd.DataFrame,
    split_results: dict[str, dict[str, dict[str, object]]],
    specialist_metrics: dict[str, float],
) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    selected_names = [LABEL_NAMES[idx] for idx in specialist_indices]

    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("Frozen MPS + QCNN Specialist Summary\n")
        handle.write("=" * 80 + "\n\n")
        handle.write(f"MPS checkpoint: {checkpoint_path}\n")
        handle.write(f"Specialist labels ({len(selected_names)}): {', '.join(selected_names)}\n")
        handle.write("Selection uses training prevalence and validation F1 only.\n")
        handle.write("Goal: targeted specialization for difficult labels, not parameter matching.\n\n")

        handle.write("Selection criteria:\n")
        handle.write(
            selection_report[selection_report["selected"]]
            .loc[
                :,
                [
                    "label_name",
                    "train_prevalence",
                    "baseline_val_f1",
                    "combined_selection_score",
                    "train_positives",
                    "val_positives",
                    "rare_rule",
                    "hard_rule",
                ],
            ]
            .to_string(index=False)
        )
        handle.write("\n\n")

        for split_name, result_pair in split_results.items():
            handle.write(f"{split_name.upper()} metrics:\n")
            baseline = result_pair["baseline"]
            handle.write(
                f"  Baseline  macro-F1={baseline['macro_f1']:.4f} "
                f"specialist-F1={baseline['specialist_f1_macro']:.4f} "
                f"micro-F1={baseline['micro_f1']:.4f}\n"
            )
            for variant_name, hybrid in result_pair.items():
                if variant_name == "baseline":
                    continue
                handle.write(
                    f"  {variant_name:<8} macro-F1={hybrid['macro_f1']:.4f} "
                    f"specialist-F1={hybrid['specialist_f1_macro']:.4f} "
                    f"micro-F1={hybrid['micro_f1']:.4f}\n"
                )
            handle.write("\n")

        handle.write("QCNN validation checkpoint metrics:\n")
        for key, value in specialist_metrics.items():
            handle.write(f"  {key}: {value:.4f}\n")


def _to_jsonable_metrics(results: dict[str, dict[str, dict[str, object]]]) -> dict[str, dict[str, dict[str, float]]]:
    json_ready: dict[str, dict[str, dict[str, float]]] = {}
    for split_name, result_pair in results.items():
        json_ready[split_name] = {}
        for model_name, metrics in result_pair.items():
            json_ready[split_name][model_name] = {
                "macro_f1": float(metrics["macro_f1"]),
                "specialist_f1_macro": float(metrics["specialist_f1_macro"]),
                "specialist_f1_micro": float(metrics["specialist_f1_micro"]),
                "micro_f1": float(metrics["micro_f1"]),
                "precision_macro": float(metrics["precision_macro"]),
                "recall_macro": float(metrics["recall_macro"]),
            }
    return json_ready


def run_hybrid_experiment(
    path_config: HybridPathConfig = HYBRID_PATH_CONFIG,
    training_config: HybridTrainingConfig = TRAINING_CONFIG,
    qcnn_config: QCNNConfig = QCNN_CONFIG,
    selection_config: SpecialistSelectionConfig = SELECTION_CONFIG,
    max_files: int | None = None,
    data_dir: str | None = None,
    checkpoint_path: str | None = None,
) -> dict[str, object]:
    _seed_everything(training_config.random_seed)

    path_config = replace(path_config)
    if data_dir is not None:
        path_config.data_dir = data_dir
    if checkpoint_path is not None:
        path_config.mps_checkpoint_path = checkpoint_path

    Path(path_config.model_dir).mkdir(parents=True, exist_ok=True)
    Path(path_config.results_dir).mkdir(parents=True, exist_ok=True)

    extract_device = _resolve_device(training_config.extractor_device)
    frozen_mps, checkpoint = _load_frozen_mps_model(
        Path(path_config.mps_checkpoint_path),
        extract_device,
    )

    X, y = load_ir_data(
        Path(path_config.data_dir),
        target_length=MPS_DATA_CONFIG.target_length,
        max_files=max_files if max_files is not None else MPS_DATA_CONFIG.max_files,
        apply_savgol=MPS_DATA_CONFIG.apply_savgol,
        savgol_window_length=MPS_DATA_CONFIG.savgol_window_length,
        savgol_polyorder=MPS_DATA_CONFIG.savgol_polyorder,
        apply_snv=MPS_DATA_CONFIG.apply_snv,
    )

    split_indices = _reconstruct_split_indices(len(X), checkpoint)
    global_threshold = float(training_config.global_threshold)

    split_cache = {
        split_name: _encode_split(
            frozen_mps,
            X[indices],
            y[indices],
            extract_device,
            batch_size=training_config.latent_batch_size,
            num_workers=training_config.num_workers,
        )
        for split_name, indices in split_indices.items()
    }
    _save_latent_cache(
        Path(path_config.latent_cache_path),
        split_cache,
        global_threshold,
        split_indices,
    )

    specialist_indices, selection_report = _select_specialist_labels(
        split_cache["train"]["labels"],
        split_cache["val"]["labels"],
        split_cache["val"]["probs"],
        global_threshold,
        selection_config,
    )
    if not specialist_indices:
        raise RuntimeError("No specialist labels could be selected for the QCNN head.")
    selection_report.to_csv(path_config.specialist_labels_path, index=False)

    specialist_model, specialist_metrics = _train_specialist_head(
        split_cache["train"]["latents"],
        split_cache["train"]["labels"][:, specialist_indices],
        split_cache["val"]["latents"],
        split_cache["val"]["labels"][:, specialist_indices],
        qcnn_config,
        training_config,
        Path(path_config.specialist_checkpoint_path),
    )

    specialist_device = _resolve_device(training_config.specialist_device)
    split_results: dict[str, dict[str, dict[str, object]]] = {}
    for split_name in ("val", "test"):
        split_data = split_cache[split_name]
        specialist_loader = _make_latent_loader(
            split_data["latents"],
            split_data["labels"][:, specialist_indices],
            batch_size=training_config.specialist_batch_size,
            shuffle=False,
            num_workers=training_config.num_workers,
        )
        _, specialist_prob = _evaluate_specialist(specialist_model, specialist_loader, specialist_device)

        baseline_metrics = _compute_metrics(
            split_data["labels"],
            split_data["probs"],
            global_threshold,
            specialist_indices,
        )
        split_results[split_name] = {"baseline": baseline_metrics}

        for strategy_name in training_config.merge_strategies:
            hybrid_prob = _merge_probabilities(
                split_data["probs"],
                specialist_prob,
                specialist_indices,
                strategy_name,
                training_config.blend_alpha,
            )
            hybrid_metrics = _compute_metrics(
                split_data["labels"],
                hybrid_prob,
                global_threshold,
                specialist_indices,
            )
            split_results[split_name][strategy_name] = hybrid_metrics

    per_class_comparison = _build_per_class_comparison(
        split_cache["train"]["labels"],
        split_results,
        specialist_indices,
    )
    per_class_comparison.to_csv(path_config.per_class_metrics_path, index=False)
    specialist_delta_table = _build_specialist_delta_table(
        per_class_comparison,
        specialist_indices,
    )
    specialist_delta_table.to_csv(path_config.specialist_delta_metrics_path, index=False)

    metrics_json = {
        "checkpoint_path": str(path_config.mps_checkpoint_path),
        "specialist_indices": specialist_indices,
        "specialist_labels": [LABEL_NAMES[idx] for idx in specialist_indices],
        "global_threshold": global_threshold,
        "selection_scope": "train+validation_only",
        "goal": "targeted specialization for difficult labels rather than parameter matching",
        "selection_config": asdict(selection_config),
        "qcnn_config": asdict(qcnn_config),
        "training_config": asdict(training_config),
        "results": _to_jsonable_metrics(split_results),
    }
    Path(path_config.metrics_path).write_text(json.dumps(metrics_json, indent=2) + "\n", encoding="utf-8")

    _write_summary(
        Path(path_config.summary_path),
        Path(path_config.mps_checkpoint_path),
        specialist_indices,
        selection_report,
        split_results,
        specialist_metrics,
    )

    return {
        "specialist_indices": specialist_indices,
        "specialist_labels": [LABEL_NAMES[idx] for idx in specialist_indices],
        "metrics": metrics_json,
        "per_class_metrics_path": path_config.per_class_metrics_path,
        "specialist_delta_metrics_path": path_config.specialist_delta_metrics_path,
        "selection_report_path": path_config.specialist_labels_path,
        "summary_path": path_config.summary_path,
        "latent_cache_path": path_config.latent_cache_path,
        "specialist_checkpoint_path": path_config.specialist_checkpoint_path,
    }
