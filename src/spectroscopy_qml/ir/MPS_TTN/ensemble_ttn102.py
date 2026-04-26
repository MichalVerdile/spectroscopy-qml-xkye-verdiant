"""Selective MPS final + TTN 10.2 ensemble for IR functional groups.

This ensemble treats the final MPS model as the baseline and selectively
overrides a subset of functional groups with TTN 10.2 predictions wherever
TTN is stronger on a shared validation split.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import types
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import KFold, train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = Path(__file__).resolve().parents[4]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spectroscopy_qml.ir.mps_encoder_final.config import TRAINING_CONFIG as MPS_TRAINING_CONFIG  # noqa: E402
from spectroscopy_qml.ir.mps_encoder_final.config import ModelConfig as LegacyMPSModelConfig  # noqa: E402
from spectroscopy_qml.ir.mps_encoder_final.data_loader import FUNCTIONAL_GROUPS  # noqa: E402
from spectroscopy_qml.ir.mps_encoder_final.model import MPSFunctionalGroupClassifier  # noqa: E402
from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment10_2_1.model import (  # noqa: E402
    load_ttn102,
)


ALL_LABEL_NAMES = list(FUNCTIONAL_GROUPS.keys())
LABEL_TO_INDEX = {name: index for index, name in enumerate(ALL_LABEL_NAMES)}

# Derived from the darker-green TTN cells in the user's comparison table.
DEFAULT_TTN_STRONGER_CLASS_NAMES = (
    "Arene",
    "Alkane",
    "Nitrile",
    "Isocyanate",
    "Ether",
    "Haloalkane",
    "Thial",
    "Sulfonic acid",
)

PRIMARY_MPS_CHECKPOINT = Path("src/spectroscopy_qml/ir/mps_encoder_final/models/mps_model_best.pt")
LEGACY_UPLOADED_MPS_CHECKPOINT = Path("src/spectroscopy_qml/ir/mps_encoder_final/MPS.pt")
DEFAULT_TTN_CHECKPOINT = Path(
    "src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment10_2"
    "/results/full_dataset_run_20260417_173715_percentile/ttn_ir_best.pt"
)
DEFAULT_TTN_CONFIG = Path(
    "src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment10_2"
    "/results/full_dataset_run_20260417_173715_percentile/run_config.json"
)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if device_arg == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def threshold_predictions(y_prob: np.ndarray, thresholds: float | np.ndarray = 0.5) -> np.ndarray:
    return (y_prob >= thresholds).astype(np.int32)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | np.ndarray]:
    return {
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_micro": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_micro": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class_f1": f1_score(y_true, y_pred, average=None, zero_division=0),
    }


def build_threshold_grid(step: float) -> np.ndarray:
    if step <= 0 or step > 1:
        raise ValueError(f"threshold step must be in (0, 1], got {step}")
    return np.arange(0.1, 0.9 + step / 2, step, dtype=np.float32)


def tune_thresholds(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    mode: str,
    target_metric: str,
    threshold_grid: np.ndarray,
) -> np.ndarray:
    n_classes = y_true.shape[1]

    if mode == "global":
        best_score = -1.0
        best_threshold = 0.5
        average = "micro" if target_metric == "f1_micro" else "macro"
        for threshold in threshold_grid:
            preds = (y_probs >= threshold).astype(np.int32)
            score = f1_score(y_true, preds, average=average, zero_division=0)
            if score > best_score:
                best_score = float(score)
                best_threshold = float(threshold)
        return np.full(n_classes, best_threshold, dtype=np.float32)

    if target_metric == "f1_micro":
        raise ValueError("threshold mode 'per_class' is incompatible with target_metric 'f1_micro'")

    thresholds = np.full(n_classes, 0.5, dtype=np.float32)
    for class_index in range(n_classes):
        best_score = -1.0
        best_threshold = 0.5
        for threshold in threshold_grid:
            preds = (y_probs[:, class_index] >= threshold).astype(np.int32)
            score = f1_score(y_true[:, class_index], preds, zero_division=0)
            if score > best_score:
                best_score = float(score)
                best_threshold = float(threshold)
        thresholds[class_index] = best_threshold
    return thresholds


def resolve_default_mps_checkpoint() -> Path:
    if PRIMARY_MPS_CHECKPOINT.exists():
        return PRIMARY_MPS_CHECKPOINT
    if LEGACY_UPLOADED_MPS_CHECKPOINT.exists():
        return LEGACY_UPLOADED_MPS_CHECKPOINT
    return PRIMARY_MPS_CHECKPOINT


def _get_model_config_kwargs(model_config) -> dict[str, object]:
    if is_dataclass(model_config):
        return asdict(model_config)
    if isinstance(model_config, dict):
        return model_config.copy()
    raise TypeError(f"Unsupported model config type in checkpoint: {type(model_config)!r}")


def parse_candidate_names(value: str | None) -> list[str]:
    if value is None or not value.strip():
        return list(DEFAULT_TTN_STRONGER_CLASS_NAMES)
    names = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [name for name in names if name not in LABEL_TO_INDEX]
    if unknown:
        raise ValueError(f"Unknown functional-group names: {unknown}")
    return names


def reconstruct_mps_checkpoint_splits(
    num_samples: int,
    checkpoint: dict,
) -> dict[str, np.ndarray]:
    dataset_indices = np.arange(num_samples, dtype=np.int64)
    trainval_indices, test_indices = train_test_split(
        dataset_indices,
        test_size=MPS_TRAINING_CONFIG.test_ratio,
        random_state=MPS_TRAINING_CONFIG.random_seed,
        shuffle=True,
    )

    best_fold = int(checkpoint.get("best_fold", 1))
    num_folds = int(checkpoint.get("cv_num_folds", MPS_TRAINING_CONFIG.num_folds))
    if num_folds < 1:
        raise ValueError(f"Checkpoint contains invalid cv_num_folds={num_folds}")

    relative_trainval_indices = np.arange(len(trainval_indices), dtype=np.int64)
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
        if best_fold < 1 or best_fold > num_folds:
            raise ValueError(
                f"Checkpoint best_fold={best_fold} is outside expected range 1..{num_folds}."
            )
        fold_splits = list(
            KFold(
                n_splits=num_folds,
                shuffle=True,
                random_state=MPS_TRAINING_CONFIG.random_seed,
            ).split(relative_trainval_indices)
        )
        train_relative_indices, val_relative_indices = fold_splits[best_fold - 1]

    return {
        "train": trainval_indices[np.asarray(train_relative_indices, dtype=np.int64)],
        "val": trainval_indices[np.asarray(val_relative_indices, dtype=np.int64)],
        "test": np.asarray(test_indices, dtype=np.int64),
    }


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    *,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(x[indices]).float(),
        torch.from_numpy(y[indices]).float(),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=True if num_workers > 0 else False,
    )


def load_mps_model(checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    legacy_package = "spectroscopy_qml.ir.mps_classifier"
    legacy_config_module = f"{legacy_package}.config"
    if legacy_config_module not in sys.modules:
        package_module = types.ModuleType(legacy_package)
        package_module.__path__ = []  # type: ignore[attr-defined]
        config_module = types.ModuleType(legacy_config_module)
        config_module.ModelConfig = LegacyMPSModelConfig
        sys.modules[legacy_package] = package_module
        sys.modules[legacy_config_module] = config_module

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = MPSFunctionalGroupClassifier(**_get_model_config_kwargs(checkpoint["config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model.to(device), checkpoint


@torch.no_grad()
def get_probabilities(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []

    for x_batch, y_batch in loader:
        logits = model(x_batch.to(device))
        probs_list.append(torch.sigmoid(logits).cpu().numpy())
        labels_list.append(y_batch.numpy().astype(np.int32))

    return np.concatenate(probs_list, axis=0), np.concatenate(labels_list, axis=0)


def select_override_indices(
    *,
    mps_per_class_f1: np.ndarray,
    ttn_per_class_f1: np.ndarray,
    val_labels: np.ndarray,
    candidate_indices: list[int],
    min_f1_gain: float,
    min_support: int,
) -> list[int]:
    override_indices: list[int] = []
    for class_index in candidate_indices:
        support = int(val_labels[:, class_index].sum())
        gain = float(ttn_per_class_f1[class_index]) - float(mps_per_class_f1[class_index])
        if support >= min_support and gain >= min_f1_gain:
            override_indices.append(class_index)
    return override_indices


def blend_probabilities(
    *,
    base_probs: np.ndarray,
    override_probs: np.ndarray,
    override_indices: list[int],
) -> np.ndarray:
    blended = np.asarray(base_probs, dtype=np.float32).copy()
    if not override_indices:
        return blended
    blended[:, override_indices] = override_probs[:, override_indices]
    return blended


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Selective ensemble: MPS final baseline + TTN 10.2 overrides for TTN-strong classes."
    )
    parser.add_argument(
        "--spectra-cache",
        type=Path,
        default=Path("data/cache/ir_spectra_len1800_snv_all.npz"),
    )
    parser.add_argument(
        "--mps-checkpoint",
        type=Path,
        default=resolve_default_mps_checkpoint(),
    )
    parser.add_argument(
        "--ttn-checkpoint",
        type=Path,
        default=DEFAULT_TTN_CHECKPOINT,
    )
    parser.add_argument(
        "--ttn-config",
        type=Path,
        default=DEFAULT_TTN_CONFIG,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/spectroscopy_qml/ir/MPS_TTN/results"),
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--candidate-names",
        type=str,
        default=",".join(DEFAULT_TTN_STRONGER_CLASS_NAMES),
        help="Comma-separated functional-group names eligible for TTN override.",
    )
    parser.add_argument("--min-f1-gain", type=float, default=0.02)
    parser.add_argument("--min-support", type=int, default=5)
    parser.add_argument("--threshold-mode", choices=["global", "per_class"], default="per_class")
    parser.add_argument(
        "--threshold-target-metric",
        choices=["f1_micro", "f1_macro", "per_class_f1"],
        default="per_class_f1",
    )
    parser.add_argument("--threshold-grid-step", type=float, default=0.02)
    return parser


def write_summary(
    summary_path: Path,
    *,
    candidate_names: list[str],
    override_names: list[str],
    min_f1_gain: float,
    min_support: int,
    mps_val_metrics: dict[str, float | np.ndarray],
    ttn_val_metrics: dict[str, float | np.ndarray],
    ensemble_val_metrics: dict[str, float | np.ndarray],
    mps_test_metrics: dict[str, float | np.ndarray],
    ttn_test_metrics: dict[str, float | np.ndarray],
    ensemble_test_metrics: dict[str, float | np.ndarray],
) -> None:
    lines = [
        "MPS + TTN 10.2 Selective Ensemble Summary",
        "=" * 80,
        f"TTN candidate classes:      {', '.join(candidate_names)}",
        f"Selected override classes:  {', '.join(override_names) if override_names else '(none)'}",
        f"Minimum F1 gain:            {min_f1_gain:.4f}",
        f"Minimum support:            {min_support}",
        "",
        "Validation metrics",
        f"  MPS final:                micro={float(mps_val_metrics['f1_micro']):.4f} macro={float(mps_val_metrics['f1_macro']):.4f}",
        f"  TTN 10.2:                 micro={float(ttn_val_metrics['f1_micro']):.4f} macro={float(ttn_val_metrics['f1_macro']):.4f}",
        f"  Selective ensemble:       micro={float(ensemble_val_metrics['f1_micro']):.4f} macro={float(ensemble_val_metrics['f1_macro']):.4f}",
        "",
        "Test metrics",
        f"  MPS final:                micro={float(mps_test_metrics['f1_micro']):.4f} macro={float(mps_test_metrics['f1_macro']):.4f}",
        f"  TTN 10.2:                 micro={float(ttn_test_metrics['f1_micro']):.4f} macro={float(ttn_test_metrics['f1_macro']):.4f}",
        f"  Selective ensemble:       micro={float(ensemble_test_metrics['f1_micro']):.4f} macro={float(ensemble_test_metrics['f1_macro']):.4f}",
    ]
    summary_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = build_parser().parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    threshold_grid = build_threshold_grid(args.threshold_grid_step)

    candidate_names = parse_candidate_names(args.candidate_names)
    candidate_indices = [LABEL_TO_INDEX[name] for name in candidate_names]

    print("Loading spectra cache...")
    cache = np.load(args.spectra_cache)
    x = cache["X"].astype(np.float32)
    y = cache["y"].astype(np.int32)
    print(f"Loaded cache X={x.shape}, y={y.shape}")

    print("Loading MPS final checkpoint...")
    mps_model, mps_checkpoint = load_mps_model(args.mps_checkpoint, device)
    split_indices = reconstruct_mps_checkpoint_splits(len(x), mps_checkpoint)

    print("Loading TTN 10.2 checkpoint...")
    ttn_config = json.loads(args.ttn_config.read_text())
    ttn_model = load_ttn102(args.ttn_checkpoint, ttn_config, device)

    val_loader = make_loader(x, y, split_indices["val"], batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = make_loader(x, y, split_indices["test"], batch_size=args.batch_size, num_workers=args.num_workers)

    print("Running validation predictions...")
    mps_val_probs, val_labels = get_probabilities(mps_model, val_loader, device)
    ttn_val_probs, val_labels_ttn = get_probabilities(ttn_model, val_loader, device)
    if not np.array_equal(val_labels, val_labels_ttn):
        raise ValueError("MPS and TTN validation labels do not match.")

    mps_thresholds = tune_thresholds(
        val_labels,
        mps_val_probs,
        args.threshold_mode,
        args.threshold_target_metric,
        threshold_grid,
    )
    ttn_thresholds = tune_thresholds(
        val_labels,
        ttn_val_probs,
        args.threshold_mode,
        args.threshold_target_metric,
        threshold_grid,
    )

    mps_val_metrics = compute_metrics(val_labels, threshold_predictions(mps_val_probs, mps_thresholds))
    ttn_val_metrics = compute_metrics(val_labels, threshold_predictions(ttn_val_probs, ttn_thresholds))

    override_indices = select_override_indices(
        mps_per_class_f1=np.asarray(mps_val_metrics["per_class_f1"], dtype=np.float32),
        ttn_per_class_f1=np.asarray(ttn_val_metrics["per_class_f1"], dtype=np.float32),
        val_labels=val_labels,
        candidate_indices=candidate_indices,
        min_f1_gain=args.min_f1_gain,
        min_support=args.min_support,
    )
    override_names = [ALL_LABEL_NAMES[index] for index in override_indices]
    print(f"Selected TTN override classes ({len(override_names)}): {override_names}")

    ensemble_val_probs = blend_probabilities(
        base_probs=mps_val_probs,
        override_probs=ttn_val_probs,
        override_indices=override_indices,
    )
    ensemble_thresholds = np.asarray(mps_thresholds, dtype=np.float32).copy()
    if override_indices:
        ensemble_thresholds[override_indices] = np.asarray(ttn_thresholds, dtype=np.float32)[override_indices]
    ensemble_val_metrics = compute_metrics(
        val_labels,
        threshold_predictions(ensemble_val_probs, ensemble_thresholds),
    )

    print("Running test predictions...")
    mps_test_probs, test_labels = get_probabilities(mps_model, test_loader, device)
    ttn_test_probs, test_labels_ttn = get_probabilities(ttn_model, test_loader, device)
    if not np.array_equal(test_labels, test_labels_ttn):
        raise ValueError("MPS and TTN test labels do not match.")

    mps_test_metrics = compute_metrics(test_labels, threshold_predictions(mps_test_probs, mps_thresholds))
    ttn_test_metrics = compute_metrics(test_labels, threshold_predictions(ttn_test_probs, ttn_thresholds))
    ensemble_test_probs = blend_probabilities(
        base_probs=mps_test_probs,
        override_probs=ttn_test_probs,
        override_indices=override_indices,
    )
    ensemble_test_metrics = compute_metrics(
        test_labels,
        threshold_predictions(ensemble_test_probs, ensemble_thresholds),
    )

    comparison_path = args.output_dir / "per_class_comparison.csv"
    with comparison_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "label_index",
                "label_name",
                "support_val",
                "support_test",
                "candidate_for_ttn",
                "selected_override",
                "mps_val_f1",
                "ttn_val_f1",
                "val_f1_gain",
                "mps_test_f1",
                "ttn_test_f1",
                "ensemble_test_f1",
            ]
        )
        mps_val_per = np.asarray(mps_val_metrics["per_class_f1"], dtype=np.float32)
        ttn_val_per = np.asarray(ttn_val_metrics["per_class_f1"], dtype=np.float32)
        mps_test_per = np.asarray(mps_test_metrics["per_class_f1"], dtype=np.float32)
        ttn_test_per = np.asarray(ttn_test_metrics["per_class_f1"], dtype=np.float32)
        ensemble_test_per = np.asarray(ensemble_test_metrics["per_class_f1"], dtype=np.float32)
        for index, name in enumerate(ALL_LABEL_NAMES):
            writer.writerow(
                [
                    index,
                    name,
                    int(val_labels[:, index].sum()),
                    int(test_labels[:, index].sum()),
                    name in candidate_names,
                    index in override_indices,
                    float(mps_val_per[index]),
                    float(ttn_val_per[index]),
                    float(ttn_val_per[index] - mps_val_per[index]),
                    float(mps_test_per[index]),
                    float(ttn_test_per[index]),
                    float(ensemble_test_per[index]),
                ]
            )

    results = {
        "candidate_names": candidate_names,
        "candidate_indices": candidate_indices,
        "override_names": override_names,
        "override_indices": override_indices,
        "min_f1_gain": float(args.min_f1_gain),
        "min_support": int(args.min_support),
        "mps_threshold_mean": float(np.mean(mps_thresholds)),
        "ttn_threshold_mean": float(np.mean(ttn_thresholds)),
        "ensemble_threshold_mean": float(np.mean(ensemble_thresholds)),
        "validation": {
            "mps": {
                "f1_micro": float(mps_val_metrics["f1_micro"]),
                "f1_macro": float(mps_val_metrics["f1_macro"]),
            },
            "ttn": {
                "f1_micro": float(ttn_val_metrics["f1_micro"]),
                "f1_macro": float(ttn_val_metrics["f1_macro"]),
            },
            "ensemble": {
                "f1_micro": float(ensemble_val_metrics["f1_micro"]),
                "f1_macro": float(ensemble_val_metrics["f1_macro"]),
            },
        },
        "test": {
            "mps": {
                "f1_micro": float(mps_test_metrics["f1_micro"]),
                "f1_macro": float(mps_test_metrics["f1_macro"]),
            },
            "ttn": {
                "f1_micro": float(ttn_test_metrics["f1_micro"]),
                "f1_macro": float(ttn_test_metrics["f1_macro"]),
            },
            "ensemble": {
                "f1_micro": float(ensemble_test_metrics["f1_micro"]),
                "f1_macro": float(ensemble_test_metrics["f1_macro"]),
            },
        },
    }
    results_path = args.output_dir / "ensemble_results.json"
    results_path.write_text(json.dumps(results, indent=2) + "\n")

    summary_path = args.output_dir / "summary.txt"
    write_summary(
        summary_path,
        candidate_names=candidate_names,
        override_names=override_names,
        min_f1_gain=args.min_f1_gain,
        min_support=args.min_support,
        mps_val_metrics=mps_val_metrics,
        ttn_val_metrics=ttn_val_metrics,
        ensemble_val_metrics=ensemble_val_metrics,
        mps_test_metrics=mps_test_metrics,
        ttn_test_metrics=ttn_test_metrics,
        ensemble_test_metrics=ensemble_test_metrics,
    )

    print("\nValidation")
    print(f"  MPS final:          micro={float(mps_val_metrics['f1_micro']):.4f} macro={float(mps_val_metrics['f1_macro']):.4f}")
    print(f"  TTN 10.2:           micro={float(ttn_val_metrics['f1_micro']):.4f} macro={float(ttn_val_metrics['f1_macro']):.4f}")
    print(f"  Selective ensemble: micro={float(ensemble_val_metrics['f1_micro']):.4f} macro={float(ensemble_val_metrics['f1_macro']):.4f}")

    print("\nTest")
    print(f"  MPS final:          micro={float(mps_test_metrics['f1_micro']):.4f} macro={float(mps_test_metrics['f1_macro']):.4f}")
    print(f"  TTN 10.2:           micro={float(ttn_test_metrics['f1_micro']):.4f} macro={float(ttn_test_metrics['f1_macro']):.4f}")
    print(f"  Selective ensemble: micro={float(ensemble_test_metrics['f1_micro']):.4f} macro={float(ensemble_test_metrics['f1_macro']):.4f}")
    print(f"\nSaved results to {results_path}")


if __name__ == "__main__":
    main()
