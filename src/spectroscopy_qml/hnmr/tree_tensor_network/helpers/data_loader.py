"""Data loading helpers for experiment 5 with reusable fixed split artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.spectroscopy_qml.hnmr.mps_classifier_hnmr.data_loader import (
    FUNCTIONAL_GROUPS,
    IRSpectraDataset,
    load_ir_data as mps_load_cnmr_data,
    multilabel_train_test_split,
)

def _validate_cache_payload(
    payload: Any,
    *,
    target_length: int,
    apply_snv: bool,
    apply_savgol: bool,
    savgol_window_length: int,
    savgol_polyorder: int,
    max_files: int | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return cached arrays when the payload matches the requested preprocessing."""
    try:
        X = payload["X"]
        y = payload["y"]
        stored_target_length = int(payload["target_length"])
        stored_apply_snv = bool(payload["apply_snv"])
        stored_apply_savgol = bool(payload["apply_savgol"])
        stored_savgol_window_length = int(payload["savgol_window_length"])
        stored_savgol_polyorder = int(payload["savgol_polyorder"])
        stored_max_files = payload["max_files"]
        stored_max_files = None if np.isnan(stored_max_files) else int(stored_max_files)
    except KeyError:
        return None

    if stored_target_length != int(target_length):
        return None
    if stored_apply_snv != bool(apply_snv):
        return None
    if stored_apply_savgol != bool(apply_savgol):
        return None
    if stored_savgol_window_length != int(savgol_window_length):
        return None
    if stored_savgol_polyorder != int(savgol_polyorder):
        return None
    if stored_max_files != max_files:
        return None

    return np.asarray(X), np.asarray(y)


def load_cnmr_data(
    data_dir: Path,
    target_length: int = 1800,
    max_files: int | None = None,
    apply_snv: bool = False,
    apply_savgol: bool = False,
    savgol_window_length: int = 11,
    savgol_polyorder: int = 3,
    cache_path: Path | None = None,
    overwrite_cache: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Load C-NMR spectra via the MPS encoder loader with optional cached arrays."""
    cache_file = Path(cache_path) if cache_path is not None else None
    if cache_file is not None and cache_file.exists() and not overwrite_cache:
        payload = np.load(cache_file, allow_pickle=False)
        cached = _validate_cache_payload(
            payload,
            target_length=target_length,
            apply_snv=apply_snv,
            apply_savgol=apply_savgol,
            savgol_window_length=savgol_window_length,
            savgol_polyorder=savgol_polyorder,
            max_files=max_files,
        )
        if cached is not None:
            X, y = cached
            print(f"Loaded cached C-NMR dataset from {cache_file}")
            print(f"Spectra shape: {X.shape}")
            print(f"Labels shape: {y.shape}")
            return X, y
        print(f"Ignoring incompatible cache artifact at {cache_file}")

    X, y = mps_load_cnmr_data(
        data_dir=data_dir,
        target_length=target_length,
        max_files=max_files,
        apply_snv=apply_snv,
        apply_savgol=apply_savgol,
        savgol_window_length=savgol_window_length,
        savgol_polyorder=savgol_polyorder,
    )

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_file,
            X=X,
            y=y,
            target_length=np.asarray(target_length, dtype=np.int64),
            apply_snv=np.asarray(apply_snv, dtype=np.bool_),
            apply_savgol=np.asarray(apply_savgol, dtype=np.bool_),
            savgol_window_length=np.asarray(savgol_window_length, dtype=np.int64),
            savgol_polyorder=np.asarray(savgol_polyorder, dtype=np.int64),
            max_files=np.asarray(np.nan if max_files is None else max_files, dtype=np.float64),
        )
        print(f"Saved C-NMR dataset cache to {cache_file}")

    return X, y

def _iterative_multilabel_sample(
    labels: np.ndarray,
    subset_size: int,
    random_seed: int,
) -> np.ndarray:
    """Approximate iterative stratification for a multilabel subset."""
    num_samples = labels.shape[0]
    if subset_size <= 0:
        return np.empty(0, dtype=int)
    if subset_size >= num_samples:
        return np.arange(num_samples, dtype=int)

    rng = np.random.default_rng(random_seed)
    selected: list[int] = []
    selected_mask = np.zeros(num_samples, dtype=bool)

    label_totals = labels.sum(axis=0).astype(float)
    desired_totals = label_totals * (subset_size / num_samples)
    current_totals = np.zeros(labels.shape[1], dtype=float)
    label_weights = 1.0 / np.maximum(label_totals, 1.0)

    while len(selected) < subset_size:
        remaining_indices = np.flatnonzero(~selected_mask)
        deficits = np.clip(desired_totals - current_totals, 0.0, None)

        if np.any(deficits > 0):
            sample_scores = labels[remaining_indices] @ (deficits * label_weights)
            best_score = float(sample_scores.max(initial=0.0))
            if best_score > 0.0:
                candidate_positions = np.flatnonzero(np.isclose(sample_scores, best_score))
                chosen_position = int(rng.choice(candidate_positions))
                chosen_index = int(remaining_indices[chosen_position])
            else:
                chosen_index = int(rng.choice(remaining_indices))
        else:
            chosen_index = int(rng.choice(remaining_indices))

        selected.append(chosen_index)
        selected_mask[chosen_index] = True
        current_totals += labels[chosen_index]

    return np.asarray(selected, dtype=int)


def _split_indices(
    labels: np.ndarray,
    split_ratio: float,
    random_seed: int,
    stratify_multilabel: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Split sample indices while optionally preserving multilabel coverage."""
    num_samples = labels.shape[0]
    split_size = int(round(num_samples * split_ratio))

    if split_size <= 0:
        return np.arange(num_samples, dtype=int), np.empty(0, dtype=int)
    if split_size >= num_samples:
        return np.empty(0, dtype=int), np.arange(num_samples, dtype=int)

    if stratify_multilabel and labels.ndim == 2 and labels.shape[1] > 0:
        indices = np.arange(num_samples, dtype=np.int64)
        try:
            keep_indices, split_indices, _, _ = multilabel_train_test_split(
                indices,
                labels,
                test_size=split_ratio,
                random_seed=random_seed,
            )
            return np.asarray(keep_indices, dtype=int), np.asarray(split_indices, dtype=int)
        except ValueError:
            split_indices = _iterative_multilabel_sample(labels, split_size, random_seed)
            split_mask = np.zeros(num_samples, dtype=bool)
            split_mask[split_indices] = True
            keep_indices = np.flatnonzero(~split_mask)
            return keep_indices, split_indices

    indices = np.arange(num_samples)
    keep_indices, split_indices = train_test_split(
        indices,
        test_size=split_ratio,
        random_state=random_seed,
        shuffle=True,
    )
    return np.asarray(keep_indices, dtype=int), np.asarray(split_indices, dtype=int)


def _validate_split_indices(
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    num_samples: int,
) -> None:
    """Validate that fixed split indices are disjoint and exhaustive."""
    all_indices = np.concatenate((train_indices, val_indices, test_indices), axis=0)
    if len(all_indices) != num_samples:
        raise ValueError(
            "Stored split does not match dataset size: "
            f"{len(all_indices)} indices for {num_samples} samples."
        )
    if len(np.unique(all_indices)) != num_samples:
        raise ValueError("Stored split contains duplicate indices.")
    if all_indices.min(initial=0) < 0 or all_indices.max(initial=-1) >= num_samples:
        raise ValueError("Stored split contains out-of-range indices.")


def load_or_create_split_indices(
    labels: np.ndarray,
    split_path: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
    stratify_multilabel: bool = True,
    overwrite: bool = False,
) -> dict[str, np.ndarray]:
    """Load a fixed split artifact or create it deterministically."""
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0.")

    split_path = Path(split_path)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    num_samples = int(labels.shape[0])
    num_labels = int(labels.shape[1]) if labels.ndim == 2 else 0

    if split_path.exists() and not overwrite:
        payload = np.load(split_path, allow_pickle=False)
        train_indices = payload["train_indices"].astype(np.int64, copy=False)
        val_indices = payload["val_indices"].astype(np.int64, copy=False)
        test_indices = payload["test_indices"].astype(np.int64, copy=False)
        stored_num_samples = int(payload["num_samples"])
        stored_num_labels = int(payload["num_labels"])
        stored_seed = int(payload["random_seed"])
        stored_train_ratio = float(payload["train_ratio"])
        stored_val_ratio = float(payload["val_ratio"])
        stored_test_ratio = float(payload["test_ratio"])

        if stored_num_samples != num_samples or stored_num_labels != num_labels:
            raise ValueError(
                "Stored split is incompatible with current dataset: "
                f"expected ({num_samples}, {num_labels}), found "
                f"({stored_num_samples}, {stored_num_labels})."
            )
        if stored_seed != int(random_seed):
            raise ValueError(
                f"Stored split seed {stored_seed} does not match requested seed {random_seed}."
            )
        if not np.isclose(
            [stored_train_ratio, stored_val_ratio, stored_test_ratio],
            [train_ratio, val_ratio, test_ratio],
        ).all():
            raise ValueError("Stored split ratios do not match the requested ratios.")

        _validate_split_indices(train_indices, val_indices, test_indices, num_samples)
        return {
            "train": train_indices,
            "val": val_indices,
            "test": test_indices,
        }

    train_val_indices, test_indices = _split_indices(
        labels,
        split_ratio=test_ratio,
        random_seed=random_seed,
        stratify_multilabel=stratify_multilabel,
    )
    remaining_labels = labels[train_val_indices]
    val_split_ratio = val_ratio / (train_ratio + val_ratio)
    train_indices_local, val_indices_local = _split_indices(
        remaining_labels,
        split_ratio=val_split_ratio,
        random_seed=random_seed + 1,
        stratify_multilabel=stratify_multilabel,
    )

    train_indices = train_val_indices[train_indices_local]
    val_indices = train_val_indices[val_indices_local]

    _validate_split_indices(train_indices, val_indices, test_indices, num_samples)
    np.savez_compressed(
        split_path,
        train_indices=train_indices.astype(np.int64, copy=False),
        val_indices=val_indices.astype(np.int64, copy=False),
        test_indices=test_indices.astype(np.int64, copy=False),
        num_samples=np.asarray(num_samples, dtype=np.int64),
        num_labels=np.asarray(num_labels, dtype=np.int64),
        random_seed=np.asarray(random_seed, dtype=np.int64),
        train_ratio=np.asarray(train_ratio, dtype=np.float32),
        val_ratio=np.asarray(val_ratio, dtype=np.float32),
        test_ratio=np.asarray(test_ratio, dtype=np.float32),
    )
    return {
        "train": train_indices.astype(np.int64, copy=False),
        "val": val_indices.astype(np.int64, copy=False),
        "test": test_indices.astype(np.int64, copy=False),
    }


def prepare_dataloaders_from_split_indices(
    X: np.ndarray,
    y: np.ndarray,
    split_indices: dict[str, np.ndarray],
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create dataloaders from a fixed split artifact."""
    train_indices = split_indices["train"]
    val_indices = split_indices["val"]
    test_indices = split_indices["test"]
    _validate_split_indices(train_indices, val_indices, test_indices, len(X))

    X_train = X[train_indices]
    y_train = y[train_indices]
    X_val = X[val_indices]
    y_val = y[val_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]

    print("\nData split:")
    print(f"  Train: {len(X_train)} samples ({len(X_train) / len(X):.1%})")
    print(f"  Val:   {len(X_val)} samples ({len(X_val) / len(X):.1%})")
    print(f"  Test:  {len(X_test)} samples ({len(X_test) / len(X):.1%})")
    print("  Split: fixed artifact")

    train_dataset = IRSpectraDataset(X_train, y_train)
    val_dataset = IRSpectraDataset(X_val, y_val)
    test_dataset = IRSpectraDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
    )
    return train_loader, val_loader, test_loader


__all__ = [
    "FUNCTIONAL_GROUPS",
    "load_cnmr_data",
    "load_or_create_split_indices",
    "prepare_dataloaders_from_split_indices",
]
