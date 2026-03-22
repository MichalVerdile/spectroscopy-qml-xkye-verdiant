"""Legacy TTN-specific data loading helpers for IR spectra."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from spectroscopy_qml.ir.mps_encoder.data_loader import (
    FUNCTIONAL_GROUPS,
    IRSpectraDataset,
    load_ir_data,
)


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
    """Split sample indices, optionally preserving rare multilabel combinations."""
    num_samples = labels.shape[0]
    split_size = int(round(num_samples * split_ratio))

    if split_size <= 0:
        return np.arange(num_samples, dtype=int), np.empty(0, dtype=int)
    if split_size >= num_samples:
        return np.empty(0, dtype=int), np.arange(num_samples, dtype=int)

    if stratify_multilabel and labels.ndim == 2 and labels.shape[1] > 0:
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


def prepare_dataloaders(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
    stratify_multilabel: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create legacy TTN train/validation/test loaders."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    train_val_indices, test_indices = _split_indices(
        y,
        split_ratio=test_ratio,
        random_seed=random_seed,
        stratify_multilabel=stratify_multilabel,
    )

    X_temp = X[train_val_indices]
    y_temp = y[train_val_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]

    val_size = val_ratio / (train_ratio + val_ratio)
    train_indices, val_indices = _split_indices(
        y_temp,
        split_ratio=val_size,
        random_seed=random_seed + 1,
        stratify_multilabel=stratify_multilabel,
    )

    X_train = X_temp[train_indices]
    y_train = y_temp[train_indices]
    X_val = X_temp[val_indices]
    y_val = y_temp[val_indices]

    print("\nData split:")
    print(f"  Train: {len(X_train)} samples ({train_ratio:.1%})")
    print(f"  Val:   {len(X_val)} samples ({val_ratio:.1%})")
    print(f"  Test:  {len(X_test)} samples ({test_ratio:.1%})")
    if stratify_multilabel:
        print("  Split: multilabel stratified")

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
    "load_ir_data",
    "prepare_dataloaders",
]
