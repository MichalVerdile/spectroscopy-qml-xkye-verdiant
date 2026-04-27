"""Shared data loading, caching, and split utilities for the MPS + TTN joint pipeline.

Both training branches (MPS and TTN 10.2) call these helpers to guarantee that
they receive *identical* preprocessed spectra and *identical* train/val/test indices.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

SRC_DIR = Path(__file__).resolve().parents[4]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spectroscopy_qml.ir.mps_encoder_final.data_loader import (  # noqa: E402
    FUNCTIONAL_GROUPS,
    IRSpectraDataset,
    load_ir_data,
    multilabel_train_test_split,
)

__all__ = [
    "FUNCTIONAL_GROUPS",
    "IRSpectraDataset",
    "load_shared_data",
    "load_or_create_split_indices",
    "split_arrays_from_indices",
    "prepare_dataloaders_from_split_indices",
]


def load_shared_data(
    data_dir: Path,
    *,
    target_length: int = 1800,
    apply_snv: bool = True,
    apply_savgol: bool = False,
    savgol_window_length: int = 11,
    savgol_polyorder: int = 3,
    max_files: int | None = None,
    cache_path: Path | None = None,
    overwrite_cache: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Load IR spectra with a single, shared preprocessing for both MPS and TTN.

    Args:
        data_dir: Directory containing raw parquet files.
        target_length: Spectrum interpolation length (default 1800).
        apply_snv: Apply Standard Normal Variate normalisation.
        apply_savgol: Apply Savitzky-Golay smoothing before SNV.
        savgol_window_length: Window length for Savitzky-Golay filter.
        savgol_polyorder: Polynomial order for Savitzky-Golay filter.
        max_files: Optional cap on the number of parquet files to load.
        cache_path: If given, save/load a .npz cache to skip re-loading.
        overwrite_cache: Force re-loading even if the cache exists.

    Returns:
        (X, y) as float32 / int32 numpy arrays.
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists() and not overwrite_cache:
            payload = np.load(cache_path)
            # Reject stale caches that were built with different preprocessing.
            mismatches = []
            if int(payload.get("target_length", -1)) != target_length:
                mismatches.append(f"target_length {int(payload['target_length'])} != {target_length}")
            if bool(payload.get("apply_snv", -1)) != apply_snv:
                mismatches.append(f"apply_snv {bool(payload['apply_snv'])} != {apply_snv}")
            if bool(payload.get("apply_savgol", -1)) != apply_savgol:
                mismatches.append(f"apply_savgol {bool(payload['apply_savgol'])} != {apply_savgol}")
            stored_max = payload.get("max_files", np.array([-1]))[()]
            req_max = -1 if max_files is None else int(max_files)
            if int(stored_max) != req_max:
                mismatches.append(f"max_files {stored_max} != {max_files}")
            if mismatches:
                raise ValueError(
                    f"Spectra cache {cache_path} was built with different settings: "
                    + "; ".join(mismatches)
                    + ". Delete it or pass --overwrite-cache."
                )
            X = payload["X"].astype(np.float32)
            y = payload["y"].astype(np.int32)
            print(f"Loading shared data from cache: {cache_path}  X={X.shape}  y={y.shape}")
            return X, y

    X, y = load_ir_data(
        data_dir=data_dir,
        target_length=target_length,
        max_files=max_files,
        apply_snv=apply_snv,
        apply_savgol=apply_savgol,
        savgol_window_length=savgol_window_length,
        savgol_polyorder=savgol_polyorder,
    )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            X=X.astype(np.float32),
            y=y.astype(np.int32),
            target_length=np.asarray(target_length, dtype=np.int64),
            apply_snv=np.asarray(apply_snv),
            apply_savgol=np.asarray(apply_savgol),
            max_files=np.asarray(-1 if max_files is None else int(max_files), dtype=np.int64),
        )
        print(f"Saved shared data cache to: {cache_path}")

    return X.astype(np.float32), y.astype(np.int32)


def load_or_create_split_indices(
    labels: np.ndarray,
    split_path: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
    overwrite: bool = False,
) -> dict[str, np.ndarray]:
    """Load a persisted split or create one with multilabel stratification.

    The split is saved as an .npz file so that both the MPS and TTN branches
    can reload the exact same indices in any order.

    Args:
        labels: Full label matrix (n_samples, n_classes).
        split_path: Where to save / load the split .npz file.
        train_ratio: Fraction for training (default 0.8).
        val_ratio: Fraction for validation (default 0.1).
        test_ratio: Fraction for testing (default 0.1).
        random_seed: Reproducibility seed.
        overwrite: Re-create the split even if the file exists.

    Returns:
        Dict with keys "train", "val", "test" → int64 index arrays.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0.")

    split_path = Path(split_path)
    num_samples = len(labels)
    num_labels = labels.shape[1] if labels.ndim == 2 else 0

    if split_path.exists() and not overwrite:
        payload = np.load(split_path, allow_pickle=False)
        train_indices = payload["train_indices"].astype(np.int64)
        val_indices = payload["val_indices"].astype(np.int64)
        test_indices = payload["test_indices"].astype(np.int64)
        stored_num_samples = int(payload["num_samples"])
        stored_num_labels = int(payload["num_labels"])
        stored_seed = int(payload["random_seed"])
        stored_train = float(payload["train_ratio"])
        stored_val = float(payload["val_ratio"])
        stored_test = float(payload["test_ratio"])

        mismatches = []
        if stored_num_samples != num_samples:
            mismatches.append(f"num_samples {stored_num_samples} != {num_samples}")
        if stored_num_labels != num_labels:
            mismatches.append(f"num_labels {stored_num_labels} != {num_labels}")
        if stored_seed != random_seed:
            mismatches.append(f"seed {stored_seed} != {random_seed}")
        if not np.isclose([stored_train, stored_val, stored_test],
                          [train_ratio, val_ratio, test_ratio]).all():
            mismatches.append(
                f"ratios {stored_train}/{stored_val}/{stored_test} "
                f"!= {train_ratio}/{val_ratio}/{test_ratio}"
            )
        if mismatches:
            raise ValueError(
                f"Split file {split_path} is incompatible with the current run: "
                + "; ".join(mismatches)
                + ". Delete it or pass --overwrite-split."
            )
        print(
            f"Loaded shared split from {split_path}:  "
            f"train={len(train_indices)}  val={len(val_indices)}  test={len(test_indices)}"
        )
        return {"train": train_indices, "val": val_indices, "test": test_indices}

    # Build the split with multilabel stratification.
    all_idx = np.arange(num_samples, dtype=np.int64).reshape(-1, 1)

    # First split: hold out test set.
    idx_trainval, idx_test, _, _ = multilabel_train_test_split(
        all_idx, labels, test_size=test_ratio, random_seed=random_seed
    )
    idx_trainval = idx_trainval[:, 0]
    idx_test = idx_test[:, 0]

    # Second split: separate train and val from the remaining pool.
    val_size = val_ratio / (train_ratio + val_ratio)
    sub_idx = idx_trainval.reshape(-1, 1)
    sub_labels = labels[idx_trainval]
    idx_train_local, idx_val_local, _, _ = multilabel_train_test_split(
        sub_idx, sub_labels, test_size=val_size, random_seed=random_seed + 1
    )
    train_indices = idx_train_local[:, 0].astype(np.int64)
    val_indices = idx_val_local[:, 0].astype(np.int64)
    test_indices = idx_test.astype(np.int64)

    _validate_split(train_indices, val_indices, test_indices, num_samples)

    split_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        split_path,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        num_samples=np.asarray(num_samples, dtype=np.int64),
        num_labels=np.asarray(num_labels, dtype=np.int64),
        random_seed=np.asarray(random_seed, dtype=np.int64),
        train_ratio=np.asarray(train_ratio, dtype=np.float32),
        val_ratio=np.asarray(val_ratio, dtype=np.float32),
        test_ratio=np.asarray(test_ratio, dtype=np.float32),
    )
    print(
        f"Created shared split at {split_path}:  "
        f"train={len(train_indices)}  val={len(val_indices)}  test={len(test_indices)}"
    )
    return {"train": train_indices, "val": val_indices, "test": test_indices}


def split_arrays_from_indices(
    X: np.ndarray,
    y: np.ndarray,
    split_indices: dict[str, np.ndarray],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Materialize train/val/test array pairs from shared split indices."""
    train_idx = split_indices["train"]
    val_idx = split_indices["val"]
    test_idx = split_indices["test"]
    _validate_split(train_idx, val_idx, test_idx, len(X))
    return {
        "train": (X[train_idx], y[train_idx]),
        "val": (X[val_idx], y[val_idx]),
        "test": (X[test_idx], y[test_idx]),
    }


def prepare_dataloaders_from_split_indices(
    X: np.ndarray,
    y: np.ndarray,
    split_indices: dict[str, np.ndarray],
    batch_size: int = 1024,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders from shared split indices.

    Args:
        X: Spectra array (n_samples, spectrum_length).
        y: Labels array (n_samples, num_classes).
        split_indices: Dict returned by :func:`load_or_create_split_indices`.
        batch_size: Batch size for all loaders.
        num_workers: Number of DataLoader workers.
        pin_memory: Pin tensors to GPU-accessible memory.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    split_arrays = split_arrays_from_indices(X, y, split_indices)
    train_ds = IRSpectraDataset(*split_arrays["train"])
    val_ds = IRSpectraDataset(*split_arrays["val"])
    test_ds = IRSpectraDataset(*split_arrays["test"])

    kw = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, shuffle=False, **kw)
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_split(
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    num_samples: int,
) -> None:
    all_idx = np.concatenate([train_indices, val_indices, test_indices])
    if len(all_idx) != num_samples:
        raise ValueError(
            f"Split contains {len(all_idx)} indices for {num_samples} samples."
        )
    if len(np.unique(all_idx)) != num_samples:
        raise ValueError("Split contains duplicate indices.")
