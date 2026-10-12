"""Shared multilabel split artifacts used by every benchmark model."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


def label_digest(labels: np.ndarray) -> str:
    """Return a stable digest that also detects a different sample order."""
    # Every loader represents the targets as binary values, but some use int32
    # and others float32. Normalize the representation before hashing.
    contiguous = np.ascontiguousarray((labels >= 0.5).astype(np.uint8))
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def create_split_artifact(
    labels: np.ndarray,
    path: Path,
    seed: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> dict[str, np.ndarray]:
    """Create one deterministic 80/10/10 multilabel-stratified split."""
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1")

    indices = np.arange(len(labels), dtype=np.int64)
    first = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=test_ratio, random_state=seed
    )
    trainval_local, test_local = next(first.split(indices, labels))
    trainval_indices = indices[trainval_local]
    test_indices = indices[test_local]

    relative_val_ratio = val_ratio / (train_ratio + val_ratio)
    second = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=relative_val_ratio, random_state=seed + 1
    )
    train_local, val_local = next(
        second.split(trainval_indices, labels[trainval_indices])
    )
    train_indices = trainval_indices[train_local]
    val_indices = trainval_indices[val_local]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        num_samples=np.asarray(len(labels), dtype=np.int64),
        num_labels=np.asarray(labels.shape[1], dtype=np.int64),
        random_seed=np.asarray(seed, dtype=np.int64),
        train_ratio=np.asarray(train_ratio),
        val_ratio=np.asarray(val_ratio),
        test_ratio=np.asarray(test_ratio),
        label_digest=np.asarray(label_digest(labels)),
    )
    return {"train": train_indices, "val": val_indices, "test": test_indices}


def load_split_artifact(
    path: Path, labels: np.ndarray, expected_seed: int | None = None
) -> dict[str, np.ndarray]:
    """Load and strictly validate a shared split against a model's labels."""
    with np.load(path, allow_pickle=False) as payload:
        required = {"train_indices", "val_indices", "test_indices", "num_samples", "num_labels", "random_seed", "label_digest"}
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"Split artifact {path} is missing fields: {sorted(missing)}")
        if int(payload["num_samples"]) != len(labels):
            raise ValueError("Split artifact sample count does not match loaded data")
        if int(payload["num_labels"]) != labels.shape[1]:
            raise ValueError("Split artifact label count does not match loaded data")
        if expected_seed is not None and int(payload["random_seed"]) != expected_seed:
            raise ValueError("Split artifact seed does not match requested seed")
        if str(payload["label_digest"]) != label_digest(labels):
            raise ValueError(
                "Split artifact label digest does not match. The model loaded a different "
                "sample set or sample order, so a fair comparison would be invalid."
            )
        result = {
            "train": payload["train_indices"].astype(np.int64),
            "val": payload["val_indices"].astype(np.int64),
            "test": payload["test_indices"].astype(np.int64),
        }

    combined = np.concatenate(list(result.values()))
    if len(combined) != len(labels) or len(np.unique(combined)) != len(labels):
        raise ValueError("Split indices must be disjoint and cover every sample exactly once")
    if combined.min(initial=0) < 0 or combined.max(initial=-1) >= len(labels):
        raise ValueError("Split artifact contains out-of-range indices")
    return result
