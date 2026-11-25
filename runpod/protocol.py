"""Infrastructure helpers. No scientific model or preprocessing overrides."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REFERENCE = "01e270ee8ad79bb0978f96a709113df3d1f3b017"
SEEDS = (41, 42, 43, 44, 45)
# CLI column, actual Parquet column, existing trainer output name, input dimension.
MODALITIES = {
    "cnmr": ("c_nmr_spectra", "c_nmr_spectra", "cnmr", 10000),
    "hnmr": ("h_nmr_spectra", "h_nmr_spectra", "hnmr", 10000),
    "ir": ("ir_spectra", "ir_spectra", "ir", 1800),
    "msms_pos": ("pos_msms", "msms_positive_40ev", "pos_msms", 10000),
    "msms_neg": ("neg_msms", "msms_negative_40ev", "neg_msms", 10000),
}
SCIENCE_PATHS = [
    "benchmark/xgb/scripts/run_xgb_baseline.py",
    "src/spectroscopy_qml/benchmarking/preprocessing.py",
    "src/spectroscopy_qml/benchmarking/splits.py",
    "scripts/create_multilabel_split.py",
    "scripts/validate_benchmark_result.py",
]


def stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    """Whole-object write/close, not rename/append/locking on Global Volumes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(data, indent=2, allow_nan=False) + "\n").encode()
    with path.open("wb") as f:
        f.write(encoded)
    if path.read_bytes() != encoded:
        raise OSError(f"Write/read verification failed: {path}")


def publish(source, target):
    """Copy closed files without chmod, hard links or atomic rename."""
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = sha256(source)
    with source.open("rb") as src, target.open("wb") as dst:
        shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
    if sha256(target) != expected:
        raise OSError(f"Persistent copy verification failed: {target}")
    return expected


def check_science(repo):
    repo = Path(repo)
    hashes = {}
    for name in SCIENCE_PATHS:
        expected = subprocess.check_output(["git", "-C", str(repo), "show", f"{REFERENCE}:{name}"])
        actual = (repo / name).read_bytes()
        if actual != expected:
            raise ValueError(f"Scientific source differs from reference: {name}")
        hashes[name] = hashlib.sha256(actual).hexdigest()
    return hashes


def run_config(modality, seed, split, split_hash, dataset_hash):
    return {
        "model": "xgboost", "modality": modality, "seed": seed,
        "input_dim": MODALITIES[modality][3], "preprocessing": "none",
        "threshold": 0.5, "threshold_rule": "probability > 0.5 (XGBClassifier.predict)",
        "n_estimators": 1000000, "early_stopping_patience": 15,
        "early_stopping_metric": "validation logloss, independently per label",
        "learning_curve_every": 20, "no_kfold": True, "device": "cuda",
        "reference_commit": REFERENCE, "split_path": str(split),
        "split_sha256": split_hash, "dataset_manifest_sha256": dataset_hash,
    }


def effective_memory_bytes():
    limits = [int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))]
    for file in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            value = Path(file).read_text().strip()
            if value != "max":
                limits.append(int(value))
        except (OSError, ValueError):
            pass
    return min(limits)
