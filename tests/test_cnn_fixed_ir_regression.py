from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CASES_FILE = REPO_ROOT / "benchmark/cnn/fixed_ir_cases.jsonl"
DEFAULT_DATA_DIR = REPO_ROOT / "data/raw"
DEFAULT_MODEL_PATH = REPO_ROOT / "benchmark/cnn/models/ir/model.keras"


def _load_jung_baseline_module() -> ModuleType:
    script_dir = REPO_ROOT / "benchmark/cnn/scripts"
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    module_path = script_dir / "run_cnn_jung_baseline.py"
    spec = importlib.util.spec_from_file_location("run_cnn_jung_baseline", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_cases() -> list[dict]:
    if not CASES_FILE.exists():
        pytest.skip(f"Missing fixed cases file: {CASES_FILE}")
    rows = []
    with CASES_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        pytest.skip(f"No cases in {CASES_FILE}")
    return rows


def _data_dir() -> Path:
    return Path(os.getenv("CNN_ANALYTICAL_DATA_DIR", str(DEFAULT_DATA_DIR)))


def _model_path() -> Path:
    return Path(os.getenv("CNN_IR_MODEL_PATH", str(DEFAULT_MODEL_PATH)))


def _load_case_spectra(cases: list[dict], interpolate_fn) -> tuple[np.ndarray, list[str]]:
    data_dir = _data_dir()
    if not data_dir.exists():
        pytest.skip(
            "IR parquet directory not found. " f"Set CNN_ANALYTICAL_DATA_DIR or create {data_dir}"
        )

    parquet_cache: dict[str, pd.DataFrame] = {}
    spectra: list[np.ndarray] = []
    ids: list[str] = []

    for case in cases:
        parquet_name = case["parquet_file"]
        row_idx = int(case["row_idx"])
        parquet_path = data_dir / parquet_name
        if not parquet_path.exists():
            pytest.skip(f"Missing parquet file for fixed case: {parquet_path}")

        if parquet_name not in parquet_cache:
            parquet_cache[parquet_name] = pd.read_parquet(parquet_path, columns=["ir_spectra"])

        row = parquet_cache[parquet_name].iloc[row_idx]
        spectra.append(np.asarray(interpolate_fn(row["ir_spectra"]), dtype=np.float32))
        ids.append(case["sample_id"])

    return np.stack(spectra), ids


def _expected_vectors(cases: list[dict], group_names: list[str]) -> np.ndarray:
    rows = []
    for case in cases:
        expected = set(case["expected_groups"])
        rows.append([1 if name in expected else 0 for name in group_names])
    return np.asarray(rows, dtype=int)


def _predict_binary(model, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    raw = model.predict(X.reshape(len(X), 600, 1), verbose=0)
    arr = np.asarray(raw)
    if np.issubdtype(arr.dtype, np.floating) and arr.min() >= 0.0 and arr.max() <= 1.0:
        return (arr >= threshold).astype(int)
    return arr.astype(int)


def _vector_to_groups(vector: np.ndarray, group_names: list[str]) -> list[str]:
    return [
        name for name, flag in zip(group_names, vector.tolist(), strict=False) if int(flag) == 1
    ]


def _split_smoke_fixed_cases(
    X: np.ndarray, y: np.ndarray, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic tiny split for 10-case smoke training."""
    if len(X) < 10:
        pytest.skip(f"Smoke test expects at least 10 fixed cases, got {len(X)}")

    idx = np.arange(len(X))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

    # Use exactly 10 fixed cases for the smoke test.
    idx = idx[:10]
    train_idx = idx[:8]
    val_idx = idx[8:9]
    test_idx = idx[9:10]

    return (
        X[train_idx],
        y[train_idx],
        X[val_idx],
        y[val_idx],
        X[test_idx],
        y[test_idx],
    )


def test_fixed_ir_cases_expected_groups_match_jung_reference() -> None:
    jung = _load_jung_baseline_module()
    group_names = list(jung.functional_groups.keys())

    mismatches: list[str] = []
    for case in _load_cases():
        calc = jung.get_functional_groups(case["smiles"])
        calc_groups = _vector_to_groups(np.asarray(calc, dtype=int), group_names)
        expected = sorted(case["expected_groups"])
        if sorted(calc_groups) != expected:
            mismatches.append(
                f"{case['sample_id']}: expected={expected}, calc={sorted(calc_groups)}"
            )

    assert not mismatches, "Expected groups mismatch in fixed_ir_cases.jsonl:\n" + "\n".join(
        mismatches
    )


@pytest.mark.slow
@pytest.mark.integration
def test_fixed_ir_cases_match_saved_cnn_model_predictions() -> None:
    model_path = _model_path()
    if not model_path.exists():
        pytest.skip(
            "No saved IR CNN model found. "
            "Set CNN_IR_MODEL_PATH or create benchmark/cnn/models/ir/model.keras"
        )

    pytest.importorskip("keras")
    from keras.models import load_model

    jung = _load_jung_baseline_module()
    group_names = list(jung.functional_groups.keys())
    cases = _load_cases()

    X, sample_ids = _load_case_spectra(cases, jung.interpolate_to_600)
    y_expected = _expected_vectors(cases, group_names)

    model = load_model(model_path, compile=False)
    y_pred = _predict_binary(model, X, threshold=0.5)

    assert y_pred.shape == y_expected.shape == (len(cases), len(group_names))

    mismatches: list[str] = []
    for sample_id, pred_vec, exp_vec in zip(sample_ids, y_pred, y_expected, strict=False):
        if not np.array_equal(pred_vec, exp_vec):
            mismatches.append(
                f"{sample_id}: expected={_vector_to_groups(exp_vec, group_names)}, "
                f"predicted={_vector_to_groups(pred_vec, group_names)}"
            )

    assert not mismatches, "CNN fixed-case regression mismatches:\n" + "\n".join(mismatches)


@pytest.mark.slow
@pytest.mark.integration
def test_smoke_train_jung_cnn_on_10_fixed_cases() -> None:
    if os.getenv("CNN_ENABLE_SMOKE_TRAIN", "0") != "1":
        pytest.skip(
            "Set CNN_ENABLE_SMOKE_TRAIN=1 to run the 10-case Jung CNN smoke train/predict test"
        )

    jung = _load_jung_baseline_module()
    group_names = list(jung.functional_groups.keys())
    cases = _load_cases()

    X, _ = _load_case_spectra(cases, jung.interpolate_to_600)
    y = _expected_vectors(cases, group_names)

    X_train, y_train, X_val, y_val, X_test, y_test = _split_smoke_fixed_cases(X, y, seed=42)

    y_pred = jung.train_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        num_fgs=len(group_names),
        aug="e",
        num=0,
        weighted=0,
    )

    y_pred = np.asarray(y_pred, dtype=int)
    assert y_pred.shape == y_test.shape == (1, len(group_names))
    assert set(np.unique(y_pred)).issubset({0, 1})
