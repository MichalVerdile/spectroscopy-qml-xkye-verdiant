"""Measurement invariance, common cohorts and leakage prevention contracts."""

import ast
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from spectroscopy_qml.benchmarking import preprocessing
from spectroscopy_qml.benchmarking.raw_guard import strict_raw_guard
from spectroscopy_qml.benchmarking.raw_labels import LABEL_NAMES, SMARTS, molecule_metadata
from spectroscopy_qml.benchmarking.raw_native import (
    COLUMNS,
    MODALITIES,
    RawCohort,
    checked_float32,
    prepare_cohort,
)
from spectroscopy_qml.benchmarking.raw_smoke import write_synthetic_parquet
from spectroscopy_qml.benchmarking.raw_splits import cohort_split, validate_split


@pytest.fixture
def raw_fixture(tmp_path):
    source = tmp_path / "data"
    lengths = write_synthetic_parquet(source, compact=True)
    cohort = prepare_cohort(source, tmp_path / "cache", expected_lengths=lengths, synthetic=True)
    return source, lengths, cohort


def test_original_measurements_order_and_all_peaks_preserved(raw_fixture):
    source, _, cohort = raw_fixture
    rows = pq.read_table(source / "synthetic_raw.parquet").to_pylist()
    assert len(rows) == len(cohort.labels)
    for i, row in enumerate(rows):
        for modality in MODALITIES:
            value = np.asarray(row[COLUMNS[modality][0]], dtype=float)
            channels = 2 if modality.startswith("msms") else 1
            value = value.reshape(-1, channels)
            mask = cohort.masks(modality)[i]
            np.testing.assert_array_equal(cohort.values(modality)[i][mask], value)
            assert mask.sum() == len(value)
            assert not cohort.values(modality)[i][~mask].any()
            assert cohort.records[i]["original_dimensions"][modality] == len(value)
            features = cohort.features(modality, [i])
            np.testing.assert_allclose(features[0, mask, :-1], value, rtol=1e-6, atol=0)
            np.testing.assert_array_equal(features[0, :, -1], mask)


def test_true_zero_peak_distinct_from_padding(tmp_path):
    source = tmp_path / "data"
    lengths = write_synthetic_parquet(source, compact=True)
    path = source / "synthetic_raw.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[0]["msms_positive_40ev"] = [[0.0, 0.0], [100.001, 7.0], [100.009, 12.0], [9999.0, 500.0]]
    pq.write_table(pa.Table.from_pylist(rows), path)
    cohort = prepare_cohort(source, tmp_path / "cache", expected_lengths=lengths, synthetic=True)
    x = cohort.features("msms_pos", [0])[0]
    np.testing.assert_array_equal(x[0], [0, 0, 1])
    assert x[1, 0] != x[2, 0]  # Previous dense binning would merge these peaks.
    assert x[3, 0] == 9999
    assert x[4, -1] == 0


@pytest.mark.parametrize("bad", [None, [float("nan")], [float("inf")], []])
def test_common_invalid_row_policy(tmp_path, bad):
    source = tmp_path / "data"
    lengths = write_synthetic_parquet(source, compact=True)
    path = source / "synthetic_raw.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[0]["ir_spectra"] = bad
    pq.write_table(pa.Table.from_pylist(rows), path)
    with pytest.raises(ValueError, match="Invalid sample"):
        prepare_cohort(source, tmp_path / "cache", expected_lengths=lengths, synthetic=True)
    cohort = prepare_cohort(
        source,
        tmp_path / "cache",
        expected_lengths=lengths,
        synthetic=True,
        invalid_policy="exclude",
    )
    assert len(cohort.labels) == len(rows) - 1
    assert cohort.manifest["rejected_count"] == 1
    assert all(len(cohort.values(m)) == len(rows) - 1 for m in MODALITIES)
    assert cohort.records[0]["row"] == 1


def test_precision_failure_is_explicit():
    for bad in [1e100, 1e-100]:
        with pytest.raises(ValueError, match="float32"):
            checked_float32(np.array([bad]))


def test_native_dimensions_cannot_be_silently_overridden(raw_fixture, tmp_path):
    source, lengths, _ = raw_fixture
    with pytest.raises(ValueError, match="cannot be overridden"):
        prepare_cohort(source, tmp_path / "other", expected_lengths=lengths)
    with pytest.raises(ValueError, match="observed"):
        prepare_cohort(source, tmp_path / "other")


def test_content_hash_detects_source_and_cache_changes(raw_fixture, tmp_path):
    source, lengths, cohort = raw_fixture
    again = prepare_cohort(source, tmp_path / "cache", expected_lengths=lengths, synthetic=True)
    assert again.manifest["dataset_hash"] == cohort.manifest["dataset_hash"]
    path = source / "synthetic_raw.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[0]["ir_spectra"][0] += 1
    pq.write_table(pa.Table.from_pylist(rows), path)
    changed = prepare_cohort(source, tmp_path / "cache", expected_lengths=lengths, synthetic=True)
    assert changed.manifest["dataset_hash"] != cohort.manifest["dataset_hash"]
    target = changed.root / "ir_values.npy"
    values = np.load(target, mmap_mode="r+")
    values[0, 0, 0] = 999
    values.flush()
    with pytest.raises(ValueError, match="changed"):
        RawCohort.open(changed.root)


def test_strict_raw_tripwire_and_loader_do_not_call_preprocessing(
    raw_fixture, tmp_path, monkeypatch
):
    source, lengths, _ = raw_fixture
    with pytest.raises(RuntimeError, match="Preprocessing"):
        with strict_raw_guard():
            preprocessing.parse_preprocessing("none")

    def forbidden(*args, **kwargs):
        raise AssertionError("Historical preprocessing called")

    for name in (
        "preprocess_spectra",
        "resample",
        "make_msms_spectrum",
        "legacy_linear_interpolate",
    ):
        monkeypatch.setattr(preprocessing, name, forbidden)
    monkeypatch.setattr(np, "interp", forbidden)
    with strict_raw_guard():
        prepare_cohort(source, tmp_path / "new_cache", expected_lengths=lengths, synthetic=True)


def test_vocabulary_matches_all_historical_families():
    root = Path(__file__).resolve().parents[1]
    paths = [
        root / "benchmark/cnn/scripts/run_cnn_jung_baseline.py",
        root / "benchmark/xgb/scripts/run_xgb_baseline.py",
    ]
    paths += list((root / "src/spectroscopy_qml/mps_classifier").glob("*/data_loader.py"))
    paths += list(
        (root / "src/spectroscopy_qml/tree_tensor_network").glob("*/helpers/data_loader.py")
    )
    for path in paths:
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(n, ast.Name) and n.id.lower() == "functional_groups"
                for n in node.targets
            ):
                value = node.value
                historical = {
                    ast.literal_eval(k): ast.literal_eval(v.args[0])
                    for k, v in zip(value.keys, value.values, strict=True)
                }
                assert list(historical.items()) == list(SMARTS.items()), str(path)
    assert len(LABEL_NAMES) == 37


@pytest.mark.parametrize("kind", ["molecule", "scaffold"])
def test_split_consistency_group_leakage_and_quality(raw_fixture, kind):
    _, _, cohort = raw_fixture
    a, b = cohort_split(cohort, 42, kind), cohort_split(cohort, 42, kind)
    assert a == b
    assert a["split_hash"] != cohort_split(cohort, 43, kind)["split_hash"]
    for grouping in ["molecule", kind]:
        validate_split(
            a["partitions"], cohort.labels, [r[f"{grouping}_id"] for r in cohort.records]
        )
    assert 0.7 <= a["quality"]["train"]["fraction"] <= 0.9
    assert all("label_support" in v for v in a["quality"].values())


def test_canonical_smiles_identity():
    assert molecule_metadata("CCO")[1] == molecule_metadata("OCC")[1]
    assert molecule_metadata("C[C@H](O)C(=O)O")[1] == molecule_metadata("C[C@@H](O)C(=O)O")[1]


def test_lfs_pointer_fails_explicitly(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data/a.parquet").write_text("version https://git-lfs.github.com/spec/v1\n")
    with pytest.raises(ValueError, match="LFS pointer"):
        prepare_cohort(tmp_path / "data", tmp_path / "cache")
