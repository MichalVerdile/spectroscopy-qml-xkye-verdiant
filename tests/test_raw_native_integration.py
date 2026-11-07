"""End-to-end, isolation and resume checks, using real framework optimization."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from spectroscopy_qml.benchmarking import raw_training
from spectroscopy_qml.benchmarking.raw_models import NativeMPS
from spectroscopy_qml.benchmarking.raw_native import prepare_cohort
from spectroscopy_qml.benchmarking.raw_smoke import write_synthetic_parquet
from spectroscopy_qml.benchmarking.raw_splits import cohort_split
from spectroscopy_qml.benchmarking.raw_training import (
    load_checkpoint,
    seed_everything,
    train_neural,
)

ROOT = Path(__file__).resolve().parents[1]


def test_all_twenty_native_end_to_end_and_safe_resume(tmp_path):
    command = [
        sys.executable,
        str(ROOT / "scripts/run_raw_native_benchmark.py"),
        "--smoke-test",
        "--phase",
        "all",
        "--epochs",
        "1",
        "--threads",
        "2",
        "--experiment-id",
        "integration",
        "--output-dir",
        str(tmp_path / "out"),
        "--cache-dir",
        str(tmp_path / "cache"),
    ]
    first = subprocess.run(command, capture_output=True, text=True, cwd=ROOT)
    assert first.returncode == 0, first.stderr
    results = list((tmp_path / "out").glob("synthetic_integration/*/seed_42/*/result.json"))
    assert len(results) == 20
    previous = {str(p): p.read_bytes() for p in results}
    splits = set()
    for modality in ("cnmr", "hnmr", "ir", "msms_pos", "msms_neg"):
        subset = [json.loads(p.read_text()) for p in results if p.parents[2].name == modality]
        assert len(subset) == 4
        assert len({r["spec"]["input_hash"] for r in subset}) == 1
        assert len({r["spec"]["dataset_hash"] for r in subset}) == 1
        splits.update(r["spec"]["split_hash"] for r in subset)
        for path in results:
            with np.load(path.with_name("predictions.npz")) as data:
                assert data["y_test"].shape == data["p_test"].shape
                assert data["y_test"].shape[1] == 37
    assert len(splits) == 1
    resumed = subprocess.run([*command, "--resume"], capture_output=True, text=True, cwd=ROOT)
    assert resumed.returncode == 0, resumed.stderr
    assert "epoch=" not in resumed.stdout and "round=" not in resumed.stdout
    assert previous == {str(p): p.read_bytes() for p in results}
    incompatible = subprocess.run(
        [*command, "--resume", "--patience", "14"], capture_output=True, text=True, cwd=ROOT
    )
    assert incompatible.returncode != 0
    assert "incompatible output reuse" in incompatible.stderr
    assert previous == {str(p): p.read_bytes() for p in results}
    aggregate = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/aggregate_raw_native.py"),
            str(tmp_path / "out/synthetic_integration"),
            "--allow-incomplete",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert aggregate.returncode == 0, aggregate.stderr
    report = json.loads((tmp_path / "out/synthetic_integration/aggregate.json").read_text())
    assert report["synthetic"] is True
    assert len(report["comparisons"]) == 20
    # Artifact corruption cannot be bypassed by --resume.
    results[0].write_text("{}")
    corrupt = subprocess.run([*command, "--resume"], capture_output=True, text=True, cwd=ROOT)
    assert corrupt.returncode != 0 and "integrity" in corrupt.stderr


def test_neural_interrupted_resume_matches_uninterrupted_training(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    lengths = write_synthetic_parquet(tmp_path / "source", compact=True)
    cohort = prepare_cohort(
        tmp_path / "source", tmp_path / "cache", synthetic=True, expected_lengths=lengths
    )
    split = cohort_split(cohort, 42)["partitions"]

    def model():
        seed_everything(42)
        return NativeMPS(37, 2, physical_dim=8, bond_dim=8, window=8, dropout=0.2)

    kwargs = dict(
        cohort=cohort,
        modality="ir",
        split=split,
        identity="test-contract",
        seed=42,
        device="cpu",
        batch_size=16,
        epochs=2,
        patience=15,
    )
    uninterrupted = tmp_path / "full"
    uninterrupted.mkdir()
    train_neural(model(), output=uninterrupted, **kwargs)
    interrupted = tmp_path / "resumed"
    interrupted.mkdir()
    save = raw_training.atomic_checkpoint

    def interrupt_after_save(path, payload):
        save(path, payload)
        if path.name == "last.pt" and payload["epoch"] == 0:
            raise RuntimeError("Simulated interruption")

    monkeypatch.setattr(raw_training, "atomic_checkpoint", interrupt_after_save)
    with pytest.raises(RuntimeError, match="Simulated"):
        train_neural(model(), output=interrupted, **kwargs)
    monkeypatch.setattr(raw_training, "atomic_checkpoint", save)
    train_neural(model(), output=interrupted, resume=True, **kwargs)
    full = load_checkpoint(uninterrupted / "last.pt")["model"]
    resumed = load_checkpoint(interrupted / "last.pt")["model"]
    for name in full:
        torch.testing.assert_close(full[name], resumed[name], rtol=0, atol=0)


def test_tuning_never_predicts_test_and_final_seals_before_test(tmp_path, monkeypatch):
    from spectroscopy_qml.benchmarking import raw_runner

    command = [
        "--smoke-test",
        "--model",
        "mps",
        "--modality",
        "ir",
        "--epochs",
        "1",
        "--device",
        "cpu",
        "--threads",
        "2",
        "--experiment-id",
        "sealed",
        "--output-dir",
        str(tmp_path / "out"),
        "--cache-dir",
        str(tmp_path / "cache"),
    ]
    predict = raw_runner.predict_partition

    def forbidden(*args, **kwargs):
        raise AssertionError("Tuning attempted calibration/test prediction")

    monkeypatch.setattr(raw_runner, "predict_partition", forbidden)
    raw_runner.main([*command, "--phase", "tune"])
    output = tmp_path / "out/synthetic_sealed/ir/seed_42/mps"
    assert (output / "selection.json").exists()
    assert not (output / "predictions.npz").exists()
    assert not (output / "test_seal.json").exists()
    partitions = json.loads((output / "split.json").read_text())["partitions"]
    calls = []

    def checked(model, cohort, modality, indices, device, batch_size, directory):
        if indices == partitions["test"]:
            assert (output / "test_seal.json").exists()
            assert (output / "thresholds.json").exists()
            calls.append("test")
        else:
            assert indices == partitions["calibration"]
            calls.append("calibration")
        return predict(model, cohort, modality, indices, device, batch_size, directory)

    monkeypatch.setattr(raw_runner, "predict_partition", checked)
    raw_runner.main([*command, "--phase", "final"])
    assert calls == ["calibration", "test"]
    raw_runner.verify_complete(output)


def test_cli_help_and_bash_entrypoint():
    for command in [
        [sys.executable, "scripts/run_raw_native_benchmark.py", "--help"],
        ["bash", "scripts/run_raw_native_benchmark.sh", "--help"],
        ["bash", "-n", "cluster/train_raw_native.sbatch"],
    ]:
        subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)


def test_xgb_external_memory_and_per_label_training(tmp_path):
    import xgboost as xgb

    from spectroscopy_qml.benchmarking.raw_training import RawRows, predict_xgb, train_xgb

    lengths = write_synthetic_parquet(tmp_path / "source", compact=True)
    cohort = prepare_cohort(
        tmp_path / "source", tmp_path / "cache", synthetic=True, expected_lengths=lengths
    )
    split = cohort_split(cohort, 42)["partitions"]
    output = tmp_path / "trial"
    output.mkdir()
    config = {"max_depth": 2, "learning_rate": 0.2, "n_estimators": 3}
    models, constants, stats = train_xgb(
        cohort,
        "ir",
        split,
        output,
        "external-test",
        seed=42,
        device="cpu",
        config=config,
        threads=2,
        memory_mode="external",
        external_batch_size=16,
    )
    assert stats["xgb_memory_mode"] == "external"
    features = cohort.features("ir", split["selection"])
    # The neural adapter and XGBoost use exactly the same canonical bytes.
    rows = RawRows(cohort, "ir", split["selection"])
    for i in range(len(rows)):
        np.testing.assert_array_equal(rows[i][0].numpy(), features[i])
    p = predict_xgb(models, features.reshape(len(features), -1), constants)
    assert p.shape == (len(features), 37) and np.isfinite(p).all()
    # Independent label-specific reference verifies target switching on the shared
    # DMatrix, using identical histogram cuts/round count on this small fixture.
    train_x = cohort.features("ir", split["train"]).reshape(len(split["train"]), -1)
    y = cohort.labels[split["train"]]
    for j in [i for i, c in enumerate(constants) if 0 < c < 1][:3]:
        dtrain = xgb.DMatrix(train_x, label=y[:, j], nthread=2)
        reference = xgb.train(
            {
                "max_depth": 2,
                "learning_rate": 0.2,
                "tree_method": "hist",
                "objective": "binary:logistic",
                "base_score": constants[j],
                "seed": 42,
                "nthread": 2,
            },
            dtrain,
            num_boost_round=stats["best_step"] + 1,
        )
        np.testing.assert_allclose(
            p[:, j],
            reference.predict(xgb.DMatrix(features.reshape(len(features), -1), nthread=2)),
            atol=1e-6,
        )


@pytest.mark.skipif(
    not os.getenv("RAW_NATIVE_REAL_PARQUET"),
    reason="Set RAW_NATIVE_REAL_PARQUET to a materialized source shard",
)
def test_real_parquet_measurements_and_metadata(tmp_path):
    import pyarrow.parquet as pq

    from spectroscopy_qml.benchmarking.raw_native import COLUMNS, MODALITIES

    path = Path(os.environ["RAW_NATIVE_REAL_PARQUET"])
    source = tmp_path / "source"
    source.mkdir()
    (source / path.name).symlink_to(path.resolve())
    cohort = prepare_cohort(
        source,
        tmp_path / "cache",
        max_samples=128,
        metadata_path=ROOT / "data/meta_data/spectrum_dimensions.json",
    )
    rows = next(
        pq.ParquetFile(path).iter_batches(
            batch_size=128, columns=[COLUMNS[m][0] for m in MODALITIES]
        )
    ).to_pylist()
    for i, row in enumerate(rows):
        for m in MODALITIES:
            original = np.asarray(row[COLUMNS[m][0]], dtype=float).reshape(
                -1, 2 if m.startswith("msms") else 1
            )
            mask = cohort.masks(m)[i]
            np.testing.assert_array_equal(cohort.values(m)[i][mask], original)
