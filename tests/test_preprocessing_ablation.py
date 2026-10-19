"""End-to-end test of the ablation aggregation on synthetic metrics files."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from spectroscopy_qml.benchmarking.preprocessing import (
    ABLATION_CONFIGS,
    EXTRA_PREPROCESSING_CONFIGS,
    SHARED_PREPROCESSING_CONFIGS,
    canonical_spec,
    metrics_fields,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEEDS = (41, 42, 43, 44, 45)
MODELS = {
    "cnn": "cnn/ir/original",
    "xgboost": "xgboost/ir/original",
    "mps": "mps/results",
    "ttn": "ttn",
}
BASE_SCORE = {"cnn": 0.80, "xgboost": 0.75, "mps": 0.70, "ttn": 0.72}
# Synthetic truth: bin_max helps every model by 0.05, SNV only helps MPS.
CONFIG_SHIFT = {
    "legacy": {},
    "shared_mssd": {},
    "resample_bin_max": {"cnn": 0.05, "xgboost": 0.05, "mps": 0.05, "ttn": 0.05},
    "norm_snv": {"mps": 0.03},
}


def _write_ablation(root: Path) -> None:
    for dim, configs in ((600, CONFIG_SHIFT), (1800, {"shared_mssd": {"cnn": -0.02}})):
        for config, shift in configs.items():
            for seed in SEEDS:
                split_path = str(root / "splits" / f"ir_seed_{seed}.npz")
                for model, subdir in MODELS.items():
                    path = (
                        root
                        / f"dim_{dim}"
                        / config
                        / "ir"
                        / f"seed_{seed}"
                        / subdir
                        / "metrics.json"
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    score = BASE_SCORE[model] + shift.get(model, 0.0) + 0.001 * (seed - 43)
                    payload = {
                        "model": model,
                        "modality": "ir",
                        "seed": seed,
                        "test_f1_micro": score,
                        "test_f1_macro": score - 0.2,
                        "split_path": split_path,
                        "input_dim": dim,
                        "input_digest": "0" * 64,
                        **metrics_fields(config, model),
                    }
                    path.write_text(json.dumps(payload))


def test_aggregation_separates_architecture_preprocessing_and_representation(tmp_path):
    _write_ablation(tmp_path)
    subprocess.run(
        [
            sys.executable,
            "scripts/aggregate_preprocessing_ablation.py",
            "--ablation-dir",
            str(tmp_path),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )
    summary = tmp_path / "summary"

    prep = pd.read_csv(summary / "preprocessing_effect.csv")
    micro = prep[prep["metric"] == "test_f1_micro"].set_index(["config", "model"])
    assert micro.loc[("resample_bin_max", "cnn"), "delta_mean"] == pytest_approx(0.05)
    assert micro.loc[("resample_bin_max", "cnn"), "seeds_improved"] == 5
    assert micro.loc[("norm_snv", "mps"), "delta_mean"] == pytest_approx(0.03)
    assert micro.loc[("norm_snv", "cnn"), "delta_mean"] == pytest_approx(0.0)
    assert set(prep["factor"]) == {"resample", "norm"}

    rep = pd.read_csv(summary / "representation_effect.csv")
    rep_micro = rep[rep["metric"] == "test_f1_micro"].set_index(["factor", "level", "model"])
    assert rep_micro.loc[("input_dim", 1800, "cnn"), "delta_mean"] == pytest_approx(-0.02)

    harm = pd.read_csv(summary / "harmonization_effect.csv")
    assert set(harm["factor"]) == {"legacy"}

    arch = pd.read_csv(summary / "architecture_effect.csv")
    ref = arch[
        (arch["config"] == "shared_mssd")
        & (arch["input_dim"] == 600)
        & (arch["metric"] == "test_f1_micro")
    ]
    assert ref.set_index("model").loc["cnn", "rank"] == 1

    sizes = pd.read_csv(summary / "effect_sizes.csv").set_index("metric")
    assert sizes.loc["test_f1_micro", "reference_winner"] == "cnn"
    assert sizes.loc["test_f1_micro", "architecture_range"] == pytest_approx(0.10)
    assert (
        (summary / "report.md")
        .read_text()
        .startswith("# Preprocessing and representation ablation")
    )


def test_multiseed_runner_passes_preprocessing_to_all_four_models():
    runner = (PROJECT_ROOT / "scripts" / "run_multiseed_benchmark.sh").read_text()
    assert runner.count('\n            --preprocessing "$PREPROCESSING"') == 4
    assert 'PREPROCESSING="${PREPROCESSING:-legacy}"' in runner


def test_precompute_runner_resolves_shared_and_legacy_configs(tmp_path):
    pd.DataFrame({"value": [1]}).to_parquet(tmp_path / "chunk.parquet")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/precompute_preprocessing_cache.py",
            "--data-dir",
            str(tmp_path),
            "--dry-run",
            "--modalities",
            "ir",
            "--input-dims",
            "600",
            "--configs",
            "shared_mssd",
            "norm_snv",
            "legacy",
        ],
        cwd=PROJECT_ROOT,
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")},
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.count("shared:") == 2
    assert "legacy: CNN/XGBoost linear; MPS Savitzky-Golay+SNV; TTN SNV" in result.stdout
    assert "Preprocessing cache plan complete." in result.stdout


def test_ablation_configs_are_one_factor_at_a_time():
    assert len(SHARED_PREPROCESSING_CONFIGS) == 13
    assert "legacy" not in SHARED_PREPROCESSING_CONFIGS
    assert list(ABLATION_CONFIGS)[0] == "legacy"
    assert not set(EXTRA_PREPROCESSING_CONFIGS) & set(ABLATION_CONFIGS)
    reference = canonical_spec("shared_mssd").split(",")
    for name, spec in {**SHARED_PREPROCESSING_CONFIGS, **EXTRA_PREPROCESSING_CONFIGS}.items():
        if name == "shared_mssd":
            continue
        changed = [a for a, b in zip(canonical_spec(spec).split(","), reference) if a != b]
        assert len(changed) == 1, (name, changed)


def test_cli_lists_exactly_the_default_shared_configs():
    result = subprocess.run(
        [sys.executable, "-m", "spectroscopy_qml.benchmarking.preprocessing", "--list"],
        cwd=PROJECT_ROOT,
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")},
        capture_output=True,
        text=True,
        check=True,
    )
    assert [line.split("\t", 1)[0] for line in result.stdout.splitlines()] == list(
        SHARED_PREPROCESSING_CONFIGS
    )


def test_aggregation_rejects_nonidentical_shared_inputs(tmp_path):
    _write_ablation(tmp_path)
    metrics = (
        tmp_path / "dim_600" / "shared_mssd" / "ir" / "seed_41" / "mps" / "results" / "metrics.json"
    )
    payload = json.loads(metrics.read_text())
    payload["input_digest"] = "f" * 64
    metrics.write_text(json.dumps(payload))
    result = subprocess.run(
        [
            sys.executable,
            "scripts/aggregate_preprocessing_ablation.py",
            "--ablation-dir",
            str(tmp_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "byte-identical preprocessed inputs" in result.stderr


def test_completed_result_validation_rejects_stale_preprocessing(tmp_path):
    split_path = tmp_path / "split.npz"
    split_path.touch()
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "model": "mps",
                "seed": 41,
                "input_dim": 600,
                "input_digest": "0" * 64,
                "preprocessing": canonical_spec("norm_snv"),
                "split_path": str(split_path),
            }
        )
    )
    command = [
        sys.executable,
        "scripts/validate_benchmark_result.py",
        "--metrics",
        str(metrics_path),
        "--model",
        "mps",
        "--input-dim",
        "600",
        "--seed",
        "41",
        "--split-path",
        str(split_path),
        "--preprocessing",
    ]
    assert subprocess.run([*command, "norm_snv"], cwd=PROJECT_ROOT).returncode == 0
    assert subprocess.run([*command, "shared_mssd"], cwd=PROJECT_ROOT).returncode != 0


def pytest_approx(value):
    import pytest

    return pytest.approx(value, abs=1e-9)


def test_ablation_runner_runs_legacy_and_skips_noop_ir_resampling(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "run_preprocessing_ablation.sh").write_text(
        (PROJECT_ROOT / "scripts" / "run_preprocessing_ablation.sh").read_text()
    )
    (scripts / "run_multiseed_benchmark.sh").write_text(
        '#!/usr/bin/env bash\necho "RUN $1 $PREPROCESSING [$MODALITIES]"\n'
    )
    (scripts / "aggregate_preprocessing_ablation.py").write_text("")
    for script in scripts.glob("*.sh"):
        script.chmod(0o755)
    (tmp_path / "src").symlink_to(PROJECT_ROOT / "src")
    result = subprocess.run(
        [str(scripts / "run_preprocessing_ablation.sh"), "600", "1800"],
        env={**os.environ, "PYTHON": sys.executable},
        capture_output=True,
        text=True,
        check=True,
    )
    runs = [line for line in result.stdout.splitlines() if line.startswith("RUN")]
    assert len(runs) == 2 * len(ABLATION_CONFIGS)
    assert "RUN 600 legacy [ir cnmr]" in runs
    assert "RUN 600 resample_fir [ir cnmr]" in runs
    assert "RUN 1800 resample_fir [cnmr]" in runs
    assert "RUN 1800 shared_mssd [ir cnmr]" in runs
