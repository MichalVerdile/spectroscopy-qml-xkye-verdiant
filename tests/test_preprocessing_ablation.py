"""End-to-end test of the ablation aggregation on synthetic metrics files."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from spectroscopy_qml.benchmarking.preprocessing import (
    ABLATION_CONFIGS,
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
    assert runner.count('--preprocessing "$PREPROCESSING"') == 4
    assert 'PREPROCESSING="${PREPROCESSING:-legacy}"' in runner


def test_ablation_configs_are_one_factor_at_a_time():
    reference = canonical_spec("shared_mssd").split(",")
    for name, spec in ABLATION_CONFIGS.items():
        if name in ("legacy", "shared_mssd"):
            continue
        changed = [a for a, b in zip(canonical_spec(spec).split(","), reference) if a != b]
        assert len(changed) == 1, (name, changed)


def pytest_approx(value):
    import pytest

    return pytest.approx(value, abs=1e-9)
