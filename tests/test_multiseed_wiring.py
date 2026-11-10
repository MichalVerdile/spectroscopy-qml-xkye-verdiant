"""Regression tests for the multi-seed benchmark entrypoint wiring."""

from __future__ import annotations

import ast
import os
import re
import subprocess
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MODALITIES = {"cnmr", "hnmr", "ir", "msms_pos", "msms_neg"}


def _literal_assignment(path: Path, variable_name: str):
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(
                isinstance(target, ast.Name) and target.id == variable_name
                for target in node.targets
            ):
                return ast.literal_eval(node.value)
    raise AssertionError(f"Assignment {variable_name!r} not found in {path}")


def test_mps_training_modules_match_refactored_package_layout():
    adapter = PROJECT_ROOT / "scripts" / "run_mps_benchmark.py"
    modules = _literal_assignment(adapter, "MPS_TRAINING_MODULES")

    assert set(modules) == EXPECTED_MODALITIES
    for module_name in modules.values():
        module_path = PROJECT_ROOT / "src" / Path(*module_name.split(".")).with_suffix(".py")
        assert module_path.is_file(), f"MPS training module does not exist: {module_path}"


def test_ttn_training_scripts_match_refactored_package_layout():
    runner = PROJECT_ROOT / "scripts" / "run_multiseed_benchmark.sh"
    runner_source = runner.read_text()
    configured_paths = set(re.findall(r'ttn_script="([^"]+)"', runner_source))
    expected_paths = {
        f"src/spectroscopy_qml/tree_tensor_network/{modality}/train.py"
        for modality in EXPECTED_MODALITIES
    }

    assert configured_paths == expected_paths
    for relative_path in configured_paths:
        assert (PROJECT_ROOT / relative_path).is_file(), (
            f"TTN script does not exist: {relative_path}"
        )


def test_multiseed_runner_defaults_cover_requested_experiment_matrix():
    runner_source = (PROJECT_ROOT / "scripts" / "run_multiseed_benchmark.sh").read_text()

    assert 'SEEDS_STRING="${SEEDS:-41 42 43 44 45}"' in runner_source
    assert 'MODALITIES_STRING="${MODALITIES:-cnmr hnmr ir msms_pos msms_neg}"' in runner_source
    assert (PROJECT_ROOT / "scripts" / "run_all_modalities_600.sh").is_file()
    assert (PROJECT_ROOT / "scripts" / "run_all_modalities_1800.sh").is_file()


def test_submit_wrapper_derives_array_from_seed_count(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_sbatch = fake_bin / "sbatch"
    fake_sbatch.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n')
    fake_sbatch.chmod(0o755)
    result = subprocess.run(
        [str(PROJECT_ROOT / "cluster" / "submit_best_preprocessing_multiseed_patience15.sh")],
        cwd=PROJECT_ROOT,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "SEED_LIST": "41 42 43 44 45",
        },
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--array=0-24%5" in result.stdout.splitlines()
    assert "--export=ALL,SEED_LIST" in result.stdout.splitlines()


def test_600_point_sweep_runners_reuse_absolute_paths():
    for model in ("cnn", "mps", "ttn", "xgb"):
        source = (PROJECT_ROOT / "scripts" / f"run_{model}_all_modalities_600.sh").read_text()
        assert 'RUNNER_PATH="$PWD/scripts/$(basename "${BASH_SOURCE[0]}")"' in source
        assert 'sweep_preprocessing_combinations "$RUNNER_PATH"' in source


def test_explicit_empty_preprocessing_does_not_start_sweep():
    command = (
        "source scripts/preprocessing_combinations.sh; "
        "if sweep_preprocessing_combinations /does/not/exist /tmp/out; "
        "then exit 9; else exit 0; fi"
    )
    result = subprocess.run(
        ["bash", "-c", command],
        cwd=PROJECT_ROOT,
        env={**os.environ, "PREPROCESSING": ""},
    )
    assert result.returncode == 0


def test_xgboost_prediction_clamps_each_target_to_fitted_rounds():
    source_path = PROJECT_ROOT / "benchmark" / "xgb" / "scripts" / "run_xgb_baseline.py"
    tree = ast.parse(source_path.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "predict_multioutput_at_step"
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {"np": np, "MultiOutputClassifier": object}
    exec(compile(module, str(source_path), "exec"), namespace)

    class Booster:
        def __init__(self, rounds):
            self.rounds = rounds

        def num_boosted_rounds(self):
            return self.rounds

    class Estimator:
        def __init__(self, rounds):
            self.booster = Booster(rounds)
            self.iteration_ranges = []

        def get_booster(self):
            return self.booster

        def predict(self, X, iteration_range):
            self.iteration_ranges.append(iteration_range)
            return np.zeros(len(X), dtype=int)

    estimators = [Estimator(2), Estimator(5)]
    model = type("Model", (), {"estimators_": estimators})()
    predictions = namespace["predict_multioutput_at_step"](model, np.zeros((3, 2)), 4)

    assert predictions.shape == (3, 2)
    assert estimators[0].iteration_ranges == [(0, 2)]
    assert estimators[1].iteration_ranges == [(0, 4)]
