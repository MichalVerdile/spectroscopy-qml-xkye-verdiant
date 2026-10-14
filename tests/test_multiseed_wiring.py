"""Regression tests for the multi-seed benchmark entrypoint wiring."""

from __future__ import annotations

import ast
import re
from pathlib import Path

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
