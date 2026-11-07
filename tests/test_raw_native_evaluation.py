import numpy as np
import pytest
from sklearn.metrics import f1_score

from spectroscopy_qml.benchmarking.raw_aggregate import describe, exact_sign_flip_p, holm_adjust
from spectroscopy_qml.benchmarking.raw_evaluation import EarlyStopping, evaluate, tune_thresholds
from spectroscopy_qml.benchmarking.raw_experiments import DEFAULTS, trial_configs


def test_threshold_search_improves_known_case_and_per_class_is_not_stub():
    y = np.tile([[0, 0], [1, 1]], (100, 1))
    p = np.tile([[0.1, 0.65], [0.3, 0.85]], (100, 1))
    fixed = tune_thresholds(y, p, "fixed")
    global_result = tune_thresholds(y, p, "global")
    per_class = tune_thresholds(y, p, "per_class")
    assert (
        evaluate(y, p, global_result["thresholds"])["f1_micro"]
        > evaluate(y, p, fixed["thresholds"])["f1_micro"]
    )
    assert per_class["thresholds"][0] < 0.5 < per_class["thresholds"][1]
    assert evaluate(y, p, per_class["thresholds"])["f1_micro"] == 1


def test_calibration_minimums_and_constant_labels():
    y = np.zeros((4, 37))
    p = np.full_like(y, 0.4)
    assert tune_thresholds(y, p)["small_set_fallback"]
    assert tune_thresholds(y, p, "per_class")["thresholds"] == [0.5] * 37
    y = np.zeros((200, 37))
    p = np.full_like(y, 0.4)
    assert tune_thresholds(y, p, "per_class")["thresholds"] == [0.5] * 37


def test_metric_consistency_and_zero_labels():
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, (51, 37))
    p = rng.random((51, 37))
    result = evaluate(y, p)
    for average in ("micro", "macro", "weighted"):
        assert result[f"f1_{average}"] == pytest.approx(f1_score(y, p >= 0.5, average=average))
    assert len(result["per_class_f1"]) == 37
    assert sum(b["count"] for b in result["reliability_bins"]) == y.size
    assert evaluate(np.zeros_like(y), np.zeros_like(p))["f1_micro"] == 0
    with pytest.raises(ValueError):
        evaluate(y, p * 3)


def test_checkpoint_selection_earliest_tie_and_patience():
    stopper = EarlyStopping(patience=3)
    assert stopper.update(0.3, 0)
    assert stopper.update(0.5, 1)
    for step in (2, 3):
        assert not stopper.update(0.5, step)
        assert not stopper.stopped
    assert not stopper.update(0.49, 4)
    assert stopper.stopped
    assert stopper.best_step == 1
    with pytest.raises(ValueError):
        stopper.update(float("nan"), 5)


@pytest.mark.parametrize("model", ["cnn", "xgb", "mps", "ttn"])
def test_search_reproducibility(model):
    assert trial_configs(model, 4) == trial_configs(model, 4)
    assert trial_configs(model, 4, 2027) != trial_configs(model, 4, 2026)
    assert trial_configs(model, 4)[0] == DEFAULTS[model]


def test_ablations_change_only_one_factor():
    for factor in ["physical_dim", "bond_dim", "mapping"]:
        for trial in trial_configs("mps", ablation=factor):
            assert {k for k in trial if trial[k] != DEFAULTS["mps"][k]} <= {factor}


def test_search_overrides_are_fixed_and_trials_are_distinct():
    configs = trial_configs("mps", 20, overrides={"physical_dim": 8, "bond_dim": 16})
    assert all(c["physical_dim"] == 8 and c["bond_dim"] == 16 for c in configs)
    assert len({tuple(sorted(c.items())) for c in configs}) == 20
    with pytest.raises(ValueError, match="distinct trials"):
        trial_configs("cnn", 2, overrides=DEFAULTS["cnn"])


def test_small_seed_inference_cannot_claim_significance():
    assert exact_sign_flip_p([1, 1, 1, 1, 1]) == 0.0625
    assert holm_adjust([0.01, 0.03, 0.04]) == pytest.approx([0.03, 0.06, 0.06])
    assert describe([1])["sd"] is None
