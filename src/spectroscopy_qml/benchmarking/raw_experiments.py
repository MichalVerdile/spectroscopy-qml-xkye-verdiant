"""Predeclared search spaces and one-factor TN architecture ablations."""

from __future__ import annotations

import numpy as np

DEFAULTS = {
    "cnn": {"width": 48, "dropout": 0.2, "learning_rate": 0.001, "weight_decay": 0.0001},
    "mps": {
        "physical_dim": 16,
        "bond_dim": 32,
        "window": 128,
        "dual_scale": False,
        "bidirectional": True,
        "mapping": "small",
        "dropout": 0.2,
        "residual": 0.1,
        "normalize": True,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
    },
    "ttn": {
        "chi": 32,
        "window": 128,
        "overlap": 0.5,
        "merge_mode": "relaxed",
        "residual": 0.1,
        "multiscale": True,
        "normalize": True,
        "mapping": "small",
        "dropout": 0.2,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
    },
    "xgb": {
        "max_depth": 6,
        "learning_rate": 0.05,
        "min_child_weight": 1.0,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "n_estimators": 300,
    },
}
SEARCH_SPACES = {
    "cnn": {
        "width": [24, 48, 96],
        "dropout": [0.1, 0.2, 0.3],
        "learning_rate": [0.0003, 0.001, 0.003],
        "weight_decay": [0.00001, 0.0001, 0.001],
    },
    "xgb": {
        "max_depth": [3, 6, 9],
        "learning_rate": [0.02, 0.05, 0.1],
        "min_child_weight": [1.0, 5.0, 10.0],
        "subsample": [0.7, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.9, 1.0],
        "reg_alpha": [0.0, 0.1, 1.0],
        "reg_lambda": [1.0, 5.0, 10.0],
        "n_estimators": [200, 400, 800],
    },
    "mps": {
        "physical_dim": [8, 16, 32, 64],
        "bond_dim": [16, 32, 64, 128],
        "window": [64, 128, 256],
        "dropout": [0.1, 0.2, 0.3],
        "learning_rate": [0.0003, 0.001, 0.003],
        "weight_decay": [0.00001, 0.0001, 0.001],
    },
    "ttn": {
        "chi": [16, 32, 64, 128],
        "window": [64, 128, 256],
        "overlap": [0.0, 0.25, 0.5],
        "dropout": [0.1, 0.2, 0.3],
        "learning_rate": [0.0003, 0.001, 0.003],
        "weight_decay": [0.00001, 0.0001, 0.001],
    },
}
ABLATIONS = {
    "mps": {
        **{k: SEARCH_SPACES["mps"][k] for k in ("physical_dim", "bond_dim", "window", "dropout")},
        "dual_scale": [False, True],
        "bidirectional": [False, True],
        "mapping": ["small", "legacy_mlp", "angle", "product"],
        "residual": [0.0, 0.1, 0.2],
        "normalize": [False, True],
    },
    "ttn": {
        **{k: SEARCH_SPACES["ttn"][k] for k in ("chi", "window", "overlap", "dropout")},
        "merge_mode": ["strict", "relaxed"],
        "residual": [0.0, 0.1, 0.2],
        "multiscale": [False, True],
        "normalize": [False, True],
        "mapping": ["small", "legacy_mlp", "angle", "product"],
    },
}


def trial_configs(model, trials=1, search_seed=2026, overrides=None, ablation=None):
    if trials < 1:
        raise ValueError("trials must be positive")
    overrides = overrides or {}
    if set(overrides) - set(DEFAULTS[model]):
        raise ValueError(
            f"Unknown {model} configuration keys: {set(overrides) - set(DEFAULTS[model])}"
        )
    base = {**DEFAULTS[model], **overrides}
    if ablation:
        if model not in ABLATIONS or ablation not in ABLATIONS[model]:
            raise ValueError(f"Unsupported one-factor ablation {model}:{ablation}")
        return [{**base, ablation: value} for value in ABLATIONS[model][ablation]]
    rng = np.random.default_rng(search_seed)
    configs = [base]
    adjustable = {k: values for k, values in SEARCH_SPACES[model].items() if k not in overrides}
    capacity = int(np.prod([len(v) for v in adjustable.values()]))
    if trials > capacity:
        raise ValueError("Requested more distinct trials than the fixed search space permits")
    while len(configs) < trials:
        config = {
            **base,
            **{k: values[int(rng.integers(len(values)))] for k, values in adjustable.items()},
        }
        if config not in configs:
            configs.append(config)
    return configs
