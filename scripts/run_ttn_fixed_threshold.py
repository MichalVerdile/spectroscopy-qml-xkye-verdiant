#!/usr/bin/env python3
"""Run a TTN training module with validation threshold tuning disabled."""

from __future__ import annotations

import argparse
import importlib
import sys

import numpy as np

TTN_TRAINING_MODULES = {
    "ir": "spectroscopy_qml.tree_tensor_network.ir.train",
    "cnmr": "spectroscopy_qml.tree_tensor_network.cnmr.train",
    "hnmr": "spectroscopy_qml.tree_tensor_network.hnmr.train",
    "msms_pos": "spectroscopy_qml.tree_tensor_network.msms_pos.train",
    "msms_neg": "spectroscopy_qml.tree_tensor_network.msms_neg.train",
}


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--modality", choices=sorted(TTN_TRAINING_MODULES), required=True)
    parser.add_argument("--fixed-threshold", type=float, required=True)
    args, training_args = parser.parse_known_args()

    if not 0.0 <= args.fixed_threshold <= 1.0:
        raise ValueError("--fixed-threshold must be between 0 and 1")

    module = importlib.import_module(TTN_TRAINING_MODULES[args.modality])

    def fixed_thresholds(_labels, probabilities, **_kwargs):
        return np.full(probabilities.shape[1], args.fixed_threshold, dtype=np.float32)

    module.tune_thresholds = fixed_thresholds
    print(f"Using fixed classification threshold: {args.fixed_threshold:.3f}")

    sys.argv = [module.__file__, *training_args]
    module.main()


if __name__ == "__main__":
    main()
