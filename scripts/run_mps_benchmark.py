#!/usr/bin/env python3
"""CLI adapter for the MPS training modules."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import torch

from spectroscopy_qml.benchmarking.preprocessing import canonical_spec

MPS_TRAINING_MODULES = {
    "ir": "spectroscopy_qml.mps_classifier.ir.train",
    "cnmr": "spectroscopy_qml.mps_classifier.cnmr.train",
    "hnmr": "spectroscopy_qml.mps_classifier.hnmr.train",
    "msms_pos": "spectroscopy_qml.mps_classifier.msms_pos.train",
    "msms_neg": "spectroscopy_qml.mps_classifier.msms_neg.train",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modality",
        choices=["ir", "cnmr", "hnmr", "msms_pos", "msms_neg"],
        required=True,
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-path", type=Path, required=True)
    parser.add_argument("--input-dim", type=int, default=600)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--preprocessing",
        default="legacy",
        help="Shared preprocessing spec or ablation config name (default: legacy MPS path)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    module_name = MPS_TRAINING_MODULES[args.modality]
    module = importlib.import_module(module_name)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.output_dir / "models"
    results_dir = args.output_dir / "results"
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    module.MODEL_CONFIG.input_dim = args.input_dim
    module.DATA_CONFIG.target_length = args.input_dim
    module.DATA_CONFIG.preprocessing = canonical_spec(args.preprocessing)
    module.TRAINING_CONFIG.random_seed = args.seed
    module.TRAINING_CONFIG.num_epochs = args.epochs
    if args.early_stopping_patience is not None:
        module.TRAINING_CONFIG.patience = args.early_stopping_patience
    module.TRAINING_CONFIG.num_folds = 1
    module.TRAINING_CONFIG.parallel_fold_workers = 1
    module.TRAINING_CONFIG.device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if module.TRAINING_CONFIG.device == "auto":
        module.TRAINING_CONFIG.device = "cpu"

    module.PATH_CONFIG.data_dir = str(args.data_dir)
    module.PATH_CONFIG.model_dir = str(model_dir)
    module.PATH_CONFIG.results_dir = str(results_dir)
    module.PATH_CONFIG.best_model_path = str(model_dir / "mps_model_best.pt")
    module.PATH_CONFIG.summary_path = str(results_dir / "summary.txt")
    module.PATH_CONFIG.training_log_path = str(results_dir / "training_log.csv")

    module.train_model(split_path=args.split_path)


if __name__ == "__main__":
    main()
