#!/usr/bin/env python3
"""Extract per-class test metrics from the archived external raw XGBoost runs."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

MODALITY_OUTPUTS = {
    "cnmr": "cnmr",
    "hnmr": "hnmr",
    "ir": "ir",
    "msms_neg": "neg_msms",
    "msms_pos": "pos_msms",
}
SEEDS = (41, 42, 43, 44, 45)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--functional-groups-reference",
        type=Path,
        required=True,
        help="Existing per-class JSON whose functional-group order is authoritative.",
    )
    return parser.parse_args()


def locate_result(root: Path, modality: str, seed: int, output_name: str) -> Path:
    matches = sorted(
        (root / modality / f"seed_{seed}" / "attempts").glob(
            f"*/artifacts/{output_name}/original/results.pickle"
        )
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one completed result for {modality}, seed {seed}; found {matches}"
        )
    return matches[0]


def extract(path: Path, modality: str, seed: int, groups: list[str]) -> dict:
    with path.open("rb") as handle:
        payload = pickle.load(handle)  # noqa: S301 - trusted experiment artifact
    y_true = np.asarray(payload["tgt"], dtype=np.int8)
    y_pred = np.asarray(payload["pred"], dtype=np.int8)
    if y_true.shape != y_pred.shape or y_true.shape[1] != len(groups):
        raise ValueError(f"Invalid prediction shapes in {path}: {y_true.shape}, {y_pred.shape}")
    tp = np.logical_and(y_true == 1, y_pred == 1).sum(axis=0)
    fp = np.logical_and(y_true == 0, y_pred == 1).sum(axis=0)
    fn = np.logical_and(y_true == 1, y_pred == 0).sum(axis=0)
    return {
        "measurement": "saved_test_predictions_per_class",
        "model": "xgboost",
        "modality": modality,
        "seed": seed,
        "functional_groups": groups,
        "support_per_class": y_true.sum(axis=0).astype(int).tolist(),
        "f1_per_class": f1_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        "precision_per_class": precision_score(
            y_true, y_pred, average=None, zero_division=0
        ).tolist(),
        "recall_per_class": recall_score(
            y_true, y_pred, average=None, zero_division=0
        ).tolist(),
        "true_positives_per_class": tp.astype(int).tolist(),
        "false_positives_per_class": fp.astype(int).tolist(),
        "false_negatives_per_class": fn.astype(int).tolist(),
        "test_f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "test_f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def main() -> None:
    args = parse_args()
    reference = json.loads(args.functional_groups_reference.read_text())
    groups = list(reference["functional_groups"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for modality, output_name in MODALITY_OUTPUTS.items():
        for seed in SEEDS:
            source = locate_result(args.archive_root, modality, seed, output_name)
            result = extract(source, modality, seed, groups)
            destination = args.output_dir / f"{modality}_seed_{seed}_xgboost.json"
            destination.write_text(json.dumps(result, indent=2) + "\n")
            print(destination)


if __name__ == "__main__":
    main()
