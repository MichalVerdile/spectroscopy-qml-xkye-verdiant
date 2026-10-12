#!/usr/bin/env python3
"""Create a shared split artifact for all models in one modality/seed run."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from spectroscopy_qml.benchmarking.splits import create_split_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--modality",
        choices=["ir", "cnmr", "hnmr", "msms_pos", "msms_neg"],
        required=True,
    )
    parser.add_argument("--input-dim", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.modality == "ir":
        from spectroscopy_qml.ir.mps_classifier.data_loader import get_functional_groups

        spectrum_column = "ir_spectra"
    elif args.modality == "cnmr":
        from spectroscopy_qml.cnmr.mps_classifier_cnmr.data_loader import get_functional_groups

        spectrum_column = "c_nmr_spectra"
    elif args.modality == "hnmr":
        from spectroscopy_qml.hnmr.mps_classifier_hnmr.data_loader import get_functional_groups

        spectrum_column = "h_nmr_spectra"
    elif args.modality == "msms_pos":
        from spectroscopy_qml.msms_pos.mps_classifier_msms_pos.data_loader import (
            get_functional_groups,
        )

        spectrum_column = "msms_positive_40ev"
    else:
        from spectroscopy_qml.msms_neg.mps_classifier_msms_neg.data_loader import (
            get_functional_groups,
        )

        spectrum_column = "msms_negative_40ev"

    label_batches = []
    for parquet_path in sorted(args.data_dir.glob("*.parquet")):
        frame = pd.read_parquet(parquet_path, columns=[spectrum_column, "smiles"])
        frame["functional_groups"] = frame["smiles"].map(get_functional_groups)
        frame = frame[frame["functional_groups"].notna()]
        frame = frame[frame[spectrum_column].notna()]
        if not frame.empty:
            label_batches.append(np.stack(frame["functional_groups"].values))
    if not label_batches:
        raise FileNotFoundError(f"No usable parquet data found in {args.data_dir}")
    labels = np.vstack(label_batches)
    splits = create_split_artifact(labels, args.output, args.seed)
    print(
        f"Created {args.output}: train={len(splits['train'])}, "
        f"val={len(splits['val'])}, test={len(splits['test'])}"
    )


if __name__ == "__main__":
    main()
