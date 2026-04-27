"""MPS training branch for the MPS + TTN joint pipeline.

Loads data via :mod:`shared_data` so that the MPS model trains on exactly the
same preprocessed spectra and the same train/val/test split as TTN 10.2.

Usage (standalone)::

    python train_mps.py \\
        --data-dir data/raw \\
        --split-path src/spectroscopy_qml/ir/MPS_TTN/results/shared_split.npz \\
        --spectra-cache src/spectroscopy_qml/ir/MPS_TTN/results/shared_spectra.npz \\
        --output-dir src/spectroscopy_qml/ir/MPS_TTN/results/mps

Or call :func:`run_mps_training` from ``run.py`` with pre-loaded data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = Path(__file__).resolve().parents[4]
for _p in (str(CURRENT_DIR), str(SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared_data import (  # noqa: E402
    load_or_create_split_indices,
    load_shared_data,
)
from spectroscopy_qml.ir.mps_encoder_final.config import (  # noqa: E402
    DATA_CONFIG,
    MODEL_CONFIG,
    PATH_CONFIG,
    TRAINING_CONFIG,
)
from spectroscopy_qml.ir.mps_encoder_final.train import train_model  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the MPS model on the shared MPS+TTN data split."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory with raw parquet files.",
    )
    parser.add_argument(
        "--spectra-cache",
        type=Path,
        default=Path("src/spectroscopy_qml/ir/MPS_TTN/results/shared_spectra.npz"),
        help="Path to the shared spectra cache (.npz with keys X, y).",
    )
    parser.add_argument(
        "--overwrite-cache",
        action="store_true",
        help="Re-load raw data even if the cache exists.",
    )
    parser.add_argument(
        "--split-path",
        type=Path,
        default=Path("src/spectroscopy_qml/ir/MPS_TTN/results/shared_split.npz"),
        help="Path to the shared split file (.npz). Created if missing.",
    )
    parser.add_argument(
        "--overwrite-split",
        action="store_true",
        help="Re-create the split even if the file exists.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/spectroscopy_qml/ir/MPS_TTN/results/mps"),
        help="Directory for MPS checkpoints and logs.",
    )
    parser.add_argument(
        "--apply-snv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply SNV normalisation (must match TTN setting).",
    )
    parser.add_argument(
        "--apply-savgol",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply Savitzky-Golay smoothing (default off for shared pipeline).",
    )
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "mps"],
        default="cuda",
        help="Device for standalone MPS runs.",
    )
    return parser


def run_mps_training(
    X: np.ndarray,
    y: np.ndarray,
    split_indices: dict[str, np.ndarray],
    output_dir: Path,
    device: str | None = None,
) -> dict:
    """Train the MPS model on pre-loaded data.

    The MPS ``train_model`` function reads its own hyperparameters from
    ``mps_encoder_final.config`` (MODEL_CONFIG, TRAINING_CONFIG, etc.).
    Output paths are redirected to *output_dir*.

    Args:
        X: Preprocessed spectra (n_samples, 1800).
        y: Label matrix (n_samples, num_classes).
        split_indices: Shared train/val/test indices created by ``run.py``.
        output_dir: Where MPS saves its checkpoint and logs.

    Returns:
        Result dict returned by :func:`~mps_encoder_final.train.train_model`.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Single-encoder mode: only the 5-site MPS, no second encoder.
    MODEL_CONFIG.use_second_encoder = False

    # Device override: allow CPU/MPS hosts to run without CUDA.
    if device is not None:
        TRAINING_CONFIG.device = device

    # Redirect MPS output paths to output_dir so they don't collide with TTN.
    PATH_CONFIG.model_dir = str(output_dir / "models")
    PATH_CONFIG.results_dir = str(output_dir / "results")
    PATH_CONFIG.best_model_path = str(output_dir / "models" / "mps_model_best.pt")
    PATH_CONFIG.summary_path = str(output_dir / "results" / "summary.txt")
    PATH_CONFIG.training_log_path = str(output_dir / "results" / "training_log.csv")

    print("\n" + "=" * 80)
    print("MPS Training Branch")
    print("=" * 80)
    print(f"Output dir:   {output_dir}")
    print(
        "Shared split: "
        f"train={len(split_indices['train'])} "
        f"val={len(split_indices['val'])} "
        f"test={len(split_indices['test'])}"
    )
    print(f"X shape:      {X.shape}")
    print(f"y shape:      {y.shape}")

    return train_model(X=X, y=y, split_indices=split_indices)


def main() -> None:
    args = build_parser().parse_args()

    X, y = load_shared_data(
        data_dir=args.data_dir,
        apply_snv=args.apply_snv,
        apply_savgol=args.apply_savgol,
        max_files=args.max_files,
        cache_path=args.spectra_cache,
        overwrite_cache=args.overwrite_cache,
    )

    split_indices = load_or_create_split_indices(
        labels=y,
        split_path=args.split_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
        overwrite=args.overwrite_split,
    )

    run_mps_training(
        X=X,
        y=y,
        split_indices=split_indices,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
