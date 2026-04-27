"""Main entry point for the MPS + TTN 10.2 joint training pipeline.

Pipeline:
    1. Load and preprocess IR spectra once (shared preprocessing).
    2. Create a single train/val/test split once (shared indices).
    3. Train MPS on that data  →  MPS branch.
    4. Train TTN 10.2 on that data  →  TTN branch.

Both models therefore see exactly the same input and the same split, enabling
a fair head-to-head comparison.

Usage::

    # Train both models
    python run.py --data-dir data/raw --model both

    # Train only one
    python run.py --data-dir data/raw --model mps
    python run.py --data-dir data/raw --model ttn

    # Reuse a cached spectra .npz and existing split
    python run.py \\
        --spectra-cache data/cache/ir_spectra_snv.npz \\
        --split-path src/spectroscopy_qml/ir/MPS_TTN/results/shared_split.npz \\
        --model both
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
    FUNCTIONAL_GROUPS,
    load_or_create_split_indices,
    load_shared_data,
)
from train_mps import run_mps_training  # noqa: E402
from train_ttn102 import build_parser as ttn_build_parser  # noqa: E402
from train_ttn102 import run_ttn102_training  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Unified MPS + TTN 10.2 training pipeline with shared preprocessing and split."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Data ----------------------------------------------------------------
    data_grp = parser.add_argument_group("Shared data")
    data_grp.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing raw parquet files.",
    )
    data_grp.add_argument(
        "--spectra-cache",
        type=Path,
        default=Path("src/spectroscopy_qml/ir/MPS_TTN/results/shared_spectra.npz"),
        help="Path to the shared spectra cache (.npz with keys X, y).",
    )
    data_grp.add_argument(
        "--overwrite-cache",
        action="store_true",
        help="Re-load raw data even if the spectra cache exists.",
    )
    data_grp.add_argument(
        "--split-path",
        type=Path,
        default=Path("src/spectroscopy_qml/ir/MPS_TTN/results/shared_split.npz"),
        help="Path to the shared split .npz file (created if absent).",
    )
    data_grp.add_argument(
        "--overwrite-split",
        action="store_true",
        help="Re-create the split even if the file exists.",
    )
    data_grp.add_argument(
        "--apply-snv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply SNV normalisation (shared by both models).",
    )
    data_grp.add_argument(
        "--apply-savgol",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply Savitzky-Golay smoothing (shared). Off by default; TTN uses Lorentzian internally.",
    )
    data_grp.add_argument("--max-files", type=int, default=None, help="Cap on parquet files to load.")
    data_grp.add_argument("--train-ratio", type=float, default=0.8)
    data_grp.add_argument("--val-ratio", type=float, default=0.1)
    data_grp.add_argument("--test-ratio", type=float, default=0.1)
    data_grp.add_argument("--seed", type=int, default=42, help="Random seed for split and training.")

    # ---- Which models --------------------------------------------------------
    parser.add_argument(
        "--model",
        choices=["mps", "ttn", "both"],
        default="both",
        help="Which model(s) to train.",
    )

    # ---- Output --------------------------------------------------------------
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/spectroscopy_qml/ir/MPS_TTN/results"),
        help="Root output directory. MPS writes to <output-dir>/mps, TTN to <output-dir>/ttn102.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps"],
        default="cuda",
        help="Default device for both branches. Override per branch only if needed.",
    )

    # ---- TTN 10.2 hyperparameters (pass-through) ----------------------------
    ttn_grp = parser.add_argument_group(
        "TTN 10.2 hyperparameters",
        description=(
            "These are forwarded to the TTN branch. "
            "Run 'python train_ttn102.py --help' for full details."
        ),
    )
    ttn_grp.add_argument("--ttn-chi", type=int, default=64)
    ttn_grp.add_argument("--ttn-epochs", type=int, default=200)
    ttn_grp.add_argument("--ttn-batch-size", type=int, default=1024)
    ttn_grp.add_argument("--ttn-lr", type=float, default=3e-4)
    ttn_grp.add_argument("--ttn-lorentz-gamma", type=float, default=3.0)
    ttn_grp.add_argument(
        "--ttn-lorentz-norm-mode",
        choices=["max_abs", "z_score", "percentile"],
        default="max_abs",
    )
    ttn_grp.add_argument(
        "--ttn-device",
        choices=["cpu", "cuda", "mps"],
        default=None,
        help="Optional override for the TTN branch. Defaults to --device.",
    )
    parser.add_argument(
        "--mps-device",
        choices=["cuda", "cpu", "mps"],
        default=None,
        help="Optional override for the MPS branch. Defaults to --device.",
    )
    ttn_grp.add_argument("--ttn-num-workers", type=int, default=0)
    ttn_grp.add_argument("--ttn-amp", action=argparse.BooleanOptionalAction, default=True)
    ttn_grp.add_argument("--ttn-compile", action=argparse.BooleanOptionalAction, default=True)

    return parser


def _build_ttn_args(
    main_args: argparse.Namespace,
    split_path: Path,
    output_dir: Path,
) -> argparse.Namespace:
    """Construct a TTN arg namespace from the main parser and defaults."""
    ttn_parser = ttn_build_parser()
    ttn_args = ttn_parser.parse_args([])  # all defaults

    # Shared data settings
    ttn_args.data_dir = main_args.data_dir
    ttn_args.spectra_cache = main_args.spectra_cache
    ttn_args.overwrite_cache = main_args.overwrite_cache
    ttn_args.split_path = split_path
    ttn_args.overwrite_split = False  # split already created by run.py
    ttn_args.apply_snv = main_args.apply_snv
    ttn_args.apply_savgol = main_args.apply_savgol
    ttn_args.max_files = main_args.max_files
    ttn_args.train_ratio = main_args.train_ratio
    ttn_args.val_ratio = main_args.val_ratio
    ttn_args.test_ratio = main_args.test_ratio
    ttn_args.seed = main_args.seed

    # TTN-specific overrides from main parser
    ttn_args.output_dir = output_dir
    ttn_args.chi = main_args.ttn_chi
    ttn_args.epochs = main_args.ttn_epochs
    ttn_args.batch_size = main_args.ttn_batch_size
    ttn_args.learning_rate = main_args.ttn_lr
    ttn_args.lorentz_gamma = main_args.ttn_lorentz_gamma
    ttn_args.lorentz_norm_mode = main_args.ttn_lorentz_norm_mode
    ttn_args.device = main_args.ttn_device or main_args.device
    ttn_args.num_workers = main_args.ttn_num_workers
    ttn_args.amp = main_args.ttn_amp
    ttn_args.compile = main_args.ttn_compile

    return ttn_args


def main() -> None:
    args = build_parser().parse_args()

    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1 — Load data (shared preprocessing)
    # ------------------------------------------------------------------
    print("=" * 80)
    print("Step 1/3: Shared data loading and preprocessing")
    print("=" * 80)
    X, y = load_shared_data(
        data_dir=args.data_dir,
        apply_snv=args.apply_snv,
        apply_savgol=args.apply_savgol,
        max_files=args.max_files,
        cache_path=args.spectra_cache,
        overwrite_cache=args.overwrite_cache,
    )
    print(f"\nShared data: X={X.shape}  y={y.shape}")

    # ------------------------------------------------------------------
    # Step 2 — Create / load shared split
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Step 2/3: Shared train/val/test split")
    print("=" * 80)
    split_indices = load_or_create_split_indices(
        labels=y,
        split_path=args.split_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
        overwrite=args.overwrite_split,
    )
    print(
        f"Split  →  train={len(split_indices['train'])}  "
        f"val={len(split_indices['val'])}  "
        f"test={len(split_indices['test'])}"
    )

    # ------------------------------------------------------------------
    # Step 3 — Train selected model(s)
    # ------------------------------------------------------------------
    train_mps = args.model in ("mps", "both")
    train_ttn = args.model in ("ttn", "both")

    if train_mps:
        print("\n" + "=" * 80)
        print("Step 3a: MPS training branch")
        print("=" * 80)
        run_mps_training(
            X=X,
            y=y,
            split_indices=split_indices,
            output_dir=args.output_dir / "mps",
            device=args.mps_device or args.device,
        )

    if train_ttn:
        print("\n" + "=" * 80)
        print("Step 3b: TTN 10.2 training branch")
        print("=" * 80)
        ttn_args = _build_ttn_args(
            main_args=args,
            split_path=args.split_path,
            output_dir=args.output_dir / "ttn102",
        )
        run_ttn102_training(X=X, y=y, split_indices=split_indices, args=ttn_args)

    print("\n" + "=" * 80)
    print("Pipeline complete.")
    if train_mps:
        print(f"  MPS results  →  {args.output_dir / 'mps'}")
    if train_ttn:
        print(f"  TTN results  →  {args.output_dir / 'ttn102'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
