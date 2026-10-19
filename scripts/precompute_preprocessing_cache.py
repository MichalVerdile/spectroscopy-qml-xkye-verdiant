#!/usr/bin/env python3
"""Precompute reusable spectrum caches before benchmark training starts."""

from __future__ import annotations

import argparse
import importlib
import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np

from spectroscopy_qml.benchmarking.preprocessing import (
    LEGACY,
    SHARED_PREPROCESSING_CONFIGS,
    canonical_spec,
    load_preprocessing_cache,
    preprocessing_cache_path,
    save_preprocessing_cache,
)

MODALITIES = ("ir", "cnmr", "hnmr", "msms_pos", "msms_neg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute preprocessing caches shared by CNN, XGBoost, MPS and TTN. "
            "No model training is started."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/preprocessed"))
    parser.add_argument("--modalities", nargs="+", choices=MODALITIES, default=list(MODALITIES))
    parser.add_argument("--input-dims", nargs="+", type=int, default=[600, 1800])
    parser.add_argument(
        "--configs",
        nargs="+",
        default=list(SHARED_PREPROCESSING_CONFIGS),
        help=(
            "Named or explicit preprocessing specs (default: all 18 shared combinations; "
            "pass legacy explicitly only for historical model-specific caches)"
        ),
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Only preprocess the first N parquet files (intended for smoke tests)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Rebuild existing caches")
    parser.add_argument(
        "--allow-low-disk",
        action="store_true",
        help="Continue even when the uncompressed cache estimate exceeds free disk space",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the work plan without reading spectra"
    )
    args = parser.parse_args()
    if any(length <= 0 for length in args.input_dims):
        parser.error("--input-dims values must be positive")
    if args.max_files is not None and args.max_files <= 0:
        parser.error("--max-files must be positive")
    if not args.data_dir.is_dir():
        parser.error(f"data directory does not exist: {args.data_dir}")
    if not list(args.data_dir.glob("*.parquet")):
        parser.error(f"no parquet files found in: {args.data_dir}")
    # Validate and canonicalize now so typos fail before the expensive first dataset load.
    args.configs = list(dict.fromkeys(canonical_spec(config) for config in args.configs))
    return args


def _dataset_row_count(data_dir: Path, max_files: int | None) -> int | None:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return None
    parquet_files = sorted(data_dir.glob("*.parquet"))
    if max_files is not None:
        parquet_files = parquet_files[:max_files]
    return sum(pq.ParquetFile(path).metadata.num_rows for path in parquet_files)


def _format_gib(size: int) -> str:
    return f"{size / 1024**3:.1f} GiB"


def _loader(modality: str) -> Callable[..., tuple[np.ndarray, np.ndarray]]:
    module = importlib.import_module(f"spectroscopy_qml.mps_classifier.{modality}.data_loader")
    return module.load_ir_data


def _load(
    loader: Callable[..., tuple[np.ndarray, np.ndarray]],
    *,
    data_dir: Path,
    cache_dir: Path,
    input_dim: int,
    max_files: int | None,
    preprocessing: str,
    overwrite: bool,
    apply_snv: bool = False,
    apply_savgol: bool = False,
    savgol_window_length: int = 11,
    savgol_polyorder: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    return loader(
        data_dir,
        target_length=input_dim,
        max_files=max_files,
        apply_snv=apply_snv,
        apply_savgol=apply_savgol,
        savgol_window_length=savgol_window_length,
        savgol_polyorder=savgol_polyorder,
        preprocessing=preprocessing,
        cache_dir=cache_dir,
        overwrite_cache=overwrite,
    )


def _save_legacy_linear_cache(
    *,
    data_dir: Path,
    cache_dir: Path,
    modality: str,
    input_dim: int,
    max_files: int | None,
    overwrite: bool,
    X: np.ndarray,
    y: np.ndarray,
) -> None:
    options = {} if max_files is None else {"max_files": max_files}
    cache_path, cache_key = preprocessing_cache_path(
        data_dir,
        consumer="legacy_linear",
        modality=modality,
        target_length=input_dim,
        preprocessing=LEGACY,
        options=options,
        cache_dir=cache_dir,
    )
    if not overwrite and load_preprocessing_cache(cache_path, cache_key) is not None:
        print(f"Loaded cached CNN/XGBoost legacy dataset from {cache_path}")
        return
    save_preprocessing_cache(cache_path, cache_key, X, y)
    print(f"Saved CNN/XGBoost legacy dataset cache to {cache_path}")


def precompute(args: argparse.Namespace) -> None:
    include_legacy = LEGACY in args.configs
    shared_configs = [config for config in args.configs if config != LEGACY]
    total = len(args.modalities) * len(args.input_dims)
    current = 0
    row_count = _dataset_row_count(args.data_dir, args.max_files)
    if row_count is not None:
        variants_per_dataset = len(shared_configs) + (3 if include_legacy else 0)
        estimated_bytes = (
            row_count
            * sum(args.input_dims)
            * len(args.modalities)
            * variants_per_dataset
            * np.dtype(np.float32).itemsize
        )
        disk_target = args.cache_dir
        while not disk_target.exists() and disk_target != disk_target.parent:
            disk_target = disk_target.parent
        available_bytes = shutil.disk_usage(disk_target).free
        print(f"Source rows (before modality filtering): {row_count:,}")
        print(
            "Estimated uncompressed spectrum footprint: "
            f"{_format_gib(estimated_bytes)} (NPZ compression may reduce this)"
        )
        print(f"Available disk space: {_format_gib(available_bytes)}")
        if not args.dry_run and estimated_bytes > available_bytes and not args.allow_low_disk:
            raise SystemExit(
                "Refusing to start because the conservative cache estimate exceeds free "
                "disk space. Select fewer --input-dims/--modalities/--configs or pass "
                "--allow-low-disk after checking storage requirements."
            )

    for input_dim in args.input_dims:
        for modality in args.modalities:
            current += 1
            print(f"\n[{current}/{total}] modality={modality} input_dim={input_dim}")
            if args.dry_run:
                for config in shared_configs:
                    print(f"  shared: {config}")
                if include_legacy:
                    print("  legacy: CNN/XGBoost linear; MPS Savitzky-Golay+SNV; TTN SNV")
                continue

            loader = _loader(modality)
            shared_mssd: tuple[np.ndarray, np.ndarray] | None = None
            for config in shared_configs:
                print(f"  Precomputing shared config: {config}")
                arrays = _load(
                    loader,
                    data_dir=args.data_dir,
                    cache_dir=args.cache_dir,
                    input_dim=input_dim,
                    max_files=args.max_files,
                    preprocessing=config,
                    overwrite=args.overwrite,
                )
                if config == canonical_spec("shared_mssd"):
                    shared_mssd = arrays
                else:
                    del arrays

            if not include_legacy:
                continue

            # CNN and XGBoost historically use linear interpolation only, which is
            # numerically the shared_mssd pipeline. Save an alias under the legacy key.
            if shared_mssd is None:
                shared_mssd = _load(
                    loader,
                    data_dir=args.data_dir,
                    cache_dir=args.cache_dir,
                    input_dim=input_dim,
                    max_files=args.max_files,
                    preprocessing="shared_mssd",
                    overwrite=args.overwrite,
                )
            _save_legacy_linear_cache(
                data_dir=args.data_dir,
                cache_dir=args.cache_dir,
                modality=modality,
                input_dim=input_dim,
                max_files=args.max_files,
                overwrite=args.overwrite,
                X=shared_mssd[0],
                y=shared_mssd[1],
            )
            del shared_mssd

            print("  Precomputing legacy MPS config")
            mps_arrays = _load(
                loader,
                data_dir=args.data_dir,
                cache_dir=args.cache_dir,
                input_dim=input_dim,
                max_files=args.max_files,
                preprocessing=LEGACY,
                overwrite=args.overwrite,
                apply_snv=True,
                apply_savgol=True,
                savgol_window_length=25,
                savgol_polyorder=4,
            )
            del mps_arrays

            print("  Precomputing legacy TTN input config")
            ttn_arrays = _load(
                loader,
                data_dir=args.data_dir,
                cache_dir=args.cache_dir,
                input_dim=input_dim,
                max_files=args.max_files,
                preprocessing=LEGACY,
                overwrite=args.overwrite,
                apply_snv=True,
                apply_savgol=False,
            )
            del ttn_arrays

    print("\nPreprocessing cache plan complete." if args.dry_run else "\nAll caches are ready.")


def main() -> None:
    precompute(parse_args())


if __name__ == "__main__":
    main()
