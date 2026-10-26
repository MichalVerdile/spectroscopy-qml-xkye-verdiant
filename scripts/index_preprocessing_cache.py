#!/usr/bin/env python3
"""Create human-readable aliases and a manifest for preprocessing caches."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from spectroscopy_qml.benchmarking.preprocessing import (
    canonical_spec,
    preprocessing_cache_path,
)

MODALITIES = ("ir", "cnmr", "hnmr", "msms_pos", "msms_neg")
FIELDNAMES = (
    "parameter",
    "effective_spec",
    "modality",
    "input_dim",
    "cache_file",
    "alias_path",
    "size_bytes",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/preprocessed"))
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--modalities", nargs="+", choices=MODALITIES, default=list(MODALITIES))
    parser.add_argument("--input-dims", nargs="+", type=int, default=[600, 1800])
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def read_manifest(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            (row["parameter"], row["modality"], row["input_dim"]): row
            for row in csv.DictReader(handle)
        }


def main() -> None:
    args = parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.cache_dir / "cache_manifest.csv"
    rows = read_manifest(manifest_path)

    for parameter in args.configs:
        effective_spec = canonical_spec(parameter)
        for input_dim in args.input_dims:
            for modality in args.modalities:
                cache_path, cache_key = preprocessing_cache_path(
                    args.data_dir,
                    consumer="shared",
                    modality=modality,
                    target_length=input_dim,
                    preprocessing=parameter,
                    cache_dir=args.cache_dir,
                )
                if not cache_path.exists():
                    if args.allow_missing:
                        continue
                    raise FileNotFoundError(cache_path)
                with np.load(cache_path, allow_pickle=False) as payload:
                    if str(payload["cache_key"]) != cache_key:
                        raise ValueError(f"Cache key mismatch in {cache_path}")

                alias_path = args.cache_dir / "by_parameter" / parameter / (
                    f"{modality}_len{input_dim}.npz"
                )
                alias_path.parent.mkdir(parents=True, exist_ok=True)
                if alias_path.is_symlink() or alias_path.exists():
                    alias_path.unlink()
                alias_path.symlink_to(cache_path.resolve())

                rows[(parameter, modality, str(input_dim))] = {
                    "parameter": parameter,
                    "effective_spec": effective_spec,
                    "modality": modality,
                    "input_dim": str(input_dim),
                    "cache_file": cache_path.name,
                    "alias_path": str(alias_path.relative_to(args.cache_dir)),
                    "size_bytes": str(cache_path.stat().st_size),
                }

    temporary_path = manifest_path.with_suffix(".csv.tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(
            sorted(
                rows.values(),
                key=lambda row: (
                    row["parameter"],
                    int(row["input_dim"]),
                    MODALITIES.index(row["modality"]),
                ),
            )
        )
    temporary_path.replace(manifest_path)
    print(f"Indexed {len(rows)} cache files in {manifest_path}")


if __name__ == "__main__":
    main()
