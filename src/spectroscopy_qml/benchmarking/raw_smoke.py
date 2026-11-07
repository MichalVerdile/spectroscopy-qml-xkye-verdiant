"""Explicitly synthetic Parquet fixtures, never scientific benchmark results."""

from pathlib import Path

import numpy as np

from .raw_native import COLUMNS, NATIVE_LENGTHS


def write_synthetic_parquet(directory: Path, *, n=96, seed=812, compact=False):
    import pyarrow as pa
    import pyarrow.parquet as pq

    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    lengths = {"cnmr": 71, "hnmr": 73, "ir": 37} if compact else NATIVE_LENGTHS
    tails = [
        "O",
        "N",
        "C(=O)O",
        "C#N",
        "Cl",
        "C(=O)C",
        "c1ccccc1",
        "S",
        "C=C",
        "C#C",
        "OC",
        "C(=O)N",
    ]
    rows = []
    for i in range(n):
        row = {"smiles": "C" * (1 + i // len(tails)) + tails[i % len(tails)]}
        for m, length in lengths.items():
            signal = rng.normal(0, 0.02, length)
            signal[(i % len(tails)) * (length // len(tails))] += 2.0
            row[COLUMNS[m][0]] = signal.tolist()
        for m in ("msms_pos", "msms_neg"):
            count = int(rng.integers(0, 20))
            row[COLUMNS[m][0]] = np.column_stack(
                [rng.uniform(0, 800, count), rng.uniform(0, 100, count)]
            ).tolist()
        rows.append(row)
    path = directory / "synthetic_raw.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    return lengths
