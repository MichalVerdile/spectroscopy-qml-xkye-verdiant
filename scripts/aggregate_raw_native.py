#!/usr/bin/env python3
"""Aggregate a single compatible raw-native experiment (five seeds by default)."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spectroscopy_qml.benchmarking.raw_aggregate import aggregate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-incomplete", action="store_true", help="Descriptive partial/smoke output only"
    )
    args = parser.parse_args()
    result = aggregate(
        args.experiment,
        args.output or args.experiment / "aggregate.json",
        allow_incomplete=args.allow_incomplete,
    )
    print(f"Aggregated {len(result['rows'])} runs; {len(result['comparisons'])} paired comparisons")
