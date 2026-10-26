#!/usr/bin/env python3
"""Return success only when an existing benchmark result matches the requested run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from spectroscopy_qml.benchmarking.preprocessing import canonical_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--preprocessing", required=True)
    parser.add_argument("--input-dim", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split-path", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        payload = json.loads(args.metrics.read_text())
        expected = {
            "model": args.model,
            "preprocessing": canonical_spec(args.preprocessing),
            "input_dim": args.input_dim,
            "seed": args.seed,
        }
        mismatches = {
            key: (payload.get(key), value)
            for key, value in expected.items()
            if payload.get(key) != value
        }
        recorded_split = payload.get("split_path")
        if recorded_split is None or Path(recorded_split).resolve() != args.split_path.resolve():
            mismatches["split_path"] = (recorded_split, str(args.split_path))
        digest = payload.get("input_digest")
        if not isinstance(digest, str) or len(digest) != 64:
            mismatches["input_digest"] = (digest, "64-character SHA-256 digest")
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        raise SystemExit(f"Invalid existing metrics {args.metrics}: {error}") from error
    if mismatches:
        details = ", ".join(
            f"{key}: recorded={actual!r}, expected={expected!r}"
            for key, (actual, expected) in mismatches.items()
        )
        raise SystemExit(f"Existing metrics do not match requested run ({details})")


if __name__ == "__main__":
    main()
