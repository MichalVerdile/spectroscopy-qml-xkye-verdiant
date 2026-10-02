#!/usr/bin/env python3
"""Aggregate multi-seed metrics, rankings, and rank frequencies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

EXPECTED_MODELS = {"cnn", "xgboost", "mps", "ttn"}
METRICS = ("test_f1_micro", "test_f1_macro")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def load_rows(experiment_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(experiment_dir.glob("*/seed_*/*/**/metrics.json")):
        payload = json.loads(path.read_text())
        # The legacy CNN/XGBoost folders call the MS/MS modalities pos_msms and
        # neg_msms. The experiment directory is the canonical source of truth.
        payload["modality"] = path.relative_to(experiment_dir).parts[0]
        payload["metrics_path"] = str(path)
        rows.append(payload)
    if not rows:
        raise FileNotFoundError(f"No metrics.json files found below {experiment_dir}")
    return rows


def validate_complete(df: pd.DataFrame) -> None:
    required = {"model", "modality", "seed", "split_path", *METRICS}
    missing_columns = required.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Metric files are missing fields: {sorted(missing_columns)}")
    duplicates = df.duplicated(["modality", "seed", "model"], keep=False)
    if duplicates.any():
        raise ValueError(
            "Duplicate model results found:\n"
            + df.loc[duplicates, ["modality", "seed", "model", "metrics_path"]].to_string(index=False)
        )
    for (modality, seed), group in df.groupby(["modality", "seed"]):
        models = set(group["model"])
        if models != EXPECTED_MODELS:
            raise ValueError(
                f"Incomplete comparison for {modality}, seed {seed}: "
                f"expected {sorted(EXPECTED_MODELS)}, found {sorted(models)}"
            )
        split_paths = {str(value) for value in group["split_path"] if pd.notna(value)}
        if len(split_paths) != 1 or group["split_path"].isna().any():
            raise ValueError(
                f"Models for {modality}, seed {seed} did not report one identical shared split: "
                f"{sorted(split_paths)}"
            )
    modality_seed_sets = {
        modality: set(group["seed"].tolist()) for modality, group in df.groupby("modality")
    }
    if len({frozenset(seeds) for seeds in modality_seed_sets.values()}) > 1:
        raise ValueError(f"Modalities use different seed sets: {modality_seed_sets}")
    for modality, group in df.groupby("modality"):
        seed_count = group["seed"].nunique()
        if seed_count < 5:
            raise ValueError(
                f"{modality} has only {seed_count} completed seeds; at least five are required"
            )


def markdown_table(df: pd.DataFrame) -> str:
    """Render a small Markdown table without the optional tabulate package."""
    columns = [str(column) for column in df.columns]
    rows = [[str(value) for value in row] for row in df.itertuples(index=False, name=None)]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.experiment_dir / "summary")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame(load_rows(args.experiment_dir))
    validate_complete(results)
    results = results.sort_values(["modality", "seed", "model"])
    results.to_csv(output_dir / "metrics_by_seed.csv", index=False)

    aggregate = (
        results.groupby(["modality", "model"])[list(METRICS)]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    aggregate.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in aggregate.columns
    ]
    aggregate.to_csv(output_dir / "aggregate_metrics.csv", index=False)

    ranking_frames = []
    for metric in METRICS:
        ranked = results[["modality", "seed", "model", metric]].copy()
        ranked = ranked.rename(columns={metric: "score"})
        ranked["metric"] = metric
        ranked["rank"] = ranked.groupby(["modality", "seed"])["score"].rank(
            method="min", ascending=False
        )
        ranking_frames.append(ranked)
    rankings = pd.concat(ranking_frames, ignore_index=True).sort_values(
        ["modality", "metric", "seed", "rank", "model"]
    )
    rankings.to_csv(output_dir / "rankings_by_seed.csv", index=False)

    observed_frequencies = (
        rankings.groupby(["modality", "metric", "model", "rank"])
        .size()
        .rename("count")
        .reset_index()
    )
    full_rank_index = pd.MultiIndex.from_product(
        [
            sorted(rankings["modality"].unique()),
            sorted(rankings["metric"].unique()),
            sorted(EXPECTED_MODELS),
            [1.0, 2.0, 3.0, 4.0],
        ],
        names=["modality", "metric", "model", "rank"],
    )
    frequencies = (
        observed_frequencies.set_index(["modality", "metric", "model", "rank"])
        .reindex(full_rank_index, fill_value=0)
        .reset_index()
    )
    seed_counts = rankings.groupby(["modality", "metric"])["seed"].nunique()
    frequencies["frequency"] = frequencies.apply(
        lambda row: row["count"] / seed_counts.loc[(row["modality"], row["metric"])],
        axis=1,
    )
    frequencies.to_csv(output_dir / "rank_frequencies.csv", index=False)

    note = (
        "Uncertainty statement: mean and standard deviation are computed across independent "
        "seed-specific data splits and training runs. Any bootstrap confidence intervals "
        "computed within a single test set quantify finite-test-set sampling uncertainty "
        "conditional on that fixed split and trained model; they do not capture variation "
        "from data splitting, initialization, or training."
    )
    (output_dir / "uncertainty_note.txt").write_text(note + "\n")

    report_lines = ["# Multi-seed benchmark summary", "", note, "", "## Mean and standard deviation", ""]
    report_lines.append(markdown_table(aggregate))
    report_lines.extend(["", "## Per-seed rankings", "", markdown_table(rankings)])
    report_lines.extend(["", "## Rank frequencies", "", markdown_table(frequencies), ""])
    (output_dir / "report.md").write_text("\n".join(report_lines))
    print(f"Wrote multi-seed summary to {output_dir}")


if __name__ == "__main__":
    main()
