#!/usr/bin/env python3
"""Separate architecture, preprocessing and input-representation effects.

Reads ``<ablation-dir>/dim_<D>/<config>/`` experiment directories written by
``scripts/run_preprocessing_ablation.sh`` and writes, under ``<ablation-dir>/summary``:

* ``architecture_effect.csv``: models compared with preprocessing and input dim
  held fixed (one block per shared config).
* ``preprocessing_effect.csv``: each downsampling / smoothing / normalization
  variant minus the reference config, paired per seed and model.
* ``representation_effect.csv``: the same for input representation and input dim.
* ``harmonization_effect.csv``: legacy per-model preprocessing minus the shared
  reference, i.e. how much of the original comparison was preprocessing.
* ``effect_sizes.csv``: per modality and metric, the spread across architectures
  next to the spread across preprocessing and representation choices.
* ``report.md`` with all of the above.

Factors are derived from the recorded preprocessing spec, not from config names:
a config belongs to the factor whose single key differs from the reference.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aggregate_multiseed_results import (  # noqa: E402
    METRICS,
    load_rows,
    markdown_table,
    validate_complete,
)

from spectroscopy_qml.benchmarking.preprocessing import (  # noqa: E402
    LEGACY,
    PreprocessingConfig,
    canonical_spec,
    parse_preprocessing,
)

PREPROCESSING_FACTORS = ("resample", "smooth", "norm")
REPRESENTATION_FACTORS = ("repr", "input_dim")
KEYS = ("modality", "model", "seed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--reference-config",
        default="shared_mssd",
        help="Config name or spec every other config is compared against",
    )
    parser.add_argument("--reference-dim", type=int, default=600)
    return parser.parse_args()


def load_ablation(ablation_dir: Path) -> pd.DataFrame:
    frames = []
    for dim_dir in sorted(ablation_dir.glob("dim_*")):
        match = re.fullmatch(r"dim_(\d+)", dim_dir.name)
        if not match or not dim_dir.is_dir():
            continue
        for config_dir in sorted(path for path in dim_dir.iterdir() if path.is_dir()):
            frame = pd.DataFrame(load_rows(config_dir))
            validate_complete(frame)
            if "preprocessing" not in frame.columns:
                raise ValueError(f"{config_dir} has metrics without a preprocessing field")
            frame["input_dim"] = int(match.group(1))
            frame["config"] = config_dir.name
            frame["preprocessing"] = frame["preprocessing"].map(canonical_spec)
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No dim_*/<config> experiment directories below {ablation_dir}")
    return pd.concat(frames, ignore_index=True)


def assign_factor(row: pd.Series, reference_spec: str, reference_dim: int) -> tuple[str, str]:
    """Return (factor, level) for one run relative to the reference."""
    spec = row["preprocessing"]
    if spec == LEGACY:
        return "legacy", LEGACY
    if spec == reference_spec:
        if row["input_dim"] == reference_dim:
            return "reference", "reference"
        return "input_dim", str(row["input_dim"])
    if row["input_dim"] != reference_dim:
        return "mixed", spec
    config = parse_preprocessing(spec)
    reference = parse_preprocessing(reference_spec)
    changed = [
        key
        for key in ("resample", "smooth", "repr", "norm")
        if getattr(config, key) != getattr(reference, key)
    ]
    if len(changed) == 1:
        return changed[0], str(getattr(config, changed[0]))
    return "mixed", spec


def paired_deltas(
    results: pd.DataFrame, reference: pd.DataFrame, factors: tuple[str, ...]
) -> pd.DataFrame:
    variants = results[results["factor"].isin(factors)]
    if variants.empty:
        return pd.DataFrame()
    merged = variants.merge(reference, on=list(KEYS), suffixes=("", "_ref"), validate="many_to_one")
    mismatched = merged["split_path"] != merged["split_path_ref"]
    if mismatched.any():
        raise ValueError(
            "Paired comparison needs identical splits; these runs used a different split:\n"
            + merged.loc[
                mismatched, ["config", "input_dim", *KEYS, "split_path", "split_path_ref"]
            ].to_string(index=False)
        )
    rows = []
    for metric in METRICS:
        merged[f"delta_{metric}"] = merged[metric] - merged[f"{metric}_ref"]
        grouped = merged.groupby(["factor", "level", "config", "input_dim", "modality", "model"])
        stats = grouped[f"delta_{metric}"].agg(["mean", "std", "count"])
        stats["seeds_improved"] = grouped[f"delta_{metric}"].apply(
            lambda values: int((values > 0).sum())
        )
        stats["metric"] = metric
        rows.append(stats.reset_index())
    table = pd.concat(rows, ignore_index=True)
    table = table.rename(columns={"mean": "delta_mean", "std": "delta_std", "count": "seeds"})
    return table.sort_values(["metric", "modality", "factor", "level", "model"]).reset_index(
        drop=True
    )


def architecture_effect(results: pd.DataFrame) -> pd.DataFrame:
    shared = results[results["factor"] != "legacy"]
    rows = []
    for metric in METRICS:
        grouped = shared.groupby(["config", "input_dim", "factor", "level", "modality", "model"])[
            metric
        ]
        stats = grouped.agg(["mean", "std", "count"]).reset_index()
        stats["metric"] = metric
        stats["rank"] = stats.groupby(["config", "input_dim", "modality"])["mean"].rank(
            method="min", ascending=False
        )
        rows.append(stats)
    table = pd.concat(rows, ignore_index=True)
    return table.sort_values(["metric", "modality", "input_dim", "config", "rank"]).reset_index(
        drop=True
    )


def effect_sizes(
    results: pd.DataFrame, arch: pd.DataFrame, reference_spec: str, reference_dim: int
) -> pd.DataFrame:
    """Compare spreads: across models (fixed inputs) vs across inputs (fixed model)."""
    rows = []
    reference_arch = arch[(arch["factor"] == "reference")]
    for metric in METRICS:
        means = (
            results[results["factor"] != "legacy"]
            .groupby(["factor", "config", "input_dim", "modality", "model"])[metric]
            .mean()
            .reset_index()
        )
        for modality, ref_block in reference_arch[reference_arch["metric"] == metric].groupby(
            "modality"
        ):
            winner = ref_block.sort_values("mean", ascending=False)["model"].iloc[0]
            row = {
                "metric": metric,
                "modality": modality,
                "reference_winner": winner,
                "architecture_range": ref_block["mean"].max() - ref_block["mean"].min(),
            }
            modality_means = means[means["modality"] == modality]
            for label, factors in (
                ("preprocessing", PREPROCESSING_FACTORS),
                ("representation", REPRESENTATION_FACTORS),
            ):
                block = modality_means[modality_means["factor"].isin((*factors, "reference"))]
                per_model_range = block.groupby("model")[metric].agg(
                    lambda values: values.max() - values.min()
                )
                n_variants = (
                    block[block["factor"] != "reference"][["config", "input_dim"]]
                    .drop_duplicates()
                    .shape[0]
                )
                row[f"{label}_range_median_over_models"] = (
                    per_model_range.median() if n_variants else float("nan")
                )
                row[f"{label}_range_max_over_models"] = (
                    per_model_range.max() if n_variants else float("nan")
                )
                row[f"{label}_variants"] = n_variants
            winners = (
                arch[
                    (arch["metric"] == metric)
                    & (arch["modality"] == modality)
                    & (arch["rank"] == 1)
                ]
                .groupby(["config", "input_dim"])["model"]
                .apply(lambda models: winner in set(models))
            )
            row["share_of_configs_with_same_winner"] = (
                float(winners.mean()) if len(winners) else float("nan")
            )
            row["configs_compared"] = int(len(winners))
            rows.append(row)
    return pd.DataFrame(rows)


def fmt(df: pd.DataFrame) -> pd.DataFrame:
    return df.astype(object).apply(
        lambda column: column.map(
            lambda value: f"{value:.4f}" if isinstance(value, float) else value
        )
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.ablation_dir / "summary")
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_spec = canonical_spec(args.reference_config)
    if reference_spec == LEGACY:
        raise ValueError("The reference must be a shared spec; legacy differs per model")

    results = load_ablation(args.ablation_dir)
    factors = results.apply(assign_factor, axis=1, args=(reference_spec, args.reference_dim))
    results["factor"] = [factor for factor, _ in factors]
    results["level"] = [level for _, level in factors]
    results.to_csv(output_dir / "ablation_metrics_by_seed.csv", index=False)

    reference = results[results["factor"] == "reference"]
    if reference.empty:
        raise ValueError(
            f"No runs with the reference spec {reference_spec!r} at input dim {args.reference_dim}"
        )
    reference = reference[[*KEYS, "split_path", *METRICS]]

    arch = architecture_effect(results)
    prep = paired_deltas(results, reference, PREPROCESSING_FACTORS)
    rep = paired_deltas(results, reference, REPRESENTATION_FACTORS)
    harm = paired_deltas(results, reference, ("legacy",))
    sizes = effect_sizes(results, arch, reference_spec, args.reference_dim)

    arch.to_csv(output_dir / "architecture_effect.csv", index=False)
    prep.to_csv(output_dir / "preprocessing_effect.csv", index=False)
    rep.to_csv(output_dir / "representation_effect.csv", index=False)
    harm.to_csv(output_dir / "harmonization_effect.csv", index=False)
    sizes.to_csv(output_dir / "effect_sizes.csv", index=False)

    lines = [
        "# Preprocessing and representation ablation",
        "",
        f"Reference: `{reference_spec}` at input dim {args.reference_dim} "
        f"(defaults: {PreprocessingConfig().spec}).",
        "All non-legacy configs give every model identical inputs. Deltas are paired per seed "
        "(same split, same seed) and reported as mean and standard deviation across seeds; "
        "`seeds_improved` counts seeds where the variant beat the reference.",
        "",
        "## Effect sizes: architecture vs preprocessing vs representation",
        "",
        "`architecture_range` is the gap between the best and worst model under the reference "
        "inputs. The preprocessing/representation ranges are the gap between the best and worst "
        "input variant for a fixed model (median and max over the four models).",
        "",
        markdown_table(fmt(sizes)),
        "",
        "## Architecture effect (inputs held fixed)",
        "",
        markdown_table(fmt(arch)),
        "",
        "## Preprocessing effect (architecture held fixed)",
        "",
        markdown_table(fmt(prep)) if not prep.empty else "_No preprocessing variants found._",
        "",
        "## Representation effect (architecture held fixed)",
        "",
        markdown_table(fmt(rep)) if not rep.empty else "_No representation variants found._",
        "",
        "## Legacy per-model preprocessing minus shared reference",
        "",
        markdown_table(fmt(harm)) if not harm.empty else "_No legacy runs found._",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines))
    print(f"Wrote ablation summary to {output_dir}")


if __name__ == "__main__":
    main()
