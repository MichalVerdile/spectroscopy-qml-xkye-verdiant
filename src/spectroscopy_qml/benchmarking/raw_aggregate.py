"""Paired, provenance-checked summaries. No choice is made from test scores."""

from __future__ import annotations

import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import t

from .raw_native import atomic_json
from .raw_runner import verify_complete

EXPECTED_SEEDS = {41, 42, 43, 44, 45}
COMPARISONS = [("mps", "cnn"), ("mps", "xgb"), ("ttn", "cnn"), ("ttn", "xgb")]


def describe(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if n > 1 else None
    half_width = float(t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if sd is not None else None
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "descriptive_t95": [mean - half_width, mean + half_width]
        if half_width is not None
        else None,
    }


def exact_sign_flip_p(differences):
    d = np.asarray(differences, dtype=float)
    if not 1 <= len(d) <= 20:
        raise ValueError("Exact sign-flip test supports 1..20 paired differences")
    observed = abs(d.mean())
    null = [abs(np.mean(d * signs)) for signs in itertools.product([-1, 1], repeat=len(d))]
    return float(np.mean(np.asarray(null) >= observed - 1e-12))


def holm_adjust(p_values):
    order = np.argsort(p_values)
    adjusted = np.zeros(len(p_values))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def aggregate(root: Path, output: Path, *, allow_incomplete=False):
    runs = []
    for path in sorted(root.glob("*/seed_*/*/result.json")):
        verify_complete(path.parent)
        runs.append(json.loads(path.read_text()))
    if not runs:
        raise ValueError("No verified final runs")
    global_keys = (
        "dataset_hash",
        "mode",
        "protocol",
        "code",
        "threshold_mode",
        "synthetic",
        "variant",
        "environment",
        "epochs",
        "patience",
        "time_limit_per_trial",
        "batch_size",
        "threads",
        "parameter_limit",
        "calibration_min_samples",
        "calibration_min_support",
    )
    reference = runs[0]["spec"]
    for run in runs:
        for key in global_keys:
            if run["spec"][key] != reference[key]:
                raise ValueError(f"Incompatible comparison: {key}")
    grouped: dict[tuple[str, int], dict[str, dict]] = defaultdict(dict)
    rows = []
    for run in runs:
        spec = run["spec"]
        pair_key = (spec["modality"], spec["seed"])
        if spec["family"] in grouped[pair_key]:
            raise ValueError("Duplicate family/seed/modality result")
        for other in grouped[pair_key].values():
            if any(other["spec"][k] != spec[k] for k in ("input_hash", "split_hash")):
                raise ValueError("Pairing requires identical raw inputs and exact sample splits")
            if other["search_trial_count"] != run["search_trial_count"]:
                raise ValueError(
                    "Unequal trial counts: do not present as equal-search-budget comparison"
                )
        grouped[pair_key][spec["family"]] = run
        count = run["winner"]["parameter_count"]
        rows.append(
            {
                "modality": spec["modality"],
                "seed": spec["seed"],
                "model": spec["family"],
                "architecture": run["winner"]["architecture"],
                "test_f1_micro": run["test_f1_micro"],
                "test_f1_macro": run["test_f1_macro"],
                "training_seconds": run["winner"]["training_seconds"],
                "search_training_seconds": run["search_training_seconds"],
                "fit_wall_seconds_this_invocation": run["winner"][
                    "fit_wall_seconds_this_invocation"
                ],
                "cpu_process_seconds_this_invocation": run["winner"][
                    "cpu_process_seconds_this_invocation"
                ],
                "search_fit_wall_seconds_this_invocation": run[
                    "search_fit_wall_seconds_this_invocation"
                ],
                "inference_seconds": run["inference_seconds"],
                "parameter_count": count,
                "tree_node_count": run["winner"].get("tree_node_count"),
                "model_bytes": run["model_bytes"],
                "gpu_peak_allocated_bytes": run["winner"]["gpu_peak_allocated_bytes"],
                "gpu_process_peak_sampled_bytes": run["winner"]["gpu_process_peak_sampled_bytes"],
                "f1_micro_per_million_parameters": run["test_f1_micro"] / (count / 1e6)
                if count
                else None,
            }
        )
    modalities = sorted({key[0] for key in grouped})
    summaries, comparisons, selected_families, per_class = [], [], [], []
    for modality in modalities:
        available_seeds = {seed for m, seed in grouped if m == modality}
        if not allow_incomplete and available_seeds != EXPECTED_SEEDS:
            raise ValueError(
                f"{modality}: expected all five seeds, found {sorted(available_seeds)}"
            )
        for seed in available_seeds:
            if not allow_incomplete and set(grouped[(modality, seed)]) != {
                "cnn",
                "xgb",
                "mps",
                "ttn",
            }:
                raise ValueError(f"{modality}/{seed}: missing models")
        for family in ("cnn", "xgb", "mps", "ttn"):
            subset = [r for r in rows if r["modality"] == modality and r["model"] == family]
            if not subset:
                continue
            summary: dict = {"modality": modality, "model": family}
            for metric in (
                "test_f1_micro",
                "test_f1_macro",
                "training_seconds",
                "search_training_seconds",
                "fit_wall_seconds_this_invocation",
                "cpu_process_seconds_this_invocation",
                "search_fit_wall_seconds_this_invocation",
                "inference_seconds",
                "model_bytes",
                "parameter_count",
                "gpu_peak_allocated_bytes",
                "gpu_process_peak_sampled_bytes",
            ):
                values = [r[metric] for r in subset if r[metric] is not None]
                summary[metric] = describe(values) if values else None
            summaries.append(summary)
            class_runs = [
                grouped[(modality, seed)][family]
                for seed in sorted(available_seeds)
                if family in grouped[(modality, seed)]
            ]
            for j in range(len(class_runs[0]["test_per_class_f1"])):
                per_class.append(
                    {
                        "modality": modality,
                        "model": family,
                        "label_index": j,
                        "f1": describe([r["test_per_class_f1"][j] for r in class_runs]),
                        "test_support": [r["test_support"][j] for r in class_runs],
                    }
                )
        for left, right in COMPARISONS:
            paired = [
                grouped[(modality, s)]
                for s in sorted(available_seeds)
                if left in grouped[(modality, s)] and right in grouped[(modality, s)]
            ]
            if not paired:
                continue
            differences = [r[left]["test_f1_micro"] - r[right]["test_f1_micro"] for r in paired]
            comparisons.append(
                {
                    "modality": modality,
                    "left": left,
                    "right": right,
                    "micro_difference": describe(differences),
                    "paired_micro_differences": differences,
                    "macro_difference": describe(
                        [r[left]["test_f1_macro"] - r[right]["test_f1_macro"] for r in paired]
                    ),
                    "sign_flip_p": exact_sign_flip_p(differences),
                }
            )
        # Family winners are selected on validation at fixed .5, independently of
        # every test result; report which family was chosen for each seed.
        for seed in sorted(available_seeds):
            g = grouped[(modality, seed)]
            if set(g) == {"cnn", "xgb", "mps", "ttn"}:
                classical = max(("cnn", "xgb"), key=lambda f: g[f]["winner"]["selection_f1_micro"])
                tensor = max(("mps", "ttn"), key=lambda f: g[f]["winner"]["selection_f1_micro"])
                selected_families.append(
                    {
                        "modality": modality,
                        "seed": seed,
                        "classical": classical,
                        "tensor": tensor,
                        "micro_difference": g[tensor]["test_f1_micro"]
                        - g[classical]["test_f1_micro"],
                        "macro_difference": g[tensor]["test_f1_macro"]
                        - g[classical]["test_f1_macro"],
                    }
                )
    adjusted = holm_adjust([c["sign_flip_p"] for c in comparisons])
    for comparison, value in zip(comparisons, adjusted, strict=True):
        comparison["holm_p_across_primary_comparisons"] = value
    family_summary = []
    for modality in modalities:
        selected = [r for r in selected_families if r["modality"] == modality]
        if selected:
            family_summary.append(
                {
                    "modality": modality,
                    "micro_difference": describe([r["micro_difference"] for r in selected]),
                    "macro_difference": describe([r["macro_difference"] for r in selected]),
                }
            )
    first_predictions = next(root.glob("*/seed_*/*/predictions.npz"))
    with np.load(first_predictions, allow_pickle=False) as data:
        label_names = data["label_names"].tolist()
    for row in per_class:
        row["label_name"] = label_names[row["label_index"]]
    result = {
        "validation_selected_family_summary": family_summary,
        "protocol": reference["protocol"],
        "synthetic": reference["synthetic"],
        "uncertainty": "Descriptive variation across repeated split/training seeds on one dataset, not independent population replications. Overlapping test sets violate independence assumptions of t/sign-flip inference. No superiority claim follows from these summaries. Five pairs have minimum two-sided exact sign-flip p=0.0625 before Holm.",
        "rows": rows,
        "summary": summaries,
        "comparisons": comparisons,
        "validation_selected_family_comparisons": selected_families,
        "per_class": per_class,
    }
    # Rare-label membership defined ONLY using training support; saved predictions
    # allow exact recomputation for each threshold policy.
    rare_rows = []
    for path in sorted(root.glob("*/seed_*/*/predictions.npz")):
        with np.load(path, allow_pickle=False) as data:
            rare = (data["train_support"] > 0) & (data["train_support"] < 100)
            calibration = json.loads((path.parent / "thresholds.json").read_text())[
                reference["threshold_mode"]
            ]
            from .raw_evaluation import evaluate

            metrics = evaluate(data["y_test"], data["p_test"], calibration["thresholds"])
            rare_rows.append(
                {
                    "run": str(path.parent.relative_to(root)),
                    "train_support_cutoff": 100,
                    "rare_label_names": data["label_names"][rare].tolist(),
                    "rare_label_macro_f1": float(np.array(metrics["per_class_f1"])[rare].mean())
                    if rare.any()
                    else None,
                    "unseen_training_label_names": data["label_names"][
                        data["train_support"] == 0
                    ].tolist(),
                }
            )
    result["rare_labels"] = rare_rows
    atomic_json(output, result)
    # Portable exact tables plus a scientific scatterplot artifact for export.
    import csv

    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for family in ("cnn", "xgb", "mps", "ttn"):
        subset = [r for r in rows if r["model"] == family]
        axes[0].scatter(
            [r["model_bytes"] for r in subset], [r["test_f1_micro"] for r in subset], label=family
        )
        neural = [r for r in subset if r["parameter_count"]]
        axes[1].scatter(
            [r["parameter_count"] for r in neural],
            [r["test_f1_micro"] for r in neural],
            label=family,
        )
    for ax, xlabel in zip(
        axes, ("Serialized model bytes", "Trainable neural parameters"), strict=True
    ):
        ax.set(xscale="log", xlabel=xlabel, ylabel="Test micro F1", ylim=(0, 1))
        ax.legend()
    fig.suptitle(
        "SYNTHETIC SMOKE — no scientific conclusion"
        if reference["synthetic"]
        else "Raw-native model efficiency"
    )
    fig.savefig(output.with_suffix(".png"), dpi=180)
    plt.close(fig)
    return result
