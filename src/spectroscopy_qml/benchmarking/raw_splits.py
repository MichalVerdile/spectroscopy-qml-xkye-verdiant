"""Deterministic grouped, approximate multilabel stratification for raw cohorts."""

from __future__ import annotations

import numpy as np

from .raw_native import RawCohort, atomic_json, digest_json

SPLIT_VERSION = "group_greedy_v1"
PARTITIONS = ("train", "selection", "calibration", "test")
RATIOS = np.array([0.8, 0.05, 0.05, 0.1])


def validate_split(split: dict, labels: np.ndarray, groups: list[str]) -> None:
    arrays = [np.asarray(split[p], dtype=np.int64) for p in PARTITIONS]
    if any(len(a) == 0 for a in arrays):
        raise ValueError("Every partition must be nonempty; need more independent molecule groups")
    all_ids = np.concatenate(arrays)
    if not np.array_equal(np.sort(all_ids), np.arange(len(labels))):
        raise ValueError("Splits must cover all samples once with no overlap")
    owners: dict[str, str] = {}
    for name, ids in zip(PARTITIONS, arrays, strict=True):
        for i in ids:
            if groups[i] in owners and owners[groups[i]] != name:
                raise ValueError("Molecule/scaffold leakage across partitions")
            owners[groups[i]] = name


def create_group_split(labels: np.ndarray, groups: list[str], seed: int) -> dict:
    if len(labels) != len(groups) or labels.ndim != 2 or not np.isin(labels, [0, 1]).all():
        raise ValueError("Invalid labels/groups")
    members: dict[str, list[int]] = {}
    for i, group in enumerate(groups):
        members.setdefault(group, []).append(i)
    if len(members) < 4:
        raise ValueError("At least four independent groups are required")
    keys = sorted(members)
    rng = np.random.default_rng(seed)
    counts = np.stack([labels[members[g]].sum(axis=0) for g in keys])
    sizes = np.array([len(members[g]) for g in keys])
    # Large/rare groups first, seeded tie-breaking. This is an approximation;
    # exact ratios/stratification are generally impossible with indivisible groups.
    rarity = (counts / np.maximum(counts.sum(axis=0), 1)).sum(axis=1)
    order = np.lexsort((rng.random(len(keys)), -sizes, -rarity))
    target_size = RATIOS * len(labels)
    target_labels = RATIOS[:, None] * counts.sum(axis=0)
    used_size = np.zeros(4)
    used_labels = np.zeros_like(target_labels, dtype=float)
    result: dict[str, list[int]] = {p: [] for p in PARTITIONS}
    for step, g in enumerate(order):
        empty = np.flatnonzero(used_size == 0)
        candidates = empty if len(order) - step <= len(empty) else np.arange(4)
        costs = []
        for p in candidates:
            # Change in globally normalized squared deficits. Size prevents rare
            # strata from absorbing most rows merely to match one positive label.
            size_delta = (
                (used_size[p] + sizes[g] - target_size[p]) ** 2
                - (used_size[p] - target_size[p]) ** 2
            ) / max(target_size[p], 1)
            label_delta = (
                (
                    (used_labels[p] + counts[g] - target_labels[p]) ** 2
                    - (used_labels[p] - target_labels[p]) ** 2
                )
                / np.maximum(target_labels[p], 1)
            ).mean()
            costs.append(size_delta + label_delta)
        best = int(candidates[int(np.argmin(costs))])
        used_size[best] += sizes[g]
        used_labels[best] += counts[g]
        result[PARTITIONS[best]].extend(members[keys[g]])
    result = {p: sorted(ids) for p, ids in result.items()}
    validate_split(result, labels, groups)
    return result


def cohort_split(cohort: RawCohort, seed: int, kind: str = "molecule") -> dict:
    if kind not in {"molecule", "scaffold"}:
        raise ValueError("Unknown split kind")
    groups = [r[f"{kind}_id"] for r in cohort.records]
    splits = create_group_split(cohort.labels, groups, seed)
    # Scaffold grouping must ALSO respect molecule identity.
    validate_split(splits, cohort.labels, [r["molecule_id"] for r in cohort.records])
    quality = {}
    prevalence = cohort.labels.mean(axis=0)
    for p, ids in splits.items():
        positive = cohort.labels[ids].sum(axis=0)
        quality[p] = {
            "n": len(ids),
            "fraction": len(ids) / len(cohort.labels),
            "label_support": positive.tolist(),
            "max_abs_prevalence_deviation": float(np.abs(positive / len(ids) - prevalence).max()),
        }
    result = {
        "version": SPLIT_VERSION,
        "seed": seed,
        "kind": kind,
        "dataset_hash": cohort.manifest["dataset_hash"],
        "partitions": splits,
        "quality": quality,
        "sample_ids": [r["sample_id"] for r in cohort.records],
    }
    result["split_hash"] = digest_json(result)
    return result
