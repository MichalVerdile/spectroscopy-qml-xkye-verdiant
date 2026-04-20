"""Experiment 11: Hybrid CNN-TTN ensemble evaluation.

Combines:
  - CNN k-fold predictions (from benchmark/cnn/models/ir/k_fold/results.pickle)
    for all 37 functional group classes.
  - TTN Head predictions (from experiment11 trained checkpoint)
    for the subset of hard classes. The TTN Head takes frozen CNN features
    (1574-dim) as input — NOT raw spectra.

The ensemble overrides the CNN's predictions at specialist class indices with
the TTN Head's predictions, then reports metrics for:
  1. CNN-only baseline
  2. TTN Head-only (on specialist classes)
  3. Hybrid ensemble (CNN + TTN override)

Usage:
    python ensemble_predict.py \\
        --specialist-dir src/.../experiment11/results/<run_dir> \\
        --cnn-pickle benchmark/cnn/models/ir/k_fold/results.pickle \\
        --features-path benchmark/cnn/features/cnn_ir_features.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = Path(__file__).resolve().parents[5]
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment5.data_loader import (  # noqa: E402
    FUNCTIONAL_GROUPS,
    load_or_create_split_indices,
)
from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment5.train import (  # noqa: E402
    compute_metrics,
    resolve_device,
    threshold_predictions,
)
from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment11.model_ttn_head import (  # noqa: E402
    TTNHead,
)

ALL_LABEL_NAMES = list(FUNCTIONAL_GROUPS.keys())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Hybrid CNN-TTN Head ensemble (Experiment 11)."
    )
    parser.add_argument(
        "--specialist-dir",
        type=Path,
        required=True,
        help="Path to the experiment11 run directory (contains ttn_head_best.pt, "
             "specialist_map.json, selected_thresholds.json, run_config.json).",
    )
    parser.add_argument(
        "--cnn-pickle",
        type=Path,
        default=Path("benchmark/cnn/models/ir/k_fold/results.pickle"),
        help="Path to CNN k-fold results.pickle.",
    )
    parser.add_argument(
        "--features-path",
        type=Path,
        default=Path("benchmark/cnn/features/cnn_ir_features.npz"),
        help="Path to cnn_ir_features.npz produced by extract_cnn_features.py.",
    )
    parser.add_argument("--split-path", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write ensemble_results.json. Defaults to --specialist-dir.",
    )
    return parser


def load_cnn_predictions(pickle_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Return (predictions, targets, test_indices_or_None) from CNN k-fold pickle.

    test_indices is reconstructed from the CNN split seed when available so that
    predictions can be aligned with the full dataset index space.
    """
    import pickle

    with pickle_path.open("rb") as f:
        data = pickle.load(f)

    if "test_predictions" in data and "test_targets" in data:
        preds = np.array(data["test_predictions"])
        targets = np.array(data["test_targets"])
    elif "pred" in data and "tgt" in data:
        preds = np.array(data["pred"])
        targets = np.array(data["tgt"])
    else:
        raise KeyError(f"Cannot find predictions/targets in {pickle_path}. Keys: {list(data.keys())}")

    if preds.dtype in (np.float32, np.float64):
        preds = (preds >= 0.5).astype(np.int32)

    # Reconstruct CNN test indices from seed if available
    cnn_test_indices = None
    seed = data.get("seed")
    train_size = data.get("train_size")
    test_size = data.get("test_size")
    if seed is not None and test_size is not None:
        n_total = int(train_size) + int(test_size)
        rng = np.random.RandomState(int(seed))
        all_idx = np.arange(n_total)
        rng.shuffle(all_idx)
        cnn_test_indices = np.sort(all_idx[int(train_size):])

    return preds.astype(np.int32), targets.astype(np.int32), cnn_test_indices


def load_specialist_config(
    specialist_dir: Path,
) -> tuple[list[int], list[str], dict]:
    map_path = specialist_dir / "specialist_map.json"
    if not map_path.exists():
        raise FileNotFoundError(f"specialist_map.json not found in {specialist_dir}")
    mapping = json.loads(map_path.read_text())
    indices = mapping["specialist_indices"]
    names = mapping["specialist_names"]
    config_path = specialist_dir / "run_config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    return indices, names, config


def load_thresholds(specialist_dir: Path) -> np.ndarray:
    threshold_path = specialist_dir / "selected_thresholds.json"
    if not threshold_path.exists():
        raise FileNotFoundError(f"selected_thresholds.json not found in {specialist_dir}")
    data = json.loads(threshold_path.read_text())
    if isinstance(data.get("thresholds"), dict):
        return np.array(list(data["thresholds"].values()), dtype=np.float32)
    if isinstance(data.get("thresholds"), list):
        return np.array(data["thresholds"], dtype=np.float32)
    raise ValueError(f"Unexpected threshold format in {threshold_path}")


def build_ttn_head(config: dict, num_labels: int) -> TTNHead:
    return TTNHead(
        cnn_feature_dim=int(config.get("cnn_feature_dim", 1574)),
        num_specialist_classes=num_labels,
        chi=int(config.get("chi", 64)),
        num_segments=int(config.get("num_segments", 16)),
        segment_dim=config.get("segment_dim", None),
        merge_mode=str(config.get("merge_mode", "relaxed")),
        merge_residual_weight=float(config.get("merge_residual_weight", 0.1)),
        merge_renormalize_output=bool(config.get("merge_renormalize_output", True)),
        segment_normalize=bool(config.get("segment_normalize", True)),
    )


@torch.no_grad()
def run_ttn_inference(
    model: TTNHead,
    features: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run TTN Head on features[indices], return (labels, probs)."""
    model.eval()
    X_t = torch.from_numpy(features[indices]).float()
    loader = DataLoader(
        TensorDataset(X_t),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    all_probs = []
    for (X_batch,) in loader:
        X_batch = X_batch.to(device)
        logits = model(X_batch)
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(all_probs, axis=0).astype(np.float32)


def print_per_class_f1(label_names: list[str], metrics: dict, prefix: str = "") -> None:
    per_class = metrics.get("per_class_f1", [])
    if per_class is None or len(per_class) == 0:
        return
    print(f"\n{prefix}Per-class F1:")
    for name, f1 in zip(label_names, per_class):
        print(f"  {name:25s}: {f1:.4f}")


def print_metrics_summary(title: str, metrics: dict) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    print(f"  F1 micro:    {metrics['f1_micro']:.4f}")
    print(f"  F1 macro:    {metrics['f1_macro']:.4f}")
    if "precision_micro" in metrics:
        print(f"  Precision:   {metrics['precision_micro']:.4f}")
        print(f"  Recall:      {metrics['recall_micro']:.4f}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_dir = args.output_dir or args.specialist_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print(f"Device: {device}")

    # ── Load specialist config ──────────────────────────────────────────────────
    specialist_indices, specialist_names, run_config = load_specialist_config(args.specialist_dir)
    num_specialist = len(specialist_indices)
    print(f"\nSpecialist classes ({num_specialist}): {specialist_names}")
    print(f"Specialist indices:         {specialist_indices}")

    best_thresholds = load_thresholds(args.specialist_dir)
    print(f"Loaded thresholds ({len(best_thresholds)}): {np.round(best_thresholds, 3).tolist()}")

    # ── Load TTN Head model ─────────────────────────────────────────────────────
    checkpoint_path = args.specialist_dir / "ttn_head_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = build_ttn_head(run_config, num_labels=num_specialist)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    print(f"\nLoaded TTN Head from {checkpoint_path}")

    # ── Load CNN features ───────────────────────────────────────────────────────
    print(f"\nLoading CNN features from {args.features_path} ...")
    if not args.features_path.exists():
        raise FileNotFoundError(
            f"Features file not found: {args.features_path}\n"
            "Run extract_cnn_features.py first."
        )
    data = np.load(args.features_path)
    features = data["features"].astype(np.float32)    # (N, 1574)
    labels_full = data["labels"].astype(np.int32)     # (N, 37)

    nan_count = np.isnan(features).sum()
    if nan_count > 0:
        print(f"WARNING: {nan_count} NaN values in CNN features — replacing with 0.")
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # Z-score normalise (same as during training)
    feat_mean = features.mean(axis=0, keepdims=True)
    feat_std  = features.std(axis=0, keepdims=True) + 1e-8
    features  = ((features - feat_mean) / feat_std).astype(np.float32)
    print(f"Features z-score normalised: {features.shape}")

    labels_specialist = labels_full[:, specialist_indices]
    N = features.shape[0]

    # ── Reproduce data split ────────────────────────────────────────────────────
    split_path = args.split_path
    if split_path is None:
        run_split = run_config.get("split_path")
        if run_split:
            split_path = Path(run_split)
        else:
            split_path = args.specialist_dir / f"data_split_seed{args.seed}_all.npz"

    if not split_path.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_path}\n"
            "Pass --split-path explicitly if the split was created elsewhere."
        )
    print(f"Using split: {split_path}")

    split_indices = load_or_create_split_indices(
        labels=labels_specialist,
        split_path=split_path,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        random_seed=args.seed,
        stratify_multilabel=True,
        overwrite=False,
    )
    test_indices = split_indices["test"]
    print(f"Test set size: {len(test_indices)}")

    # ── Run TTN Head on test set ────────────────────────────────────────────────
    print("\nRunning TTN Head inference...")
    ttn_test_probs = run_ttn_inference(
        model, features, test_indices,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    ttn_test_probs = np.nan_to_num(ttn_test_probs, nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)
    ttn_test_labels = labels_specialist[test_indices].astype(np.int32)
    ttn_test_preds = threshold_predictions(ttn_test_probs, best_thresholds).astype(np.int32)
    ttn_metrics = compute_metrics(ttn_test_labels, ttn_test_preds)

    print_metrics_summary("TTN Head (specialist classes only)", ttn_metrics)
    print_per_class_f1(specialist_names, ttn_metrics, prefix="TTN Head ")

    # ── Load CNN predictions ────────────────────────────────────────────────────
    print(f"\nLoading CNN predictions from {args.cnn_pickle}...")
    if not args.cnn_pickle.exists():
        raise FileNotFoundError(f"CNN pickle not found: {args.cnn_pickle}")

    cnn_preds_all, cnn_targets_all, cnn_test_indices = load_cnn_predictions(args.cnn_pickle)
    print(f"CNN predictions shape: {cnn_preds_all.shape}, targets shape: {cnn_targets_all.shape}")

    if cnn_preds_all.shape[0] == N:
        # Full-dataset pickle: index directly
        cnn_preds_test = cnn_preds_all[test_indices]
        cnn_targets_test = cnn_targets_all[test_indices]
        ttn_eval_local = np.arange(len(test_indices))  # all TTN test samples
    elif cnn_test_indices is not None:
        # Partial pickle (e.g. CNN k-fold test set): find overlap with TTN test set
        print(f"CNN pickle covers {len(cnn_test_indices)} of {N} samples — finding overlap with TTN test set...")
        cnn_idx_set = {int(i): pos for pos, i in enumerate(cnn_test_indices)}
        overlap_ttn_local = []   # positions in test_indices
        overlap_cnn_local = []   # positions in cnn_preds_all
        for local_pos, global_idx in enumerate(test_indices):
            if global_idx in cnn_idx_set:
                overlap_ttn_local.append(local_pos)
                overlap_cnn_local.append(cnn_idx_set[global_idx])
        overlap_ttn_local = np.array(overlap_ttn_local)
        overlap_cnn_local = np.array(overlap_cnn_local)
        if len(overlap_ttn_local) == 0:
            raise ValueError(
                "No overlap between CNN test set and TTN test set. "
                "Use the same seed/split or provide a full-dataset CNN pickle."
            )
        print(f"Overlap: {len(overlap_ttn_local)} samples (of {len(test_indices)} TTN test samples)")
        cnn_preds_test = cnn_preds_all[overlap_cnn_local]
        cnn_targets_test = cnn_targets_all[overlap_cnn_local]
        ttn_eval_local = overlap_ttn_local  # restrict TTN results to overlap
    else:
        raise ValueError(
            f"CNN pickle has {cnn_preds_all.shape[0]} rows but features file has {N} rows, "
            "and no split seed found to reconstruct indices. Cannot align predictions."
        )

    # Restrict TTN predictions to overlap set
    ttn_test_probs_eval = ttn_test_probs[ttn_eval_local]
    ttn_test_labels_eval = ttn_test_labels[ttn_eval_local]
    ttn_test_preds_eval = threshold_predictions(ttn_test_probs_eval, best_thresholds).astype(np.int32)

    # Sanity check: targets must agree
    if not np.array_equal(cnn_targets_test, labels_full[test_indices[ttn_eval_local]]):
        raise ValueError(
            "CNN pickle targets do not match features labels at the overlap indices. "
            "The CNN pickle and features file must come from the same dataset."
        )

    # Full 37-class CNN metrics on overlap test set
    cnn_metrics_full = compute_metrics(cnn_targets_test, cnn_preds_test)
    print_metrics_summary("CNN Baseline (all 37 classes, overlap test set)", cnn_metrics_full)
    print_per_class_f1(ALL_LABEL_NAMES, cnn_metrics_full, prefix="CNN ")

    # CNN on specialist classes only
    cnn_preds_specialist = cnn_preds_test[:, specialist_indices]
    cnn_targets_specialist = cnn_targets_test[:, specialist_indices]
    cnn_specialist_metrics = compute_metrics(cnn_targets_specialist, cnn_preds_specialist)
    print_metrics_summary("CNN on specialist classes only", cnn_specialist_metrics)
    print_per_class_f1(specialist_names, cnn_specialist_metrics, prefix="CNN specialist ")

    # ── Build hybrid ensemble predictions ───────────────────────────────────────
    ensemble_preds = cnn_preds_test.copy()
    for local_idx, global_idx in enumerate(specialist_indices):
        ensemble_preds[:, global_idx] = ttn_test_preds_eval[:, local_idx]

    ensemble_metrics = compute_metrics(cnn_targets_test, ensemble_preds)
    print_metrics_summary("Hybrid Ensemble (CNN + TTN override)", ensemble_metrics)
    print_per_class_f1(ALL_LABEL_NAMES, ensemble_metrics, prefix="Ensemble ")

    # ── Summary comparison ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  COMPARISON SUMMARY")
    print("=" * 60)
    print(f"  {'Model':<35} {'F1 micro':>9} {'F1 macro':>9}")
    print(f"  {'-'*35} {'-'*9} {'-'*9}")
    print(f"  {'CNN baseline (all 37)':<35} {cnn_metrics_full['f1_micro']:>9.4f} {cnn_metrics_full['f1_macro']:>9.4f}")
    print(f"  {'TTN Head (hard classes)':<35} {ttn_metrics['f1_micro']:>9.4f} {ttn_metrics['f1_macro']:>9.4f}")
    print(f"  {'Hybrid ensemble':<35} {ensemble_metrics['f1_micro']:>9.4f} {ensemble_metrics['f1_macro']:>9.4f}")

    delta_micro = ensemble_metrics["f1_micro"] - cnn_metrics_full["f1_micro"]
    delta_macro = ensemble_metrics["f1_macro"] - cnn_metrics_full["f1_macro"]
    sign_micro = "+" if delta_micro >= 0 else ""
    sign_macro = "+" if delta_macro >= 0 else ""
    print(f"\n  Ensemble vs CNN delta:              {sign_micro}{delta_micro:.4f} micro, {sign_macro}{delta_macro:.4f} macro")

    print("\n  Specialist class F1 comparison:")
    print(f"  {'Class':<25} {'CNN':>8} {'TTN':>8} {'Ensemble':>10} {'Delta':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*10} {'-'*8}")
    ttn_per_class = compute_metrics(ttn_test_labels_eval, ttn_test_preds_eval).get("per_class_f1", [])
    ensemble_per_class = ensemble_metrics.get("per_class_f1", [])
    cnn_specialist_per_class = cnn_specialist_metrics.get("per_class_f1", [])
    for local_idx, (name, global_idx) in enumerate(zip(specialist_names, specialist_indices)):
        cnn_f1 = cnn_specialist_per_class[local_idx] if local_idx < len(cnn_specialist_per_class) else float("nan")
        ttn_f1 = ttn_per_class[local_idx] if local_idx < len(ttn_per_class) else float("nan")
        ens_f1 = ensemble_per_class[global_idx] if global_idx < len(ensemble_per_class) else float("nan")
        delta = ens_f1 - cnn_f1
        sign = "+" if delta >= 0 else ""
        print(f"  {name:<25} {cnn_f1:>8.4f} {ttn_f1:>8.4f} {ens_f1:>10.4f} {sign}{delta:>7.4f}")

    # ── Save results ────────────────────────────────────────────────────────────
    results = {
        "cnn_baseline": {
            "f1_micro": float(cnn_metrics_full["f1_micro"]),
            "f1_macro": float(cnn_metrics_full["f1_macro"]),
        },
        "ttn_head": {
            "f1_micro": float(ttn_metrics["f1_micro"]),
            "f1_macro": float(ttn_metrics["f1_macro"]),
            "per_class_f1": {n: float(f) for n, f in zip(specialist_names, ttn_per_class)},
        },
        "hybrid_ensemble": {
            "f1_micro": float(ensemble_metrics["f1_micro"]),
            "f1_macro": float(ensemble_metrics["f1_macro"]),
            "delta_vs_cnn_micro": float(delta_micro),
            "delta_vs_cnn_macro": float(delta_macro),
        },
        "specialist_indices": specialist_indices,
        "specialist_names": specialist_names,
    }
    out_path = output_dir / "ensemble_results.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
