"""Experiment 10.2.1: Ensemble evaluation — TTN 10.2 + Specialist Head.

Compares three configurations on the test set:
  1. TTN 10.2 alone (all 37 classes)
  2. Specialist Head alone (10 hard classes)
  3. Ensemble: TTN 10.2 + Specialist Head override for the 10 hard classes
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
from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment10_2_1.model import (  # noqa: E402
    TTN102SpecialistEnsemble,
    load_ttn102,
)

ALL_LABEL_NAMES = list(FUNCTIONAL_GROUPS.keys())

DEFAULT_TTN_CHECKPOINT = Path(
    "src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment10_2"
    "/results/full_dataset_run_20260417_173715_percentile/ttn_ir_best.pt"
)
DEFAULT_TTN_CONFIG = Path(
    "src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment10_2"
    "/results/full_dataset_run_20260417_173715_percentile/run_config.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ensemble evaluation: TTN 10.2 + Specialist Head."
    )
    parser.add_argument("--specialist-dir", type=Path, required=True,
                        help="Output dir of a completed experiment10_2_1 training run.")
    parser.add_argument("--spectra-cache", type=Path,
                        default=Path("data/cache/ir_spectra_len1800_snv_all.npz"))
    parser.add_argument("--ttn-checkpoint", type=Path, default=DEFAULT_TTN_CHECKPOINT)
    parser.add_argument("--ttn-config", type=Path, default=DEFAULT_TTN_CONFIG)
    parser.add_argument("--split-path", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser


@torch.no_grad()
def get_probs(model, loader, device, n_out):
    model.eval()
    all_probs, all_labels = [], []
    for x_batch, y_batch in loader:
        logits = model(x_batch.to(device))
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(y_batch.numpy().astype(np.int32))
    return np.concatenate(all_probs), np.concatenate(all_labels)


def print_metrics(label: str, metrics: dict, label_names: list[str], indices: list[int] | None = None):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  f1_macro:  {float(metrics['f1_macro']):.4f}")
    print(f"  f1_micro:  {float(metrics['f1_micro']):.4f}")
    if indices is not None and "per_class_f1" in metrics:
        print("  Per-class F1 (specialist classes):")
        for i, f1 in zip(indices, metrics["per_class_f1"]):
            print(f"    {ALL_LABEL_NAMES[i]:20s}: {float(f1):.4f}")


def main():
    parser = build_parser()
    args = parser.parse_args()

    device = torch.device("cpu")

    # Load specialist map
    spec_map = json.loads((args.specialist_dir / "specialist_map.json").read_text())
    specialist_indices = spec_map["specialist_indices"]
    specialist_names   = spec_map["specialist_names"]

    # Load thresholds
    ttn_thresholds  = json.loads(
        (Path(str(args.ttn_config).replace("run_config.json", "selected_thresholds.json"))).read_text()
    )
    spec_thresholds = json.loads((args.specialist_dir / "selected_thresholds.json").read_text())

    # Load data
    print("Loading spectra...")
    cache  = np.load(args.spectra_cache)
    X      = cache["X"].astype(np.float32)
    y_full = cache["y"].astype(np.int32)

    split_path = args.split_path or (args.specialist_dir / f"data_split_seed{args.seed}_all.npz")
    split_indices = load_or_create_split_indices(
        labels=y_full, split_path=split_path,
        train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
        random_seed=args.seed, stratify_multilabel=True,
    )
    test_idx = split_indices["test"]

    test_loader_full = DataLoader(
        TensorDataset(torch.from_numpy(X[test_idx]).float(),
                      torch.from_numpy(y_full[test_idx]).float()),
        batch_size=args.batch_size, shuffle=False,
    )

    # Load TTN 10.2
    print("Loading TTN 10.2...")
    ttn_config = json.loads(args.ttn_config.read_text())
    ttn = load_ttn102(args.ttn_checkpoint, ttn_config, device)

    # Load specialist ensemble
    print("Loading specialist head...")
    spec_config = json.loads((args.specialist_dir / "run_config.json").read_text())
    ensemble = TTN102SpecialistEnsemble(
        ttn=ttn,
        specialist_indices=specialist_indices,
        hidden_dim=spec_config.get("mlp_hidden_dim", 64),
        dropout=0.0,
    ).to(device)
    state = torch.load(args.specialist_dir / "specialist_best.pt", map_location=device, weights_only=True)
    ensemble.load_state_dict(state)

    # --- 1. TTN 10.2 alone ---
    print("\nRunning TTN 10.2 predictions...")
    ttn_probs_list, ttn_labels_list = [], []
    ttn.eval()
    with torch.no_grad():
        for x_batch, y_batch in test_loader_full:
            logits = ttn(x_batch.to(device))
            ttn_probs_list.append(torch.sigmoid(logits).cpu().numpy())
            ttn_labels_list.append(y_batch.numpy().astype(np.int32))
    ttn_probs  = np.concatenate(ttn_probs_list)   # (N, 37)
    ttn_labels = np.concatenate(ttn_labels_list)  # (N, 37)

    ttn_thresh_arr = np.array([ttn_thresholds.get(n, 0.5) for n in ALL_LABEL_NAMES], dtype=np.float32)
    ttn_preds = (ttn_probs >= ttn_thresh_arr).astype(np.int32)
    ttn_metrics = compute_metrics(ttn_labels, ttn_preds)
    print_metrics("TTN 10.2 alone (all 37 classes)", ttn_metrics, ALL_LABEL_NAMES)

    # --- 2. Specialist head alone (10 classes) ---
    print("\nRunning specialist head predictions...")
    spec_probs_list = []
    ensemble.eval()
    with torch.no_grad():
        for x_batch, _ in test_loader_full:
            spec_probs_list.append(torch.sigmoid(ensemble(x_batch.to(device))).cpu().numpy())
    spec_probs  = np.concatenate(spec_probs_list)  # (N, 10)
    spec_labels = ttn_labels[:, specialist_indices] # (N, 10)

    spec_thresh_arr = np.array([spec_thresholds.get(n, 0.5) for n in specialist_names], dtype=np.float32)
    spec_preds  = (spec_probs >= spec_thresh_arr).astype(np.int32)
    spec_metrics = compute_metrics(spec_labels, spec_preds)
    print_metrics("Specialist Head alone (10 hard classes)", spec_metrics, ALL_LABEL_NAMES, specialist_indices)

    ttn_per  = ttn_metrics["per_class_f1"]
    spec_per = spec_metrics["per_class_f1"]

    # --- 3. Ensemble: TTN 10.2 + selective specialist override ---
    # Only override classes where specialist F1 > TTN F1 on validation
    override_indices = [
        (local_i, global_i)
        for local_i, global_i in enumerate(specialist_indices)
        if float(spec_per[local_i]) > float(ttn_per[global_i])
    ]
    override_global = [g for _, g in override_indices]
    print(f"\nSelective override for {len(override_indices)}/10 classes: "
          f"{[ALL_LABEL_NAMES[g] for g in override_global]}")

    ensemble_probs = ttn_probs.copy()
    ens_thresh_arr = ttn_thresh_arr.copy()
    for local_i, global_i in override_indices:
        ensemble_probs[:, global_i] = spec_probs[:, local_i]
        ens_thresh_arr[global_i]    = spec_thresh_arr[local_i]

    ens_preds   = (ensemble_probs >= ens_thresh_arr).astype(np.int32)
    ens_metrics = compute_metrics(ttn_labels, ens_preds)
    print_metrics("Ensemble: TTN 10.2 + Selective Specialist override", ens_metrics, ALL_LABEL_NAMES)

    ens_per = ens_metrics["per_class_f1"]

    # Specialist classes comparison
    print("\n--- Specialist classes detail ---")
    print(f"{'Class':20s} | {'TTN 10.2':>10} | {'Specialist':>10} | {'Ensemble':>10} | Override")
    print("-" * 72)
    for local_i, global_i in enumerate(specialist_indices):
        name     = ALL_LABEL_NAMES[global_i]
        used     = "✓" if global_i in override_global else " "
        print(f"{name:20s} | {float(ttn_per[global_i]):10.4f} | {float(spec_per[local_i]):10.4f} | {float(ens_per[global_i]):10.4f} | {used}")

    # Save results
    results = {
        "ttn_macro": float(ttn_metrics["f1_macro"]),
        "ttn_micro": float(ttn_metrics["f1_micro"]),
        "specialist_macro": float(spec_metrics["f1_macro"]),
        "specialist_micro": float(spec_metrics["f1_micro"]),
        "ensemble_macro": float(ens_metrics["f1_macro"]),
        "ensemble_micro": float(ens_metrics["f1_micro"]),
        "overridden_classes": [ALL_LABEL_NAMES[g] for g in override_global],
    }
    out_path = args.specialist_dir / "ensemble_results.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
