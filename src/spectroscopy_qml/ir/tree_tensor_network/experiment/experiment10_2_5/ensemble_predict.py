"""Experiment 10.2.5: Ensemble evaluation — TTN 10.2 + Quanvolutional Specialist Heads."""

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
    threshold_predictions,
)
from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment10_2_5.model import (  # noqa: E402
    TTN102QuanvEnsemble,
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
        description="Ensemble evaluation: TTN 10.2 + Quanvolutional Specialist Heads (Exp 10.2.5)."
    )
    parser.add_argument("--specialist-dir", type=Path, required=True)
    parser.add_argument("--spectra-cache", type=Path,
                        default=Path("data/cache/ir_spectra_len1800_snv_all.npz"))
    parser.add_argument("--ttn-checkpoint", type=Path, default=DEFAULT_TTN_CHECKPOINT)
    parser.add_argument("--ttn-config", type=Path, default=DEFAULT_TTN_CONFIG)
    parser.add_argument("--split-path", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    return parser


@torch.no_grad()
def get_probs_ttn(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    for x_batch, y_batch in loader:
        logits = model(x_batch)
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(y_batch.numpy().astype(np.int32))
    return np.concatenate(all_probs), np.concatenate(all_labels)


@torch.no_grad()
def get_probs_specialist(model, loader):
    model.eval()
    all_probs = []
    for x_batch, _ in loader:
        logits = model(x_batch)
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(all_probs)


def print_metrics(label, metrics, indices=None):
    print(f"\n{'='*60}\n  {label}\n{'='*60}")
    print(f"  f1_macro:  {float(metrics['f1_macro']):.4f}")
    print(f"  f1_micro:  {float(metrics['f1_micro']):.4f}")
    if indices is not None and "per_class_f1" in metrics:
        print("  Per-class F1 (specialist classes):")
        for i, f1 in zip(indices, metrics["per_class_f1"]):
            print(f"    {ALL_LABEL_NAMES[i]:20s}: {float(f1):.4f}")


def main():
    args = build_parser().parse_args()
    device = torch.device("cpu")

    spec_map = json.loads((args.specialist_dir / "specialist_map.json").read_text())
    specialist_indices = spec_map["specialist_indices"]
    specialist_names   = spec_map["specialist_names"]

    ttn_thresh_path = Path(str(args.ttn_config).replace("run_config.json", "selected_thresholds.json"))
    ttn_thresholds  = json.loads(ttn_thresh_path.read_text())
    spec_thresholds = json.loads((args.specialist_dir / "selected_thresholds.json").read_text())

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
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X[test_idx]).float(),
                      torch.from_numpy(y_full[test_idx]).float()),
        batch_size=args.batch_size, shuffle=False,
    )

    print("Loading TTN 10.2...")
    ttn_config = json.loads(args.ttn_config.read_text())
    ttn = load_ttn102(args.ttn_checkpoint, ttn_config, device)

    print("Loading Quanvolutional specialist ensemble...")
    spec_config = json.loads((args.specialist_dir / "run_config.json").read_text())
    ensemble = TTN102QuanvEnsemble(
        ttn=ttn,
        specialist_indices=specialist_indices,
        patch_size=spec_config.get("patch_size", 4),
        stride=spec_config.get("stride", 2),
        n_filters=spec_config.get("n_filters", 8),
        quanv_seed=spec_config.get("quanv_seed", 42),
        conv_channels=spec_config.get("conv_channels", 32),
        hidden_dim=spec_config.get("hidden_dim", 64),
        dropout=0.0,
        use_quanv=not spec_config.get("use_trainable_conv", False),
    ).to(device)
    state = torch.load(args.specialist_dir / "specialist_best.pt", map_location=device, weights_only=True)
    ensemble.load_state_dict(state)

    # --- 1. TTN alone ---
    ttn_probs, ttn_labels = get_probs_ttn(ttn, test_loader)
    ttn_thresh_arr = np.array([ttn_thresholds.get(n, 0.5) for n in ALL_LABEL_NAMES], dtype=np.float32)
    ttn_metrics = compute_metrics(ttn_labels, (ttn_probs >= ttn_thresh_arr).astype(np.int32))
    print_metrics("TTN 10.2 alone (all 37 classes)", ttn_metrics)

    # --- 2. Quanvolutional specialist alone ---
    spec_probs   = get_probs_specialist(ensemble, test_loader)
    spec_labels  = ttn_labels[:, specialist_indices]
    spec_thresh_arr = np.array([spec_thresholds.get(n, 0.5) for n in specialist_names], dtype=np.float32)
    spec_metrics = compute_metrics(spec_labels, (spec_probs >= spec_thresh_arr).astype(np.int32))
    print_metrics(
        f"Quanvolutional Specialist Heads alone ({len(specialist_indices)} hard classes)",
        spec_metrics,
        specialist_indices,
    )

    ttn_per  = ttn_metrics["per_class_f1"]
    spec_per = spec_metrics["per_class_f1"]

    # --- 3. Selective ensemble ---
    override_indices = [
        (local_i, global_i)
        for local_i, global_i in enumerate(specialist_indices)
        if float(spec_per[local_i]) > float(ttn_per[global_i])
    ]
    override_global = [g for _, g in override_indices]
    print(f"\nSelective override {len(override_indices)}/{len(specialist_indices)}: "
          f"{[ALL_LABEL_NAMES[g] for g in override_global]}")

    ensemble_probs = ttn_probs.copy()
    ens_thresh_arr = ttn_thresh_arr.copy()
    for local_i, global_i in override_indices:
        ensemble_probs[:, global_i] = spec_probs[:, local_i]
        ens_thresh_arr[global_i]    = spec_thresh_arr[local_i]

    ens_metrics = compute_metrics(ttn_labels, (ensemble_probs >= ens_thresh_arr).astype(np.int32))
    print_metrics("Ensemble: TTN 10.2 + Quanvolutional Specialist Override", ens_metrics)

    ens_per = ens_metrics["per_class_f1"]
    print("\n--- Specialist classes detail ---")
    print(f"{'Class':20s} | {'TTN 10.2':>10} | {'Quanv Head':>10} | {'Ensemble':>10} | Override")
    print("-" * 72)
    for local_i, global_i in enumerate(specialist_indices):
        name = ALL_LABEL_NAMES[global_i]
        used = "yes" if global_i in override_global else "no"
        print(f"{name:20s} | {float(ttn_per[global_i]):10.4f} | {float(spec_per[local_i]):10.4f} | {float(ens_per[global_i]):10.4f} | {used}")

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
