"""Re-tune TTN 10.2 thresholds for hard classes using F-beta (recall-weighted).

For the 10 hard specialist classes, maximize F2 (beta=2 weights recall 2x over
precision) instead of F1. All other classes keep their original thresholds.
Evaluates on test set and prints a comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import fbeta_score, f1_score
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
)
from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment10_2_2.model import (  # noqa: E402
    load_ttn102,
)

ALL_LABEL_NAMES = list(FUNCTIONAL_GROUPS.keys())

HARD_CLASS_INDICES = [1, 13, 14, 18, 19, 21, 24, 28, 33, 35]

DEFAULT_TTN_CHECKPOINT = Path(
    "src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment10_2"
    "/results/full_dataset_run_20260417_173715_percentile/ttn_ir_best.pt"
)
DEFAULT_TTN_CONFIG = Path(
    "src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment10_2"
    "/results/full_dataset_run_20260417_173715_percentile/run_config.json"
)
DEFAULT_THRESHOLDS = Path(
    "src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment10_2"
    "/results/full_dataset_run_20260417_173715_percentile/selected_thresholds.json"
)
DEFAULT_SPLIT = Path(
    "src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment10_2_1"
    "/results/data_split_seed42_all.npz"
)


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--ttn-checkpoint", type=Path, default=DEFAULT_TTN_CHECKPOINT)
    p.add_argument("--ttn-config",     type=Path, default=DEFAULT_TTN_CONFIG)
    p.add_argument("--thresholds",     type=Path, default=DEFAULT_THRESHOLDS)
    p.add_argument("--spectra-cache",  type=Path,
                   default=Path("data/cache/ir_spectra_len1800_snv_all.npz"))
    p.add_argument("--split-path",     type=Path, default=DEFAULT_SPLIT)
    p.add_argument("--beta",           type=float, default=2.0,
                   help="F-beta: beta>1 favours recall, beta<1 favours precision.")
    p.add_argument("--threshold-step", type=float, default=0.02)
    p.add_argument("--hard-indices",   type=str, default=None,
                   help="Comma-separated class indices to re-tune. Default: 10 hard classes.")
    p.add_argument("--output",         type=Path, default=None,
                   help="Save new thresholds JSON. Defaults to thresholds_beta<beta>.json next to original.")
    p.add_argument("--device",         default="cpu")
    p.add_argument("--batch-size",     type=int, default=1024)
    p.add_argument("--seed",           type=int, default=42)
    return p


@torch.no_grad()
def get_probs(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for x_batch, y_batch in loader:
        logits = model(x_batch.to(device))
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(y_batch.numpy().astype(np.int32))
    return np.concatenate(all_probs).astype(np.float32), np.concatenate(all_labels)


def tune_fbeta(y_true_col, probs_col, beta, step):
    """Find threshold that maximises F-beta for a single class."""
    best_thresh, best_score = 0.5, -1.0
    for t in np.arange(step, 1.0, step):
        preds = (probs_col >= t).astype(np.int32)
        score = fbeta_score(y_true_col, preds, beta=beta, zero_division=0)
        if score > best_score:
            best_score, best_thresh = score, float(t)
    return best_thresh, best_score


def main():
    args = build_parser().parse_args()

    hard_indices = (
        [int(x) for x in args.hard_indices.split(",")]
        if args.hard_indices else HARD_CLASS_INDICES
    )
    hard_names = [ALL_LABEL_NAMES[i] for i in hard_indices]

    device = resolve_device(args.device)
    print(f"Device: {device}")

    print("Loading TTN 10.2...")
    ttn_config = json.loads(args.ttn_config.read_text())
    ttn = load_ttn102(args.ttn_checkpoint, ttn_config, device)

    print("Loading spectra...")
    cache  = np.load(args.spectra_cache)
    X      = cache["X"].astype(np.float32)
    y_full = cache["y"].astype(np.int32)

    split = load_or_create_split_indices(
        labels=y_full, split_path=args.split_path,
        train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
        random_seed=args.seed, stratify_multilabel=True,
    )

    def make_loader(idx):
        return DataLoader(
            TensorDataset(torch.from_numpy(X[idx]).float(),
                          torch.from_numpy(y_full[idx]).float()),
            batch_size=args.batch_size, shuffle=False,
        )

    print("Running TTN predictions on val set...")
    val_probs, val_labels = get_probs(ttn, make_loader(split["val"]), device)

    # Load original thresholds
    orig_thresholds = json.loads(args.thresholds.read_text())
    new_thresholds  = dict(orig_thresholds)

    print(f"\nRe-tuning {len(hard_indices)} hard classes with F{args.beta:.0f} (beta={args.beta}):")
    print(f"{'Class':20s} | {'Old thresh':>10} | {'New thresh':>10} | {'Val F2':>8}")
    print("-" * 60)
    for name, idx in zip(hard_names, hard_indices):
        old_t = orig_thresholds.get(name, 0.5)
        new_t, fb = tune_fbeta(val_labels[:, idx], val_probs[:, idx], args.beta, args.threshold_step)
        new_thresholds[name] = new_t
        print(f"{name:20s} | {old_t:10.4f} | {new_t:10.4f} | {fb:8.4f}")

    # Evaluate on test set
    print("\nRunning TTN predictions on test set...")
    test_probs, test_labels = get_probs(ttn, make_loader(split["test"]), device)

    orig_arr = np.array([orig_thresholds.get(n, 0.5) for n in ALL_LABEL_NAMES], dtype=np.float32)
    new_arr  = np.array([new_thresholds.get(n, 0.5)  for n in ALL_LABEL_NAMES], dtype=np.float32)

    orig_metrics = compute_metrics(test_labels, (test_probs >= orig_arr).astype(np.int32))
    new_metrics  = compute_metrics(test_labels, (test_probs >= new_arr).astype(np.int32))

    print(f"\n{'':30s} | {'f1_micro':>8} | {'f1_macro':>8}")
    print("-" * 52)
    print(f"{'Original thresholds':30s} | {float(orig_metrics['f1_micro']):8.4f} | {float(orig_metrics['f1_macro']):8.4f}")
    print(f"{'Re-tuned (F'+str(int(args.beta))+' hard classes)':30s} | {float(new_metrics['f1_micro']):8.4f} | {float(new_metrics['f1_macro']):8.4f}")

    print(f"\n--- Hard class detail (test set) ---")
    print(f"{'Class':20s} | {'F1 orig':>8} | {'F1 new':>8} | {'Delta':>8}")
    print("-" * 55)
    for name, idx in zip(hard_names, hard_indices):
        f1_orig = float(orig_metrics["per_class_f1"][idx])
        f1_new  = float(new_metrics["per_class_f1"][idx])
        print(f"{name:20s} | {f1_orig:8.4f} | {f1_new:8.4f} | {f1_new - f1_orig:+8.4f}")

    # Save new thresholds
    out_path = args.output or (
        args.thresholds.parent / f"selected_thresholds_beta{int(args.beta)}.json"
    )
    out_path.write_text(json.dumps(new_thresholds, indent=2) + "\n")
    print(f"\nNew thresholds saved to {out_path}")


if __name__ == "__main__":
    main()
