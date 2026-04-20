"""Evaluate CNN baseline plus linear / MLP / QCNN specialist-head overrides."""

from __future__ import annotations

import argparse
import json
import pickle
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
from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment11_qcnn_head.model import (  # noqa: E402
    SpecialistHeadEnsemble,
)

ALL_LABEL_NAMES = list(FUNCTIONAL_GROUPS.keys())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate CNN + specialist-head ensemble on hard classes."
    )
    parser.add_argument("--specialist-dir", type=Path, required=True)
    parser.add_argument(
        "--cnn-pickle",
        type=Path,
        default=Path("benchmark/cnn/models/ir/k_fold/results.pickle"),
    )
    parser.add_argument(
        "--features-path",
        type=Path,
        default=Path("benchmark/cnn/features/cnn_ir_features.npz"),
    )
    parser.add_argument("--split-path", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Blend factor: 1.0 = hard replace with specialist probs.")
    return parser


def resolve_head_device(requested_device: str, head_type: str) -> torch.device:
    device = resolve_device(requested_device)
    if head_type == "qcnn" and device.type != "cpu":
        print("QCNN head uses PennyLane default.qubit; falling back to CPU.")
        return torch.device("cpu")
    return device


def load_cnn_predictions(pickle_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with pickle_path.open("rb") as handle:
        data = pickle.load(handle)
    if "test_predictions" in data and "test_targets" in data:
        preds = np.asarray(data["test_predictions"])
        targets = np.asarray(data["test_targets"])
    elif "pred" in data and "tgt" in data:
        preds = np.asarray(data["pred"])
        targets = np.asarray(data["tgt"])
    else:
        raise KeyError(f"Unexpected CNN pickle format: {list(data.keys())}")
    return preds.astype(np.float32), targets.astype(np.int32)


def load_thresholds(run_dir: Path) -> np.ndarray:
    data = json.loads((run_dir / "selected_thresholds.json").read_text())
    thresholds = data["thresholds"]
    if isinstance(thresholds, dict):
        return np.asarray(list(thresholds.values()), dtype=np.float32)
    return np.asarray(thresholds, dtype=np.float32)


def build_model_from_run(run_dir: Path, num_specialists: int) -> SpecialistHeadEnsemble:
    config = json.loads((run_dir / "run_config.json").read_text())
    return SpecialistHeadEnsemble(
        num_specialist_classes=num_specialists,
        head_type=config["head_type"],
        input_dim=int(config.get("cnn_feature_dim", 1574)),
        mlp_hidden_dim=int(config.get("mlp_hidden_dim", 128)),
        qcnn_qubits=int(config.get("qcnn_qubits", 8)),
        qcnn_projection_hidden_dim=int(config.get("qcnn_projection_hidden_dim", 64)),
        dropout=float(config.get("dropout", 0.2)),
    )


@torch.no_grad()
def run_specialist_inference(
    model: SpecialistHeadEnsemble,
    features: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> np.ndarray:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(features[indices]).float()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    all_probs = []
    model.eval()
    for (x_batch,) in loader:
        all_probs.append(model(x_batch.to(device), apply_sigmoid=True).cpu().numpy())
    return np.concatenate(all_probs, axis=0).astype(np.float32)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    specialist_map = json.loads((args.specialist_dir / "specialist_map.json").read_text())
    specialist_indices = specialist_map["specialist_indices"]
    specialist_names = specialist_map["specialist_names"]
    head_type = specialist_map.get("head_type", "qcnn")
    device = resolve_head_device(args.device, head_type)
    thresholds = load_thresholds(args.specialist_dir)

    features_npz = np.load(args.features_path)
    features = features_npz["features"].astype(np.float32)
    labels_full = features_npz["labels"].astype(np.int32)
    split_path = args.split_path or (args.specialist_dir / f"data_split_seed{args.seed}_all.npz")
    split_indices = load_or_create_split_indices(
        labels=labels_full,
        split_path=split_path,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        random_seed=args.seed,
        stratify_multilabel=True,
        overwrite=False,
    )
    test_indices = split_indices["test"]

    cnn_probs, cnn_targets = load_cnn_predictions(args.cnn_pickle)
    if cnn_probs.shape[0] != len(test_indices):
        raise ValueError(
            f"CNN predictions length {cnn_probs.shape[0]} does not match test split {len(test_indices)}."
        )

    model = build_model_from_run(args.specialist_dir, num_specialists=len(specialist_indices))
    state = torch.load(args.specialist_dir / "specialist_head_best.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model = model.to(device)

    specialist_probs = run_specialist_inference(
        model=model,
        features=features,
        indices=test_indices,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    specialist_preds = threshold_predictions(specialist_probs, thresholds)

    cnn_preds = (cnn_probs >= 0.5).astype(np.int32)
    ensemble_probs = cnn_probs.copy()
    for local_idx, global_idx in enumerate(specialist_indices):
        ensemble_probs[:, global_idx] = (
            args.alpha * specialist_probs[:, local_idx]
            + (1.0 - args.alpha) * ensemble_probs[:, global_idx]
        )
    ensemble_preds = (ensemble_probs >= 0.5).astype(np.int32)

    labels_test = labels_full[test_indices]
    cnn_metrics = compute_metrics(labels_test, cnn_preds)
    ensemble_metrics = compute_metrics(labels_test, ensemble_preds)
    specialist_metrics = compute_metrics(
        labels_test[:, specialist_indices],
        specialist_preds,
    )

    payload = {
        "specialist_names": specialist_names,
        "specialist_indices": specialist_indices,
        "alpha": args.alpha,
        "cnn_metrics": {
            k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else float(v))
            for k, v in cnn_metrics.items()
        },
        "specialist_metrics": {
            k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else float(v))
            for k, v in specialist_metrics.items()
        },
        "ensemble_metrics": {
            k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else float(v))
            for k, v in ensemble_metrics.items()
        },
    }
    out_path = args.specialist_dir / "ensemble_results.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"CNN f1_micro:      {cnn_metrics['f1_micro']:.4f}")
    print(f"Ensemble f1_micro: {ensemble_metrics['f1_micro']:.4f}")
    print(f"Ensemble f1_macro: {ensemble_metrics['f1_macro']:.4f}")
    print(f"Saved ensemble report to {out_path}")


if __name__ == "__main__":
    main()
