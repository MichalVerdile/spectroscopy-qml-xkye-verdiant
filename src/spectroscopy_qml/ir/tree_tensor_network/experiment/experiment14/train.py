"""Train Experiment 14 reduction-before-quantum ablations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment5.losses import build_loss  # noqa: E402
from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment5.train import (  # noqa: E402
    EarlyStopping,
    build_threshold_grid,
    compute_metrics,
    get_pos_weight,
    resolve_device,
    select_early_stopping_score,
    threshold_predictions,
    tune_thresholds,
    write_epoch_details,
    write_threshold_artifact,
)
from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment14.model import (  # noqa: E402
    Experiment14Classifier,
    Experiment14FeatureExtractor,
    PCACompressor,
    RawSubsampleCompressor,
    SharedClassicalHead,
    SharedQuantumHead,
    TNSequenceCompressor,
    count_trainable_parameters,
)


VARIANT_LABELS = {
    "raw_quantum": "A: raw subsampling -> quantum",
    "pca_quantum": "B: PCA -> quantum",
    "tn_quantum": "C: TN compression -> quantum",
    "tn_classical": "D: TN compression -> classical",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Experiment 14: compare raw/PCA/TN compression before a shared quantum head."
    )
    parser.add_argument(
        "--spectra-cache",
        type=Path,
        default=Path("data/cache/ir_spectra_len1800_snv_all.npz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment14/results"),
    )
    parser.add_argument("--feature-cache-path", type=Path, default=None)
    parser.add_argument("--overwrite-feature-cache", action="store_true")
    parser.add_argument("--split-path", type=Path, default=None)
    parser.add_argument("--overwrite-split", action="store_true")
    parser.add_argument(
        "--variant",
        choices=["raw_quantum", "pca_quantum", "tn_quantum", "tn_classical"],
        default="tn_quantum",
    )
    parser.add_argument(
        "--feature-source",
        choices=["raw", "sg_no_norm", "sg_z_score", "voigt_z_score"],
        default="raw",
    )
    parser.add_argument("--include-raw-channel", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sg-window-length", type=int, default=11)
    parser.add_argument("--sg-polyorder", type=int, default=3)
    parser.add_argument("--voigt-gamma-l", type=float, default=3.0)
    parser.add_argument("--voigt-gamma-g", type=float, default=2.0)
    parser.add_argument("--voigt-eta", type=float, default=0.5)
    parser.add_argument("--voigt-kernel-half-width", type=int, default=20)
    parser.add_argument("--compression-dim", type=int, default=32)
    parser.add_argument("--tn-site-length", type=int, default=16)
    parser.add_argument("--compressor-dropout", type=float, default=0.1)
    parser.add_argument("--quantum-qubits", type=int, default=4)
    parser.add_argument("--quantum-layers", type=int, default=3)
    parser.add_argument("--quantum-adapter-hidden-dim", type=int, default=32)
    parser.add_argument("--quantum-head-hidden-dim", type=int, default=32)
    parser.add_argument("--classical-head-hidden-dims", type=int, nargs="*", default=[64, 32])
    parser.add_argument("--head-dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--feature-batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--loss-type", choices=["bce", "focal"], default="bce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--pos-weight-power", type=float, default=0.5)
    parser.add_argument("--pos-weight-max", type=float, default=None)
    parser.add_argument("--threshold-mode", choices=["global", "per_class"], default="per_class")
    parser.add_argument(
        "--threshold-target-metric",
        choices=["f1_micro", "f1_macro", "per_class_f1"],
        default="per_class_f1",
    )
    parser.add_argument("--threshold-grid-step", type=float, default=0.02)
    parser.add_argument(
        "--early-stopping-metric",
        choices=["f1_micro", "f1_macro", "blended_f1"],
        default="blended_f1",
    )
    parser.add_argument("--early-stopping-blend-alpha", type=float, default=0.5)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--min-epochs-before-stopping", type=int, default=25)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--check-subset-size", type=int, default=2048)
    parser.add_argument("--check-only", action="store_true")
    return parser


def build_feature_extractor(args: argparse.Namespace) -> Experiment14FeatureExtractor:
    return Experiment14FeatureExtractor(
        feature_source=args.feature_source,
        include_raw_channel=args.include_raw_channel,
        sg_window_length=args.sg_window_length,
        sg_polyorder=args.sg_polyorder,
        voigt_gamma_l=args.voigt_gamma_l,
        voigt_gamma_g=args.voigt_gamma_g,
        voigt_eta=args.voigt_eta,
        voigt_kernel_half_width=args.voigt_kernel_half_width,
    )


def extract_sequence_features(
    extractor: Experiment14FeatureExtractor,
    x_np: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> Tensor:
    extractor = extractor.to(device)
    extractor.eval()
    outputs: list[Tensor] = []
    with torch.no_grad():
        for start in range(0, x_np.shape[0], batch_size):
            stop = min(start + batch_size, x_np.shape[0])
            batch = torch.as_tensor(x_np[start:stop], dtype=torch.float32, device=device)
            outputs.append(extractor(batch).cpu())
    return torch.cat(outputs, dim=0)


def build_feature_cache_key(args: argparse.Namespace) -> dict[str, object]:
    return {
        "spectra_cache": str(args.spectra_cache.resolve()),
        "feature_source": args.feature_source,
        "include_raw_channel": bool(args.include_raw_channel),
        "sg_window_length": int(args.sg_window_length),
        "sg_polyorder": int(args.sg_polyorder),
        "voigt_gamma_l": float(args.voigt_gamma_l),
        "voigt_gamma_g": float(args.voigt_gamma_g),
        "voigt_eta": float(args.voigt_eta),
        "voigt_kernel_half_width": int(args.voigt_kernel_half_width),
    }


def resolve_feature_cache_path(args: argparse.Namespace) -> Path:
    if args.feature_cache_path is not None:
        return Path(args.feature_cache_path)

    key = json.dumps(build_feature_cache_key(args), sort_keys=True).encode("utf-8")
    digest = hashlib.sha1(key).hexdigest()[:12]
    return args.spectra_cache.parent / f"{args.spectra_cache.stem}_experiment14_features_{digest}.pt"


def load_or_extract_sequence_features(
    extractor: Experiment14FeatureExtractor,
    x_np: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    cache_path: Path | None,
    overwrite_cache: bool,
    cache_key: dict[str, object],
) -> Tensor:
    expected_metadata = {
        "cache_key": cache_key,
        "x_shape": tuple(int(v) for v in x_np.shape),
    }

    if cache_path is not None and cache_path.exists() and not overwrite_cache:
        payload = torch.load(cache_path, map_location="cpu")
        if payload.get("metadata") == expected_metadata:
            sequence_tensor = payload["sequence_tensor"].to(torch.float32).cpu()
            print(f"Loaded feature cache: {cache_path}")
            return sequence_tensor
        print(f"Ignoring stale feature cache: {cache_path}")

    sequence_tensor = extract_sequence_features(extractor, x_np, device, batch_size)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"metadata": expected_metadata, "sequence_tensor": sequence_tensor.cpu()}, cache_path)
        print(f"Saved feature cache: {cache_path}")
    return sequence_tensor


def make_check_only_subset(
    x_np: np.ndarray,
    y_np: np.ndarray,
    *,
    subset_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if subset_size <= 0:
        raise ValueError("check-subset-size must be positive.")

    num_samples = int(x_np.shape[0])
    if subset_size >= num_samples:
        return x_np, y_np

    rng = np.random.default_rng(seed)
    subset_indices = np.sort(rng.choice(num_samples, size=int(subset_size), replace=False))
    return x_np[subset_indices], y_np[subset_indices]


def make_check_only_split_indices(
    num_samples: int,
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, np.ndarray]:
    if num_samples < 3:
        raise ValueError("check-only needs at least 3 samples to form train/val/test splits.")

    rng = np.random.default_rng(seed)
    indices = np.arange(num_samples, dtype=np.int64)
    rng.shuffle(indices)

    test_count = max(1, int(round(num_samples * test_ratio)))
    val_count = max(1, int(round(num_samples * val_ratio)))
    train_count = num_samples - test_count - val_count
    if train_count <= 0:
        train_count = 1
        remaining = num_samples - train_count
        val_count = max(1, remaining // 2)
        test_count = remaining - val_count
    if test_count <= 0:
        test_count = 1
        train_count = max(1, train_count - 1)

    train_end = train_count
    val_end = train_end + val_count
    return {
        "train": np.sort(indices[:train_end]),
        "val": np.sort(indices[train_end:val_end]),
        "test": np.sort(indices[val_end:]),
    }


def _train_indices_digest(train_indices: np.ndarray) -> str:
    payload = np.asarray(train_indices, dtype=np.int64).tobytes()
    return hashlib.sha1(payload).hexdigest()


def fit_pca_compressor(
    train_sequence: Tensor,
    compression_dim: int,
    seed: int,
) -> tuple[PCACompressor, dict[str, float]]:
    flat_train = train_sequence.flatten(start_dim=1).numpy()
    pca = PCA(n_components=compression_dim, random_state=seed)
    pca.fit(flat_train)
    info = {
        "n_components": float(pca.n_components_),
        "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
    }
    return PCACompressor.from_sklearn(pca), info


def load_or_fit_pca_compressor(
    train_sequence: Tensor,
    compression_dim: int,
    seed: int,
    *,
    train_indices: np.ndarray,
    artifact_path: Path | None,
) -> tuple[PCACompressor, dict[str, float]]:
    train_signature = _train_indices_digest(train_indices)
    expected_metadata = {
        "compression_dim": int(compression_dim),
        "seed": int(seed),
        "train_indices_sha1": train_signature,
        "train_shape": tuple(int(v) for v in train_sequence.shape),
    }

    if artifact_path is not None and artifact_path.exists():
        with artifact_path.open("rb") as handle:
            payload = pickle.load(handle)
        if payload.get("metadata") == expected_metadata:
            compressor = PCACompressor(
                mean=torch.from_numpy(payload["mean"]),
                components=torch.from_numpy(payload["components"]),
            )
            print(f"Loaded PCA artifact: {artifact_path}")
            return compressor, payload["info"]
        print(f"Ignoring stale PCA artifact: {artifact_path}")

    compressor, info = fit_pca_compressor(train_sequence, compression_dim, seed)
    if artifact_path is not None:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        with artifact_path.open("wb") as handle:
            pickle.dump(
                {
                    "metadata": expected_metadata,
                    "mean": compressor.mean.cpu().numpy(),
                    "components": compressor.components.cpu().numpy(),
                    "info": info,
                },
                handle,
            )
        print(f"Saved PCA artifact: {artifact_path}")
    return compressor, info


def build_model(
    args: argparse.Namespace,
    sequence_tensor: Tensor,
    train_indices: np.ndarray,
    *,
    pca_artifact_path: Path | None = None,
) -> tuple[Experiment14Classifier, dict[str, float] | None]:
    sequence_length = int(sequence_tensor.shape[1])
    feature_dim = int(sequence_tensor.shape[2])
    compression_dim = int(args.compression_dim)
    variant = args.variant
    pca_info: dict[str, float] | None = None

    if variant == "raw_quantum":
        compressor = RawSubsampleCompressor(output_dim=compression_dim)
        head = SharedQuantumHead(
            input_dim=compression_dim,
            num_labels=len(FUNCTIONAL_GROUPS),
            n_qubits=args.quantum_qubits,
            n_layers=args.quantum_layers,
            adapter_hidden_dim=args.quantum_adapter_hidden_dim,
            head_hidden_dim=args.quantum_head_hidden_dim,
            dropout=args.head_dropout,
        )
    elif variant == "pca_quantum":
        compressor, pca_info = load_or_fit_pca_compressor(
            sequence_tensor[train_indices],
            compression_dim,
            args.seed,
            train_indices=train_indices,
            artifact_path=pca_artifact_path,
        )
        head = SharedQuantumHead(
            input_dim=compression_dim,
            num_labels=len(FUNCTIONAL_GROUPS),
            n_qubits=args.quantum_qubits,
            n_layers=args.quantum_layers,
            adapter_hidden_dim=args.quantum_adapter_hidden_dim,
            head_hidden_dim=args.quantum_head_hidden_dim,
            dropout=args.head_dropout,
        )
    elif variant == "tn_quantum":
        compressor = TNSequenceCompressor(
            sequence_length=sequence_length,
            feature_dim=feature_dim,
            output_dim=compression_dim,
            site_length=args.tn_site_length,
            dropout=args.compressor_dropout,
        )
        head = SharedQuantumHead(
            input_dim=compression_dim,
            num_labels=len(FUNCTIONAL_GROUPS),
            n_qubits=args.quantum_qubits,
            n_layers=args.quantum_layers,
            adapter_hidden_dim=args.quantum_adapter_hidden_dim,
            head_hidden_dim=args.quantum_head_hidden_dim,
            dropout=args.head_dropout,
        )
    elif variant == "tn_classical":
        compressor = TNSequenceCompressor(
            sequence_length=sequence_length,
            feature_dim=feature_dim,
            output_dim=compression_dim,
            site_length=args.tn_site_length,
            dropout=args.compressor_dropout,
        )
        head = SharedClassicalHead(
            input_dim=compression_dim,
            num_labels=len(FUNCTIONAL_GROUPS),
            hidden_dims=tuple(args.classical_head_hidden_dims),
            dropout=args.head_dropout,
        )
    else:
        raise ValueError(f"Unsupported variant: {variant}")

    return Experiment14Classifier(compressor=compressor, head=head), pca_info


def make_loader(
    sequence_tensor: Tensor,
    labels: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    dataset = TensorDataset(
        sequence_tensor[indices],
        torch.as_tensor(labels[indices], dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: AdamW,
    criterion: nn.Module,
    device: torch.device,
    grad_clip_norm: float | None,
) -> tuple[float, dict[str, float | np.ndarray]]:
    model.train()
    total_loss = 0.0
    probs_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        if grad_clip_norm is not None and grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()

        total_loss += loss.item() * features.size(0)
        probs_list.append(torch.sigmoid(logits).detach().cpu().numpy())
        labels_list.append(labels.detach().cpu().numpy())

    probs = np.concatenate(probs_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return total_loss / len(loader.dataset), compute_metrics(labels, threshold_predictions(probs, 0.5))


@torch.no_grad()
def evaluate_with_probs(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    probs_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        logits = model(features)
        total_loss += criterion(logits, labels).item() * features.size(0)
        probs_list.append(torch.sigmoid(logits).cpu().numpy())
        labels_list.append(labels.cpu().numpy())

    probs = np.concatenate(probs_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return total_loss / len(loader.dataset), labels, probs


def write_summary(
    summary_path: Path,
    *,
    args: argparse.Namespace,
    best_epoch: int,
    best_score: float,
    completed_epochs: int,
    elapsed_seconds: float,
    parameter_count: int,
    test_loss: float,
    test_metrics: dict[str, float | np.ndarray],
    pca_info: dict[str, float] | None,
) -> None:
    lines = [
        "Experiment14 Summary",
        "=" * 80,
        f"Variant:                   {VARIANT_LABELS[args.variant]}",
        f"Feature source:            {args.feature_source}",
        f"Compression dim:           {args.compression_dim}",
        f"Quantum qubits:            {args.quantum_qubits}",
        f"Quantum layers:            {args.quantum_layers}",
        f"Trainable parameters:      {parameter_count}",
        f"Completed epochs:          {completed_epochs}/{args.epochs}",
        f"Elapsed seconds:           {elapsed_seconds:.2f}",
        f"Best epoch:                {best_epoch}",
        f"Best early-stop score:     {best_score:.6f}",
        f"Test loss:                 {test_loss:.6f}",
        f"Test f1_micro:             {float(test_metrics['f1_micro']):.6f}",
        f"Test f1_macro:             {float(test_metrics['f1_macro']):.6f}",
    ]
    if pca_info is not None:
        lines.append(f"PCA explained variance:    {pca_info['explained_variance_ratio_sum']:.6f}")
    lines.append("")
    lines.append("Per-class F1:")
    for name, value in zip(FUNCTIONAL_GROUPS, np.asarray(test_metrics["per_class_f1"]).tolist(), strict=False):
        lines.append(f"  {name:20s}: {float(value):.6f}")
    summary_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = build_parser().parse_args()

    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_path = args.split_path or (args.output_dir / f"data_split_seed{args.seed}_all.npz")
    checkpoint_path = args.output_dir / "experiment14_best.pt"
    config_path = args.output_dir / "run_config.json"
    log_path = args.output_dir / "training_log.csv"
    details_path = args.output_dir / "training_details.jsonl"
    summary_path = args.output_dir / "summary.txt"
    threshold_path = args.output_dir / "selected_thresholds.json"
    pca_artifact_path = args.output_dir / "pca_artifact.pkl"
    feature_cache_path = resolve_feature_cache_path(args)

    config_path.write_text(json.dumps(vars(args), indent=2, default=str) + "\n")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    if args.variant in {"raw_quantum", "pca_quantum", "tn_quantum"} and device.type != "cpu":
        print("Quantum head uses PennyLane default.qubit; falling back to CPU.")
        device = torch.device("cpu")
    print(f"Variant: {VARIANT_LABELS[args.variant]}")
    print(f"Device: {device}")

    cache = np.load(args.spectra_cache)
    x_np = cache["X"].astype(np.float32)
    y_np = cache["y"].astype(np.int32)
    print(f"Loaded cache X={x_np.shape}, y={y_np.shape}")

    if args.check_only:
        x_np, y_np = make_check_only_subset(
            x_np,
            y_np,
            subset_size=max(args.check_subset_size, args.batch_size, args.compression_dim + 8),
            seed=args.seed,
        )
        print(f"Check-only subset: X={x_np.shape}, y={y_np.shape}")
        split_indices = make_check_only_split_indices(
            x_np.shape[0],
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )
    else:
        split_indices = load_or_create_split_indices(
            labels=y_np,
            split_path=split_path,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            random_seed=args.seed,
            stratify_multilabel=True,
            overwrite=args.overwrite_split,
        )

    sequence_extractor = build_feature_extractor(args)
    sequence_tensor = load_or_extract_sequence_features(
        sequence_extractor,
        x_np,
        device=device,
        batch_size=args.feature_batch_size,
        cache_path=None if args.check_only else feature_cache_path,
        overwrite_cache=args.overwrite_feature_cache,
        cache_key=build_feature_cache_key(args),
    )
    print(f"Extracted sequence tensor: {tuple(sequence_tensor.shape)}")

    model, pca_info = build_model(
        args,
        sequence_tensor,
        split_indices["train"],
        pca_artifact_path=None if args.check_only else pca_artifact_path,
    )

    train_loader = make_loader(
        sequence_tensor,
        y_np,
        split_indices["train"],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = make_loader(
        sequence_tensor,
        y_np,
        split_indices["val"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = make_loader(
        sequence_tensor,
        y_np,
        split_indices["test"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = model.to(device)
    parameter_count = count_trainable_parameters(model)
    print(f"Trainable parameters: {parameter_count:,}")

    if args.check_only:
        xb, yb = next(iter(train_loader))
        logits = model(xb.to(device))
        print(f"Check-only OK: X={tuple(xb.shape)}, y={tuple(yb.shape)}, logits={tuple(logits.shape)}")
        return

    pos_weight = get_pos_weight(
        y_np[split_indices["train"]],
        device,
        power=args.pos_weight_power,
        max_value=args.pos_weight_max,
    )
    criterion = build_loss(args.loss_type, pos_weight=pos_weight, focal_gamma=args.focal_gamma)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.9, patience=5, min_lr=1e-6)
    threshold_grid = build_threshold_grid(args.threshold_grid_step)
    early_stopping = EarlyStopping(
        patience=args.early_stopping_patience,
        min_delta=args.early_stopping_min_delta,
    )

    best_state = None
    best_thresholds = None
    best_score = float("-inf")
    best_epoch = 0
    completed_epochs = 0
    start_time = time.time()

    with log_path.open("w", newline="") as csv_file, details_path.open("w") as details_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            ["epoch", "train_loss", "train_f1_macro", "val_loss", "val_f1_micro", "val_f1_macro", "score", "lr"]
        )

        for epoch in range(1, args.epochs + 1):
            train_loss, train_metrics = train_epoch(
                model, train_loader, optimizer, criterion, device, args.grad_clip_norm
            )
            val_loss, val_labels, val_probs = evaluate_with_probs(model, val_loader, criterion, device)
            thresholds = tune_thresholds(
                val_labels,
                val_probs,
                args.threshold_mode,
                args.threshold_target_metric,
                threshold_grid,
            )
            val_metrics = compute_metrics(val_labels, threshold_predictions(val_probs, thresholds))
            score = select_early_stopping_score(
                val_metrics, args.early_stopping_metric, args.early_stopping_blend_alpha
            )
            scheduler.step(score)

            writer.writerow(
                [
                    epoch,
                    float(train_loss),
                    float(train_metrics["f1_macro"]),
                    float(val_loss),
                    float(val_metrics["f1_micro"]),
                    float(val_metrics["f1_macro"]),
                    float(score),
                    float(optimizer.param_groups[0]["lr"]),
                ]
            )
            csv_file.flush()
            write_epoch_details(
                details_file,
                epoch=epoch,
                label_names=list(FUNCTIONAL_GROUPS.keys()),
                thresholds=thresholds,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                early_stopping_score=score,
            )

            print(
                f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"val_f1_micro={val_metrics['f1_micro']:.4f} | val_f1_macro={val_metrics['f1_macro']:.4f}"
            )

            if score > best_score:
                best_score = float(score)
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                best_thresholds = thresholds.copy()
                torch.save(best_state, checkpoint_path)

            completed_epochs = epoch
            if epoch >= args.min_epochs_before_stopping and early_stopping(score):
                break

    model.load_state_dict(best_state)
    model.to(device)
    test_loss, test_labels, test_probs = evaluate_with_probs(model, test_loader, criterion, device)
    test_metrics = compute_metrics(test_labels, threshold_predictions(test_probs, best_thresholds))

    write_threshold_artifact(
        threshold_path,
        label_names=list(FUNCTIONAL_GROUPS.keys()),
        thresholds=best_thresholds,
        threshold_mode=args.threshold_mode,
        threshold_target_metric=args.threshold_target_metric,
        best_epoch=best_epoch,
    )
    write_summary(
        summary_path,
        args=args,
        best_epoch=best_epoch,
        best_score=best_score,
        completed_epochs=completed_epochs,
        elapsed_seconds=time.time() - start_time,
        parameter_count=parameter_count,
        test_loss=test_loss,
        test_metrics=test_metrics,
        pca_info=pca_info,
    )

    print("\nTest metrics")
    print(f"  f1_micro:  {float(test_metrics['f1_micro']):.4f}")
    print(f"  f1_macro:  {float(test_metrics['f1_macro']):.4f}")


if __name__ == "__main__":
    main()
