"""Training entry point for experiment 12 compression baselines."""

from __future__ import annotations

import argparse
import csv
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
    load_ir_data,
    load_or_create_split_indices,
)
from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment5.losses import build_loss  # noqa: E402
from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment5.train import (  # noqa: E402
    EarlyStopping,
    build_threshold_grid,
    compute_metrics,
    count_available_data_files,
    get_pos_weight,
    resolve_device,
    resolve_used_file_count,
    score_threshold_metrics,
    select_early_stopping_score,
    threshold_predictions,
    tune_thresholds,
    write_epoch_details,
    write_threshold_artifact,
)
from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment12.model import (  # noqa: E402
    Experiment12FeatureExtractor,
    MLPClassifier,
    TNCompressionClassifier,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train experiment12: compare flat MLP, PCA->MLP, and TN-compression->MLP on engineered IR features."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment12/results"),
    )
    parser.add_argument("--split-path", type=Path, default=None)
    parser.add_argument("--overwrite-split", action="store_true")
    parser.add_argument("--input-dim", type=int, default=1800)
    parser.add_argument("--num-labels", type=int, default=len(FUNCTIONAL_GROUPS))
    parser.add_argument("--apply-snv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument(
        "--feature-source",
        choices=["sg_no_norm", "sg_z_score", "voigt_z_score"],
        default="sg_no_norm",
    )
    parser.add_argument("--include-raw-channel", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sg-window-length", type=int, default=11)
    parser.add_argument("--sg-polyorder", type=int, default=3)
    parser.add_argument("--voigt-gamma-l", type=float, default=3.0)
    parser.add_argument("--voigt-gamma-g", type=float, default=2.0)
    parser.add_argument("--voigt-eta", type=float, default=0.5)
    parser.add_argument("--voigt-kernel-half-width", type=int, default=20)
    parser.add_argument("--model-type", choices=["mlp", "pca_mlp", "tn"], default="tn")
    parser.add_argument("--batch-size", type=int, default=256)
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
    parser.add_argument("--hidden-dims", type=int, nargs="*", default=[512, 256])
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--pca-components", type=int, default=256)
    parser.add_argument("--tn-bond-dim", type=int, default=128)
    parser.add_argument("--tn-site-length", type=int, default=16)
    parser.add_argument("--tn-head-hidden-dims", type=int, nargs="*", default=[256])
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--feature-batch-size", type=int, default=512)
    return parser


def resolve_cache_path(args: argparse.Namespace) -> Path | None:
    if args.cache_path is not None:
        return args.cache_path
    sample_scope = "all" if args.max_files is None else f"files{max(int(args.max_files), 0)}"
    cache_name = (
        f"ir_data_len{args.input_dim}_{'snv' if args.apply_snv else 'raw'}_{sample_scope}.npz"
    )
    return args.output_dir / cache_name


def describe_args(args: argparse.Namespace, split_path: Path) -> None:
    print("=" * 80)
    print("Experiment12 Compression Baselines")
    print("=" * 80)
    print(f"Data dir:                 {args.data_dir}")
    print(f"Output dir:               {args.output_dir}")
    print(f"Split path:               {split_path}")
    print(f"Input dim:                {args.input_dim}")
    print(f"Num labels:               {args.num_labels}")
    print(f"Apply SNV:                {args.apply_snv}")
    print(f"Feature source:           {args.feature_source}")
    print(f"Include raw channel:      {args.include_raw_channel}")
    print(f"Model type:               {args.model_type}")
    print(f"Batch size:               {args.batch_size}")
    print(f"Epochs:                   {args.epochs}")
    print(f"Hidden dims:              {args.hidden_dims}")
    print(f"Dropout:                  {args.dropout}")
    if args.model_type == "pca_mlp":
        print(f"PCA components:           {args.pca_components}")
    if args.model_type == "tn":
        print(f"TN bond dim:              {args.tn_bond_dim}")
        print(f"TN site length:           {args.tn_site_length}")
        print(f"TN head dims:             {args.tn_head_hidden_dims}")


def build_feature_extractor(args: argparse.Namespace) -> Experiment12FeatureExtractor:
    return Experiment12FeatureExtractor(
        feature_source=args.feature_source,
        include_raw_channel=args.include_raw_channel,
        sg_window_length=args.sg_window_length,
        sg_polyorder=args.sg_polyorder,
        voigt_gamma_l=args.voigt_gamma_l,
        voigt_gamma_g=args.voigt_gamma_g,
        voigt_eta=args.voigt_eta,
        voigt_kernel_half_width=args.voigt_kernel_half_width,
    )


def extract_feature_sequence(
    extractor: Experiment12FeatureExtractor,
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


def build_model(args: argparse.Namespace, feature_tensor: Tensor) -> nn.Module:
    num_labels = args.num_labels
    if args.model_type in {"mlp", "pca_mlp"}:
        input_dim = int(feature_tensor.shape[1])
        return MLPClassifier(
            input_dim=input_dim,
            num_labels=num_labels,
            hidden_dims=tuple(args.hidden_dims),
            dropout=args.dropout,
        )
    if args.model_type == "tn":
        return TNCompressionClassifier(
            sequence_length=int(feature_tensor.shape[1]),
            feature_dim=int(feature_tensor.shape[2]),
            num_labels=num_labels,
            bond_dim=args.tn_bond_dim,
            site_length=args.tn_site_length,
            head_hidden_dims=tuple(args.tn_head_hidden_dims),
            dropout=args.dropout,
        )
    raise ValueError(f"Unsupported model_type: {args.model_type}")


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
    metrics = compute_metrics(labels, threshold_predictions(probs, 0.5))
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, metrics


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
        loss = criterion(logits, labels)
        total_loss += loss.item() * features.size(0)
        probs_list.append(torch.sigmoid(logits).cpu().numpy())
        labels_list.append(labels.cpu().numpy())

    probs = np.concatenate(probs_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, labels, probs


def make_loader(
    features: Tensor,
    labels: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    dataset = TensorDataset(
        features[indices],
        torch.as_tensor(labels[indices], dtype=torch.float32),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def write_summary(
    summary_path: Path,
    *,
    args: argparse.Namespace,
    split_path: Path,
    used_data_files: int,
    total_data_files: int,
    completed_epochs: int,
    elapsed_seconds: float,
    best_epoch: int,
    best_score: float,
    best_val_loss: float,
    test_loss: float,
    test_metrics: dict[str, float | np.ndarray],
    pca_info: dict[str, float] | None = None,
) -> None:
    lines = [
        "Experiment12 Summary",
        "=" * 80,
        f"Model type:                {args.model_type}",
        f"Feature source:            {args.feature_source}",
        f"Include raw channel:       {args.include_raw_channel}",
        f"Apply SNV:                 {args.apply_snv}",
        f"Data files used:           {used_data_files}/{total_data_files}",
        f"Split path:                {split_path}",
        f"Completed epochs:          {completed_epochs}/{args.epochs}",
        f"Elapsed seconds:           {elapsed_seconds:.2f}",
        f"Best epoch:                {best_epoch}",
        f"Best early-stop score:     {best_score:.6f}",
        f"Best val loss:             {best_val_loss:.6f}",
        f"Test loss:                 {test_loss:.6f}",
        f"Test f1_micro:             {float(test_metrics['f1_micro']):.6f}",
        f"Test f1_macro:             {float(test_metrics['f1_macro']):.6f}",
        f"Test precision_micro:      {float(test_metrics['precision_micro']):.6f}",
        f"Test recall_micro:         {float(test_metrics['recall_micro']):.6f}",
    ]
    if pca_info is not None:
        lines.extend(
            [
                f"PCA components:            {int(pca_info['n_components'])}",
                f"PCA explained variance:    {pca_info['explained_variance_ratio_sum']:.6f}",
            ]
        )
    lines.append("")
    lines.append("Per-class F1:")
    for name, value in zip(FUNCTIONAL_GROUPS, np.asarray(test_metrics["per_class_f1"]).tolist(), strict=False):
        lines.append(f"  {name:20s}: {float(value):.6f}")
    summary_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0.")
    if args.threshold_mode == "per_class" and args.threshold_target_metric == "f1_micro":
        raise ValueError("per_class thresholds are incompatible with f1_micro target.")
    if not args.data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    total_data_files = count_available_data_files(args.data_dir)
    used_data_files = resolve_used_file_count(total_data_files, args.max_files)
    split_suffix = "all" if args.max_files is None else f"files{used_data_files}"
    split_path = args.split_path or (args.output_dir / f"data_split_seed{args.seed}_{split_suffix}.npz")
    describe_args(args, split_path)

    config_path = args.output_dir / "run_config.json"
    log_path = args.output_dir / "training_log.csv"
    details_path = args.output_dir / "training_details.jsonl"
    checkpoint_path = args.output_dir / "exp12_best.pt"
    threshold_path = args.output_dir / "selected_thresholds.json"
    summary_path = args.output_dir / "summary.txt"
    pca_path = args.output_dir / "pca_model.pkl"
    config_path.write_text(json.dumps(vars(args), indent=2, default=str) + "\n")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    print(f"Device: {device}")

    X, y = load_ir_data(
        data_dir=args.data_dir,
        target_length=args.input_dim,
        max_files=args.max_files,
        apply_snv=args.apply_snv,
        cache_path=resolve_cache_path(args),
        overwrite_cache=args.overwrite_cache,
    )
    if y.shape[1] != args.num_labels:
        raise RuntimeError(f"Loaded labels have width {y.shape[1]}, expected {args.num_labels}.")

    split_indices = load_or_create_split_indices(
        labels=y,
        split_path=split_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
        stratify_multilabel=True,
        overwrite=args.overwrite_split,
    )

    extractor = build_feature_extractor(args)
    print("Extracting engineered features...")
    feature_sequence = extract_feature_sequence(
        extractor=extractor,
        x_np=X,
        device=device,
        batch_size=args.feature_batch_size,
    )
    print(f"Feature tensor: {tuple(feature_sequence.shape)}")

    pca_info: dict[str, float] | None = None
    if args.model_type == "tn":
        feature_tensor = feature_sequence
    else:
        flat_features = feature_sequence.flatten(start_dim=1)
        if args.model_type == "pca_mlp":
            n_train = len(split_indices["train"])
            max_components = min(args.pca_components, n_train, flat_features.shape[1])
            if max_components <= 0:
                raise ValueError("pca_components must be positive after clipping.")
            pca = PCA(n_components=max_components, svd_solver="randomized", random_state=args.seed)
            pca.fit(flat_features[split_indices["train"]].numpy())
            transformed = pca.transform(flat_features.numpy()).astype(np.float32, copy=False)
            feature_tensor = torch.from_numpy(transformed)
            pca_info = {
                "n_components": float(max_components),
                "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
            }
            with pca_path.open("wb") as handle:
                pickle.dump(pca, handle)
        else:
            feature_tensor = flat_features

    train_loader = make_loader(
        feature_tensor, y, split_indices["train"],
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
    )
    val_loader = make_loader(
        feature_tensor, y, split_indices["val"],
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
    )
    test_loader = make_loader(
        feature_tensor, y, split_indices["test"],
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
    )

    train_labels = y[split_indices["train"]]
    pos_weight = get_pos_weight(
        train_labels,
        device,
        power=args.pos_weight_power,
        max_value=args.pos_weight_max,
    )
    criterion = build_loss(
        loss_name=args.loss_type,
        pos_weight=pos_weight,
        focal_gamma=args.focal_gamma,
    )
    model = build_model(args, feature_tensor).to(device)

    if args.check_only:
        batch_features, batch_labels = next(iter(train_loader))
        with torch.no_grad():
            logits = model(batch_features.to(device))
        print(
            "Check-only OK: "
            f"batch_features={tuple(batch_features.shape)}, "
            f"batch_labels={tuple(batch_labels.shape)}, logits={tuple(logits.shape)}"
        )
        return

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.9,
        patience=5,
        min_lr=1e-6,
    )
    threshold_grid = build_threshold_grid(args.threshold_grid_step)
    early_stopping = EarlyStopping(
        patience=args.early_stopping_patience,
        min_delta=args.early_stopping_min_delta,
    )

    best_epoch = 0
    best_val_loss = float("inf")
    best_score = float("-inf")
    best_state_dict: dict[str, Tensor] | None = None
    best_thresholds: np.ndarray | None = None
    completed_epochs = 0
    start_time = time.time()

    with log_path.open("w", newline="") as csv_file, details_path.open("w") as details_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_f1_micro",
                "train_f1_macro",
                "val_loss",
                "val_f1_micro",
                "val_f1_macro",
                "early_stopping_score",
                "selected_metric_score",
                "lr",
            ]
        )

        for epoch in range(1, args.epochs + 1):
            train_loss, train_metrics = train_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                grad_clip_norm=args.grad_clip_norm,
            )
            val_loss, val_labels, val_probs = evaluate_with_probs(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
            )

            thresholds = tune_thresholds(
                y_true=val_labels,
                y_probs=val_probs,
                threshold_mode=args.threshold_mode,
                target_metric=args.threshold_target_metric,
                threshold_grid=threshold_grid,
            )
            val_metrics = compute_metrics(val_labels, threshold_predictions(val_probs, thresholds))
            selected_metric_score = score_threshold_metrics(val_metrics, args.threshold_target_metric)
            early_score = select_early_stopping_score(
                val_metrics,
                args.early_stopping_metric,
                args.early_stopping_blend_alpha,
            )
            scheduler.step(early_score)

            writer.writerow(
                [
                    epoch,
                    float(train_loss),
                    float(train_metrics["f1_micro"]),
                    float(train_metrics["f1_macro"]),
                    float(val_loss),
                    float(val_metrics["f1_micro"]),
                    float(val_metrics["f1_macro"]),
                    float(early_score),
                    float(selected_metric_score),
                    float(optimizer.param_groups[0]["lr"]),
                ]
            )
            csv_file.flush()
            write_epoch_details(
                details_file,
                epoch=epoch,
                label_names=FUNCTIONAL_GROUPS,
                thresholds=thresholds,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                early_stopping_score=early_score,
            )

            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_f1_micro={val_metrics['f1_micro']:.4f} | "
                f"val_f1_macro={val_metrics['f1_macro']:.4f} | "
                f"score={early_score:.4f}"
            )

            if early_score > best_score:
                best_epoch = epoch
                best_score = float(early_score)
                best_val_loss = float(val_loss)
                best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_thresholds = thresholds.copy()
                torch.save(best_state_dict, checkpoint_path)

            completed_epochs = epoch
            if epoch >= args.min_epochs_before_stopping and early_stopping(early_score):
                break

    if best_state_dict is None or best_thresholds is None:
        raise RuntimeError("Training finished without a valid best checkpoint.")

    model.load_state_dict(best_state_dict)
    model.to(device)

    test_loss, test_labels, test_probs = evaluate_with_probs(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )
    test_metrics = compute_metrics(test_labels, threshold_predictions(test_probs, best_thresholds))

    write_threshold_artifact(
        threshold_path=threshold_path,
        label_names=FUNCTIONAL_GROUPS,
        thresholds=best_thresholds,
        threshold_mode=args.threshold_mode,
        threshold_target_metric=args.threshold_target_metric,
        best_epoch=best_epoch,
    )
    write_summary(
        summary_path,
        args=args,
        split_path=split_path,
        used_data_files=used_data_files,
        total_data_files=total_data_files,
        completed_epochs=completed_epochs,
        elapsed_seconds=time.time() - start_time,
        best_epoch=best_epoch,
        best_score=best_score,
        best_val_loss=best_val_loss,
        test_loss=test_loss,
        test_metrics=test_metrics,
        pca_info=pca_info,
    )

    print("\nTest metrics")
    print(f"  f1_micro:        {float(test_metrics['f1_micro']):.4f}")
    print(f"  f1_macro:        {float(test_metrics['f1_macro']):.4f}")
    print(f"  precision_micro: {float(test_metrics['precision_micro']):.4f}")
    print(f"  recall_micro:    {float(test_metrics['recall_micro']):.4f}")


if __name__ == "__main__":
    main()
