"""Train the Shpetim IR specialist quantum model on the 10 weakest TTN-10.2 classes."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from spectroscopy_qml.ir.mps_encoder.data_loader import load_ir_data, prepare_dataloaders
from spectroscopy_qml.ir.qml.diagnostics import run_diagnostics
from spectroscopy_qml.ir.qml.quantum_model_shpetim import (
    AngleEncodingClassifier,
    N_SPECIALIST_CLASSES,
    TTN10_2_HARD_CLASS_INDICES,
    TTN10_2_HARD_CLASS_NAMES,
    select_specialist_labels,
)


class FocalBCELoss(nn.Module):
    """Focal BCE for multi-label specialist training."""

    def __init__(
        self,
        pos_weight: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal_weight = alpha_t * (1.0 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the Shpetim specialist quantum model on the 10 weakest TTN-10.2 classes."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/spectroscopy_qml/ir/qml/results/shpetim_specialist"),
    )
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--target-len", type=int, default=1800)
    parser.add_argument("--apply-snv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pos-weight-max", type=float, default=50.0)
    parser.add_argument("--use-focal-loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--check-only", action="store_true")
    return parser


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_pos_weight(
    y_train: np.ndarray,
    device: torch.device,
    clamp_max: float,
) -> torch.Tensor:
    n = y_train.shape[0]
    pos = y_train.sum(axis=0).clip(min=1)
    neg = n - pos
    weight = (neg / pos).clip(max=clamp_max)
    return torch.as_tensor(weight, dtype=torch.float32, device=device)


def tune_thresholds_per_class(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    grid: np.ndarray | None = None,
) -> np.ndarray:
    if grid is None:
        grid = np.arange(0.05, 0.96, 0.05)

    thresholds = np.full(y_true.shape[1], 0.5, dtype=np.float32)
    for class_idx in range(y_true.shape[1]):
        best_threshold = 0.5
        best_f1 = -1.0
        for threshold in grid:
            preds = (y_probs[:, class_idx] >= threshold).astype(np.int32)
            f1 = f1_score(y_true[:, class_idx], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(threshold)
        thresholds[class_idx] = best_threshold
    return thresholds


def _prepare_flux(spectra: torch.Tensor, device: torch.device) -> torch.Tensor:
    return spectra.to(device).unsqueeze(1)


def train_epoch(
    model: AngleEncodingClassifier,
    loader,
    criterion: nn.Module,
    optimizer: Adam,
    device: torch.device,
    thresholds: np.ndarray,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for spectra, labels in loader:
        flux = _prepare_flux(spectra, device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(flux)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += float(loss.item()) * flux.size(0)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.append((probs >= thresholds).astype(np.int32))
        all_labels.append(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    train_f1 = f1_score(
        np.vstack(all_labels),
        np.vstack(all_preds),
        average="micro",
        zero_division=0,
    )
    return avg_loss, float(train_f1)


@torch.no_grad()
def evaluate(
    model: AngleEncodingClassifier,
    loader,
    criterion: nn.Module,
    device: torch.device,
    thresholds: np.ndarray,
    *,
    return_probs: bool = False,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for spectra, labels in loader:
        flux = _prepare_flux(spectra, device)
        labels = labels.to(device)
        logits = model(flux)
        loss = criterion(logits, labels)
        total_loss += float(loss.item()) * flux.size(0)
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    probs = np.vstack(all_probs)
    labels = np.vstack(all_labels)
    preds = (probs >= thresholds).astype(np.int32)
    f1_micro = f1_score(labels, preds, average="micro", zero_division=0)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    if return_probs:
        return avg_loss, float(f1_micro), float(f1_macro), labels, probs
    return avg_loss, float(f1_micro), float(f1_macro), labels, preds


def main() -> None:
    args = build_parser().parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = resolve_device(args.device)
    output_dir = args.output_dir
    model_dir = output_dir / "models"
    results_dir = output_dir / "artifacts"
    best_model_path = model_dir / "shpetim_specialist_best.pt"
    log_csv = results_dir / "training_log.csv"
    summary_txt = results_dir / "summary.txt"
    thresholds_path = results_dir / "selected_thresholds.json"
    run_config_path = results_dir / "run_config.json"
    specialist_map_path = results_dir / "specialist_map.json"

    print("=" * 72)
    print("Shpetim specialist quantum model")
    print("=" * 72)
    print(f"Device: {device}")
    print(f"Specialist indices: {TTN10_2_HARD_CLASS_INDICES}")
    print(f"Specialist names:   {', '.join(TTN10_2_HARD_CLASS_NAMES)}")

    X, y_full = load_ir_data(
        args.data_dir,
        target_length=args.target_len,
        max_files=args.max_files,
        apply_snv=args.apply_snv,
        cache_path=args.cache_path,
        overwrite_cache=args.overwrite_cache,
    )
    y_specialist = select_specialist_labels(torch.as_tensor(y_full, dtype=torch.float32)).numpy()

    print(f"Dataset: {X.shape[0]:,} samples | spectra shape={X.shape} | labels shape={y_specialist.shape}")

    train_loader, val_loader, test_loader = prepare_dataloaders(
        X,
        y_specialist,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = AngleEncodingClassifier(
        num_classes=N_SPECIALIST_CLASSES,
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        n_scalars=6,
        dropout=args.dropout,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    if args.check_only:
        spectra, labels = next(iter(train_loader))
        flux = _prepare_flux(spectra, device)
        model.eval()
        with torch.no_grad():
            logits = model(flux)
        print(f"Check-only OK: flux={tuple(flux.shape)} labels={tuple(labels.shape)} logits={tuple(logits.shape)}")
        return

    train_labels = np.vstack([batch_labels.numpy() for _, batch_labels in train_loader])
    pos_weight = compute_pos_weight(train_labels, device, args.pos_weight_max)
    print(
        "pos_weight "
        f"min={pos_weight.min().item():.1f} "
        f"max={pos_weight.max().item():.1f} "
        f"mean={pos_weight.mean().item():.1f} "
        f"(clamped at {args.pos_weight_max})"
    )

    if args.use_focal_loss:
        criterion: nn.Module = FocalBCELoss(
            pos_weight=pos_weight,
            alpha=args.focal_alpha,
            gamma=args.focal_gamma,
        )
        loss_name = f"Focal(alpha={args.focal_alpha}, gamma={args.focal_gamma})"
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss_name = "BCEWithLogitsLoss"

    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        **vars(args),
        "device_resolved": str(device),
        "specialist_indices": list(TTN10_2_HARD_CLASS_INDICES),
        "specialist_names": list(TTN10_2_HARD_CLASS_NAMES),
        "num_classes": N_SPECIALIST_CLASSES,
    }
    run_config_path.write_text(json.dumps(run_config, indent=2, default=str) + "\n")
    specialist_map_path.write_text(
        json.dumps(
            {
                "specialist_indices": list(TTN10_2_HARD_CLASS_INDICES),
                "specialist_names": list(TTN10_2_HARD_CLASS_NAMES),
            },
            indent=2,
        )
        + "\n"
    )

    with log_csv.open("w", newline="") as handle:
        csv.writer(handle).writerow(
            [
                "epoch",
                "train_loss",
                "train_f1_micro",
                "val_loss",
                "val_f1_micro",
                "val_f1_macro",
                "blended_f1",
                "mean_threshold",
                "lr",
            ]
        )

    thresholds = np.full(N_SPECIALIST_CLASSES, 0.5, dtype=np.float32)
    best_score = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    best_thresholds = thresholds.copy()
    no_improve = 0
    started_at = time.time()
    last_epoch = 0

    print(f"Training epochs={args.epochs} batch={args.batch_size} patience={args.patience}")

    for epoch in range(1, args.epochs + 1):
        last_epoch = epoch
        epoch_started = time.time()

        train_loss, train_f1 = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            thresholds,
        )

        _, _, _, val_labels, val_probs = evaluate(
            model,
            val_loader,
            criterion,
            device,
            thresholds,
            return_probs=True,
        )
        thresholds = tune_thresholds_per_class(val_labels, val_probs)
        val_preds = (val_probs >= thresholds).astype(np.int32)
        val_f1_micro = f1_score(val_labels, val_preds, average="micro", zero_division=0)
        val_f1_macro = f1_score(val_labels, val_preds, average="macro", zero_division=0)
        blended = 0.5 * val_f1_micro + 0.5 * val_f1_macro

        val_loss, _, _, _, _ = evaluate(
            model,
            val_loader,
            criterion,
            device,
            thresholds,
        )
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_seconds = time.time() - epoch_started

        print(
            f"Ep {epoch:03d}/{args.epochs} "
            f"loss={train_loss:.4f} "
            f"tr_f1={train_f1:.4f} "
            f"val_mic={val_f1_micro:.4f} "
            f"val_mac={val_f1_macro:.4f} "
            f"blend={blended:.4f} "
            f"lr={current_lr:.1e} "
            f"({epoch_seconds:.1f}s)"
        )

        with log_csv.open("a", newline="") as handle:
            csv.writer(handle).writerow(
                [
                    epoch,
                    train_loss,
                    train_f1,
                    val_loss,
                    val_f1_micro,
                    val_f1_macro,
                    blended,
                    float(thresholds.mean()),
                    current_lr,
                ]
            )

        if blended > best_score:
            best_score = float(blended)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_thresholds = thresholds.copy()
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": best_state,
                    "thresholds": best_thresholds,
                    "val_f1_micro": float(val_f1_micro),
                    "val_f1_macro": float(val_f1_macro),
                    "blended_f1": float(blended),
                    "specialist_indices": list(TTN10_2_HARD_CLASS_INDICES),
                    "specialist_names": list(TTN10_2_HARD_CLASS_NAMES),
                },
                best_model_path,
            )
            print(f"  New best blended={best_score:.4f}")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping after {args.patience} epochs without improvement.")
                break

    total_seconds = time.time() - started_at
    if best_state is not None:
        model.load_state_dict(best_state)
        thresholds = best_thresholds

    test_loss, test_f1_micro, test_f1_macro, test_labels, test_preds = evaluate(
        model,
        test_loader,
        criterion,
        device,
        thresholds,
    )
    print(
        f"Test metrics: loss={test_loss:.4f} "
        f"f1_micro={test_f1_micro:.4f} "
        f"f1_macro={test_f1_macro:.4f}"
    )

    diagnostics = run_diagnostics(
        test_labels,
        test_preds,
        out_dir=results_dir,
        model_name="Shpetim specialist quantum model",
        class_names=TTN10_2_HARD_CLASS_NAMES,
        benchmark_f1=None,
    )

    thresholds_payload = {
        name: float(threshold)
        for name, threshold in zip(TTN10_2_HARD_CLASS_NAMES, thresholds, strict=False)
    }
    thresholds_path.write_text(json.dumps(thresholds_payload, indent=2) + "\n")

    with summary_txt.open("w") as handle:
        handle.write("Shpetim Specialist Quantum Training Summary\n")
        handle.write("=" * 46 + "\n")
        handle.write(f"samples        : {X.shape[0]}\n")
        handle.write(f"classes        : {N_SPECIALIST_CLASSES}\n")
        handle.write(f"class_indices  : {','.join(str(i) for i in TTN10_2_HARD_CLASS_INDICES)}\n")
        handle.write(f"class_names    : {', '.join(TTN10_2_HARD_CLASS_NAMES)}\n")
        handle.write(f"batch_size     : {args.batch_size}\n")
        handle.write(f"epochs_run     : {last_epoch}\n")
        handle.write(f"loss           : {loss_name}\n")
        handle.write(f"best_blended   : {best_score:.4f}\n")
        handle.write(f"test_f1_micro  : {test_f1_micro:.4f}\n")
        handle.write(f"test_f1_macro  : {test_f1_macro:.4f}\n")
        handle.write(f"test_loss      : {test_loss:.4f}\n")
        handle.write(f"training_time  : {total_seconds:.1f}s\n")
        handle.write("\nPer-class test F1\n")
        for row in diagnostics["per_class"]:
            handle.write(f"  {row['name']}: {row['f1']:.4f}\n")

    print(f"Summary -> {summary_txt}")
    print(f"Thresholds -> {thresholds_path}")
    print(f"Model -> {best_model_path}")


if __name__ == "__main__":
    main()
