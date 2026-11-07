"""Common neural trainer and synchronized 37-label XGBoost trainer.

Selection uses ONLY selection-partition micro F1 at .5. Calibration and test
arrays are never passed into the fitting functions. Checkpoints are atomic.
"""

from __future__ import annotations

import os
import random
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .raw_evaluation import EarlyStopping, evaluate, micro_f1
from .raw_native import RawCohort, atomic_json, file_sha256
from .raw_resources import measure_gpu_memory


def seed_everything(seed: int):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def atomic_checkpoint(path: Path, payload):
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".pt")
    os.close(fd)
    try:
        torch.save(payload, name)
        os.replace(name, path)
        atomic_json(path.with_suffix(".sha256.json"), {"sha256": file_sha256(path)})
    finally:
        Path(name).unlink(missing_ok=True)


def load_checkpoint(path: Path):
    import json

    expected = json.loads(path.with_suffix(".sha256.json").read_text())["sha256"]
    if file_sha256(path) != expected:
        raise ValueError(f"Checkpoint integrity failure: {path}")
    # Only locally generated, hash-checked checkpoints are supported.
    return torch.load(path, map_location="cpu", weights_only=False)


class RawRows(Dataset):
    def __init__(self, cohort: RawCohort, modality: str, indices):
        self.values = cohort.values(modality)
        self.masks = cohort.masks(modality)
        self.labels = cohort.labels
        self.indices = np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        from .raw_native import pack_native

        index = self.indices[position]
        x = pack_native(self.values[index], self.masks[index])
        return torch.from_numpy(x), torch.from_numpy(self.labels[index].astype(np.float32))


def batches(cohort, modality, indices, batch_size, seed=None):
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    return DataLoader(
        RawRows(cohort, modality, indices),
        batch_size=batch_size,
        shuffle=seed is not None,
        generator=generator,
        num_workers=0,
    )


@torch.no_grad()
def predict_neural(model, loader, device):
    model.eval()
    predictions, truths = [], []
    for x, y in loader:
        logits = model(x.to(device))
        if not torch.isfinite(logits).all():
            raise ValueError("Nonfinite model logits")
        predictions.append(logits.sigmoid().cpu().numpy())
        truths.append(y.numpy())
    return np.concatenate(truths), np.concatenate(predictions)


@measure_gpu_memory
def train_neural(
    model,
    cohort,
    modality,
    split,
    output,
    identity,
    *,
    seed,
    device,
    batch_size=64,
    epochs=100,
    patience=15,
    learning_rate=1e-3,
    weight_decay=1e-4,
    clip_norm=1.0,
    time_limit=None,
    resume=False,
):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    stopper = EarlyStopping(patience)
    criterion = nn.BCEWithLogitsLoss()
    selection_loader = batches(cohort, modality, split["selection"], batch_size)
    path = output / "last.pt"
    history, start_epoch, elapsed, best_state = [], 0, 0.0, None
    if resume and path.exists():
        state = load_checkpoint(path)
        if state["identity"] != identity:
            raise ValueError("Resume identity mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        stopper = EarlyStopping(**state["stopper"])
        history, start_epoch, elapsed = state["history"], state["epoch"] + 1, state["elapsed"]
        best_state = state["best_model"]
        torch.set_rng_state(state["torch_rng"])
        if str(device).startswith("cuda"):
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    sync(device)
    session_start = time.perf_counter()
    for epoch in range(start_epoch, epochs):
        if stopper.stopped or (time_limit is not None and elapsed >= time_limit):
            break
        model.train()
        total_loss, total_count, gradient_max = 0.0, 0, 0.0
        for x, y in batches(cohort, modality, split["train"], batch_size, seed + epoch):
            optimizer.zero_grad(set_to_none=True)
            logits = model(x.to(device))
            loss = criterion(logits, y.to(device))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss; no automatic rescaling or retry")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), clip_norm, error_if_nonfinite=True
            )
            gradient_max = max(gradient_max, float(norm))
            optimizer.step()
            total_loss += float(loss.detach()) * len(x)
            total_count += len(x)
        truth, probabilities = predict_neural(model, selection_loader, device)
        metrics = evaluate(truth, probabilities)
        improved = stopper.update(metrics["f1_micro"], epoch)
        if improved:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        scheduler.step()
        sync(device)
        now = time.perf_counter()
        elapsed += now - session_start
        history.append(
            {
                "epoch": epoch,
                "train_bce": total_loss / total_count,
                "selection_bce": metrics["bce"],
                "selection_f1_micro": metrics["f1_micro"],
                "gradient_norm_max_before_clip": gradient_max,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": elapsed,
            }
        )
        atomic_checkpoint(
            path,
            {
                "identity": identity,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "stopper": asdict(stopper),
                "epoch": epoch,
                "best_model": best_state,
                "history": history,
                "elapsed": elapsed,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all()
                if str(device).startswith("cuda")
                else [],
                "numpy_rng": np.random.get_state(),
                "python_rng": random.getstate(),
            },
        )
        # Include checkpoint I/O in the recorded resource budget.
        session_start = now
        print(
            f"{modality} {model.architecture} epoch={epoch + 1} selection_f1={metrics['f1_micro']:.5f}",
            flush=True,
        )
    if best_state is None:
        raise ValueError("No valid checkpoint was produced")
    model.load_state_dict(best_state)
    atomic_checkpoint(output / "best.pt", {"identity": identity, "model": best_state})
    atomic_json(output / "history.json", history)
    return {
        "selection_f1_micro": stopper.best,
        "best_step": stopper.best_step,
        "training_seconds": elapsed,
        "selection_checks": len(history),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device))
        if str(device).startswith("cuda")
        else None,
        "gpu_memory_scope": "PyTorch allocated peak, includes optimizer; CPU runs null",
        "stop_reason": "patience"
        if stopper.stopped
        else ("time_budget" if time_limit and elapsed >= time_limit else "epoch_limit"),
    }


def xgb_matrix(cohort, modality, indices, output: Path):
    """Disk-backed full flattened features. DMatrix still has its own RAM cost."""
    shape = cohort.manifest["modalities"][modality]["shape"]
    width = shape[1] * (shape[2] + 1)
    matrix = np.lib.format.open_memmap(
        output, mode="w+", dtype=np.float32, shape=(len(indices), width)
    )
    for start in range(0, len(indices), 64):
        rows = indices[start : start + 64]
        matrix[start : start + len(rows)] = cohort.features(modality, rows).reshape(len(rows), -1)
    matrix.flush()
    return matrix


def predict_xgb(models, matrix, constants):
    import xgboost as xgb

    dmatrix = xgb.DMatrix(matrix, nthread=1)
    return np.column_stack(
        [
            np.full(len(matrix), constants[j], dtype=np.float32)
            if model is None
            else model.predict(dmatrix)
            for j, model in enumerate(models)
        ]
    )


@measure_gpu_memory
def train_xgb(
    cohort,
    modality,
    split,
    output,
    identity,
    *,
    seed,
    device,
    config,
    patience=15,
    time_limit=None,
    resume=False,
    threads=4,
    memory_mode="auto",
    external_batch_size=4096,
    external_on_host=True,
):
    """One round per nonconstant label, then ONE joint micro-F1 selection check.

    Each label receives its own train/validation DMatrix target. No per-label
    test/validation feedback or log-loss early stopping changes the joint checkpoint.
    Resume restarts an interrupted trial deterministically (XGBoost RNG internals
    cannot be restored portably); completed trials are reused by the runner.
    """
    import xgboost as xgb

    start = time.perf_counter()
    shape = cohort.manifest["modalities"][modality]["shape"]
    estimated_bytes = (
        (len(split["train"]) + len(split["selection"])) * shape[1] * (shape[2] + 1) * 4
    )
    use_external = memory_mode == "external" or (
        memory_mode == "auto" and estimated_bytes > 1024**3
    )
    matrix_paths = []
    train_x = val_x = None
    if use_external:
        from .raw_xgb_memory import external_matrices

        dtrain, dval = external_matrices(
            cohort, modality, split, output, device, threads, external_batch_size, external_on_host
        )
    else:
        matrix_paths = [output / "train_features.npy", output / "selection_features.npy"]
        train_x = xgb_matrix(cohort, modality, split["train"], matrix_paths[0])
        val_x = xgb_matrix(cohort, modality, split["selection"], matrix_paths[1])
        dtrain, dval = xgb.DMatrix(train_x, nthread=threads), xgb.DMatrix(val_x, nthread=threads)
    train_y, val_y = cohort.labels[split["train"]], cohort.labels[split["selection"]]
    constants = [float(train_y[:, j].mean()) for j in range(train_y.shape[1])]
    params = {k: v for k, v in config.items() if k != "n_estimators"}
    params.update(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device=str(device),
        seed=seed,
        nthread=threads,
        verbosity=0,
    )
    stopper = EarlyStopping(patience)
    history, best_raw = [], None
    # Two matrices are reused by setting target per label, avoiding 37 copies of
    # native 10k-column data. The training loop updates all labels synchronously.
    models = [
        None
        if c in (0, 1)
        else xgb.Booster(params={**params, "base_score": c}, cache=[dtrain, dval])
        for c in constants
    ]
    try:
        for step in range(config.get("n_estimators", 300)):
            if stopper.stopped or (
                time_limit and step > 0 and time.perf_counter() - start >= time_limit
            ):
                break
            probabilities = []
            losses = []
            train_losses = []
            for j, model in enumerate(models):
                if model is None:
                    probabilities.append(np.full(len(val_y), constants[j]))
                    losses.append(
                        float(evaluate(val_y[:, j : j + 1], probabilities[-1][:, None])["bce"])
                    )
                    train_losses.append(0.0)
                    continue
                dtrain.set_label(train_y[:, j])
                dval.set_label(val_y[:, j])
                model.update(dtrain, step)
                p = model.predict(dval)
                probabilities.append(p)
                losses.append(float(evaluate(val_y[:, j : j + 1], p[:, None])["bce"]))
                train_p = np.clip(model.predict(dtrain), 1e-7, 1 - 1e-7)
                train_losses.append(
                    float(
                        -(
                            train_y[:, j] * np.log(train_p)
                            + (1 - train_y[:, j]) * np.log1p(-train_p)
                        ).mean()
                    )
                )
            score = micro_f1(val_y, np.column_stack(probabilities) >= 0.5)
            if stopper.update(score, step):
                best_raw = [
                    bytes(m.save_raw(raw_format="ubj")) if m is not None else None for m in models
                ]
            history.append(
                {
                    "boosting_round": step + 1,
                    "selection_f1_micro": score,
                    "selection_bce": float(np.mean(losses)),
                    "train_bce": float(np.mean(train_losses)),
                    "elapsed_seconds": time.perf_counter() - start,
                }
            )
            print(f"{modality} xgb round={step + 1} selection_f1={score:.5f}", flush=True)
        best_models = []
        for j, raw in enumerate(best_raw or []):
            model = None
            if raw is not None:
                model = xgb.Booster(params={"device": str(device), "nthread": threads})
                model.load_model(bytearray(raw))
                model.save_model(output / f"label_{j}.ubj")
            best_models.append(model)
        if len(best_models) != train_y.shape[1]:
            raise ValueError("No XGBoost checkpoint produced")
        atomic_json(
            output / "xgb.json",
            {
                "identity": identity,
                "constants": constants,
                "files": {p.name: file_sha256(p) for p in output.glob("label_*.ubj")},
            },
        )
        atomic_json(output / "history.json", history)
        node_count = sum(len(m.trees_to_dataframe()) for m in best_models if m is not None)
        # Tree nodes are not differentiable neural parameters; report separately.
        return (
            best_models,
            constants,
            {
                "selection_f1_micro": stopper.best,
                "best_step": stopper.best_step,
                "training_seconds": time.perf_counter() - start,
                "selection_checks": len(history),
                "parameter_count": None,
                "tree_node_count": node_count,
                "gpu_peak_allocated_bytes": None,
                "gpu_memory_scope": "not instrumented for XGBoost; never report torch allocator as XGB memory",
                "resume_semantics": "restart incomplete trial; completed trials reused",
                "xgb_memory_mode": "external" if use_external else "ram",
                "raw_feature_bytes": estimated_bytes,
                "stop_reason": "patience"
                if stopper.stopped
                else (
                    "time_budget"
                    if time_limit and time.perf_counter() - start >= time_limit
                    else "round_limit"
                ),
            },
        )
    finally:
        del train_x, val_x, dtrain, dval
        for path in matrix_paths:
            path.unlink(missing_ok=True)


def load_xgb(path: Path, device="cpu", threads=4):
    import json

    import xgboost as xgb

    meta = json.loads((path / "xgb.json").read_text())
    for name, digest in meta["files"].items():
        if file_sha256(path / name) != digest:
            raise ValueError("XGBoost checkpoint hash mismatch")
    models = []
    for j, c in enumerate(meta["constants"]):
        model = None
        if c not in (0, 1):
            model = xgb.Booster(params={"device": str(device), "nthread": threads})
            model.load_model(path / f"label_{j}.ubj")
        models.append(model)
    return models, meta["constants"]
