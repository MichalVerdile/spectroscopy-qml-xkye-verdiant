#!/usr/bin/env python3
"""Run tiny CUDA smoke tests for all four benchmark model families."""

from __future__ import annotations

import importlib

import numpy as np


def import_training_modules() -> None:
    modules = [
        "benchmark.cnn.scripts.run_cnn_jung_baseline",
        "benchmark.xgb.scripts.run_xgb_baseline",
        "spectroscopy_qml.mps_classifier.cnmr.train",
        "spectroscopy_qml.mps_classifier.hnmr.train",
        "spectroscopy_qml.mps_classifier.ir.train",
        "spectroscopy_qml.mps_classifier.msms_pos.train",
        "spectroscopy_qml.mps_classifier.msms_neg.train",
        "spectroscopy_qml.tree_tensor_network.cnmr.train",
        "spectroscopy_qml.tree_tensor_network.hnmr.train",
        "spectroscopy_qml.tree_tensor_network.ir.train",
        "spectroscopy_qml.tree_tensor_network.msms_pos.train",
        "spectroscopy_qml.tree_tensor_network.msms_neg.train",
    ]
    for module in modules:
        importlib.import_module(module)
        print(f"IMPORT OK: {module}", flush=True)


def smoke_cnn() -> None:
    import tensorflow as tf
    from keras import Sequential
    from keras.layers import Conv1D, Dense, GlobalAveragePooling1D, Input

    devices = tf.config.list_physical_devices("GPU")
    if not devices:
        raise RuntimeError("CNN smoke test cannot see a TensorFlow GPU")
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)

    model = Sequential(
        [
            Input(shape=(32, 1)),
            Conv1D(4, 3, activation="relu"),
            GlobalAveragePooling1D(),
            Dense(3, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy")
    x = np.random.default_rng(41).normal(size=(8, 32, 1)).astype(np.float32)
    y = np.random.default_rng(42).integers(0, 2, size=(8, 3)).astype(np.float32)
    loss = float(model.train_on_batch(x, y))
    print(f"CNN SMOKE OK: loss={loss:.6f}", flush=True)


def smoke_xgboost() -> None:
    from xgboost import XGBClassifier

    rng = np.random.default_rng(43)
    x = rng.normal(size=(64, 12)).astype(np.float32)
    y = (x[:, 0] + x[:, 1] > 0).astype(np.int32)
    model = XGBClassifier(
        n_estimators=2,
        max_depth=2,
        tree_method="hist",
        device="cuda",
        n_jobs=1,
        verbosity=1,
    )
    model.fit(x, y)
    prediction = model.predict(x[:4])
    if prediction.shape != (4,):
        raise RuntimeError(f"Unexpected XGBoost output shape: {prediction.shape}")
    print("XGBOOST SMOKE OK", flush=True)


def backward_step(model, x, target) -> float:
    import torch

    model.train()
    logits = model(x)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    loss.backward()
    return float(loss.detach().cpu())


def smoke_mps() -> None:
    import torch

    from spectroscopy_qml.mps_classifier.cnmr.model import MPSFunctionalGroupClassifier

    model = MPSFunctionalGroupClassifier(
        input_dim=60,
        num_sites=5,
        physical_dim=4,
        bond_dim=4,
        num_classes=3,
        dropout_rate=0.0,
        classifier_head="mps",
        num_sites_2=10,
        physical_dim_2=4,
        bond_dim_2=4,
    ).cuda()
    x = torch.randn(2, 60, device="cuda")
    target = torch.randint(0, 2, (2, 3), device="cuda", dtype=torch.float32)
    loss = backward_step(model, x, target)
    print(f"MPS SMOKE OK: loss={loss:.6f}", flush=True)


def smoke_ttn() -> None:
    import torch

    from spectroscopy_qml.tree_tensor_network.cnmr.model import TTNCnmrClassifier10_2

    model = TTNCnmrClassifier10_2(
        num_labels=3,
        chi=8,
        input_dim=600,
        segment_window_size=48,
        segment_stride=43,
        lorentz_kernel_half_width=5,
    ).cuda()
    x = torch.randn(2, 600, device="cuda")
    target = torch.randint(0, 2, (2, 3), device="cuda", dtype=torch.float32)
    loss = backward_step(model, x, target)
    print(f"TTN SMOKE OK: loss={loss:.6f}", flush=True)


def main() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is unavailable")
    print(f"CUDA OK: {torch.cuda.get_device_name(0)}", flush=True)
    import_training_modules()
    smoke_cnn()
    smoke_xgboost()
    smoke_mps()
    smoke_ttn()
    print("ALL FOUR MODEL SMOKE TESTS PASSED", flush=True)


if __name__ == "__main__":
    main()
