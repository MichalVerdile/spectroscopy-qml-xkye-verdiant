"""Extract frozen CNN features (1574-dim) from the pretrained IR CNN model.

Loads the Keras IR CNN (benchmark/cnn/models/ir/k_fold/ir_model.keras),
truncates it at the last Dense layer (1574-dim feature vector), runs all
IR spectra through it, and saves the result as a .npz file:

    cnn_features.npz
        features  : float32  (N, 1574)
        labels    : int32    (N, 37)
        split_indices : stored separately (use the existing split .npz)

Usage:
    python extract_cnn_features.py \\
        --cnn-model   benchmark/cnn/models/ir/k_fold/ir_model.keras \\
        --data-dir    data/raw \\
        --output-path benchmark/cnn/features/cnn_ir_features.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = Path(__file__).resolve().parents[5]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment5.data_loader import (  # noqa: E402
    load_ir_data,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract 1574-dim CNN features from pretrained IR CNN."
    )
    parser.add_argument(
        "--cnn-model",
        type=Path,
        default=Path("benchmark/cnn/models/ir/k_fold/ir_model.keras"),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("benchmark/cnn/features/cnn_ir_features.npz"),
    )
    parser.add_argument("--input-dim", type=int, default=1800,
                        help="TTN input resolution (spectra are resampled to 600 for CNN).")
    parser.add_argument("--cnn-input-dim", type=int, default=600,
                        help="CNN expects 600-point spectra.")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Limit number of parquet files loaded (for quick tests).")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--apply-snv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache-path", type=Path, default=None)
    return parser


def resample_to_600(X: np.ndarray) -> np.ndarray:
    """Downsample from 1800 to 600 via simple averaging of triplets."""
    if X.shape[1] == 600:
        return X
    if X.shape[1] % 600 != 0:
        from scipy.interpolate import interp1d
        old_x = np.linspace(0, 1, X.shape[1])
        new_x = np.linspace(0, 1, 600)
        out = np.zeros((X.shape[0], 600), dtype=np.float32)
        for i in range(X.shape[0]):
            out[i] = interp1d(old_x, X[i])(new_x)
        return out
    factor = X.shape[1] // 600
    return X.reshape(X.shape[0], 600, factor).mean(axis=2).astype(np.float32)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.cnn_model.exists():
        raise FileNotFoundError(f"CNN model not found: {args.cnn_model}")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load Keras model and build feature extractor ────────────────────────────
    print(f"Loading CNN model from {args.cnn_model} ...")
    import keras
    from keras.layers import Dense as _KerasDense

    # Keras 3.9 serialises quantization_config=None but Dense.__init__ rejects it.
    _orig_from_config = _KerasDense.from_config.__func__
    @classmethod  # type: ignore[misc]
    def _patched_from_config(cls, config):
        config.pop("quantization_config", None)
        return _orig_from_config(cls, config)
    _KerasDense.from_config = _patched_from_config

    full_model = keras.models.load_model(str(args.cnn_model), compile=False)
    full_model.summary()

    # Find the Dense layer with the largest units < 37 (the 1574-dim feature layer).
    # Architecture: Dense(4927) → Dense(2785) → Dense(1574) → Dense(37)
    feature_layer = None
    for layer in reversed(full_model.layers):
        if isinstance(layer, _KerasDense) and layer.units != 37:
            feature_layer = layer
            break
    if feature_layer is None:
        raise RuntimeError("Could not locate the 1574-dim feature Dense layer.")
    print(f"\nFeature layer: {feature_layer.name}  units={feature_layer.units}")

    extractor = keras.Model(
        inputs=full_model.input,
        outputs=feature_layer.output,
    )
    print("Feature extractor ready.")

    # ── Load IR spectra ─────────────────────────────────────────────────────────
    print("\nLoading IR spectra...")
    X_1800, y = load_ir_data(
        data_dir=args.data_dir,
        target_length=args.input_dim,
        max_files=args.max_files,
        apply_snv=args.apply_snv,
        cache_path=args.cache_path,
        overwrite_cache=False,
    )
    print(f"Loaded X shape: {X_1800.shape}, y shape: {y.shape}")

    # CNN was trained on 600-point spectra
    print(f"Resampling {X_1800.shape[1]} → 600 points ...")
    X_600 = resample_to_600(X_1800)
    print(f"Resampled shape: {X_600.shape}")

    # CNN input: (N, 600, 1)
    X_cnn = X_600.reshape(-1, 600, 1).astype(np.float32)

    # ── Extract features in batches ─────────────────────────────────────────────
    print(f"\nExtracting features (batch_size={args.batch_size}) ...")
    n = X_cnn.shape[0]
    num_batches = (n + args.batch_size - 1) // args.batch_size
    feature_list = []

    for i in range(num_batches):
        start = i * args.batch_size
        end = min(start + args.batch_size, n)
        batch = X_cnn[start:end]
        feats = extractor(batch, training=False).numpy()
        feature_list.append(feats)
        if (i + 1) % 10 == 0 or (i + 1) == num_batches:
            print(f"  Batch {i+1}/{num_batches}  ({end}/{n} samples)")

    features = np.concatenate(feature_list, axis=0).astype(np.float32)
    nan_count = np.isnan(features).sum()
    if nan_count > 0:
        print(f"WARNING: {nan_count} NaN values in extracted features (likely zero-variance spectra). Replacing with 0.")
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"\nExtracted features shape: {features.shape}")

    # ── Save ────────────────────────────────────────────────────────────────────
    np.savez_compressed(
        args.output_path,
        features=features,
        labels=y.astype(np.int32),
    )
    print(f"\nSaved to {args.output_path}")
    print(f"  features : {features.shape}  float32")
    print(f"  labels   : {y.shape}  int32")


if __name__ == "__main__":
    main()
