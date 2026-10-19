"""Shared, configurable spectrum preprocessing for the CNN, XGBoost, MPS and TTN runs.

Every model historically shipped its own preprocessing:

* CNN and XGBoost: linear interpolation only.
* MPS: linear interpolation, Savitzky-Golay smoothing (25, 4), SNV.
* TTN: linear interpolation, SNV.

``legacy`` keeps exactly those per-model code paths so published results stay
reproducible. Any other spec routes every model through :func:`preprocess_spectra`
so that all four architectures see byte-identical inputs, which is what the
preprocessing/representation ablations need.

A spec is a comma separated ``key=value`` string, applied in this order::

    resample=<linear|bin_mean|bin_max|fir>
    smooth=<none|savgol:<window>:<polyorder>|gaussian:<sigma>|moving_average:<window>
            |lorentz:<gamma>[:<half_width>]>
    repr=<intensity|sqrt|log1p|derivative>
    norm=<none|snv|minmax|max|area|l2|quantile|pqn>

``lorentz`` convolves with a sum-normalised Lorentzian kernel (HWHM ``gamma`` in
data points, truncated at ``+-half_width``, default 15) with mirror padding, like the
TTN ``LorentzianFeatureMap``. ``quantile`` and ``pqn`` are cross-sample methods.
Direct calls to :func:`preprocess_spectra` retain the historical whole-batch
behaviour. Training loaders defer these methods, fit their reference on the fixed
training split, and then transform train, validation and test without refitting.

Omitted keys take the values of ``shared_mssd`` (linear, none, intensity, none),
which is the original MSSD protocol the paper describes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.ndimage import convolve1d, gaussian_filter1d, uniform_filter1d
from scipy.signal import resample_poly, savgol_filter

LEGACY = "legacy"

RESAMPLE_METHODS = ("linear", "bin_mean", "bin_max", "fir")
SMOOTH_METHODS = ("none", "savgol", "gaussian", "moving_average", "lorentz")
REPRESENTATIONS = ("intensity", "sqrt", "log1p", "derivative")
NORMALIZATIONS = ("none", "snv", "minmax", "max", "area", "l2", "quantile", "pqn")
# Normalizations that use statistics of the whole batch instead of a single spectrum.
CROSS_SAMPLE_NORMALIZATIONS = ("quantile", "pqn")

_SPEC_KEYS = ("resample", "smooth", "repr", "norm")
_EPS = 1e-8
_LORENTZ_DEFAULT_HALF_WIDTH = 15
_CACHE_SCHEMA_VERSION = 2


def _parse_smooth(smooth: str) -> tuple[str, tuple[float, ...]]:
    name, *raw_params = smooth.split(":")
    if name not in SMOOTH_METHODS:
        raise ValueError(f"Unknown smoothing {smooth!r}; use {SMOOTH_METHODS}")
    try:
        params = tuple(float(value) for value in raw_params)
    except ValueError as error:
        raise ValueError(f"Non-numeric smoothing parameter in {smooth!r}") from error
    expected = {
        "none": (0,),
        "savgol": (2,),
        "gaussian": (1,),
        "moving_average": (1,),
        "lorentz": (1, 2),
    }[name]
    if len(params) not in expected:
        counts = " or ".join(str(count) for count in expected)
        raise ValueError(f"Smoothing {name!r} takes {counts} parameter(s), got {smooth!r}")
    if name == "lorentz":
        if params[0] <= 0:
            raise ValueError(f"lorentz gamma must be positive, got {smooth!r}")
        if len(params) == 2 and (not params[1].is_integer() or params[1] < 1):
            raise ValueError(f"lorentz half_width must be a positive integer, got {smooth!r}")
    if name == "savgol":
        window, polyorder = params
        if not window.is_integer() or not polyorder.is_integer():
            raise ValueError(f"savgol window and polyorder must be integers, got {smooth!r}")
        if window < 3 or int(window) % 2 == 0 or polyorder < 0 or polyorder >= window:
            raise ValueError(
                f"savgol needs an odd window >= 3 and 0 <= polyorder < window, got {smooth!r}"
            )
    if name == "moving_average" and not params[0].is_integer():
        raise ValueError(f"moving_average window must be an integer, got {smooth!r}")
    if name in ("gaussian", "moving_average") and params[0] <= 0:
        raise ValueError(f"Smoothing parameter must be positive, got {smooth!r}")
    return name, params


@dataclass(frozen=True)
class PreprocessingConfig:
    resample: str = "linear"
    smooth: str = "none"
    repr: str = "intensity"
    norm: str = "none"

    def __post_init__(self) -> None:
        if self.resample not in RESAMPLE_METHODS:
            raise ValueError(f"Unknown resample method {self.resample!r}; use {RESAMPLE_METHODS}")
        _parse_smooth(self.smooth)
        if self.repr not in REPRESENTATIONS:
            raise ValueError(f"Unknown representation {self.repr!r}; use {REPRESENTATIONS}")
        if self.norm not in NORMALIZATIONS:
            raise ValueError(f"Unknown normalization {self.norm!r}; use {NORMALIZATIONS}")

    @property
    def spec(self) -> str:
        """Canonical spec string; identical configs always give the same string."""
        return ",".join(f"{key}={getattr(self, key)}" for key in _SPEC_KEYS)

    @property
    def short_hash(self) -> str:
        return hashlib.sha256(self.spec.encode("ascii")).hexdigest()[:10]


# What each model did before this module existed, expressed as a shared spec.
# Used only for reporting: legacy runs still execute the original code paths.
LEGACY_EFFECTIVE_SPECS = {
    "cnn": PreprocessingConfig().spec,
    "xgboost": PreprocessingConfig().spec,
    "mps": PreprocessingConfig(smooth="savgol:25:4", norm="snv").spec,
    "ttn": PreprocessingConfig(norm="snv").spec,
}

# Default ablation: the reference plus one variant per factor the reviewers named
# (downsampling, smoothing/filtering, normalization). Every entry is applied
# identically to every spectrum and model architecture. Input representation is
# covered by running the same configs at 600 and 1800 points.
SHARED_PREPROCESSING_CONFIGS: dict[str, str] = {
    # MSSD protocol the paper used: linear interpolation, nothing else
    "shared_mssd": PreprocessingConfig().spec,
    # Max-pooling keeps narrow C-NMR/MS peaks that linear interpolation can miss
    "resample_bin_max": PreprocessingConfig(resample="bin_max").spec,
    # Lorentzian kernel of the paper's TTN feature map (gamma 3, half width 15)
    "smooth_lorentz": PreprocessingConfig(smooth="lorentz:3:15").spec,
    # SNV: standard spectroscopic normalization, used by the old MPS/TTN pipelines
    "norm_snv": PreprocessingConfig(norm="snv").spec,
}
# Further one-factor variants, usable by name (CONFIGS=...) but not run by default.
EXTRA_PREPROCESSING_CONFIGS: dict[str, str] = {
    "resample_bin_mean": PreprocessingConfig(resample="bin_mean").spec,
    "resample_fir": PreprocessingConfig(resample="fir").spec,
    "smooth_savgol": PreprocessingConfig(smooth="savgol:25:4").spec,
    "smooth_gaussian": PreprocessingConfig(smooth="gaussian:2").spec,
    "smooth_moving_average": PreprocessingConfig(smooth="moving_average:5").spec,
    "norm_minmax": PreprocessingConfig(norm="minmax").spec,
    "norm_area": PreprocessingConfig(norm="area").spec,
    "norm_pqn": PreprocessingConfig(norm="pqn").spec,
    "norm_max": PreprocessingConfig(norm="max").spec,
    "norm_l2": PreprocessingConfig(norm="l2").spec,
    "norm_quantile": PreprocessingConfig(norm="quantile").spec,
    "repr_sqrt": PreprocessingConfig(repr="sqrt").spec,
    "repr_derivative": PreprocessingConfig(repr="derivative").spec,
    "repr_log1p": PreprocessingConfig(repr="log1p").spec,
}
# ``legacy`` (per-model historical preprocessing) on request: CONFIGS="legacy ...".
ABLATION_CONFIGS: dict[str, str] = {
    "legacy": LEGACY,
    **SHARED_PREPROCESSING_CONFIGS,
}
# Every name parse_preprocessing() accepts.
NAMED_CONFIGS: dict[str, str] = {**ABLATION_CONFIGS, **EXTRA_PREPROCESSING_CONFIGS}


def parse_preprocessing(spec: str | None) -> PreprocessingConfig | None:
    """Parse a spec or ablation config name. Returns ``None`` for ``legacy``."""
    if spec is None:
        return None
    spec = spec.strip()
    if spec in ("", LEGACY):
        return None
    if spec in NAMED_CONFIGS:
        return parse_preprocessing(NAMED_CONFIGS[spec])
    values: dict[str, str] = {}
    for part in spec.split(","):
        key, sep, value = part.partition("=")
        key, value = key.strip(), value.strip()
        if not sep or key not in _SPEC_KEYS:
            raise ValueError(
                f"Invalid preprocessing spec part {part!r} in {spec!r}; "
                f"expected key=value with key in {_SPEC_KEYS}"
            )
        if key in values:
            raise ValueError(f"Duplicate key {key!r} in preprocessing spec {spec!r}")
        values[key] = value
    return PreprocessingConfig(**values)


def canonical_spec(spec: str | None) -> str:
    config = parse_preprocessing(spec)
    return LEGACY if config is None else config.spec


def effective_spec(spec: str | None, model: str) -> str:
    """Spec actually applied for ``model``, resolving ``legacy`` per model."""
    config = parse_preprocessing(spec)
    return LEGACY_EFFECTIVE_SPECS[model] if config is None else config.spec


def metrics_fields(spec: str | None, model: str) -> dict[str, str]:
    """Fields every model writes into metrics.json for the ablation aggregation."""
    return {
        "preprocessing": canonical_spec(spec),
        "preprocessing_effective": effective_spec(spec, model),
    }


def array_digest(array: np.ndarray) -> str:
    """Stable digest used to prove that every model received identical inputs."""
    contiguous = np.ascontiguousarray(np.asarray(array, dtype=np.float32))
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def dataset_fingerprint(data_dir: Path) -> str:
    """Fingerprint a parquet dataset without hashing many gigabytes of file contents.

    The resolved directory plus each parquet file's relative name, size and nanosecond
    modification time are sufficient to invalidate caches after normal dataset updates.
    """
    data_dir = Path(data_dir).resolve()
    digest = hashlib.sha256(str(data_dir).encode("utf-8"))
    parquet_files = sorted(data_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")
    for path in parquet_files:
        stat = path.stat()
        digest.update(path.name.encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


@lru_cache(maxsize=1)
def preprocessing_code_fingerprint() -> str:
    """Hash this implementation so code changes automatically invalidate caches."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def preprocessing_cache_path(
    data_dir: Path,
    *,
    consumer: str,
    modality: str,
    target_length: int,
    preprocessing: str | None,
    options: dict[str, object] | None = None,
    cache_dir: Path | None = None,
) -> tuple[Path, str]:
    """Return a content-addressed cache path and its validation key.

    The key changes when the source parquet metadata or any preprocessing input
    changes.  Keeping the key inside the payload as well as in the filename makes
    stale or accidentally renamed cache files safe to ignore.
    """
    metadata = {
        "schema_version": _CACHE_SCHEMA_VERSION,
        "consumer": consumer,
        "modality": modality,
        "target_length": int(target_length),
        "preprocessing": canonical_spec(preprocessing),
        "data_fingerprint": dataset_fingerprint(data_dir),
        "code_fingerprint": preprocessing_code_fingerprint(),
        "options": options or {},
    }
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":"), default=str)
    cache_key = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    root = Path("data/cache/preprocessed") if cache_dir is None else Path(cache_dir)
    safe_consumer = consumer.replace("/", "_")
    safe_modality = modality.replace("/", "_")
    filename = f"{safe_consumer}_{safe_modality}_len{target_length}_{cache_key[:16]}.npz"
    return root / filename, cache_key


def load_preprocessing_cache(
    cache_path: Path, cache_key: str
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load cached ``X``/``y`` arrays when the embedded key is compatible."""
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as payload:
            if str(payload["cache_key"]) != cache_key:
                return None
            X = np.asarray(payload["X"])
            y = np.asarray(payload["y"])
    except (OSError, ValueError, KeyError):
        return None
    if X.ndim != 2 or y.ndim != 2 or len(X) != len(y):
        return None
    return X, y


def save_preprocessing_cache(
    cache_path: Path,
    cache_key: str,
    X: np.ndarray,
    y: np.ndarray,
) -> None:
    """Atomically save preprocessed inputs and labels as a compressed NPZ."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=cache_path.parent, prefix=f".{cache_path.stem}-", suffix=".npz", delete=False
        ) as temporary:
            temporary_path = temporary.name
        np.savez_compressed(
            temporary_path,
            X=np.asarray(X),
            y=np.asarray(y),
            cache_key=np.asarray(cache_key),
        )
        os.replace(temporary_path, cache_path)
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def legacy_linear_interpolate(spec: object, target_length: int) -> np.ndarray | None:
    """Historical CNN/XGBoost resampling, kept bit-exact for ``legacy`` runs.

    Unlike ``resample=linear`` it casts to float32 *before* interpolating, so the
    two differ in the last bit; legacy caches must therefore be built with this.
    """
    from scipy.interpolate import interp1d

    if spec is None:
        return None
    values = np.asarray(spec, dtype=np.float32).flatten()
    if len(values) == 0:
        return np.zeros(target_length, dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if len(values) == 1:
        return np.full(target_length, values[0], dtype=np.float32)
    old_x = np.arange(len(values))
    new_x = np.linspace(old_x.min(), old_x.max(), target_length)
    return interp1d(old_x, values, kind="linear")(new_x).astype(np.float32)


def make_msms_spectrum(spectrum: object, length: int = 10_000) -> np.ndarray | None:
    """Convert an MS/MS peak list into the shared dense representation."""
    if spectrum is None:
        return None
    dense = np.zeros(length, dtype=np.float32)
    for peak in spectrum:
        if peak is None or len(peak) < 2:
            continue
        position = int(float(peak[0]) * 10)
        position = min(max(position, 0), length - 1)
        dense[position] = float(peak[1])
    return dense


def _bin_edges(source_length: int, target_length: int) -> np.ndarray:
    if target_length > source_length:
        raise ValueError(
            f"Binning cannot upsample ({source_length} -> {target_length}); "
            "use resample=linear or resample=fir"
        )
    return np.floor(np.linspace(0, source_length, target_length + 1)).astype(np.int64)


def resample(spectra: np.ndarray, target_length: int, method: str) -> np.ndarray:
    """Resample a (n_samples, n_points) batch to ``target_length`` points."""
    source_length = spectra.shape[1]
    if method == "linear":
        if source_length == target_length:
            return spectra.copy()
        old_x = np.arange(source_length)
        new_x = np.linspace(0, source_length - 1, target_length)
        # Same grid as the legacy interp1d(kind="linear") code.
        return np.stack([np.interp(new_x, old_x, row) for row in spectra])
    if method in ("bin_mean", "bin_max"):
        starts = _bin_edges(source_length, target_length)[:-1]
        if method == "bin_mean":
            sums = np.add.reduceat(spectra, starts, axis=1)
            counts = np.diff(np.append(starts, source_length))
            return sums / counts
        return np.maximum.reduceat(spectra, starts, axis=1)
    if method == "fir":
        # Polyphase resampling applies a Kaiser-windowed FIR low-pass filter
        # before decimation, so peaks are not aliased.
        ratio = Fraction(target_length, source_length)
        out = resample_poly(spectra, ratio.numerator, ratio.denominator, axis=1)
        if out.shape[1] >= target_length:
            return out[:, :target_length]
        return np.pad(out, ((0, 0), (0, target_length - out.shape[1])), mode="edge")
    raise ValueError(f"Unknown resample method {method!r}")


def smooth(spectra: np.ndarray, smooth_spec: str) -> np.ndarray:
    name, params = _parse_smooth(smooth_spec)
    if name == "none":
        return spectra
    if name == "savgol":
        window, polyorder = int(params[0]), int(params[1])
        if spectra.shape[1] < window:
            return spectra
        return savgol_filter(spectra, window_length=window, polyorder=polyorder, axis=1)
    if name == "gaussian":
        return gaussian_filter1d(spectra, sigma=params[0], axis=1, mode="nearest")
    if name == "lorentz":
        gamma = params[0]
        half_width = int(params[1]) if len(params) == 2 else _LORENTZ_DEFAULT_HALF_WIDTH
        positions = np.arange(-half_width, half_width + 1, dtype=np.float64)
        kernel = 1.0 / (1.0 + (positions / gamma) ** 2)
        kernel /= kernel.sum()
        # "mirror" is numpy's "reflect" padding (edge sample not repeated); the kernel is symmetric.
        return convolve1d(spectra, kernel, axis=1, mode="mirror")
    return uniform_filter1d(spectra, size=int(params[0]), axis=1, mode="nearest")


def represent(spectra: np.ndarray, representation: str) -> np.ndarray:
    if representation == "intensity":
        return spectra
    if representation == "sqrt":
        return np.sign(spectra) * np.sqrt(np.abs(spectra))
    if representation == "log1p":
        return np.sign(spectra) * np.log1p(np.abs(spectra))
    if representation == "derivative":
        if spectra.shape[1] < 2:
            return np.zeros_like(spectra)
        return np.gradient(spectra, axis=1)
    raise ValueError(f"Unknown representation {representation!r}")


def _quantile_normalize(spectra: np.ndarray) -> np.ndarray:
    """Give every spectrum the mean sorted-value distribution of the batch."""
    mean_quantiles = np.sort(spectra, axis=1).mean(axis=0)
    ranks = np.argsort(np.argsort(spectra, axis=1), axis=1)
    return mean_quantiles[ranks]


def _pqn_normalize(spectra: np.ndarray) -> np.ndarray:
    """Probabilistic quotient normalization against the batch median spectrum."""
    # Step 1: total-intensity (row sum) normalization.
    row_sums = spectra.sum(axis=1, keepdims=True)
    tic_normalized = spectra / np.where(row_sums > _EPS, row_sums, 1.0)
    # Step 2: reference spectrum is the median profile of the batch.
    reference = np.median(tic_normalized, axis=0)
    # Step 3: per-spectrum dilution factor is the median quotient to the reference.
    mask = (tic_normalized > _EPS) & (reference > _EPS)
    quotients = np.full(tic_normalized.shape, np.nan)
    np.divide(tic_normalized, reference, out=quotients, where=mask)
    all_nan = ~mask.any(axis=1)
    dilution = np.ones(len(spectra))
    if not all_nan.all():
        dilution[~all_nan] = np.nanmedian(quotients[~all_nan], axis=1)
    dilution = np.where(dilution > _EPS, dilution, 1.0)
    return tic_normalized / dilution[:, None]


@dataclass(frozen=True)
class CrossSampleNormalizationState:
    """Training-only reference required to transform quantile or PQN inputs."""

    method: str
    reference: np.ndarray


def fit_cross_sample_normalization(
    train_spectra: np.ndarray, method: str
) -> CrossSampleNormalizationState:
    """Fit a cross-sample normalization reference using training spectra only."""
    spectra = np.asarray(train_spectra, dtype=np.float64)
    if spectra.ndim != 2 or len(spectra) == 0:
        raise ValueError("Cross-sample normalization needs a non-empty 2D training array")
    if method == "quantile":
        reference = np.sort(spectra, axis=1).mean(axis=0)
    elif method == "pqn":
        row_sums = spectra.sum(axis=1, keepdims=True)
        tic_normalized = spectra / np.where(row_sums > _EPS, row_sums, 1.0)
        reference = np.median(tic_normalized, axis=0)
    else:
        raise ValueError(f"{method!r} is not a cross-sample normalization")
    return CrossSampleNormalizationState(method=method, reference=reference)


def transform_cross_sample_normalization(
    spectra: np.ndarray, state: CrossSampleNormalizationState
) -> np.ndarray:
    """Apply a fitted training reference to arbitrary spectra without refitting."""
    values = np.asarray(spectra, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(state.reference):
        raise ValueError("Spectra width does not match the fitted normalization reference")
    if state.method == "quantile":
        ranks = np.argsort(np.argsort(values, axis=1), axis=1)
        transformed = state.reference[ranks]
    elif state.method == "pqn":
        row_sums = values.sum(axis=1, keepdims=True)
        tic_normalized = values / np.where(row_sums > _EPS, row_sums, 1.0)
        mask = (tic_normalized > _EPS) & (state.reference > _EPS)
        quotients = np.full(tic_normalized.shape, np.nan)
        np.divide(tic_normalized, state.reference, out=quotients, where=mask)
        all_nan = ~mask.any(axis=1)
        dilution = np.ones(len(values))
        if not all_nan.all():
            dilution[~all_nan] = np.nanmedian(quotients[~all_nan], axis=1)
        dilution = np.where(dilution > _EPS, dilution, 1.0)
        transformed = tic_normalized / dilution[:, None]
    else:
        raise ValueError(f"Unknown fitted normalization {state.method!r}")
    return np.nan_to_num(transformed, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def apply_train_fitted_normalization(
    spectra: np.ndarray,
    preprocessing: str | PreprocessingConfig | None,
    train_indices: np.ndarray,
) -> np.ndarray:
    """Fit quantile/PQN on ``train_indices`` and transform the complete dataset."""
    config = (
        preprocessing
        if isinstance(preprocessing, PreprocessingConfig)
        else parse_preprocessing(preprocessing)
    )
    if config is None or config.norm not in CROSS_SAMPLE_NORMALIZATIONS:
        return spectra
    indices = np.asarray(train_indices, dtype=np.int64)
    state = fit_cross_sample_normalization(np.asarray(spectra)[indices], config.norm)
    return transform_cross_sample_normalization(spectra, state)


def normalize(spectra: np.ndarray, method: str) -> np.ndarray:
    """Normalize a batch; constant spectra map to zeros, never NaN.

    ``quantile`` and ``pqn`` depend on the other spectra in ``spectra``.
    """
    if method == "none":
        return spectra
    if method == "quantile":
        return _quantile_normalize(spectra)
    if method == "pqn":
        return _pqn_normalize(spectra)
    if method == "snv":
        centered = spectra - spectra.mean(axis=1, keepdims=True)
        scale = spectra.std(axis=1, keepdims=True)
    elif method == "minmax":
        low = spectra.min(axis=1, keepdims=True)
        centered = spectra - low
        scale = spectra.max(axis=1, keepdims=True) - low
    elif method == "max":
        centered = spectra
        scale = np.abs(spectra).max(axis=1, keepdims=True)
    elif method == "area":
        centered = spectra
        scale = np.abs(spectra).sum(axis=1, keepdims=True)
    elif method == "l2":
        centered = spectra
        scale = np.linalg.norm(spectra, axis=1, keepdims=True)
    else:
        raise ValueError(f"Unknown normalization {method!r}")
    return np.where(scale < _EPS, centered * 0.0, centered / np.maximum(scale, _EPS))


def _apply(
    spectra: np.ndarray,
    config: PreprocessingConfig,
    target_length: int,
    apply_norm: bool = True,
) -> np.ndarray:
    spectra = resample(spectra, target_length, config.resample)
    spectra = smooth(spectra, config.smooth)
    spectra = represent(spectra, config.repr)
    return normalize(spectra, config.norm) if apply_norm else spectra


def preprocess_spectra(
    spectra: Sequence[np.ndarray] | np.ndarray,
    config: PreprocessingConfig,
    target_length: int,
    *,
    defer_cross_sample: bool = False,
) -> np.ndarray:
    """Turn dense raw spectra (any lengths) into a float32 (n, target_length) array."""
    rows = [
        np.nan_to_num(np.asarray(spec, dtype=np.float64).ravel(), nan=0.0, posinf=0.0, neginf=0.0)
        for spec in spectra
    ]
    # Cross-sample normalizations run once over all spectra (whatever their raw length),
    # so they need the un-normalized rows in full precision.
    cross_sample = config.norm in CROSS_SAMPLE_NORMALIZATIONS
    output = np.zeros((len(rows), target_length), dtype=np.float64 if cross_sample else np.float32)
    by_length: dict[int, list[int]] = {}
    for index, row in enumerate(rows):
        by_length.setdefault(len(row), []).append(index)
    for length, indices in by_length.items():
        if length == 0:
            continue
        batch = np.stack([rows[i] for i in indices])
        if length == 1:
            batch = np.repeat(batch, 2, axis=1)
        output[indices] = _apply(
            batch,
            config,
            target_length,
            apply_norm=not cross_sample,
        )
    if cross_sample and not defer_cross_sample:
        valid = np.array([len(row) > 0 for row in rows], dtype=bool)
        if valid.any():
            output[valid] = normalize(output[valid], config.norm)
    if cross_sample:
        output = output.astype(np.float32)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve preprocessing ablation configs.")
    parser.add_argument("name", nargs="?", help="Ablation config name or spec")
    parser.add_argument(
        "--list", action="store_true", help="List the 4 default shared preprocessing configs"
    )
    parser.add_argument(
        "--include-legacy",
        action="store_true",
        help="Include the model-specific historical legacy configuration in --list",
    )
    parser.add_argument(
        "--include-extra",
        action="store_true",
        help="Also list the extra configs that are not run by default",
    )
    args = parser.parse_args()
    if args.list or args.name is None:
        configs = dict(ABLATION_CONFIGS if args.include_legacy else SHARED_PREPROCESSING_CONFIGS)
        if args.include_extra:
            configs.update(EXTRA_PREPROCESSING_CONFIGS)
        for name, spec in configs.items():
            print(f"{name}\t{spec}")
        return
    print(canonical_spec(args.name))


if __name__ == "__main__":
    main()
