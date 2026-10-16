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
    smooth=<none|savgol:<window>:<polyorder>|gaussian:<sigma>|moving_average:<window>>
    repr=<intensity|sqrt|log1p|derivative>
    norm=<none|snv|minmax|max|area|l2>

Omitted keys take the values of ``shared_mssd`` (linear, none, intensity, none),
which is the original MSSD protocol the paper describes.
"""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import resample_poly, savgol_filter

LEGACY = "legacy"

RESAMPLE_METHODS = ("linear", "bin_mean", "bin_max", "fir")
SMOOTH_METHODS = ("none", "savgol", "gaussian", "moving_average")
REPRESENTATIONS = ("intensity", "sqrt", "log1p", "derivative")
NORMALIZATIONS = ("none", "snv", "minmax", "max", "area", "l2")

_SPEC_KEYS = ("resample", "smooth", "repr", "norm")
_EPS = 1e-8


def _parse_smooth(smooth: str) -> tuple[str, tuple[float, ...]]:
    name, *raw_params = smooth.split(":")
    if name not in SMOOTH_METHODS:
        raise ValueError(f"Unknown smoothing {smooth!r}; use {SMOOTH_METHODS}")
    try:
        params = tuple(float(value) for value in raw_params)
    except ValueError as error:
        raise ValueError(f"Non-numeric smoothing parameter in {smooth!r}") from error
    expected = {"none": 0, "savgol": 2, "gaussian": 1, "moving_average": 1}[name]
    if len(params) != expected:
        raise ValueError(f"Smoothing {name!r} takes {expected} parameter(s), got {smooth!r}")
    if name == "savgol" and (params[0] < 3 or params[1] >= params[0]):
        raise ValueError(f"savgol needs window >= 3 and polyorder < window, got {smooth!r}")
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

# One-factor-at-a-time design around ``shared_mssd``. Every config except
# ``legacy`` feeds identical inputs to all four models.
ABLATION_CONFIGS: dict[str, str] = {
    "legacy": LEGACY,
    "shared_mssd": PreprocessingConfig().spec,
    # Downsampling method
    "resample_bin_mean": PreprocessingConfig(resample="bin_mean").spec,
    "resample_bin_max": PreprocessingConfig(resample="bin_max").spec,
    "resample_fir": PreprocessingConfig(resample="fir").spec,
    # Smoothing / signal filtering
    "smooth_savgol": PreprocessingConfig(smooth="savgol:25:4").spec,
    "smooth_gaussian": PreprocessingConfig(smooth="gaussian:2").spec,
    # Normalization
    "norm_snv": PreprocessingConfig(norm="snv").spec,
    "norm_minmax": PreprocessingConfig(norm="minmax").spec,
    "norm_area": PreprocessingConfig(norm="area").spec,
    # Input representation
    "repr_sqrt": PreprocessingConfig(repr="sqrt").spec,
    "repr_derivative": PreprocessingConfig(repr="derivative").spec,
}


def parse_preprocessing(spec: str | None) -> PreprocessingConfig | None:
    """Parse a spec or ablation config name. Returns ``None`` for ``legacy``."""
    if spec is None:
        return None
    spec = spec.strip()
    if spec in ("", LEGACY):
        return None
    if spec in ABLATION_CONFIGS:
        return parse_preprocessing(ABLATION_CONFIGS[spec])
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


def normalize(spectra: np.ndarray, method: str) -> np.ndarray:
    """Per-spectrum normalization; constant spectra map to zeros, never NaN."""
    if method == "none":
        return spectra
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


def _apply(spectra: np.ndarray, config: PreprocessingConfig, target_length: int) -> np.ndarray:
    spectra = resample(spectra, target_length, config.resample)
    spectra = smooth(spectra, config.smooth)
    spectra = represent(spectra, config.repr)
    return normalize(spectra, config.norm)


def preprocess_spectra(
    spectra: Sequence[np.ndarray] | np.ndarray,
    config: PreprocessingConfig,
    target_length: int,
) -> np.ndarray:
    """Turn dense raw spectra (any lengths) into a float32 (n, target_length) array."""
    rows = [
        np.nan_to_num(np.asarray(spec, dtype=np.float64).ravel(), nan=0.0, posinf=0.0, neginf=0.0)
        for spec in spectra
    ]
    output = np.zeros((len(rows), target_length), dtype=np.float32)
    by_length: dict[int, list[int]] = {}
    for index, row in enumerate(rows):
        by_length.setdefault(len(row), []).append(index)
    for length, indices in by_length.items():
        if length == 0:
            continue
        batch = np.stack([rows[i] for i in indices])
        if length == 1:
            batch = np.repeat(batch, 2, axis=1)
        output[indices] = _apply(batch, config, target_length).astype(np.float32)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve preprocessing ablation configs.")
    parser.add_argument("name", nargs="?", help="Ablation config name or spec")
    parser.add_argument("--list", action="store_true", help="List ablation config names")
    args = parser.parse_args()
    if args.list or args.name is None:
        for name, spec in ABLATION_CONFIGS.items():
            print(f"{name}\t{spec}")
        return
    print(canonical_spec(args.name))


if __name__ == "__main__":
    main()
