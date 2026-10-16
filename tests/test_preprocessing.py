"""Tests for the shared preprocessing used by the ablation benchmark."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from spectroscopy_qml.benchmarking.preprocessing import (
    ABLATION_CONFIGS,
    LEGACY,
    LEGACY_EFFECTIVE_SPECS,
    PreprocessingConfig,
    array_digest,
    canonical_spec,
    dataset_fingerprint,
    effective_spec,
    make_msms_spectrum,
    metrics_fields,
    parse_preprocessing,
    preprocess_spectra,
)

RNG = np.random.default_rng(0)


def _legacy_mps(spectrum: np.ndarray, target_length: int) -> np.ndarray:
    """The interpolate -> savgol(25, 4) -> SNV path of the MPS data loaders."""
    old_x = np.arange(len(spectrum))
    new_x = np.linspace(0, len(spectrum) - 1, target_length)
    out = interp1d(old_x, spectrum, kind="linear")(new_x)
    out = savgol_filter(out, window_length=25, polyorder=4)
    return (out - out.mean()) / out.std()


def test_legacy_and_empty_specs_mean_legacy():
    assert parse_preprocessing(None) is None
    assert parse_preprocessing("legacy") is None
    assert canonical_spec("") == LEGACY


def test_spec_is_canonical_regardless_of_key_order():
    a = canonical_spec("norm=snv,resample=fir")
    b = canonical_spec("resample=fir,norm=snv")
    assert a == b == "resample=fir,smooth=none,repr=intensity,norm=snv"


def test_config_names_resolve_to_specs():
    for name, spec in ABLATION_CONFIGS.items():
        assert canonical_spec(name) == canonical_spec(spec)


@pytest.mark.parametrize(
    "spec",
    ["resample=cubic", "smooth=savgol:4", "smooth=savgol:5:5", "norm=zscore", "foo=bar", "norm"],
)
def test_invalid_specs_fail_fast(spec):
    with pytest.raises(ValueError):
        parse_preprocessing(spec)


@pytest.mark.parametrize(
    "spec",
    ["smooth=savgol:24:4", "smooth=savgol:25.5:4", "smooth=savgol:25:-1", "smooth=moving_average:0.5"],
)
def test_integer_filter_parameters_are_validated(spec):
    with pytest.raises(ValueError):
        parse_preprocessing(spec)


def test_effective_spec_resolves_legacy_per_model():
    assert effective_spec("legacy", "mps") == LEGACY_EFFECTIVE_SPECS["mps"]
    assert effective_spec("norm_snv", "cnn") == canonical_spec("norm=snv")
    assert metrics_fields("legacy", "ttn") == {
        "preprocessing": "legacy",
        "preprocessing_effective": "resample=linear,smooth=none,repr=intensity,norm=snv",
    }


def test_shared_spec_reproduces_legacy_mps_pipeline():
    spectra = RNG.random((4, 1800))
    config = parse_preprocessing(LEGACY_EFFECTIVE_SPECS["mps"])
    ours = preprocess_spectra(spectra, config, 600)
    expected = np.stack([_legacy_mps(row, 600) for row in spectra])
    np.testing.assert_allclose(ours, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("method", ["linear", "bin_mean", "bin_max", "fir"])
@pytest.mark.parametrize("source_length", [1800, 10000])
def test_resampling_methods_hit_target_length(method, source_length):
    out = preprocess_spectra(
        RNG.random((3, source_length)), PreprocessingConfig(resample=method), 600
    )
    assert out.shape == (3, 600)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()


def test_bin_max_keeps_narrow_peak_that_linear_interpolation_misses():
    # A one-point C-NMR-like peak between two linear-interpolation sample points.
    spectrum = np.zeros(10000)
    spectrum[5016] = 1.0
    linear = preprocess_spectra([spectrum], PreprocessingConfig(resample="linear"), 600)
    binned = preprocess_spectra([spectrum], PreprocessingConfig(resample="bin_max"), 600)
    assert linear.max() < 0.5
    assert binned.max() == pytest.approx(1.0)


def test_bin_mean_preserves_total_intensity_per_bin():
    spectrum = RNG.random(1800)
    out = preprocess_spectra([spectrum], PreprocessingConfig(resample="bin_mean"), 600)
    np.testing.assert_allclose(out[0], spectrum.reshape(600, 3).mean(axis=1), rtol=1e-6)


def test_binning_refuses_to_upsample():
    with pytest.raises(ValueError):
        preprocess_spectra([RNG.random(100)], PreprocessingConfig(resample="bin_mean"), 600)


@pytest.mark.parametrize("norm", ["snv", "minmax", "max", "area", "l2"])
def test_normalizations_handle_constant_and_empty_spectra(norm):
    config = PreprocessingConfig(norm=norm)
    spectra = [np.zeros(1800), np.array([]), np.array([np.nan, 1.0, 2.0])]
    if norm in ("snv", "minmax"):
        spectra[0] = np.full(1800, 3.0)  # no variation left after centering
    out = preprocess_spectra(spectra, config, 600)
    assert np.isfinite(out).all()
    np.testing.assert_array_equal(out[0], 0.0)
    np.testing.assert_array_equal(out[1], 0.0)


def test_normalization_results():
    spectrum = RNG.random(1800) * 5 + 1
    snv = preprocess_spectra([spectrum], PreprocessingConfig(norm="snv"), 600)[0]
    assert snv.mean() == pytest.approx(0.0, abs=1e-5)
    assert snv.std() == pytest.approx(1.0, abs=1e-4)
    minmax = preprocess_spectra([spectrum], PreprocessingConfig(norm="minmax"), 600)[0]
    assert minmax.min() == pytest.approx(0.0) and minmax.max() == pytest.approx(1.0)
    area = preprocess_spectra([spectrum], PreprocessingConfig(norm="area"), 600)[0]
    assert np.abs(area).sum() == pytest.approx(1.0, rel=1e-4)


@pytest.mark.parametrize("smooth", ["savgol:25:4", "gaussian:2", "moving_average:5"])
def test_smoothing_reduces_noise(smooth):
    clean = np.sin(np.linspace(0, 6, 600))
    noisy = clean + RNG.normal(scale=0.2, size=600)
    out = preprocess_spectra([noisy], PreprocessingConfig(smooth=smooth), 600)[0]
    assert np.abs(out - clean).mean() < np.abs(noisy - clean).mean()


@pytest.mark.parametrize("representation", ["intensity", "sqrt", "log1p", "derivative"])
def test_representations_keep_shape(representation):
    out = preprocess_spectra(RNG.random((2, 1800)), PreprocessingConfig(repr=representation), 600)
    assert out.shape == (2, 600)
    assert np.isfinite(out).all()


def test_mixed_lengths_are_processed_per_length():
    out = preprocess_spectra(
        [RNG.random(1800), RNG.random(10000), RNG.random(1800)], PreprocessingConfig(), 600
    )
    assert out.shape == (3, 600)


def test_array_digest_normalizes_dtype_but_detects_content_changes():
    values = np.arange(12).reshape(3, 4)
    assert array_digest(values) == array_digest(values.astype(np.float64))
    changed = values.copy()
    changed[0, 0] = 99
    assert array_digest(values) != array_digest(changed)


def test_shared_msms_conversion_handles_invalid_and_out_of_range_peaks():
    dense = make_msms_spectrum([None, [1.2], [-1.0, 2.0], [2000.0, 3.0]])
    assert dense.dtype == np.float32
    assert dense[0] == 2.0
    assert dense[-1] == 3.0
    assert make_msms_spectrum(None) is None


def test_dataset_fingerprint_changes_with_parquet_metadata(tmp_path):
    parquet = tmp_path / "chunk.parquet"
    parquet.write_bytes(b"first")
    first = dataset_fingerprint(tmp_path)
    parquet.write_bytes(b"longer second value")
    assert dataset_fingerprint(tmp_path) != first
