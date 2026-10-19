"""Tests for the shared preprocessing used by the ablation benchmark."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from spectroscopy_qml.benchmarking.preprocessing import (
    ABLATION_CONFIGS,
    CROSS_SAMPLE_NORMALIZATIONS,
    LEGACY,
    LEGACY_EFFECTIVE_SPECS,
    PreprocessingConfig,
    apply_train_fitted_normalization,
    array_digest,
    canonical_spec,
    dataset_fingerprint,
    effective_spec,
    load_preprocessing_cache,
    make_msms_spectrum,
    metrics_fields,
    parse_preprocessing,
    preprocess_spectra,
    preprocessing_cache_path,
    preprocessing_code_fingerprint,
    save_preprocessing_cache,
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
    [
        "smooth=savgol:24:4",
        "smooth=savgol:25.5:4",
        "smooth=savgol:25:-1",
        "smooth=moving_average:0.5",
    ],
)
def test_integer_filter_parameters_are_validated(spec):
    with pytest.raises(ValueError):
        parse_preprocessing(spec)


def test_effective_spec_resolves_legacy_per_model():
    assert effective_spec("legacy", "mps") == LEGACY_EFFECTIVE_SPECS["mps"]
    assert effective_spec("norm_none", "cnn") == canonical_spec("smooth=lorentz:3:15")
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


@pytest.mark.parametrize(
    "smooth", ["savgol:25:4", "gaussian:2", "moving_average:5", "lorentz:3", "lorentz:3:15"]
)
def test_smoothing_reduces_noise(smooth):
    clean = np.sin(np.linspace(0, 6, 600))
    noisy = clean + RNG.normal(scale=0.2, size=600)
    out = preprocess_spectra([noisy], PreprocessingConfig(smooth=smooth), 600)[0]
    assert np.abs(out - clean).mean() < np.abs(noisy - clean).mean()


def test_lorentz_smoothing_matches_ttn_feature_map_kernel():
    torch = pytest.importorskip("torch")
    from spectroscopy_qml.tree_tensor_network.cnmr.model import LorentzianFeatureMap

    spectra = RNG.random((3, 300)).astype(np.float32)
    feature_map = LorentzianFeatureMap(gamma=3.0, kernel_half_width=15, norm_mode="max_abs")
    expected = feature_map._apply_kernel(torch.from_numpy(spectra), feature_map._k0).numpy()
    out = preprocess_spectra(spectra, PreprocessingConfig(smooth="lorentz:3:15"), 300)
    np.testing.assert_allclose(out, expected, atol=1e-5)


@pytest.mark.parametrize("smooth", ["lorentz:0", "lorentz:3:0", "lorentz:3:2.5", "lorentz:1:2:3"])
def test_invalid_lorentz_parameters_fail_fast(smooth):
    with pytest.raises(ValueError):
        PreprocessingConfig(smooth=smooth)


def test_quantile_normalization_matches_original_script():
    spectra = RNG.random((6, 600)) * 10
    expected = np.mean(np.sort(spectra, axis=1), axis=0)[
        np.argsort(np.argsort(spectra, axis=1), axis=1)
    ]
    out = preprocess_spectra(spectra, PreprocessingConfig(norm="quantile"), 600)
    np.testing.assert_allclose(out, expected, rtol=1e-5)
    # Every spectrum ends up with the same value distribution.
    np.testing.assert_allclose(np.sort(out, axis=1), np.sort(out, axis=1)[:1].repeat(6, axis=0))


def test_pqn_normalization_matches_original_script():
    spectra = RNG.random((8, 600)) + 0.1
    tic = spectra / spectra.sum(axis=1, keepdims=True)
    reference = np.median(tic, axis=0)
    expected = tic / np.median(tic / reference, axis=1, keepdims=True)
    out = preprocess_spectra(spectra, PreprocessingConfig(norm="pqn"), 600)
    np.testing.assert_allclose(out, expected, rtol=1e-4)


def test_pqn_removes_dilution_differences():
    profile = RNG.random(600) + 0.1
    spectra = np.stack([profile * factor for factor in (1.0, 2.0, 5.0)] + [profile])
    out = preprocess_spectra(spectra, PreprocessingConfig(norm="pqn"), 600)
    np.testing.assert_allclose(out, out[0:1].repeat(4, axis=0), rtol=1e-4)


@pytest.mark.parametrize("norm", ["quantile", "pqn"])
def test_cross_sample_normalizations_are_finite_with_degenerate_inputs(norm):
    spectra = [RNG.random(1800), np.zeros(1800), np.array([]), RNG.random(10000), np.array([2.0])]
    out = preprocess_spectra(spectra, PreprocessingConfig(norm=norm), 600)
    assert out.shape == (5, 600)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()
    np.testing.assert_array_equal(out[2], 0.0)


def test_cross_sample_normalization_spans_mixed_raw_lengths():
    spectra = [RNG.random(1800), RNG.random(10000), RNG.random(1800)]
    out = preprocess_spectra(spectra, PreprocessingConfig(norm="quantile"), 600)
    np.testing.assert_allclose(np.sort(out, axis=1), np.sort(out, axis=1)[:1].repeat(3, axis=0))


@pytest.mark.parametrize("norm", CROSS_SAMPLE_NORMALIZATIONS)
def test_deferred_cross_sample_normalization_keeps_split_independent_base(norm):
    spectra = RNG.random((5, 1800)) + 0.1
    deferred = preprocess_spectra(
        spectra,
        PreprocessingConfig(norm=norm),
        600,
        defer_cross_sample=True,
    )
    expected = preprocess_spectra(spectra, PreprocessingConfig(norm="none"), 600)
    np.testing.assert_array_equal(deferred, expected)


@pytest.mark.parametrize("norm", CROSS_SAMPLE_NORMALIZATIONS)
def test_train_fitted_normalization_does_not_fit_on_validation_or_test(norm):
    base = RNG.random((6, 20)) + 0.1
    changed_test = base.copy()
    changed_test[3:] *= 10_000
    train_indices = np.array([0, 1, 2])

    first = apply_train_fitted_normalization(base, f"norm={norm}", train_indices)
    second = apply_train_fitted_normalization(changed_test, f"norm={norm}", train_indices)

    np.testing.assert_array_equal(first[train_indices], second[train_indices])
    assert first.dtype == second.dtype == np.float32


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


def test_preprocessing_cache_round_trip_and_validation(tmp_path):
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    (data_dir / "chunk.parquet").write_bytes(b"source metadata")
    cache_dir = tmp_path / "cache"
    cache_path, cache_key = preprocessing_cache_path(
        data_dir,
        consumer="mps",
        modality="ir",
        target_length=600,
        preprocessing="norm_none",
        options={"max_files": None},
        cache_dir=cache_dir,
    )
    X = RNG.random((3, 600)).astype(np.float32)
    y = RNG.integers(0, 2, size=(3, 37), dtype=np.int32)

    assert load_preprocessing_cache(cache_path, cache_key) is None
    save_preprocessing_cache(cache_path, cache_key, X, y)
    cached = load_preprocessing_cache(cache_path, cache_key)
    assert cached is not None
    np.testing.assert_array_equal(cached[0], X)
    np.testing.assert_array_equal(cached[1], y)
    assert load_preprocessing_cache(cache_path, "wrong-key") is None


def test_preprocessing_cache_key_changes_with_configuration(tmp_path):
    (tmp_path / "chunk.parquet").write_bytes(b"data")
    common = {
        "data_dir": tmp_path,
        "consumer": "cnn",
        "modality": "ir",
        "target_length": 600,
        "cache_dir": tmp_path / "cache",
    }
    first, _ = preprocessing_cache_path(preprocessing="shared_mssd", **common)
    changed, _ = preprocessing_cache_path(preprocessing="norm_none", **common)
    assert first != changed


def test_preprocessing_cache_key_contains_code_fingerprint(tmp_path, monkeypatch):
    (tmp_path / "chunk.parquet").write_bytes(b"data")
    kwargs = {
        "data_dir": tmp_path,
        "consumer": "shared",
        "modality": "ir",
        "target_length": 600,
        "preprocessing": "shared_mssd",
        "cache_dir": tmp_path / "cache",
    }
    first, _ = preprocessing_cache_path(**kwargs)
    monkeypatch.setattr(
        "spectroscopy_qml.benchmarking.preprocessing.preprocessing_code_fingerprint",
        lambda: "changed-code",
    )
    changed, _ = preprocessing_cache_path(**kwargs)
    assert first != changed
    assert len(preprocessing_code_fingerprint()) == 64


def test_mps_loader_populates_and_reuses_shared_cache(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    (data_dir / "chunk.parquet").write_bytes(b"fingerprint only")
    cache_dir = tmp_path / "cache"

    from spectroscopy_qml.mps_classifier.ir import data_loader

    frame = data_loader.pd.DataFrame(
        {
            "ir_spectra": [np.arange(8, dtype=np.float32), np.arange(8, dtype=np.float32) + 1],
            "smiles": ["CC", "CO"],
        }
    )
    monkeypatch.setattr(data_loader.pd, "read_parquet", lambda *args, **kwargs: frame.copy())
    monkeypatch.setattr(
        data_loader,
        "get_functional_groups",
        lambda smiles: np.zeros(len(data_loader.FUNCTIONAL_GROUPS), dtype=np.int32),
    )
    first = data_loader.load_ir_data(
        data_dir,
        target_length=4,
        preprocessing="shared_mssd",
        cache_dir=cache_dir,
    )

    def fail_if_raw_data_is_read(*args, **kwargs):
        raise AssertionError("raw parquet should not be read after the cache is populated")

    monkeypatch.setattr(data_loader.pd, "read_parquet", fail_if_raw_data_is_read)
    second = data_loader.load_ir_data(
        data_dir,
        target_length=4,
        preprocessing="shared_mssd",
        cache_dir=cache_dir,
    )
    np.testing.assert_array_equal(second[0], first[0])
    np.testing.assert_array_equal(second[1], first[1])
    assert len(list(cache_dir.glob("shared_ir_len4_*.npz"))) == 1


def test_legacy_linear_interpolate_matches_historical_cnn_code():
    from spectroscopy_qml.benchmarking.preprocessing import legacy_linear_interpolate

    def historical(spec, target_length):
        spec = np.asarray(spec, dtype=np.float32).flatten()
        spec = np.nan_to_num(spec, nan=0.0, posinf=0.0, neginf=0.0)
        old_x = np.arange(len(spec))
        new_x = np.linspace(min(old_x), max(old_x), target_length)
        return interp1d(old_x, spec)(new_x).astype(np.float32)

    for length in (1800, 10000):
        spectrum = RNG.random(length) * 3.7
        np.testing.assert_array_equal(
            legacy_linear_interpolate(spectrum, 600), historical(spectrum, 600)
        )
    assert legacy_linear_interpolate(None, 600) is None
    np.testing.assert_array_equal(legacy_linear_interpolate([], 600), np.zeros(600, np.float32))
