"""Tests for experiments 10.2, 10.3, 10.4, and 10.5 feature maps and models.

Covers the invariants that were previously unprotected:
  - Zero/flat input produces near-zero derivative channels (pos_emb fix).
  - Smoothed feature maps reduce high-frequency energy vs raw finite-differences.
  - Forward pass produces finite outputs with correct shapes.
  - Max-abs normalization scale behaviour.
"""

from __future__ import annotations

import torch
import pytest

from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment10_2.model import (
    LorentzianFeatureMap,
    TTNIRClassifier10_2,
)
from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment10_3.model import (
    VoigtFeatureMap,
    TTNIRClassifier10_3,
)
from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment10_4.model import (
    SavitzkyGolayFeatureMap,
    TTNIRClassifier10_4,
)
from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment10_5.model import (
    SavitzkyGolayFeatureMapNoNorm,
    TTNIRClassifier10_5,
)
from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment6.model import (
    SpectralDerivativeFeatureMap,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def hf_energy(channel: torch.Tensor) -> float:
    """High-frequency energy proxy: std of (channel - 7-point box smooth)."""
    # Simple box filter as HF estimator
    kernel = torch.ones(1, 1, 7) / 7.0
    import torch.nn.functional as F
    padded = F.pad(channel.unsqueeze(1), (3, 3), mode="reflect")
    smooth = F.conv1d(padded, kernel).squeeze(1)
    return float((channel - smooth).pow(2).mean().sqrt())


# ── LorentzianFeatureMap ───────────────────────────────────────────────────────

class TestLorentzianFeatureMap:
    def test_output_shape(self) -> None:
        fm = LorentzianFeatureMap()
        x = torch.randn(4, 1800)
        out = fm(x)
        assert out.shape == (4, 1800, 3)

    def test_zero_input_derivatives_near_zero(self) -> None:
        """Analytical derivative kernels of zero input must yield zero channels."""
        fm = LorentzianFeatureMap(gamma=3.0, kernel_half_width=15)
        x = torch.zeros(1, 1800)
        d1 = fm._apply_kernel(x, fm._k1)
        d2 = fm._apply_kernel(x, fm._k2)
        assert d1.abs().max().item() < 1e-6, "d1 of zero input must be zero"
        assert d2.abs().max().item() < 1e-6, "d2 of zero input must be zero"

    def test_reduces_hf_energy_vs_raw(self) -> None:
        """Lorentzian-smoothed d1/d2 must have lower HF energy than raw finite-diff."""
        torch.manual_seed(0)
        x = torch.randn(8, 1800)
        fm_raw = SpectralDerivativeFeatureMap()
        fm_lor = LorentzianFeatureMap(gamma=3.0, kernel_half_width=15)
        raw_out = fm_raw(x)
        lor_out = fm_lor(x)
        for ch in [1, 2]:
            e_raw = hf_energy(raw_out[:, :, ch])
            e_lor = hf_energy(lor_out[:, :, ch])
            assert e_lor < e_raw, (
                f"Lorentzian ch{ch} HF energy {e_lor:.5f} not less than raw {e_raw:.5f}"
            )

    def test_ramp_d1_positive(self) -> None:
        """Derivative of a rising ramp must be positive (sign check for conv1d flip)."""
        fm = LorentzianFeatureMap(gamma=3.0, kernel_half_width=15)
        x = torch.linspace(0.0, 1.0, 1800).unsqueeze(0)
        d1 = fm._apply_kernel(x, fm._k1)
        assert d1[0, 100:-100].mean().item() > 0, "d1 of ramp must be positive"

    def test_all_finite(self) -> None:
        fm = LorentzianFeatureMap()
        out = fm(torch.randn(4, 1800))
        assert torch.isfinite(out).all()


# ── VoigtFeatureMap ───────────────────────────────────────────────────────────

class TestVoigtFeatureMap:
    def test_output_shape(self) -> None:
        fm = VoigtFeatureMap()
        assert fm(torch.randn(4, 1800)).shape == (4, 1800, 3)

    def test_zero_input_derivatives_near_zero(self) -> None:
        fm = VoigtFeatureMap(gamma_l=3.0, gamma_g=2.0, eta=0.5, kernel_half_width=20)
        x = torch.zeros(1, 1800)
        d1 = fm._apply_kernel(x, fm._k1)
        d2 = fm._apply_kernel(x, fm._k2)
        assert d1.abs().max().item() < 1e-6
        assert d2.abs().max().item() < 1e-6

    def test_reduces_hf_energy_vs_raw(self) -> None:
        torch.manual_seed(0)
        x = torch.randn(8, 1800)
        fm_raw = SpectralDerivativeFeatureMap()
        fm_voi = VoigtFeatureMap()
        raw_out = fm_raw(x)
        voi_out = fm_voi(x)
        for ch in [1, 2]:
            assert hf_energy(voi_out[:, :, ch]) < hf_energy(raw_out[:, :, ch])

    def test_ramp_d1_positive(self) -> None:
        """Derivative of a rising ramp must be positive (sign check for conv1d flip)."""
        fm = VoigtFeatureMap(gamma_l=3.0, gamma_g=2.0, eta=0.5, kernel_half_width=20)
        x = torch.linspace(0.0, 1.0, 1800).unsqueeze(0)
        d1 = fm._apply_kernel(x, fm._k1)
        assert d1[0, 100:-100].mean().item() > 0, "d1 of ramp must be positive"

    def test_all_finite(self) -> None:
        assert torch.isfinite(VoigtFeatureMap()(torch.randn(4, 1800))).all()


# ── SavitzkyGolayFeatureMap ───────────────────────────────────────────────────

class TestSavitzkyGolayFeatureMap:
    def test_output_shape(self) -> None:
        fm = SavitzkyGolayFeatureMap()
        assert fm(torch.randn(4, 1800)).shape == (4, 1800, 3)

    def test_zero_input_derivatives_near_zero(self) -> None:
        """SG d1/d2 of a zero spectrum must be exactly zero (polynomial fit of zeros)."""
        fm = SavitzkyGolayFeatureMap(window_length=11, polyorder=3)
        x = torch.zeros(1, 1800)
        d1 = fm._apply_kernel(x, fm._k1)
        d2 = fm._apply_kernel(x, fm._k2)
        assert d1.abs().max().item() < 1e-5
        assert d2.abs().max().item() < 1e-5

    def test_reduces_hf_energy_vs_raw(self) -> None:
        torch.manual_seed(0)
        x = torch.randn(8, 1800)
        fm_raw = SpectralDerivativeFeatureMap()
        fm_sg  = SavitzkyGolayFeatureMap()
        raw_out = fm_raw(x)
        sg_out  = fm_sg(x)
        for ch in [1, 2]:
            assert hf_energy(sg_out[:, :, ch]) < hf_energy(raw_out[:, :, ch])

    def test_ramp_d1_positive(self) -> None:
        """Derivative of a rising ramp must be positive (sign check for conv1d flip)."""
        fm = SavitzkyGolayFeatureMap(window_length=11, polyorder=3)
        x = torch.linspace(0.0, 1.0, 1800).unsqueeze(0)
        d1 = fm._apply_kernel(x, fm._k1)
        assert d1[0, 100:-100].mean().item() > 0, "d1 of ramp must be positive"

    def test_invalid_even_window_raises(self) -> None:
        with pytest.raises(ValueError, match="odd"):
            SavitzkyGolayFeatureMap(window_length=10)

    def test_all_finite(self) -> None:
        assert torch.isfinite(SavitzkyGolayFeatureMap()(torch.randn(4, 1800))).all()


class TestSavitzkyGolayFeatureMapNoNorm:
    def test_output_shape(self) -> None:
        fm = SavitzkyGolayFeatureMapNoNorm()
        assert fm(torch.randn(4, 1800)).shape == (4, 1800, 3)

    def test_zero_input_derivatives_near_zero(self) -> None:
        fm = SavitzkyGolayFeatureMapNoNorm(window_length=11, polyorder=3)
        x = torch.zeros(1, 1800)
        d1 = fm._apply_kernel(x, fm._k1)
        d2 = fm._apply_kernel(x, fm._k2)
        assert d1.abs().max().item() < 1e-5
        assert d2.abs().max().item() < 1e-5

    def test_leaves_channels_unscaled_after_filtering(self) -> None:
        fm = SavitzkyGolayFeatureMapNoNorm(window_length=11, polyorder=3)
        x = torch.randn(2, 1800)
        out = fm(x)
        assert torch.allclose(out[:, :, 0], fm._apply_kernel(x, fm._k0))
        assert torch.allclose(out[:, :, 1], fm._apply_kernel(x, fm._k1))
        assert torch.allclose(out[:, :, 2], fm._apply_kernel(x, fm._k2))

    def test_ramp_d1_positive(self) -> None:
        fm = SavitzkyGolayFeatureMapNoNorm(window_length=11, polyorder=3)
        x = torch.linspace(0.0, 1.0, 1800).unsqueeze(0)
        d1 = fm._apply_kernel(x, fm._k1)
        assert d1[0, 100:-100].mean().item() > 0

    def test_invalid_even_window_raises(self) -> None:
        with pytest.raises(ValueError, match="odd"):
            SavitzkyGolayFeatureMapNoNorm(window_length=10)

    def test_all_finite(self) -> None:
        assert torch.isfinite(SavitzkyGolayFeatureMapNoNorm()(torch.randn(4, 1800))).all()


# ── Model forward pass tests ──────────────────────────────────────────────────

@pytest.mark.parametrize("ModelCls,kwargs", [
    (TTNIRClassifier10_2, {}),
    (TTNIRClassifier10_3, {}),
    (TTNIRClassifier10_4, {}),
    (TTNIRClassifier10_5, {}),
])
def test_model_forward_shape_and_finite(ModelCls, kwargs) -> None:
    model = ModelCls(num_labels=5, chi=8, input_dim=1800, **kwargs)
    x = torch.randn(3, 1800)
    logits = model(x)
    assert logits.shape == (3, 5)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("ModelCls", [
    TTNIRClassifier10_2,
    TTNIRClassifier10_3,
    TTNIRClassifier10_4,
    TTNIRClassifier10_5,
])
def test_zero_spectrum_derivative_channels_not_dominated_by_pos_emb(ModelCls) -> None:
    """After the pos_emb fix: derivative channels (ch 1, ch 2) for zero input must
    come from the zero spectrum, not from the positional embedding.
    The un-normalised values must be (near) zero before max-abs scaling.
    We verify by checking that ch1 and ch2 in the feature_map output are zero
    when passing x=0 directly to the feature map (bypassing forward)."""
    model = ModelCls(num_labels=5, chi=8, input_dim=1800)
    model.eval()
    x_zero = torch.zeros(1, 1800)
    with torch.no_grad():
        fm_out = model.feature_map(x_zero)  # (1, 1800, 3)
    # Raw smoothed/SG output of zero spectrum must be zero
    assert fm_out[:, :, 1].abs().max().item() < 1e-5, "d1 channel not zero for zero input"
    assert fm_out[:, :, 2].abs().max().item() < 1e-5, "d2 channel not zero for zero input"


@pytest.mark.parametrize("ModelCls", [
    TTNIRClassifier10_2,
    TTNIRClassifier10_3,
    TTNIRClassifier10_4,
    TTNIRClassifier10_5,
])
def test_sigmoid_output_in_unit_interval(ModelCls) -> None:
    model = ModelCls(num_labels=5, chi=8, input_dim=1800)
    probs = model(torch.randn(2, 1800), apply_sigmoid=True)
    assert (probs >= 0).all() and (probs <= 1).all()
