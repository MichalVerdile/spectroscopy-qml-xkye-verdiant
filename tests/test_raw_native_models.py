import copy

import numpy as np
import pytest
import torch

from spectroscopy_qml.benchmarking.raw_models import NativeCNN, NativeMPS, NativeTTN
from spectroscopy_qml.benchmarking.raw_models.neural import SegmentEncoder, build_model
from spectroscopy_qml.benchmarking.raw_training import seed_everything


@pytest.mark.parametrize("model", [NativeCNN, NativeMPS, NativeTTN])
@pytest.mark.parametrize("modality", ["cnmr", "hnmr", "ir", "msms_pos", "msms_neg"])
def test_forward_backward_finite_on_native_shapes(model, modality):
    torch.set_num_threads(2)
    seed_everything(42)
    length = 10000 if modality in {"cnmr", "hnmr"} else (1800 if modality == "ir" else 19)
    channels = 3 if modality.startswith("msms") else 2
    x = torch.randn(2, length, channels)
    x[..., -1] = 1
    if channels == 3:
        x[0, 7:] = 0
        x[1] = 0  # Empty observed peak list is supported and remains finite.
    network = model(length, channels, width=12, physical_dim=8, bond_dim=8, chi=8)
    logits = network(x)
    assert logits.shape == (2, 37)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in network.parameters() if p.grad is not None)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in network.parameters())


@pytest.mark.parametrize("mapping", ["small", "legacy_mlp", "angle", "product"])
def test_feature_map_covers_tail_and_all_channels(mapping):
    encoder = SegmentEncoder(37, 3, 8, window=8, stride=5, mapping=mapping, dropout=0)
    x = torch.randn(2, 37, 3, requires_grad=True)
    with torch.no_grad():
        x[..., -1] = 1
    h, mask = encoder(x)
    (h[..., 0].sum()).backward()
    assert mask.all()
    assert (x.grad[..., :-1].abs().sum(dim=(0, 2)) > 0).all()


@pytest.mark.parametrize("mode", ["strict", "relaxed"])
def test_ttn_odd_tree_and_reproducibility(mode):
    seed_everything(44)
    x = torch.randn(2, 37, 2)
    x[..., -1] = 1
    net = NativeTTN(37, 2, window=8, overlap=0, chi=8, merge_mode=mode, dropout=0)
    net.eval()
    a = net(x)
    b = copy.deepcopy(net)(x)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    a.sum().backward()
    if mode == "strict":
        q = torch.linalg.qr(net.merges[0].weight)[0]
        torch.testing.assert_close(q.T @ q, torch.eye(8), atol=1e-5, rtol=1e-5)


def test_historical_ttn_not_mislabelled_as_raw_and_large_models_not_silently_shrunk():
    with pytest.raises(ValueError, match="Lorentz"):
        build_model("ttn", 1800, 2, {}, variant="baseline")
    with pytest.raises(ValueError, match="Historical CNN"):
        build_model("cnn", 10000, 2, {}, variant="baseline")


@pytest.mark.parametrize("physical", [8, 16, 32, 64])
@pytest.mark.parametrize("bond", [16, 32, 64, 128])
def test_requested_mps_dimension_grid_is_trainable(physical, bond):
    torch.set_num_threads(2)
    x = torch.randn(2, 37, 2)
    x[..., -1] = 1
    model = NativeMPS(37, 2, physical_dim=physical, bond_dim=bond, window=16)
    loss = model(x).square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@pytest.mark.parametrize("chi", [16, 32, 64, 128])
@pytest.mark.parametrize("mode", ["strict", "relaxed"])
def test_requested_ttn_chi_grid_is_trainable(chi, mode):
    torch.set_num_threads(2)
    x = torch.randn(2, 37, 2)
    x[..., -1] = 1
    model = NativeTTN(37, 2, chi=chi, window=16, merge_mode=mode)
    loss = model(x).square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
