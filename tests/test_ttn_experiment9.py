from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment9.model import (
    DirectSegmentLeafEncoder,
    TTNIRClassifier9,
)

import torch


def test_experiment9_forward_shapes() -> None:
    model = TTNIRClassifier9(num_labels=5, chi=8, input_dim=1800)
    x = torch.randn(3, 1800)
    logits = model(x)
    assert logits.shape == (3, 5)
    assert torch.isfinite(logits).all()


def test_direct_segment_leaf_encoder_has_no_hidden_projection() -> None:
    encoder = DirectSegmentLeafEncoder(max_segment_length=48, chi=64, feature_dim=3)
    assert encoder.proj.weight.shape == (64, 144)
    assert encoder.skip.weight.shape == (64, 144)
    assert hasattr(encoder, "activation")
    assert not hasattr(encoder, "fc1")
    assert not hasattr(encoder, "fc2")
