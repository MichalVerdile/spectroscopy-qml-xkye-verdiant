import torch

from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment10.model import (
    FastDirectIsometricMerge,
    TTNIRClassifier10,
)


def test_experiment10_forward_shapes() -> None:
    model = TTNIRClassifier10(num_labels=5, chi=8, input_dim=1800)
    x = torch.randn(3, 1800)
    logits = model(x)
    assert logits.shape == (3, 5)
    assert torch.isfinite(logits).all()


def test_experiment10_has_no_leaf_encoder_and_direct_first_merge() -> None:
    model = TTNIRClassifier10(num_labels=5, chi=64, input_dim=1800, segment_window_size=48, segment_stride=43)
    assert not hasattr(model, "leaf_encoder")
    assert model.segment_state_dim == 144
    first_merge = model.merge_levels[0]
    assert isinstance(first_merge, FastDirectIsometricMerge)
    assert first_merge.input_dim == 144
    assert first_merge.output_dim == 64
    assert first_merge.raw_isometry.shape == (144 * 144, 64)
