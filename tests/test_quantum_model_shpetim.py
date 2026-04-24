from __future__ import annotations

import torch

from spectroscopy_qml.ir.qml.quantum_model_shpetim import (
    ALL_FUNCTIONAL_GROUP_NAMES,
    AngleEncodingClassifier,
    N_SPECIALIST_CLASSES,
    TTN10_2_HARD_CLASS_INDICES,
    TTN10_2_HARD_CLASS_NAMES,
    select_specialist_labels,
)


def test_shpetim_model_defaults_to_ttn10_2_hard_classes() -> None:
    model = AngleEncodingClassifier(n_layers=1)

    assert model.num_classes == 10
    assert model.specialist_indices == TTN10_2_HARD_CLASS_INDICES
    assert model.specialist_class_names == TTN10_2_HARD_CLASS_NAMES
    assert model.head[-1].out_features == 10


def test_select_specialist_labels_projects_full_ir_targets() -> None:
    labels = torch.arange(2 * len(ALL_FUNCTIONAL_GROUP_NAMES), dtype=torch.float32).reshape(
        2, len(ALL_FUNCTIONAL_GROUP_NAMES)
    )

    specialist = select_specialist_labels(labels)

    assert specialist.shape == (2, N_SPECIALIST_CLASSES)
    expected = labels[:, list(TTN10_2_HARD_CLASS_INDICES)]
    assert torch.equal(specialist, expected)
