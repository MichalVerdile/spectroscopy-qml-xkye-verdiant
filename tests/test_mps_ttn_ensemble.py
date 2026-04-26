from __future__ import annotations

import numpy as np

from spectroscopy_qml.ir.MPS_TTN.ensemble_ttn102 import (
    blend_probabilities,
    select_override_indices,
)


def test_select_override_indices_respects_gain_support_and_candidates() -> None:
    mps = np.array([0.80, 0.60, 0.20, 0.70], dtype=np.float32)
    ttn = np.array([0.84, 0.61, 0.40, 0.90], dtype=np.float32)
    labels = np.array(
        [
            [1, 1, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.int32,
    )

    selected = select_override_indices(
        mps_per_class_f1=mps,
        ttn_per_class_f1=ttn,
        val_labels=labels,
        candidate_indices=[0, 1, 2, 3],
        min_f1_gain=0.03,
        min_support=1,
    )

    assert selected == [0, 3]

    selected_with_support = select_override_indices(
        mps_per_class_f1=mps,
        ttn_per_class_f1=ttn,
        val_labels=labels,
        candidate_indices=[0, 1, 2, 3],
        min_f1_gain=0.03,
        min_support=2,
    )
    assert selected_with_support == [0, 3]


def test_blend_probabilities_overrides_only_selected_columns() -> None:
    base = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    override = np.array([[0.9, 0.8, 0.7], [0.6, 0.5, 0.4]], dtype=np.float32)

    blended = blend_probabilities(base_probs=base, override_probs=override, override_indices=[0, 2])

    expected = np.array([[0.9, 0.2, 0.7], [0.6, 0.5, 0.4]], dtype=np.float32)
    assert np.allclose(blended, expected)
