"""Compatibility wrapper for experiment 2 legacy data loading helpers."""

from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment2.legacy_data_loader import (
    FUNCTIONAL_GROUPS,
    load_ir_data,
    prepare_dataloaders,
)

__all__ = [
    "FUNCTIONAL_GROUPS",
    "load_ir_data",
    "prepare_dataloaders",
]
