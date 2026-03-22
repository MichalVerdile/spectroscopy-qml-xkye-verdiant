"""Compatibility wrapper for experiment 3 data loading helpers."""

from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment3.data_loader import (
    FUNCTIONAL_GROUPS,
    load_ir_data,
    prepare_dataloaders,
)

__all__ = [
    "FUNCTIONAL_GROUPS",
    "load_ir_data",
    "prepare_dataloaders",
]
