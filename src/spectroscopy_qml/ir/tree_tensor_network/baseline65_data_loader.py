"""Compatibility wrapper for experiment 1 baseline65 data loading helpers."""

from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment1.baseline65_data_loader import (
    FUNCTIONAL_GROUPS,
    IRSpectraDataset,
    load_ir_data,
    prepare_dataloaders,
)

__all__ = [
    "FUNCTIONAL_GROUPS",
    "IRSpectraDataset",
    "load_ir_data",
    "prepare_dataloaders",
]
