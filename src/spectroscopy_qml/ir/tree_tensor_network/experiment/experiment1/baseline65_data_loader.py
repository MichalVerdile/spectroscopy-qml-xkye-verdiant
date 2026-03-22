"""Data-loading entry points for the restored 0.65 TTN baseline."""

from spectroscopy_qml.ir.mps_encoder.data_loader import (
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
