#!/usr/bin/env python
"""Quick test script to verify the training pipeline works."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from torch.utils.data import DataLoader

from spectroscopy_qml.data import IRFunctionalGroupDataset, create_data_splits
from spectroscopy_qml.models import MPSEncoder
from spectroscopy_qml.training import (
    FunctionalGroupClassifier,
    TrainingConfig,
    train_model,
)


def main() -> None:
    print("Quick training test...")
    dataset = IRFunctionalGroupDataset("data/raw", target_length=512, max_chunks=1)
    train, val, test = create_data_splits(dataset)

    train_loader = DataLoader(train, batch_size=32, shuffle=True)
    val_loader = DataLoader(val, batch_size=32)

    # Test with MPS encoder
    encoder = MPSEncoder(
        input_length=512, num_sites=64, physical_dim=4, bond_dim=8, embedding_dim=128
    )
    model = FunctionalGroupClassifier(encoder, num_classes=37, embedding_dim=128)

    config = TrainingConfig(
        learning_rate=1e-3,
        batch_size=32,
        num_epochs=3,  # Just 3 epochs for testing
        patience=10,
        log_interval=1,
    )

    pos_weight = dataset.get_label_weights()
    metrics = train_model(model, train_loader, val_loader, config, pos_weight)

    print(f"\nFinal train loss: {metrics.train_losses[-1]:.4f}")
    print(f"Final val F1 micro: {metrics.val_f1_micro[-1]:.4f}")
    print("\nTraining pipeline test passed!")


if __name__ == "__main__":
    main()
