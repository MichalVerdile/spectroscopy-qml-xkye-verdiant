"""
Quick start script for MPS Functional Group Classifier.

This script provides a simple interface to train and evaluate the model.

Usage:
    Install the package first: pip install -e .
    Or set PYTHONPATH: export PYTHONPATH=src
"""

import argparse


def main():
    parser = argparse.ArgumentParser(
        description="MPS Functional Group Classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train the model
  python run.py --train

  # Evaluate the model
  python run.py --evaluate

  # Train and then evaluate
  python run.py --train --evaluate
        """,
    )

    parser.add_argument("--train", action="store_true", help="Train the model")

    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")

    parser.add_argument(
        "--check", action="store_true", help="Check model architecture and dependencies"
    )

    args = parser.parse_args()

    # If no arguments, show help
    if not (args.train or args.evaluate or args.check):
        parser.print_help()
        return

    # Check dependencies and model
    if args.check:
        print("=" * 80)
        print("Checking Dependencies and Model")
        print("=" * 80)

        try:
            import torch

            print(f"✓ PyTorch {torch.__version__}")
            print(f"  CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"  CUDA version: {torch.version.cuda}")
                print(f"  GPU: {torch.cuda.get_device_name(0)}")
        except ImportError as e:
            print(f"✗ PyTorch not found: {e}")
            return

        try:
            import numpy

            print(f"✓ NumPy {numpy.__version__}")
        except ImportError as e:
            print(f"✗ NumPy not found: {e}")
            return

        try:
            import pandas

            print(f"✓ Pandas {pandas.__version__}")
        except ImportError as e:
            print(f"✗ Pandas not found: {e}")
            return

        try:
            import sklearn

            print(f"✓ Scikit-learn {sklearn.__version__}")
        except ImportError as e:
            print(f"✗ Scikit-learn not found: {e}")
            return

        try:
            import rdkit

            print(f"✓ RDKit {rdkit.__version__}")
        except ImportError as e:
            print(f"✗ RDKit not found: {e}")
            return

        # Check model
        try:
            from spectroscopy_qml.mps_encoder import MODEL_CONFIG, MPSFunctionalGroupClassifier

            model = MPSFunctionalGroupClassifier()
            params = model.get_num_parameters()
            print("\n✓ Model loaded successfully")
            print(f"  Total parameters: {params:,}")

            # Test forward pass
            test_input = torch.randn(2, MODEL_CONFIG.input_dim)
            output = model(test_input)
            print(f"  Test forward pass: {test_input.shape} → {output.shape}")
            print("✓ All checks passed!")
        except Exception as e:
            print(f"✗ Model check failed: {e}")
            import traceback

            traceback.print_exc()
            return

    # Train
    if args.train:
        print("\n" + "=" * 80)
        print("Starting Training")
        print("=" * 80 + "\n")

        try:
            from spectroscopy_qml.mps_encoder.train import train_model

            train_model()
        except Exception as e:
            print(f"Training failed: {e}")
            import traceback

            traceback.print_exc()
            return

    # Evaluate
    if args.evaluate:
        print("\n" + "=" * 80)
        print("Starting Evaluation")
        print("=" * 80 + "\n")

        try:
            # Call evaluate with default args
            import click

            from spectroscopy_qml.mps_encoder.evaluate import main as evaluate_main

            ctx = click.Context(evaluate_main)
            ctx.invoke(evaluate_main)
        except Exception as e:
            print(f"Evaluation failed: {e}")
            import traceback

            traceback.print_exc()
            return

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
