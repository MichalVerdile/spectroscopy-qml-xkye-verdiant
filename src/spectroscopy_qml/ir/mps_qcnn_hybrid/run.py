import argparse
import sys
from pathlib import Path

src_dir = Path(__file__).parents[3]
sys.path.insert(0, str(src_dir))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Frozen MPS + QCNN specialist experiment for rare or hard IR functional groups",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional limit on parquet files to load for quick debugging.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override the IR data directory.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Override the trained MPS checkpoint path.",
    )
    args = parser.parse_args()

    from spectroscopy_qml.ir.mps_qcnn_hybrid.train import run_hybrid_experiment

    result = run_hybrid_experiment(
        max_files=args.max_files,
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint,
    )
    print("Hybrid experiment completed.")
    print(f"Summary: {result['summary_path']}")
    print(f"Metrics: {result['metrics']['results']}")


if __name__ == "__main__":
    main()
