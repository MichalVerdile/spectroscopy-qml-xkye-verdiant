"""Run focused TTN sweeps with segment settings prioritized before chi."""

from __future__ import annotations

import argparse
import csv
import itertools
import subprocess
import sys
from pathlib import Path


SRC_DIR = Path(__file__).parents[3]
REPO_ROOT = SRC_DIR.parent
TRAIN_SCRIPT = SRC_DIR / "spectroscopy_qml" / "ir" / "tree_tensor_network" / "train.py"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep TTN segment settings before expanding chi."
    )
    parser.add_argument(
        "--base-output-dir",
        type=Path,
        default=Path("src/spectroscopy_qml/ir/tree_tensor_network/sweeps"),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--learning-rates", nargs="+", type=float, default=[3e-4, 5e-4])
    parser.add_argument("--chis", nargs="+", type=int, default=[64])
    parser.add_argument("--segment-window-sizes", nargs="+", type=int, default=[32, 64, 96])
    parser.add_argument("--segment-strides", nargs="+", type=int, default=[16, 32, 48])
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument(
        "--threshold-metric",
        choices=["f1_micro", "f1_macro", "per_class_f1"],
        default="f1_micro",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def format_float(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def build_run_name(window_size: int, stride: int, chi: int, learning_rate: float) -> str:
    return (
        f"win{window_size}"
        f"_stride{stride}"
        f"_chi{chi}"
        f"_lr{format_float(learning_rate)}"
    )


def build_command(args: argparse.Namespace, output_dir: Path, config: dict[str, object]) -> list[str]:
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--data-dir",
        str(args.data_dir),
        "--output-dir",
        str(output_dir),
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--early-stopping-patience",
        str(args.early_stopping_patience),
        "--learning-rate",
        str(config["learning_rate"]),
        "--chi",
        str(config["chi"]),
        "--segment-window-size",
        str(config["segment_window_size"]),
        "--segment-stride",
        str(config["segment_stride"]),
        "--weight-decay",
        str(args.weight_decay),
        "--threshold-metric",
        args.threshold_metric,
        "--seed",
        str(args.seed),
    ]
    if args.max_files is not None:
        command.extend(["--max-files", str(args.max_files)])
    return command


def parse_summary(summary_path: Path) -> dict[str, str]:
    metrics: dict[str, str] = {}
    if not summary_path.exists():
        return metrics

    for line in summary_path.read_text().splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metrics[key.strip().lower().replace(" ", "_")] = value.strip()
    return metrics


def iter_segment_configs(args: argparse.Namespace) -> list[tuple[int, int]]:
    if len(args.segment_window_sizes) != len(args.segment_strides):
        raise ValueError("segment_window_sizes and segment_strides must have the same length.")
    return list(zip(args.segment_window_sizes, args.segment_strides, strict=True))


def iter_configs(args: argparse.Namespace) -> list[dict[str, object]]:
    configs = [
        {
            "segment_window_size": window_size,
            "segment_stride": stride,
            "chi": chi,
            "learning_rate": learning_rate,
        }
        for (window_size, stride), chi, learning_rate in itertools.product(
            iter_segment_configs(args),
            args.chis,
            args.learning_rates,
        )
    ]
    if args.limit is not None:
        configs = configs[: args.limit]
    return configs


def main() -> None:
    args = build_parser().parse_args()
    args.base_output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.base_output_dir / "sweep_results.csv"

    configs = iter_configs(args)
    print(f"Planned runs: {len(configs)}")

    with results_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_name",
                "status",
                "segment_window_size",
                "segment_stride",
                "chi",
                "learning_rate",
                "best_epoch",
                "best_val_f1",
                "best_val_loss",
                "test_f1_micro",
                "elapsed_seconds",
                "output_dir",
                "command",
            ],
        )
        writer.writeheader()

        for index, config in enumerate(configs, start=1):
            run_name = build_run_name(
                window_size=int(config["segment_window_size"]),
                stride=int(config["segment_stride"]),
                chi=int(config["chi"]),
                learning_rate=float(config["learning_rate"]),
            )
            output_dir = args.base_output_dir / run_name
            summary_path = output_dir / "summary.txt"
            command = build_command(args, output_dir, config)

            print(f"[{index}/{len(configs)}] {run_name}")

            status = "dry_run"
            if summary_path.exists() and not args.overwrite:
                print("  Skipping existing run")
                status = "skipped"
            elif args.dry_run:
                print(f"  {' '.join(command)}")
            else:
                output_dir.mkdir(parents=True, exist_ok=True)
                completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
                status = "ok" if completed.returncode == 0 else f"failed_{completed.returncode}"

            summary = parse_summary(summary_path)
            writer.writerow(
                {
                    "run_name": run_name,
                    "status": status,
                    "segment_window_size": config["segment_window_size"],
                    "segment_stride": config["segment_stride"],
                    "chi": config["chi"],
                    "learning_rate": config["learning_rate"],
                    "best_epoch": summary.get("best_epoch", ""),
                    "best_val_f1": summary.get("best_val_f1", ""),
                    "best_val_loss": summary.get("best_val_loss", ""),
                    "test_f1_micro": summary.get("f1_micro", ""),
                    "elapsed_seconds": summary.get("elapsed_seconds", ""),
                    "output_dir": str(output_dir),
                    "command": " ".join(command),
                }
            )
            handle.flush()

    print(f"Results written to {results_path}")


if __name__ == "__main__":
    main()
