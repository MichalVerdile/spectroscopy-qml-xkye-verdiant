"""Run focused hyperparameter sweeps for experiment 6."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[5]
TRAIN_SCRIPT = CURRENT_DIR / "train.py"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep experiment 6 hyperparameters to improve F1."
    )
    parser.add_argument("--preset", choices=["coarse", "fine"], default="coarse")
    parser.add_argument(
        "--base-output-dir",
        type=Path,
        default=Path("src/spectroscopy_qml/ir/tree_tensor_network/experiment/experiment6/results/sweeps"),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--split-path", type=Path, default=None)
    parser.add_argument("--apply-snv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--min-epochs-before-stopping", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--learning-rates", nargs="+", type=float, default=[5e-4, 1e-3, 2e-3])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1024, 2048, 4096])
    parser.add_argument("--weight-decays", nargs="+", type=float, default=[1e-6, 1e-5])
    parser.add_argument("--leaf-dropouts", nargs="+", type=float, default=[0.0, 0.1])
    parser.add_argument("--readout-dropouts", nargs="+", type=float, default=[0.0, 0.1])
    parser.add_argument("--merge-residual-weights", nargs="+", type=float, default=[0.1, 0.15, 0.2])
    parser.add_argument("--threshold-grid-steps", nargs="+", type=float, default=[0.05])
    parser.add_argument("--chis", nargs="+", type=int, default=[64])
    parser.add_argument("--segment-window-sizes", nargs="+", type=int, default=[64])
    parser.add_argument("--segment-strides", nargs="+", type=int, default=[32])
    parser.add_argument(
        "--ranking-metric",
        choices=["best_early_stop_score", "test_f1_micro", "test_f1_macro"],
        default="test_f1_micro",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def format_float(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def normalize_summary_key(key: str) -> str:
    return key.strip().lower().replace(" ", "_").replace("-", "_")


def build_run_name(config: dict[str, object]) -> str:
    return (
        f"lr{format_float(float(config['learning_rate']))}"
        f"_bs{int(config['batch_size'])}"
        f"_wd{format_float(float(config['weight_decay']))}"
        f"_ld{format_float(float(config['leaf_dropout']))}"
        f"_rd{format_float(float(config['readout_dropout']))}"
        f"_mrw{format_float(float(config['merge_residual_weight']))}"
        f"_tgs{format_float(float(config['threshold_grid_step']))}"
        f"_chi{int(config['chi'])}"
        f"_win{int(config['segment_window_size'])}"
        f"_stride{int(config['segment_stride'])}"
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
        "--epochs",
        str(args.epochs),
        "--early-stopping-patience",
        str(args.early_stopping_patience),
        "--min-epochs-before-stopping",
        str(args.min_epochs_before_stopping),
        "--num-workers",
        str(args.num_workers),
        "--seed",
        str(args.seed),
        "--batch-size",
        str(config["batch_size"]),
        "--learning-rate",
        str(config["learning_rate"]),
        "--weight-decay",
        str(config["weight_decay"]),
        "--leaf-dropout",
        str(config["leaf_dropout"]),
        "--readout-dropout",
        str(config["readout_dropout"]),
        "--merge-residual-weight",
        str(config["merge_residual_weight"]),
        "--threshold-grid-step",
        str(config["threshold_grid_step"]),
        "--chi",
        str(config["chi"]),
        "--segment-window-size",
        str(config["segment_window_size"]),
        "--segment-stride",
        str(config["segment_stride"]),
    ]
    command.append("--apply-snv" if args.apply_snv else "--no-apply-snv")

    if args.cache_path is not None:
        command.extend(["--cache-path", str(args.cache_path)])
    if args.split_path is not None:
        command.extend(["--split-path", str(args.split_path)])
    if args.max_files is not None:
        command.extend(["--max-files", str(args.max_files)])
    if args.amp is not None:
        command.append("--amp" if args.amp else "--no-amp")
    if args.compile is not None:
        command.append("--compile" if args.compile else "--no-compile")
    return command


def parse_summary(summary_path: Path) -> dict[str, str]:
    metrics: dict[str, str] = {}
    if not summary_path.exists():
        return metrics

    for line in summary_path.read_text().splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metrics[normalize_summary_key(key)] = value.strip()
    return metrics


def iter_configs(args: argparse.Namespace) -> list[dict[str, object]]:
    configs = [
        {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "leaf_dropout": leaf_dropout,
            "readout_dropout": readout_dropout,
            "merge_residual_weight": merge_residual_weight,
            "threshold_grid_step": threshold_grid_step,
            "chi": chi,
            "segment_window_size": segment_window_size,
            "segment_stride": segment_stride,
        }
        for learning_rate, batch_size, weight_decay, leaf_dropout, readout_dropout, merge_residual_weight, threshold_grid_step, chi, segment_window_size, segment_stride in itertools.product(
            args.learning_rates,
            args.batch_sizes,
            args.weight_decays,
            args.leaf_dropouts,
            args.readout_dropouts,
            args.merge_residual_weights,
            args.threshold_grid_steps,
            args.chis,
            args.segment_window_sizes,
            args.segment_strides,
        )
    ]
    if args.limit is not None:
        configs = configs[: args.limit]
    return configs


def build_run_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    if args.device == "mps":
        env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    return env


def apply_preset(args: argparse.Namespace) -> None:
    if args.preset != "fine":
        return

    args.learning_rates = [4e-4, 5e-4, 6e-4]
    args.batch_sizes = [1024]
    args.weight_decays = [1e-6, 3e-6, 1e-5]
    args.leaf_dropouts = [0.05, 0.1, 0.15]
    args.readout_dropouts = [0.0, 0.05]
    args.merge_residual_weights = [0.1, 0.15]
    args.threshold_grid_steps = [0.02]
    args.chis = [64]
    args.segment_window_sizes = [64]
    args.segment_strides = [32]
    if args.limit is None:
        args.limit = 36


def parse_float_metric(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def write_best_run_artifacts(
    base_output_dir: Path,
    ranking_metric: str,
    rows: list[dict[str, str]],
) -> None:
    successful_rows = [row for row in rows if row.get("status") == "ok"]
    if not successful_rows:
        return

    best_row = max(successful_rows, key=lambda row: parse_float_metric(row, ranking_metric))
    best_json_path = base_output_dir / "best_run.json"
    best_txt_path = base_output_dir / "best_run.txt"
    best_json_path.write_text(json.dumps(best_row, indent=2) + "\n")

    lines = [
        f"Ranking metric: {ranking_metric}",
        f"Run name:       {best_row['run_name']}",
        f"Status:         {best_row['status']}",
        f"Best epoch:     {best_row.get('best_epoch', '')}",
        f"Best score:     {best_row.get('best_early_stop_score', '')}",
        f"Test f1_micro:  {best_row.get('test_f1_micro', '')}",
        f"Test f1_macro:  {best_row.get('test_f1_macro', '')}",
        f"Output dir:     {best_row.get('output_dir', '')}",
        f"Command:        {best_row.get('command', '')}",
    ]
    best_txt_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = build_parser().parse_args()
    apply_preset(args)
    args.base_output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.base_output_dir / "sweep_results.csv"
    configs = iter_configs(args)
    run_env = build_run_env(args)
    all_rows: list[dict[str, str]] = []

    print(f"Planned runs: {len(configs)}")

    with results_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_name",
                "status",
                "learning_rate",
                "batch_size",
                "weight_decay",
                "leaf_dropout",
                "readout_dropout",
                "merge_residual_weight",
                "threshold_grid_step",
                "chi",
                "segment_window_size",
                "segment_stride",
                "best_epoch",
                "best_early_stop_score",
                "best_val_loss",
                "test_f1_micro",
                "test_f1_macro",
                "test_precision_micro",
                "test_recall_micro",
                "elapsed_seconds",
                "output_dir",
                "command",
            ],
        )
        writer.writeheader()

        for index, config in enumerate(configs, start=1):
            run_name = build_run_name(config)
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
                completed = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    check=False,
                    env=run_env,
                )
                status = "ok" if completed.returncode == 0 else f"failed_{completed.returncode}"

            summary = parse_summary(summary_path)
            writer.writerow(
                {
                    "run_name": run_name,
                    "status": status,
                    "learning_rate": config["learning_rate"],
                    "batch_size": config["batch_size"],
                    "weight_decay": config["weight_decay"],
                    "leaf_dropout": config["leaf_dropout"],
                    "readout_dropout": config["readout_dropout"],
                    "merge_residual_weight": config["merge_residual_weight"],
                    "threshold_grid_step": config["threshold_grid_step"],
                    "chi": config["chi"],
                    "segment_window_size": config["segment_window_size"],
                    "segment_stride": config["segment_stride"],
                    "best_epoch": summary.get("best_epoch", ""),
                    "best_early_stop_score": summary.get("best_early_stop_score", ""),
                    "best_val_loss": summary.get("best_val_loss", ""),
                    "test_f1_micro": summary.get("test_f1_micro", ""),
                    "test_f1_macro": summary.get("test_f1_macro", ""),
                    "test_precision_micro": summary.get("test_precision_micro", ""),
                    "test_recall_micro": summary.get("test_recall_micro", ""),
                    "elapsed_seconds": summary.get("elapsed_seconds", ""),
                    "output_dir": str(output_dir),
                    "command": " ".join(command),
                }
            )
            all_rows.append(
                {
                    "run_name": run_name,
                    "status": status,
                    "learning_rate": str(config["learning_rate"]),
                    "batch_size": str(config["batch_size"]),
                    "weight_decay": str(config["weight_decay"]),
                    "leaf_dropout": str(config["leaf_dropout"]),
                    "readout_dropout": str(config["readout_dropout"]),
                    "merge_residual_weight": str(config["merge_residual_weight"]),
                    "threshold_grid_step": str(config["threshold_grid_step"]),
                    "chi": str(config["chi"]),
                    "segment_window_size": str(config["segment_window_size"]),
                    "segment_stride": str(config["segment_stride"]),
                    "best_epoch": summary.get("best_epoch", ""),
                    "best_early_stop_score": summary.get("best_early_stop_score", ""),
                    "best_val_loss": summary.get("best_val_loss", ""),
                    "test_f1_micro": summary.get("test_f1_micro", ""),
                    "test_f1_macro": summary.get("test_f1_macro", ""),
                    "test_precision_micro": summary.get("test_precision_micro", ""),
                    "test_recall_micro": summary.get("test_recall_micro", ""),
                    "elapsed_seconds": summary.get("elapsed_seconds", ""),
                    "output_dir": str(output_dir),
                    "command": " ".join(command),
                }
            )
            handle.flush()

    write_best_run_artifacts(args.base_output_dir, args.ranking_metric, all_rows)
    print(f"Results written to {results_path}")


if __name__ == "__main__":
    main()
