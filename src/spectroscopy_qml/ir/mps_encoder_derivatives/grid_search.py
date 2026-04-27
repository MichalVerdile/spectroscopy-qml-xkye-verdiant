import csv
import gc
from dataclasses import asdict
from itertools import product
from pathlib import Path

import torch

from spectroscopy_qml.ir.mps_encoder_derivatives.config import (
    DATA_CONFIG,
    GRID_SEARCH_CONFIG,
    MODEL_CONFIG,
    PATH_CONFIG,
    TRAINING_CONFIG,
)
from spectroscopy_qml.ir.mps_encoder_derivatives.data_loader import load_ir_data
from spectroscopy_qml.ir.mps_encoder_derivatives.train import train_model


def _set_attrs(target, updates: dict) -> None:
    """Set multiple attributes on a dataclass-like object."""
    for key, value in updates.items():
        setattr(target, key, value)


def _collect_grid() -> dict[str, tuple]:
    """Return all hyperparameter value lists for exhaustive search."""
    return {
        "num_sites": GRID_SEARCH_CONFIG.num_sites_values,
        "physical_dim": GRID_SEARCH_CONFIG.physical_dim_values,
        "bond_dim": GRID_SEARCH_CONFIG.bond_dim_values,
        "dropout_rate": GRID_SEARCH_CONFIG.dropout_rate_values,
        "batch_size": GRID_SEARCH_CONFIG.batch_size_values,
        "learning_rate": GRID_SEARCH_CONFIG.learning_rate_values,
        "weight_decay": GRID_SEARCH_CONFIG.weight_decay_values,
        "num_epochs": GRID_SEARCH_CONFIG.num_epochs_values,
        "patience": GRID_SEARCH_CONFIG.patience_values,
        "lr_scheduler_factor": GRID_SEARCH_CONFIG.lr_scheduler_factor_values,
        "lr_scheduler_patience": GRID_SEARCH_CONFIG.lr_scheduler_patience_values,
    }


def _extract_score(result: dict) -> float:
    """Extract ranking score from train_model result using configured metric."""
    metric_name = GRID_SEARCH_CONFIG.optimize_metric

    if metric_name.startswith("test_"):
        key = metric_name.replace("test_", "", 1)
        return float(result["test_metrics"][key])

    if metric_name.startswith("val_"):
        key = metric_name.replace("val_", "", 1)
        return float(result["best_val_metrics"][key])

    raise ValueError(
        "optimize_metric must start with 'test_' or 'val_', "
        f"got: {metric_name}"
    )


def run_grid_search() -> None:
    """Run exhaustive grid search over all configured hyperparameter combinations."""
    original_model_cfg = asdict(MODEL_CONFIG)
    original_training_cfg = asdict(TRAINING_CONFIG)
    original_path_cfg = asdict(PATH_CONFIG)

    grid = _collect_grid()
    keys = list(grid.keys())

    all_combinations = []
    for values in product(*(grid[key] for key in keys)):
        combo = dict(zip(keys, values))
        # Model constraint: input_dim must be divisible by num_sites.
        if MODEL_CONFIG.input_dim % combo["num_sites"] != 0:
            continue
        all_combinations.append(combo)

    total = len(all_combinations)
    if total == 0:
        raise RuntimeError("No valid hyperparameter combinations after constraints filtering.")

    grid_results_path = Path(GRID_SEARCH_CONFIG.grid_search_results_csv)
    grid_results_path.parent.mkdir(parents=True, exist_ok=True)

    # Single file that holds the best model found so far across all runs.
    # Every run writes its checkpoint here first; if it is not better it is
    # deleted immediately so only one .pt file ever lives on disk at a time.
    global_best_model_path = grid_results_path.parent / "best_model.pt"

    rows = []
    best_row = None

    print("=" * 80)
    print("Exhaustive Hyperparameter Grid Search")
    print("=" * 80)
    print(f"Total combinations to run: {total}")
    print(f"Optimization metric: {GRID_SEARCH_CONFIG.optimize_metric}")
    print("Running all combinations... this can take a long time.")

    # Load dataset once and reuse for all hyperparameter combinations.
    print("\nLoading dataset once for all runs...")
    data_dir = Path(PATH_CONFIG.data_dir)
    if not data_dir.exists():
        project_root = Path(__file__).parents[4]
        data_dir = project_root / "data" / "raw"
        print(f"Processed data not found, using raw data from: {data_dir}")

    X, y = load_ir_data(
        data_dir,
        target_length=DATA_CONFIG.target_length,
        max_files=DATA_CONFIG.max_files,
        apply_savgol=DATA_CONFIG.apply_savgol,
        savgol_window_length=DATA_CONFIG.savgol_window_length,
        savgol_polyorder=DATA_CONFIG.savgol_polyorder,
        apply_snv=DATA_CONFIG.apply_snv,
    )
    print(f"Dataset loaded once: X shape={X.shape}, y shape={y.shape}")

    try:
        for idx, combo in enumerate(all_combinations, start=1):
            run_id = f"run_{idx:05d}"
            print("\n" + "-" * 80)
            print(f"[{idx}/{total}] {run_id}")
            print(f"Params: {combo}")
            print("-" * 80)

            model_updates = {
                "num_sites": combo["num_sites"],
                "physical_dim": combo["physical_dim"],
                "bond_dim": combo["bond_dim"],
                "dropout_rate": combo["dropout_rate"],
            }
            training_updates = {
                "batch_size": combo["batch_size"],
                "learning_rate": combo["learning_rate"],
                "weight_decay": combo["weight_decay"],
                "num_epochs": combo["num_epochs"],
                "patience": combo["patience"],
                "lr_scheduler_factor": combo["lr_scheduler_factor"],
                "lr_scheduler_patience": combo["lr_scheduler_patience"],
            }

            # Results (logs, summary) are kept per-run; the model checkpoint
            # is written to a temporary path and promoted or deleted below.
            run_temp_model_path = grid_results_path.parent / f"_temp_{run_id}.pt"
            run_results_dir = Path(original_path_cfg["results_dir"]) / "grid_search" / run_id
            run_results_dir.mkdir(parents=True, exist_ok=True)

            _set_attrs(MODEL_CONFIG, model_updates)
            _set_attrs(TRAINING_CONFIG, training_updates)
            _set_attrs(
                PATH_CONFIG,
                {
                    "model_dir": str(grid_results_path.parent),
                    "results_dir": str(run_results_dir),
                    "best_model_path": str(run_temp_model_path),
                    "summary_path": str(run_results_dir / "summary.txt"),
                    "training_log_path": str(run_results_dir / "training_log.csv"),
                },
            )

            result = train_model(X=X, y=y)
            # train_model already cleans up its own objects; do a second pass
            # here to ensure the grid-search process doesn't accumulate memory.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            score = _extract_score(result)

            row = {
                "run_id": run_id,
                "score": score,
                "best_epoch": result["best_epoch"],
                "num_epochs_trained": result["num_epochs_trained"],
                "best_val_loss": result["best_val_loss"],
                "test_loss": result["test_loss"],
                "test_f1_micro": result["test_metrics"]["f1_micro"],
                "test_f1_macro": result["test_metrics"]["f1_macro"],
                "test_precision_micro": result["test_metrics"]["precision_micro"],
                "test_recall_micro": result["test_metrics"]["recall_micro"],
                "val_f1_micro": result["best_val_metrics"]["f1_micro"],
                "val_f1_macro": result["best_val_metrics"]["f1_macro"],
                "model_path": str(global_best_model_path),
                "summary_path": result["summary_path"],
                "training_log_path": result["training_log_path"],
            }
            row.update(combo)
            rows.append(row)

            is_new_best = best_row is None or (
                row["score"] > best_row["score"]
                if GRID_SEARCH_CONFIG.maximize_metric
                else row["score"] < best_row["score"]
            )

            if is_new_best:
                # Promote temp checkpoint to the single global best file.
                if global_best_model_path.exists():
                    global_best_model_path.unlink()
                run_temp_model_path.rename(global_best_model_path)
                best_row = row
                print(f"  ★ New global best – model saved to {global_best_model_path}")
            else:
                # Not better: delete checkpoint immediately to free disk space.
                if run_temp_model_path.exists():
                    run_temp_model_path.unlink()

            # Persist intermediate leaderboard after each run.
            sorted_rows = sorted(
                rows,
                key=lambda r: r["score"],
                reverse=GRID_SEARCH_CONFIG.maximize_metric,
            )
            fieldnames = list(sorted_rows[0].keys())
            with open(grid_results_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(sorted_rows)

            print(f"Completed {run_id} with score={score:.6f}")

    finally:
        _set_attrs(MODEL_CONFIG, original_model_cfg)
        _set_attrs(TRAINING_CONFIG, original_training_cfg)
        _set_attrs(PATH_CONFIG, original_path_cfg)

    summary_path = Path(GRID_SEARCH_CONFIG.grid_search_summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("Exhaustive Hyperparameter Grid Search Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total combinations executed: {total}\n")
        f.write(f"Optimization metric: {GRID_SEARCH_CONFIG.optimize_metric}\n")
        f.write(f"Results CSV: {grid_results_path}\n\n")

        if best_row is not None:
            f.write("Best run:\n")
            f.write(f"  run_id: {best_row['run_id']}\n")
            f.write(f"  score: {best_row['score']:.6f}\n")
            f.write(f"  test_f1_micro: {best_row['test_f1_micro']:.6f}\n")
            f.write(f"  test_f1_macro: {best_row['test_f1_macro']:.6f}\n")
            f.write(f"  val_f1_micro: {best_row['val_f1_micro']:.6f}\n")
            f.write(f"  val_f1_macro: {best_row['val_f1_macro']:.6f}\n\n")

            f.write("Best hyperparameters:\n")
            for key in keys:
                f.write(f"  {key}: {best_row[key]}\n")

    print("\n" + "=" * 80)
    print("Grid search completed")
    print(f"Results leaderboard: {grid_results_path}")
    print(f"Summary: {summary_path}")
    if best_row is not None:
        print(f"Best run: {best_row['run_id']} (score={best_row['score']:.6f})")
    print("=" * 80)


if __name__ == "__main__":
    run_grid_search()
