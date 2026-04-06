from __future__ import annotations

from pathlib import Path

from spectroscopy_qml.ir.tree_tensor_network.experiment.experiment6.sweep import (
    REPO_ROOT,
    TRAIN_SCRIPT,
    apply_preset,
    build_command,
    build_run_name,
    iter_configs,
    iter_segment_layouts,
    parse_summary,
    write_best_run_artifacts,
)


class _Args:
    data_dir = Path("data/raw")
    cache_path = Path("data/cache/ir_spectra_len1800_snv_files1.npz")
    split_path = Path("results/split.npz")
    apply_snv = True
    device = "mps"
    amp = False
    compile = False
    epochs = 10
    early_stopping_patience = 4
    min_epochs_before_stopping = 6
    num_workers = 0
    seed = 42
    max_files = 1
    learning_rates = [1e-3, 2e-3]
    batch_sizes = [1024, 2048]
    weight_decays = [1e-6]
    leaf_dropouts = [0.0]
    readout_dropouts = [0.1]
    merge_residual_weights = [0.15]
    threshold_grid_steps = [0.05]
    chis = [64]
    segment_window_sizes = [64]
    segment_strides = [58]
    pair_segment_layouts = False
    ranking_metric = "test_f1_micro"
    limit = None
    preset = "coarse"


def test_build_run_name_is_stable() -> None:
    run_name = build_run_name(
        {
            "learning_rate": 1e-3,
            "batch_size": 2048,
            "weight_decay": 1e-6,
            "leaf_dropout": 0.1,
            "readout_dropout": 0.1,
            "merge_residual_weight": 0.15,
            "threshold_grid_step": 0.05,
            "chi": 64,
            "segment_window_size": 64,
            "segment_stride": 58,
        }
    )
    assert run_name == "lr0p001_bs2048_wd1em06_ld0p1_rd0p1_mrw0p15_tgs0p05_chi64_win64_stride58"


def test_build_command_includes_cache_split_and_runtime_flags() -> None:
    command = build_command(
        _Args(),
        Path("out/run"),
        {
            "learning_rate": 2e-3,
            "batch_size": 1024,
            "weight_decay": 1e-6,
            "leaf_dropout": 0.0,
            "readout_dropout": 0.1,
            "merge_residual_weight": 0.15,
            "threshold_grid_step": 0.02,
            "chi": 64,
            "segment_window_size": 64,
            "segment_stride": 58,
        },
    )

    command_str = " ".join(command)
    assert "--cache-path data/cache/ir_spectra_len1800_snv_files1.npz" in command_str
    assert "--split-path results/split.npz" in command_str
    assert "--max-files 1" in command_str
    assert "--no-amp" in command_str
    assert "--no-compile" in command_str
    assert "--apply-snv" in command_str
    assert "--batch-size 1024" in command_str
    assert "--learning-rate 0.002" in command_str
    assert "--threshold-grid-step 0.02" in command_str


def test_iter_configs_returns_cartesian_product() -> None:
    configs = iter_configs(_Args())
    assert len(configs) == 4


def test_iter_segment_layouts_can_pair_window_and_stride() -> None:
    args = _Args()
    args.segment_window_sizes = [48, 64, 80]
    args.segment_strides = [43, 58, 72]
    args.pair_segment_layouts = True

    layouts = iter_segment_layouts(args)

    assert layouts == [(48, 43), (64, 58), (80, 72)]


def test_iter_segment_layouts_rejects_mismatched_paired_lists() -> None:
    args = _Args()
    args.segment_window_sizes = [48, 64]
    args.segment_strides = [43]
    args.pair_segment_layouts = True

    try:
        iter_segment_layouts(args)
    except ValueError as exc:
        assert "same length" in str(exc)
    else:
        raise AssertionError("Expected iter_segment_layouts to reject mismatched paired lists.")


def test_parse_summary_extracts_experiment6_metrics(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.txt"
    summary_path.write_text(
        "TTN IR Experiment6 Summary\n"
        "Best epoch:                34\n"
        "Best early-stop score:     0.533978\n"
        "Best val loss:             0.763326\n"
        "Test f1_micro:             0.715624\n"
        "Elapsed seconds:           50.77\n"
    )

    parsed = parse_summary(summary_path)

    assert parsed["best_epoch"] == "34"
    assert parsed["best_early_stop_score"] == "0.533978"
    assert parsed["best_val_loss"] == "0.763326"
    assert parsed["test_f1_micro"] == "0.715624"
    assert parsed["elapsed_seconds"] == "50.77"


def test_repo_root_and_train_script_resolve_inside_repo() -> None:
    assert TRAIN_SCRIPT.exists()
    assert (REPO_ROOT / "data").exists()


def test_apply_preset_fine_replaces_search_space() -> None:
    args = _Args()
    args.preset = "fine"
    args.limit = None
    apply_preset(args)

    assert args.learning_rates == [4e-4, 5e-4, 6e-4]
    assert args.batch_sizes == [1024]
    assert args.threshold_grid_steps == [0.02]
    assert args.segment_window_sizes == [48, 64, 80]
    assert args.segment_strides == [43, 58, 72]
    assert args.pair_segment_layouts is True
    assert args.limit == 36


def test_write_best_run_artifacts_uses_requested_metric(tmp_path: Path) -> None:
    rows = [
        {
            "run_name": "run_a",
            "status": "ok",
            "best_epoch": "10",
            "best_early_stop_score": "0.55",
            "test_f1_micro": "0.74",
            "test_f1_macro": "0.30",
            "output_dir": "out/a",
            "command": "python train.py --a",
        },
        {
            "run_name": "run_b",
            "status": "ok",
            "best_epoch": "12",
            "best_early_stop_score": "0.53",
            "test_f1_micro": "0.76",
            "test_f1_macro": "0.29",
            "output_dir": "out/b",
            "command": "python train.py --b",
        },
    ]

    write_best_run_artifacts(tmp_path, "test_f1_micro", rows)

    best_txt = (tmp_path / "best_run.txt").read_text()
    best_json = (tmp_path / "best_run.json").read_text()
    assert "run_b" in best_txt
    assert "\"run_name\": \"run_b\"" in best_json
