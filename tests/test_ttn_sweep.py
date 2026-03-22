from __future__ import annotations

from pathlib import Path

from spectroscopy_qml.ir.tree_tensor_network.sweep import (
    build_command,
    build_run_name,
    iter_segment_configs,
    parse_summary,
)


class _Args:
    base_output_dir = Path("tmp")
    data_dir = Path("data/raw")
    device = "cpu"
    batch_size = 8
    epochs = 5
    early_stopping_patience = 3
    segment_window_sizes = [32, 64, 96]
    segment_strides = [16, 32, 48]
    weight_decay = 1e-6
    threshold_metric = "f1_micro"
    seed = 42
    max_files = 2


def test_build_run_name_is_stable() -> None:
    run_name = build_run_name(window_size=64, stride=32, chi=64, learning_rate=5e-4)
    assert run_name == "win64_stride32_chi64_lr0p0005"


def test_build_command_includes_segment_parameters() -> None:
    command = build_command(
        _Args(),
        Path("out/run"),
        {
            "segment_window_size": 96,
            "segment_stride": 48,
            "chi": 128,
            "learning_rate": 3e-4,
        },
    )

    command_str = " ".join(command)
    assert "--segment-window-size 96" in command_str
    assert "--segment-stride 48" in command_str
    assert "--chi 128" in command_str
    assert "--learning-rate 0.0003" in command_str
    assert "--threshold-metric f1_micro" in command_str


def test_iter_segment_configs_uses_paired_window_stride_values() -> None:
    configs = iter_segment_configs(_Args())

    assert configs == [(32, 16), (64, 32), (96, 48)]


def test_parse_summary_extracts_metrics(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.txt"
    summary_path.write_text(
        "TTN IR Training Summary\n"
        "Best epoch:      7\n"
        "Best val F1:     0.712300\n"
        "Best val loss:   0.456700\n"
        "Elapsed seconds: 123.40\n"
        "f1_micro: 0.701000\n"
    )

    parsed = parse_summary(summary_path)

    assert parsed["best_epoch"] == "7"
    assert parsed["best_val_f1"] == "0.712300"
    assert parsed["best_val_loss"] == "0.456700"
    assert parsed["elapsed_seconds"] == "123.40"
    assert parsed["f1_micro"] == "0.701000"
