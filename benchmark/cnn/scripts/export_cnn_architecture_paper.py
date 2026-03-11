import json
import tempfile
import zipfile
from pathlib import Path

import click
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Conv1D, Dense, Flatten, MaxPooling1D
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle


def _shape_no_batch(shape):
    if isinstance(shape, list):
        shape = shape[0]
    return tuple(shape[1:]) if shape is not None else None


def _layer_shape_no_batch(layer):
    shape = getattr(layer, "output_shape", None)
    if shape is None and hasattr(layer, "output"):
        shape = layer.output.shape
    return _shape_no_batch(shape)


def _format_shape(shape):
    if shape is None:
        return "n/a"
    return " x ".join(str(v) for v in shape)


def _stack_depth(channels):
    if channels <= 2:
        return 2
    if channels <= 8:
        return 4
    if channels <= 32:
        return 7
    return 9


def _draw_feature_stack(ax, x, y, width, height, depth, color, title, shape_text):
    dx = 0.11
    dy = 0.07
    for i in range(depth):
        ax.add_patch(
            Rectangle(
                (x + i * dx, y + i * dy),
                width,
                height,
                linewidth=1.0,
                edgecolor="#4f5d75",
                facecolor=color,
                alpha=0.85 if i == depth - 1 else 0.35,
            )
        )
    ax.text(
        x + width / 2, y + height + depth * dy + 0.26, title, ha="center", fontsize=9, weight="bold"
    )
    ax.text(x + width / 2, y - 0.26, shape_text, ha="center", fontsize=9)
    front_x = x + (depth - 1) * dx
    front_y = y + (depth - 1) * dy
    return x + width + depth * dx, (front_x, front_y, width, height)


def _force_monotonic_shrink(base_w, base_h, prev_w, prev_h, min_w=0.58, min_h=0.52, shrink=0.88):
    # Keep true shape trend from data, but enforce a clear visual shrinking progression.
    w = min(base_w, prev_w * shrink)
    h = min(base_h, prev_h * shrink)
    return max(min_w, w), max(min_h, h)


def _draw_focus_window(
    ax,
    face_x,
    face_y,
    face_w,
    face_h,
    rel_x=0.58,
    rel_y=0.52,
    size_frac=0.30,
):
    win_w = max(0.10, face_w * size_frac)
    win_h = max(0.10, face_h * size_frac)
    x = face_x + rel_x * face_w - win_w / 2.0
    y = face_y + rel_y * face_h - win_h / 2.0
    ax.add_patch(
        Rectangle(
            (x, y),
            win_w,
            win_h,
            linewidth=1.2,
            edgecolor="#d9534f",
            facecolor="none",
            linestyle=(0, (2, 2)),
            zorder=4,
        )
    )
    return x + win_w / 2.0, y + win_h / 2.0


def _load_real_ir_spectrum(ir_data_root: Path):
    parquet_files = sorted(ir_data_root.glob("*.parquet"))
    if not parquet_files:
        return None

    for parquet_file in parquet_files[:8]:
        try:
            df = pd.read_parquet(parquet_file, columns=["ir_spectra"])
            for spectrum in df["ir_spectra"]:
                if spectrum is None:
                    continue
                arr = np.asarray(spectrum, dtype=float)
                if arr.size > 20 and np.isfinite(arr).all():
                    return arr
        except Exception:
            continue
    return None


def _draw_ir_spectrum_in_box(ax, x, y, width, height, spectrum):
    if spectrum is None:
        xx = [
            x + 0.10 * width,
            x + 0.35 * width,
            x + 0.52 * width,
            x + 0.68 * width,
            x + 0.90 * width,
        ]
        yy = [
            y + 0.65 * height,
            y + 0.25 * height,
            y + 0.72 * height,
            y + 0.36 * height,
            y + 0.58 * height,
        ]
        ax.plot(xx, yy, color="#1f2937", linewidth=1.2)
        return

    arr = np.asarray(spectrum, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return

    points = 160
    idx = np.linspace(0, arr.size - 1, points).astype(int)
    y_vals = arr[idx]
    y_min = float(np.min(y_vals))
    y_max = float(np.max(y_vals))
    if abs(y_max - y_min) < 1e-12:
        y_norm = np.full_like(y_vals, 0.5)
    else:
        y_norm = (y_vals - y_min) / (y_max - y_min)

    x_plot = np.linspace(x + 0.07 * width, x + 0.93 * width, len(y_norm))
    y_plot = y + 0.12 * height + (0.76 * height) * y_norm
    ax.plot(x_plot, y_plot, color="#111827", linewidth=1.3)
    ax.plot(
        [x + 0.07 * width, x + 0.93 * width],
        [y + 0.12 * height, y + 0.12 * height],
        color="#9ca3af",
        linewidth=0.7,
    )


def _draw_flatten_column(ax, x, y_mid, n_rect=7):
    rect_w = 0.24
    rect_h = 0.17
    gap = 0.11
    total_h = n_rect * rect_h + (n_rect - 1) * gap
    y_top = y_mid + total_h / 2.0
    y_positions = []
    for i in range(n_rect):
        y0 = y_top - (i + 1) * rect_h - i * gap
        ax.add_patch(
            Rectangle(
                (x, y0),
                rect_w,
                rect_h,
                linewidth=0.9,
                edgecolor="#6b7280",
                facecolor="#9ca3af",
                alpha=0.9,
            )
        )
        y_positions.append(y0 + rect_h / 2.0)

    ax.text(x + rect_w / 2.0, y_top + 0.25, "Flatten layer", ha="center", fontsize=9, weight="bold")
    return x + rect_w, y_positions


def _draw_neuron_column(
    ax,
    x_center,
    y_mid,
    n_nodes,
    title,
    units_text,
    color,
    edge_color,
    radius=0.11,
    max_show=9,
    label_output=False,
):
    if n_nodes <= max_show:
        shown_indices = list(range(n_nodes))
        truncated = False
    else:
        head = max_show // 2
        tail = max_show - head
        shown_indices = list(range(head)) + list(range(n_nodes - tail, n_nodes))
        truncated = True

    y_top = y_mid + 1.35
    y_bottom = y_mid - 1.35
    y_positions = []

    if truncated:
        upper_count = len(shown_indices) // 2
        lower_count = len(shown_indices) - upper_count
        upper_ys = [y_top - i * 0.38 for i in range(upper_count)]
        lower_ys = [y_bottom + i * 0.38 for i in range(lower_count - 1, -1, -1)]
        y_positions = upper_ys + lower_ys
    else:
        if len(shown_indices) == 1:
            y_positions = [y_mid]
        else:
            step = (y_top - y_bottom) / (len(shown_indices) - 1)
            y_positions = [y_top - i * step for i in range(len(shown_indices))]

    for idx, y in zip(shown_indices, y_positions):
        ax.add_patch(
            Circle(
                (x_center, y), radius=radius, facecolor=color, edgecolor=edge_color, linewidth=1.0
            )
        )
        if label_output:
            label = str(idx) if idx < 3 or idx == n_nodes - 1 else ""
            if label:
                ax.text(x_center + 0.26, y, label, va="center", fontsize=9, weight="bold")

    if truncated:
        ax.text(x_center, y_mid, "⋮", ha="center", va="center", fontsize=14, color="#111827")
        if label_output:
            ax.text(x_center + 0.26, y_mid, "…", va="center", fontsize=11, weight="bold")

    text_box = {
        "boxstyle": "round,pad=0.2",
        "facecolor": "#f8fafc",
        "edgecolor": "none",
        "alpha": 0.95,
    }
    ax.text(
        x_center,
        y_top + 0.30,
        title,
        ha="center",
        fontsize=8,
        weight="bold",
        bbox=text_box,
    )
    ax.text(
        x_center,
        y_bottom - 0.32,
        units_text,
        ha="center",
        fontsize=8,
        weight="bold",
        bbox=text_box,
    )
    shown_count = min(n_nodes, max_show)
    if n_nodes > shown_count:
        ax.text(
            x_center,
            y_bottom - 0.58,
            f"shown: {shown_count}",
            ha="center",
            fontsize=7.5,
            color="#4b5563",
            bbox=text_box,
        )
    return y_positions


def _connect_columns(ax, x1, ys1, x2, ys2, color="#b9c0ca", lw=0.6, alpha=0.8):
    for y1 in ys1:
        for y2 in ys2:
            ax.plot(
                [x1 + 0.11, x2 - 0.11], [y1, y2], color=color, linewidth=lw, alpha=alpha, zorder=0
            )


def _connect_focus_windows(ax, p1, p2):
    ax.plot(
        [p1[0], p2[0]],
        [p1[1], p2[1]],
        color="#d9534f",
        linewidth=1.1,
        linestyle=(0, (2, 2)),
        alpha=0.9,
        zorder=3,
    )


def _arrow(ax, x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.2,
            color="#6b7280",
        )
    )


def _strip_quantization_config(obj):
    if isinstance(obj, dict):
        obj.pop("quantization_config", None)
        for value in obj.values():
            _strip_quantization_config(value)
    elif isinstance(obj, list):
        for item in obj:
            _strip_quantization_config(item)


def load_model_compat(model_path: Path):
    try:
        return keras.models.load_model(model_path, compile=False)
    except Exception as err:
        short_err = str(err).splitlines()[0]
        print(f"Standard load_model failed, trying compatibility loader: {short_err}")
        with zipfile.ZipFile(model_path, "r") as archive:
            config = json.loads(archive.read("config.json"))
            _strip_quantization_config(config)
            model = keras.models.model_from_json(json.dumps(config))
            with tempfile.TemporaryDirectory() as tmp_dir:
                weights_path = Path(tmp_dir) / "model.weights.h5"
                weights_path.write_bytes(archive.read("model.weights.h5"))
                model.load_weights(weights_path)
        return model


def export_paper_diagram(model, out_file: Path, title: str, ir_data_root: Path):
    input_shape = _shape_no_batch(model.input_shape)
    conv_layers = [layer for layer in model.layers if isinstance(layer, Conv1D)]
    pool_layers = [layer for layer in model.layers if isinstance(layer, MaxPooling1D)]
    flatten_layers = [layer for layer in model.layers if isinstance(layer, Flatten)]
    dense_layers = [layer for layer in model.layers if isinstance(layer, Dense)]

    fig, ax = plt.subplots(figsize=(20, 6.8), dpi=220)
    ax.set_xlim(0, 22.0)
    ax.set_ylim(0, 5)
    ax.axis("off")
    fig.patch.set_facecolor("#f2f3f5")
    ax.set_facecolor("#f2f3f5")

    ax.text(0.3, 4.55, title, fontsize=13, weight="bold", color="#111827")

    x = 0.5
    y = 1.75

    # Input block
    ax.add_patch(
        Rectangle((x, y), 1.0, 1.0, linewidth=1.2, edgecolor="#6b7280", facecolor="#ffffff")
    )
    ir_spectrum = _load_real_ir_spectrum(ir_data_root)
    _draw_ir_spectrum_in_box(ax, x, y, 1.0, 1.0, ir_spectrum)
    input_txt = f"INPUT\n({_format_shape(input_shape)})"
    ax.text(x + 0.5, y - 0.35, input_txt, ha="center", fontsize=9, weight="bold")
    ax.text(
        x + 0.5,
        y - 0.53,
        "Input spectrum\n(from .parquet)",
        ha="center",
        fontsize=8,
        color="#374151",
    )
    prev_x = x + 1.0
    prev_focus_center = _draw_focus_window(
        ax, x, y, 1.0, 1.0, rel_x=0.52, rel_y=0.55, size_frac=0.30
    )
    next_prev_focus_center = None

    # Conv/Pool stacks
    stage_x = 2.0
    conv_pool_pairs = min(len(conv_layers), len(pool_layers))
    prev_block_w = 1.45
    prev_block_h = 1.30
    for i in range(conv_pool_pairs):
        conv = conv_layers[i]
        conv_shape = _layer_shape_no_batch(conv)
        conv_len = conv_shape[0] if conv_shape else 1
        conv_ch = conv_shape[1] if conv_shape and len(conv_shape) > 1 else 1
        conv_base_w = 0.72 + 0.85 * min(conv_len, 600) / 600.0
        conv_base_h = 0.58 + 0.62 * min(conv_ch, 64) / 64.0
        conv_w, conv_h = _force_monotonic_shrink(
            conv_base_w, conv_base_h, prev_block_w, prev_block_h
        )
        prev_block_w, prev_block_h = conv_w, conv_h

        _arrow(ax, prev_x + 0.1, 2.25, stage_x - 0.12, 2.25)
        conv_end, conv_front = _draw_feature_stack(
            ax,
            stage_x,
            1.45,
            conv_w,
            conv_h,
            _stack_depth(conv_ch),
            "#b8d0eb",
            f"Conv_{i + 1} + ReLU\n({conv.filters} filters, k={conv.kernel_size[0]})",
            f"({_format_shape(conv_shape)})",
        )
        conv_focus_center = _draw_focus_window(
            ax,
            conv_front[0],
            conv_front[1],
            conv_front[2],
            conv_front[3],
            rel_x=0.52 if i % 2 == 0 else 0.62,
            rel_y=0.58 if i % 2 == 0 else 0.44,
            size_frac=0.28,
        )
        _connect_focus_windows(ax, prev_focus_center, conv_focus_center)
        next_prev_focus_center = conv_focus_center
        prev_x = conv_end
        stage_x = conv_end + 0.55

        pool = pool_layers[i]
        pool_shape = _layer_shape_no_batch(pool)
        pool_len = pool_shape[0] if pool_shape else 1
        pool_ch = pool_shape[1] if pool_shape and len(pool_shape) > 1 else 1
        pool_base_w = 0.60 + 0.80 * min(pool_len, 600) / 600.0
        pool_base_h = 0.50 + 0.56 * min(pool_ch, 64) / 64.0
        pool_w, pool_h = _force_monotonic_shrink(
            pool_base_w, pool_base_h, prev_block_w, prev_block_h
        )
        prev_block_w, prev_block_h = pool_w, pool_h

        _arrow(ax, prev_x + 0.06, 2.25, stage_x - 0.12, 2.25)
        pool_end, pool_front = _draw_feature_stack(
            ax,
            stage_x,
            1.5,
            pool_w,
            pool_h,
            _stack_depth(pool_ch),
            "#c8e6c9",
            f"MaxPool_{i + 1}\n(p={pool.pool_size[0]}, s={pool.strides[0]})",
            f"({_format_shape(pool_shape)})",
        )
        pool_focus_center = _draw_focus_window(
            ax,
            pool_front[0],
            pool_front[1],
            pool_front[2],
            pool_front[3],
            rel_x=0.55 if i % 2 == 0 else 0.45,
            rel_y=0.46 if i % 2 == 0 else 0.58,
            size_frac=0.30,
        )
        _connect_focus_windows(ax, next_prev_focus_center, pool_focus_center)
        prev_focus_center = pool_focus_center
        prev_x = pool_end
        stage_x = pool_end + 0.65

    # Flatten
    if flatten_layers:
        flat = flatten_layers[0]
        flat_shape = _layer_shape_no_batch(flat)
        _arrow(ax, prev_x + 0.1, 2.25, stage_x + 0.1, 2.25)
        flat_end, flat_centers = _draw_flatten_column(ax, stage_x + 0.15, 2.25, n_rect=7)
        ax.text(stage_x + 0.25, 1.20, f"({_format_shape(flat_shape)})", ha="left", fontsize=8.5)
        for target_y in flat_centers:
            ax.plot(
                [prev_x + 0.04, stage_x + 0.15],
                [2.25, target_y],
                color="#111827",
                linewidth=0.8,
                linestyle=(0, (4, 3)),
                alpha=0.9,
                zorder=0,
            )
        prev_x = flat_end + 0.05
        stage_x = prev_x + 0.95

    # Dense layers in the same visual style as the reference image:
    # vertical neuron columns with explicit connections and ellipsis.
    hidden_dense = dense_layers[:-1] if len(dense_layers) > 1 else []
    dense_x_positions = []
    dense_y_positions = []
    dense_spacing = 1.9
    for idx, dense in enumerate(hidden_dense):
        dense_x = stage_x + idx * dense_spacing
        dense_x_positions.append(dense_x)
        ys = _draw_neuron_column(
            ax,
            dense_x,
            2.25,
            dense.units,
            f"fc_{idx + 3}\nFully-Connected\n(ReLU)",
            f"Total: {dense.units} neurons",
            color="#7ac45f",
            edge_color="#3d5a40",
            radius=0.105,
            max_show=9,
            label_output=False,
        )
        dense_y_positions.append(ys)
    if dense_x_positions:
        # Fan-in lines from flatten block to first dense column.
        x_flat = prev_x + 0.1
        for anchor_y in [1.75, 2.25, 2.75]:
            for target_y in dense_y_positions[0]:
                ax.plot(
                    [x_flat, dense_x_positions[0] - 0.12],
                    [anchor_y, target_y],
                    color="#c3c9d2",
                    linewidth=0.6,
                    alpha=0.9,
                    zorder=0,
                )
        _arrow(ax, prev_x + 0.1, 2.25, dense_x_positions[0] - 0.22, 2.25)
        for i in range(len(dense_x_positions) - 1):
            _connect_columns(
                ax,
                dense_x_positions[i],
                dense_y_positions[i],
                dense_x_positions[i + 1],
                dense_y_positions[i + 1],
            )
            _arrow(ax, dense_x_positions[i] + 0.18, 2.25, dense_x_positions[i + 1] - 0.18, 2.25)
        prev_x = dense_x_positions[-1] + 0.25

    # Output layer
    if dense_layers:
        out = dense_layers[-1]
        out_x = min(prev_x + 1.55, 21.0)
        out_ys = _draw_neuron_column(
            ax,
            out_x,
            2.25,
            out.units,
            "OUTPUT\n(sigmoid)",
            f"Total: {out.units} labels",
            color="#ff6b57",
            edge_color="#a34242",
            radius=0.11,
            max_show=9,
            label_output=True,
        )
        if dense_x_positions:
            _connect_columns(ax, dense_x_positions[-1], dense_y_positions[-1], out_x, out_ys)
        _arrow(ax, prev_x + 0.1, 2.25, out_x - 0.2, 2.25)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved paper-style architecture: {out_file}")


@click.command()
@click.option(
    "--model_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=False,
    help="Path to one saved Keras model (.keras)",
)
@click.option(
    "--out_file",
    type=click.Path(dir_okay=False, path_type=Path),
    required=False,
    help="Output PNG path (default: <model_stem>_paper_architecture.png in model folder)",
)
@click.option(
    "--title",
    type=str,
    default="CNN Architecture Overview",
    show_default=True,
)
@click.option(
    "--all_models/--single_model",
    default=False,
    help="Export for all .keras models below --models_root (default: False)",
)
@click.option(
    "--models_root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("benchmark/cnn/models"),
    show_default=True,
    help="Root directory scanned when --all_models is enabled",
)
@click.option(
    "--ir_data_root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("data/raw"),
    show_default=True,
    help="Directory with parquet files used to draw a real IR spectrum in the input block",
)
def main(
    model_path: Path | None,
    out_file: Path | None,
    title: str,
    all_models: bool,
    models_root: Path,
    ir_data_root: Path,
):
    if all_models:
        model_paths = sorted(models_root.rglob("*.keras"))
        if not model_paths:
            raise click.ClickException(f"No .keras files found under: {models_root}")
        for current_model_path in model_paths:
            model = load_model_compat(current_model_path)
            auto_title = f"CNN Architecture - {current_model_path.parent.parent.name} ({current_model_path.parent.name})"
            final_out = (
                current_model_path.parent / f"{current_model_path.stem}_paper_architecture.png"
            )
            export_paper_diagram(model, final_out, auto_title, ir_data_root)
        return

    if model_path is None:
        raise click.ClickException(
            "Provide --model_path for single-model export or use --all_models."
        )

    model = load_model_compat(model_path)
    final_out = (
        out_file
        if out_file is not None
        else model_path.parent / f"{model_path.stem}_paper_architecture.png"
    )
    export_paper_diagram(model, final_out, title, ir_data_root)


if __name__ == "__main__":
    main()
