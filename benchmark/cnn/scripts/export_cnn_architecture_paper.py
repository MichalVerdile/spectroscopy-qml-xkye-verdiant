import json
import tempfile
import zipfile
from pathlib import Path

import click
import keras
import matplotlib.pyplot as plt
from keras.layers import Conv1D, Dense, Flatten, MaxPooling1D
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle


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
    return x + width + depth * dx


def _draw_dense_column(ax, x, y_mid, n_nodes, title, units_text, color="#7fb069", radius=0.08):
    display_nodes = min(n_nodes, 7)
    spacing = 0.22
    total_h = (display_nodes - 1) * spacing
    start_y = y_mid - total_h / 2

    for i in range(display_nodes):
        ax.add_patch(
            Circle((x, start_y + i * spacing), radius=radius, facecolor=color, edgecolor="#3d5a40")
        )

    if n_nodes > display_nodes:
        ax.text(x, start_y - 0.15, "...", ha="center", va="center", fontsize=12)

    ax.text(x, y_mid + total_h / 2 + 0.3, title, ha="center", fontsize=9, weight="bold")
    ax.text(x, y_mid - total_h / 2 - 0.32, units_text, ha="center", fontsize=9)


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


def export_paper_diagram(model, out_file: Path, title: str):
    input_shape = _shape_no_batch(model.input_shape)
    conv_layers = [layer for layer in model.layers if isinstance(layer, Conv1D)]
    pool_layers = [layer for layer in model.layers if isinstance(layer, MaxPooling1D)]
    flatten_layers = [layer for layer in model.layers if isinstance(layer, Flatten)]
    dense_layers = [layer for layer in model.layers if isinstance(layer, Dense)]

    fig, ax = plt.subplots(figsize=(15, 6), dpi=200)
    ax.set_xlim(0, 16)
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
    input_txt = f"INPUT\n({_format_shape(input_shape)})"
    ax.text(x + 0.5, y - 0.35, input_txt, ha="center", fontsize=9, weight="bold")
    prev_x = x + 1.0

    # Conv/Pool stacks
    stage_x = 2.0
    conv_pool_pairs = min(len(conv_layers), len(pool_layers))
    for i in range(conv_pool_pairs):
        conv = conv_layers[i]
        conv_shape = _layer_shape_no_batch(conv)
        conv_len = conv_shape[0] if conv_shape else 1
        conv_ch = conv_shape[1] if conv_shape and len(conv_shape) > 1 else 1
        conv_w = 0.85 + 0.7 * min(conv_len, 600) / 600.0
        conv_h = 0.62 + 0.8 * min(conv_ch, 64) / 64.0

        _arrow(ax, prev_x + 0.1, 2.25, stage_x - 0.12, 2.25)
        conv_end = _draw_feature_stack(
            ax,
            stage_x,
            1.45,
            conv_w,
            conv_h,
            _stack_depth(conv_ch),
            "#b8d0eb",
            f"Conv_{i + 1}\n({conv.filters} filters, k={conv.kernel_size[0]})",
            f"({_format_shape(conv_shape)})",
        )
        prev_x = conv_end
        stage_x = conv_end + 0.55

        pool = pool_layers[i]
        pool_shape = _layer_shape_no_batch(pool)
        pool_len = pool_shape[0] if pool_shape else 1
        pool_ch = pool_shape[1] if pool_shape and len(pool_shape) > 1 else 1
        pool_w = 0.7 + 0.6 * min(pool_len, 600) / 600.0
        pool_h = 0.55 + 0.75 * min(pool_ch, 64) / 64.0

        _arrow(ax, prev_x + 0.06, 2.25, stage_x - 0.12, 2.25)
        pool_end = _draw_feature_stack(
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
        prev_x = pool_end
        stage_x = pool_end + 0.65

    # Flatten
    if flatten_layers:
        flat = flatten_layers[0]
        flat_shape = _layer_shape_no_batch(flat)
        _arrow(ax, prev_x + 0.1, 2.25, stage_x + 0.1, 2.25)
        ax.add_patch(
            Rectangle(
                (stage_x, 1.7),
                1.0,
                1.1,
                linewidth=1.1,
                edgecolor="#8a6d3b",
                facecolor="#f4d6a0",
                angle=-26,
            )
        )
        ax.text(
            stage_x + 0.95, 2.95, "Flatten", ha="center", fontsize=9, weight="bold", rotation=-26
        )
        ax.text(stage_x + 0.62, 1.15, f"({_format_shape(flat_shape)})", ha="center", fontsize=9)
        prev_x = stage_x + 1.05
        stage_x = prev_x + 0.8

    # Dense layers (all but final output)
    hidden_dense = dense_layers[:-1] if len(dense_layers) > 1 else []
    dense_x_positions = []
    for idx, dense in enumerate(hidden_dense):
        dense_x = stage_x + idx * 0.65
        dense_x_positions.append(dense_x)
        _draw_dense_column(
            ax,
            dense_x,
            2.25,
            dense.units,
            f"Dense_{idx + 1}",
            f"{dense.units} units",
            color="#8bcf7a",
            radius=0.08,
        )
    if dense_x_positions:
        _arrow(ax, prev_x + 0.1, 2.25, dense_x_positions[0] - 0.2, 2.25)
        for i in range(len(dense_x_positions) - 1):
            _arrow(ax, dense_x_positions[i] + 0.12, 2.25, dense_x_positions[i + 1] - 0.12, 2.25)
        prev_x = dense_x_positions[-1] + 0.1

    # Output layer
    if dense_layers:
        out = dense_layers[-1]
        out_x = min(prev_x + 1.1, 15.0)
        _draw_dense_column(
            ax,
            out_x,
            2.25,
            out.units,
            "OUTPUT\n(sigmoid)",
            f"{out.units} labels",
            color="#ff7f7f",
            radius=0.08,
        )
        _arrow(ax, prev_x + 0.1, 2.25, out_x - 0.18, 2.25)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved paper-style architecture: {out_file}")


@click.command()
@click.option(
    "--model_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to saved Keras model (.keras)",
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
def main(model_path: Path, out_file: Path | None, title: str):
    model = load_model_compat(model_path)
    final_out = (
        out_file
        if out_file is not None
        else model_path.parent / f"{model_path.stem}_paper_architecture.png"
    )
    export_paper_diagram(model, final_out, title)


if __name__ == "__main__":
    main()
