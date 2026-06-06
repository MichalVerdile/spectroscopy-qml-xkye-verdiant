import json
import subprocess
import tempfile
import zipfile
from pathlib import Path

import click
import keras
import pandas as pd
from keras.utils import plot_model


def _shape_to_string(shape):
    if shape is None:
        return "n/a"
    if isinstance(shape, list):
        return "; ".join(_shape_to_string(s) for s in shape)
    if hasattr(shape, "as_list"):
        values = shape.as_list()
    else:
        values = list(shape)
    return "(" + ", ".join("None" if v is None else str(v) for v in values) + ")"


def _neurons_from_shape(shape):
    if shape is None:
        return None
    if isinstance(shape, list):
        counts = [_neurons_from_shape(s) for s in shape]
        if any(c is None for c in counts):
            return None
        return sum(counts)
    if hasattr(shape, "as_list"):
        values = shape.as_list()
    else:
        values = list(shape)

    neurons = 1
    has_dims = False
    for dim in values[1:]:
        if dim is None:
            return None
        neurons *= int(dim)
        has_dims = True
    return neurons if has_dims else None


def export_architecture_artifacts(model, out_path: Path, model_stem: str):
    out_path.mkdir(parents=True, exist_ok=True)

    summary_lines = []
    model.summary(print_fn=lambda x: summary_lines.append(x))
    summary_path = out_path / f"{model_stem}_summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n")
    print(f"Saved summary: {summary_path}")

    layer_rows = []
    for layer in model.layers:
        output_shape = getattr(layer, "output_shape", None)
        if output_shape is None and hasattr(layer, "output"):
            output_shape = layer.output.shape

        layer_rows.append(
            {
                "layer_name": layer.name,
                "layer_type": layer.__class__.__name__,
                "output_shape": _shape_to_string(output_shape),
                "neurons": _neurons_from_shape(output_shape),
                "params": layer.count_params(),
            }
        )

    layers_path = out_path / f"{model_stem}_layers.csv"
    pd.DataFrame(layer_rows).to_csv(layers_path, index=False)
    print(f"Saved layer table: {layers_path}")

    png_path = out_path / f"{model_stem}_architecture.png"
    try:
        plot_model(
            model,
            to_file=str(png_path),
            show_shapes=True,
            show_layer_names=True,
            expand_nested=True,
            dpi=160,
        )
        print(f"Saved architecture graph: {png_path}")
    except Exception as e:
        print(f"Could not create architecture PNG (plot_model): {e}")


def maybe_launch_netron(model_path: Path, launch_netron: bool):
    if not launch_netron:
        print(f"To open in Netron manually: netron {model_path}")
        return

    try:
        subprocess.Popen(["netron", str(model_path)])
        print(f"Netron launched for model: {model_path}")
    except FileNotFoundError:
        print("Netron not found. Install with: pip install netron")
        print(f"Then run: netron {model_path}")
    except Exception as e:
        print(f"Could not launch Netron automatically: {e}")
        print(f"Try manually: netron {model_path}")


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


@click.command()
@click.option(
    "--model_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to saved Keras model (.keras)",
)
@click.option(
    "--out_path",
    type=click.Path(file_okay=False, path_type=Path),
    required=False,
    help="Output folder for exported files (default: model folder)",
)
@click.option(
    "--model_stem",
    type=str,
    required=False,
    help="Prefix for exported filenames (default: model filename stem)",
)
@click.option(
    "--launch_netron/--no-launch_netron",
    default=False,
    help="Launch Netron automatically (default: False)",
)
def main(model_path: Path, out_path: Path | None, model_stem: str | None, launch_netron: bool):
    out_dir = out_path if out_path is not None else model_path.parent
    stem = model_stem if model_stem is not None else model_path.stem

    print(f"Loading model: {model_path}")
    model = load_model_compat(model_path)
    export_architecture_artifacts(model, out_dir, stem)
    maybe_launch_netron(model_path, launch_netron)


if __name__ == "__main__":
    main()
