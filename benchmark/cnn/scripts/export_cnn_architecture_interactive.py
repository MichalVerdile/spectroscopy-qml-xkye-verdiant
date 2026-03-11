import base64
import json
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path

import click
import keras
import numpy as np
import pandas as pd
from keras.layers import Conv1D, Dense, Flatten, MaxPooling1D
from rdkit import Chem
from rdkit.Chem import Draw

FUNCTIONAL_GROUP_LABELS = [
    "Acid anhydride",
    "Acyl halide",
    "Alcohol",
    "Aldehyde",
    "Alkane",
    "Alkene",
    "Alkyne",
    "Amide",
    "Amine",
    "Arene",
    "Azo compound",
    "Carbamate",
    "Carboxylic acid",
    "Enamine",
    "Enol",
    "Ester",
    "Ether",
    "Haloalkane",
    "Hydrazine",
    "Hydrazone",
    "Imide",
    "Imine",
    "Isocyanate",
    "Isothiocyanate",
    "Ketone",
    "Nitrile",
    "Phenol",
    "Phosphine",
    "Sulfide",
    "Sulfonamide",
    "Sulfonate",
    "Sulfone",
    "Sulfonic acid",
    "Sulfoxide",
    "Thial",
    "Thioamide",
    "Thiol",
]


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
    except Exception:
        with zipfile.ZipFile(model_path, "r") as archive:
            config = json.loads(archive.read("config.json"))
            _strip_quantization_config(config)
            model = keras.models.model_from_json(json.dumps(config))
            with tempfile.TemporaryDirectory() as tmp_dir:
                weights_path = Path(tmp_dir) / "model.weights.h5"
                weights_path.write_bytes(archive.read("model.weights.h5"))
                model.load_weights(weights_path)
        return model


def _shape_no_batch(shape):
    if isinstance(shape, list):
        shape = shape[0]
    if shape is None:
        return None
    return [None if v is None else int(v) for v in shape[1:]]


def extract_layers(model):
    layers = []
    for layer in model.layers:
        output_shape = getattr(layer, "output_shape", None)
        if output_shape is None and hasattr(layer, "output"):
            output_shape = layer.output.shape
        shape = _shape_no_batch(output_shape)
        layer_type = layer.__class__.__name__

        if layer_type == "Conv1D":
            layers.append(
                {
                    "name": layer.name,
                    "kind": "conv",
                    "title": f"Conv + ReLU ({layer.filters})",
                    "shape": shape,
                    "kernel": int(layer.kernel_size[0]),
                    "stride": int(layer.strides[0]),
                    "params": int(layer.count_params()),
                }
            )
        elif layer_type == "MaxPooling1D":
            layers.append(
                {
                    "name": layer.name,
                    "kind": "pool",
                    "title": "MaxPool",
                    "shape": shape,
                    "pool": int(layer.pool_size[0]),
                    "stride": int(layer.strides[0]),
                    "params": int(layer.count_params()),
                }
            )
        elif layer_type == "Flatten":
            layers.append(
                {
                    "name": layer.name,
                    "kind": "flatten",
                    "title": "Flatten",
                    "shape": shape,
                    "params": int(layer.count_params()),
                }
            )
        elif layer_type == "Dense":
            act = getattr(layer, "activation", None)
            act_name = getattr(act, "__name__", "linear")
            layers.append(
                {
                    "name": layer.name,
                    "kind": "dense",
                    "title": f"Dense ({act_name})",
                    "shape": shape,
                    "units": int(layer.units),
                    "params": int(layer.count_params()),
                }
            )

    return layers


def _downsample(values: np.ndarray, n: int = 240) -> list[float]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size <= n:
        return arr.tolist()
    idx = np.linspace(0, arr.size - 1, n).astype(int)
    return arr[idx].tolist()


def _to_input_length(values: np.ndarray, target_len: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == target_len:
        return arr
    x_old = np.linspace(0.0, 1.0, arr.size)
    x_new = np.linspace(0.0, 1.0, target_len)
    return np.interp(x_new, x_old, arr)


def load_real_ir_input(ir_data_root: Path, target_len: int):
    parquet_files = sorted(ir_data_root.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {ir_data_root}")

    for parquet_file in parquet_files:
        try:
            df = pd.read_parquet(parquet_file, columns=["ir_spectra", "smiles"])
        except Exception:
            continue
        for _, row in df.iterrows():
            spectrum = row["ir_spectra"]
            if spectrum is None:
                continue
            arr = np.asarray(spectrum, dtype=float)
            if arr.size < 10 or not np.isfinite(arr).all():
                continue
            arr = _to_input_length(arr, target_len)
            smiles = row.get("smiles", "")
            return arr.reshape(1, target_len, 1), str(smiles) if smiles is not None else ""

    raise RuntimeError("Could not find a valid ir_spectra sample in parquet files.")


def compute_layer_signals(model, layers: list[dict], input_batch: np.ndarray):
    # Use only visualized layers so slider and signal view are aligned.
    layer_names = [lay["name"] for lay in layers]
    keras_layers = [layer for layer in model.layers if layer.name in layer_names]
    probe = keras.Model(inputs=model.input, outputs=[layer.output for layer in keras_layers])
    outputs = probe.predict(input_batch, verbose=0)
    if not isinstance(outputs, list):
        outputs = [outputs]

    signals = {}
    for meta, out in zip(layers, outputs):
        arr = np.asarray(out)
        if arr.ndim == 3:
            # 1D feature maps: show first channel as temporal activation trace.
            vec = arr[0, :, 0]
        elif arr.ndim == 2:
            vec = arr[0, :]
        else:
            vec = arr.reshape(-1)
        signals[meta["name"]] = _downsample(vec, n=240)
    return signals


def smiles_to_data_url(smiles: str):
    if not smiles:
        return ""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        img = Draw.MolToImage(mol, size=(220, 140))
        buff = BytesIO()
        img.save(buff, format="PNG")
        encoded = base64.b64encode(buff.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    except Exception:
        return ""


def render_html(
    model_name: str,
    layers: list[dict],
    input_signal: list[float],
    layer_signals: dict,
    smiles: str,
    mol_image_url: str,
    spectrum_type: str,
    output_probs: list[float],
    output_binary: list[int],
    functional_groups: list[str],
):
    layers_json = json.dumps(layers)
    model_name_json = json.dumps(model_name)
    input_signal_json = json.dumps(input_signal)
    layer_signals_json = json.dumps(layer_signals)
    smiles_json = json.dumps(smiles)
    mol_image_url_json = json.dumps(mol_image_url)
    spectrum_type_json = json.dumps(spectrum_type)
    output_probs_json = json.dumps(output_probs)
    output_binary_json = json.dumps(output_binary)
    functional_groups_json = json.dumps(functional_groups)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Interactive CNN Viewer</title>
  <style>
    :root {{
      --bg: #0f172a;
      --panel: #111827;
      --text: #e5e7eb;
      --muted: #94a3b8;
      --conv: #4da3ff;
      --pool: #5ccf8e;
      --flat: #f4b76c;
      --dense: #c792ea;
      --out: #ff7b7b;
      --accent: #38bdf8;
    }}
    html, body {{
      margin: 0; padding: 0; background: radial-gradient(circle at 10% 10%, #1f2937, var(--bg));
      color: var(--text); font-family: "Avenir Next", "Segoe UI", sans-serif;
    }}
    .wrap {{
      max-width: 1200px; margin: 0 auto; padding: 20px;
    }}
    h1 {{ margin: 0 0 8px 0; font-size: 22px; }}
    .sub {{ color: var(--muted); margin-bottom: 14px; }}
    .panel {{
      background: color-mix(in oklab, var(--panel), transparent 10%);
      border: 1px solid #334155; border-radius: 14px; padding: 14px;
    }}
    canvas {{
      width: 100%; height: 520px; border-radius: 10px; background: linear-gradient(180deg, #0b1220, #0e1729);
      display: block;
    }}
    .controls {{
      display: grid; grid-template-columns: 1fr auto auto; gap: 10px; align-items: center; margin-top: 12px;
    }}
    .controls input[type="range"] {{ width: 100%; }}
    button {{
      border: 1px solid #334155; background: #0b1324; color: var(--text); border-radius: 9px; padding: 8px 12px;
      cursor: pointer;
    }}
    button:hover {{ border-color: var(--accent); }}
    .meta {{
      margin-top: 12px; display: grid; grid-template-columns: 1fr 1fr; gap: 12px;
    }}
    .signal {{
      margin-top: 12px; background: #0b1324; border: 1px solid #253145; border-radius: 10px; padding: 10px;
    }}
    #signalCanvas {{
      width: 100%; height: 220px; border-radius: 8px; background: #0b1220; display: block;
    }}
    .card {{
      background: #0b1324; border: 1px solid #253145; border-radius: 10px; padding: 10px;
    }}
    .card h3 {{ margin: 0 0 8px 0; font-size: 14px; color: #cbd5e1; }}
    .kv {{ font-size: 13px; color: var(--muted); line-height: 1.5; white-space: pre-wrap; word-break: break-all; }}
    .smilesRow {{ display: grid; grid-template-columns: 1fr auto; gap: 8px; align-items: start; }}
    .smilesBox {{
      width: 100%; min-height: 72px; resize: vertical; border-radius: 8px; padding: 8px;
      background: #0a1220; border: 1px solid #334155; color: #e5e7eb; font: 12px/1.35 monospace;
      white-space: pre-wrap; word-break: break-all;
    }}
    .copyBtn {{
      border: 1px solid #334155; background: #0b1324; color: var(--text); border-radius: 8px;
      padding: 8px 10px; cursor: pointer; font-size: 12px;
    }}
    .fgList {{
      margin-top: 6px; max-height: 260px; overflow: auto; border: 1px solid #334155; border-radius: 8px;
      background: #0a1220;
    }}
    .fgRow {{
      display: grid; grid-template-columns: 1fr 70px; gap: 8px; align-items: center;
      padding: 6px 8px; border-bottom: 1px solid #1e293b; font-size: 12px;
    }}
    .fgRow:last-child {{ border-bottom: none; }}
    .fgName {{ color: #cbd5e1; word-break: break-word; }}
    .fgVal {{ color: #e5e7eb; text-align: center; font-weight: 700; }}
    .legend {{
      margin-top: 10px; display: flex; gap: 14px; color: var(--muted); font-size: 12px; flex-wrap: wrap;
    }}
    .dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; margin-right: 6px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Interactive CNN Architecture</h1>
    <div class="sub" id="sub"></div>
    <div class="panel">
      <div class="signal">
        <h3 style="margin:0 0 8px 0;font-size:14px;color:#cbd5e1;">Real Input Trace From .parquet and Current Layer Activation</h3>
        <canvas id="signalCanvas" width="1120" height="220"></canvas>
      </div>
      <canvas id="scene" width="1120" height="520"></canvas>
      <div class="controls">
        <input id="step" type="range" min="0" max="0" value="0" />
        <button id="prev">Prev</button>
        <button id="play">Play</button>
      </div>
      <div class="legend">
        <span><span class="dot" style="background: var(--conv)"></span>Conv</span>
        <span><span class="dot" style="background: var(--pool)"></span>Pool</span>
        <span><span class="dot" style="background: var(--flat)"></span>Flatten</span>
        <span><span class="dot" style="background: var(--dense)"></span>Dense</span>
        <span><span class="dot" style="background: var(--out)"></span>Output Dense</span>
      </div>
      <div class="meta">
        <div class="card">
          <h3>Current Step</h3>
          <div class="kv" id="current"></div>
        </div>
        <div class="card">
          <h3>How To Use</h3>
          <div class="kv">Use the slider to scrub layer-by-layer.
Press Play to animate progression.
Blue connector highlights the active transition.</div>
        </div>
      </div>
      <div class="meta" style="grid-template-columns: 1fr; margin-top: 10px;">
        <div class="card">
          <h3>Input Sample Metadata</h3>
          <div class="kv" id="sampleMeta"></div>
          <div class="smilesRow" style="margin-top:8px;">
            <textarea id="smilesBox" class="smilesBox" readonly></textarea>
            <button id="copySmiles" class="copyBtn">Copy SMILES</button>
          </div>
        </div>
      </div>
      <div class="meta" style="grid-template-columns: 1fr; margin-top: 10px;">
        <div class="card">
          <h3>Model Output For This Input</h3>
          <div class="kv">probabilities (sigmoid):</div>
          <div class="smilesRow" style="margin-top:6px;">
            <textarea id="outputProbBox" class="smilesBox" readonly></textarea>
            <button id="copyOutputProb" class="copyBtn">Copy Probs</button>
          </div>
          <div class="kv" style="margin-top:8px;">binary output (&gt; 0.5): Functional Group (vertikal) mit zugehörigem 0/1</div>
          <div class="smilesRow" style="margin-top:6px;">
            <div id="fgBinaryList" class="fgList"></div>
            <button id="copyOutputBin" class="copyBtn">Copy Binary</button>
          </div>
        </div>
      </div>
    </div>
  </div>
  <script>
    const modelName = {model_name_json};
    const layers = {layers_json};
    const inputSignal = {input_signal_json};
    const layerSignals = {layer_signals_json};
    const smiles = {smiles_json};
    const molImageUrl = {mol_image_url_json};
    const spectrumType = {spectrum_type_json};
    const outputProbs = {output_probs_json};
    const outputBinary = {output_binary_json};
    const functionalGroups = {functional_groups_json};
    const canvas = document.getElementById('scene');
    const ctx = canvas.getContext('2d');
    const signalCanvas = document.getElementById('signalCanvas');
    const sctx = signalCanvas.getContext('2d');
    const molImg = new Image();
    molImg.src = molImageUrl;
    const stepEl = document.getElementById('step');
    const prevBtn = document.getElementById('prev');
    const playBtn = document.getElementById('play');
    const currentEl = document.getElementById('current');
    const subEl = document.getElementById('sub');
    const sampleMetaEl = document.getElementById('sampleMeta');
    const smilesBox = document.getElementById('smilesBox');
    const copySmilesBtn = document.getElementById('copySmiles');
    const outputProbBox = document.getElementById('outputProbBox');
    const fgBinaryList = document.getElementById('fgBinaryList');
    const copyOutputProbBtn = document.getElementById('copyOutputProb');
    const copyOutputBinBtn = document.getElementById('copyOutputBin');

    subEl.textContent = `Model: ${{modelName}} | Spectrum: ${{spectrumType}}`;
    stepEl.max = Math.max(0, layers.length - 1);
    sampleMetaEl.textContent = `spectrum: ${{spectrumType}}`;
    smilesBox.value = smiles || 'n/a';
    outputProbBox.value = outputProbs.map(v => Number(v).toFixed(6)).join(', ');
    fgBinaryList.innerHTML = '';
    for (let i = 0; i < functionalGroups.length; i++) {{
      const row = document.createElement('div');
      row.className = 'fgRow';
      const name = document.createElement('div');
      name.className = 'fgName';
      name.textContent = functionalGroups[i];
      const val = document.createElement('div');
      val.className = 'fgVal';
      val.textContent = String(outputBinary[i] ?? '');
      row.appendChild(name);
      row.appendChild(val);
      fgBinaryList.appendChild(row);
    }}

    function wireCopy(button, getValue, defaultText) {{
      button.addEventListener('click', async () => {{
        try {{
          await navigator.clipboard.writeText(getValue());
          button.textContent = 'Copied';
        }} catch (_) {{
          button.textContent = 'Select & Copy';
        }}
        setTimeout(() => {{ button.textContent = defaultText; }}, 1200);
      }});
    }}

    copySmilesBtn.addEventListener('click', async () => {{
      try {{
        await navigator.clipboard.writeText(smilesBox.value);
        copySmilesBtn.textContent = 'Copied';
      }} catch (_) {{
        smilesBox.select();
        document.execCommand('copy');
        copySmilesBtn.textContent = 'Copied';
      }}
      setTimeout(() => {{ copySmilesBtn.textContent = 'Copy SMILES'; }}, 1200);
    }});
    wireCopy(copyOutputProbBtn, () => outputProbBox.value, 'Copy Probs');
    wireCopy(
      copyOutputBinBtn,
      () => functionalGroups.map((fg, i) => `${{fg}}\t${{outputBinary[i] ?? ''}}`).join('\\n'),
      'Copy Binary'
    );

    let step = 0;
    let playing = false;
    let timer = null;

    function colorFor(layer, isOutput) {{
      if (layer.kind === 'conv') return '#4da3ff';
      if (layer.kind === 'pool') return '#5ccf8e';
      if (layer.kind === 'flatten') return '#f4b76c';
      if (layer.kind === 'dense') return isOutput ? '#ff7b7b' : '#c792ea';
      return '#94a3b8';
    }}

    function shapeText(shape) {{
      if (!shape) return 'n/a';
      return '(' + shape.map(v => v === null ? 'None' : v).join(', ') + ')';
    }}

    function drawStack(x, y, w, h, depth, color) {{
      for (let i = 0; i < depth; i++) {{
        const dx = i * 7, dy = i * -4;
        ctx.fillStyle = color;
        ctx.globalAlpha = i === depth - 1 ? 0.88 : 0.35;
        ctx.fillRect(x + dx, y + dy, w, h);
        ctx.strokeStyle = '#2f4860';
        ctx.globalAlpha = 1;
        ctx.strokeRect(x + dx, y + dy, w, h);
      }}
    }}

    function drawDense(x, y, units, color, isOutput) {{
      const shown = Math.min(units, 10);
      const top = y - 95;
      for (let i = 0; i < shown; i++) {{
        const yy = top + i * (190 / Math.max(1, shown - 1));
        ctx.beginPath();
        ctx.arc(x, yy, 8, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.strokeStyle = '#374151';
        ctx.stroke();
        if (isOutput && (i < 3 || i === shown - 1)) {{
          ctx.fillStyle = '#e5e7eb';
          ctx.font = '12px sans-serif';
          ctx.fillText(String(i), x + 14, yy + 4);
        }}
      }}
      if (units > shown) {{
        ctx.fillStyle = '#cbd5e1';
        ctx.font = '20px sans-serif';
        ctx.fillText('⋮', x - 5, y + 8);
      }}
    }}

    function layoutFor(layer, i, total) {{
      const startX = 80;
      const gap = 86;
      const x = startX + i * gap;
      const y = 290;
      if (layer.kind === 'conv' || layer.kind === 'pool') {{
        const len = layer.shape && layer.shape[0] ? layer.shape[0] : 100;
        const ch = layer.shape && layer.shape[1] ? layer.shape[1] : 8;
        const w = Math.max(42, Math.min(92, 40 + (len / 600) * 52));
        const h = Math.max(36, Math.min(100, 30 + (ch / 64) * 70));
        const depth = Math.max(3, Math.min(9, Math.round(ch / 8)));
        return {{x, y, w, h, depth}};
      }}
      if (layer.kind === 'flatten') return {{x, y, w: 28, h: 130, depth: 1}};
      return {{x, y, w: 1, h: 1, depth: 1}};
    }}

    function draw() {{
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#0f172a';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      const nodes = layers.map((layer, i) => layoutFor(layer, i, layers.length));

      for (let i = 0; i < layers.length - 1; i++) {{
        const a = nodes[i], b = nodes[i + 1];
        const ax = a.x + (a.w || 16) + 8;
        const ay = a.y - (a.h || 16) / 2;
        const bx = b.x - 8;
        const by = b.y - (b.h || 16) / 2;
        ctx.beginPath();
        ctx.moveTo(ax, ay);
        ctx.lineTo(bx, by);
        ctx.strokeStyle = i === step ? '#38bdf8' : '#334155';
        ctx.lineWidth = i === step ? 2.2 : 1.0;
        ctx.stroke();
      }}

      layers.forEach((layer, i) => {{
        const isOutputDense = layer.kind === 'dense' && i === layers.length - 1;
        const color = colorFor(layer, isOutputDense);
        const n = nodes[i];
        const active = i <= step;
        ctx.globalAlpha = active ? 1.0 : 0.28;

        if (layer.kind === 'conv' || layer.kind === 'pool') {{
          drawStack(n.x, n.y - n.h / 2, n.w, n.h, n.depth, color);
        }} else if (layer.kind === 'flatten') {{
          for (let k = 0; k < 7; k++) {{
            const yy = n.y - 64 + k * 19;
            ctx.fillStyle = color;
            ctx.fillRect(n.x, yy, 26, 12);
            ctx.strokeStyle = '#6b7280';
            ctx.strokeRect(n.x, yy, 26, 12);
          }}
        }} else if (layer.kind === 'dense') {{
          drawDense(n.x + 12, n.y - 4, layer.units || 0, color, isOutputDense);
        }}

        ctx.globalAlpha = 1;
        ctx.fillStyle = '#e5e7eb';
        ctx.font = '12px sans-serif';
        ctx.fillText(layer.title, n.x - 4, 410);
        ctx.fillStyle = '#94a3b8';
        ctx.fillText(shapeText(layer.shape), n.x - 4, 428);
      }});

      const cur = layers[step] || null;
      if (cur) {{
        const extras = [];
        if (cur.kernel) extras.push(`kernel: ${{cur.kernel}}`);
        if (cur.pool) extras.push(`pool: ${{cur.pool}}`);
        if (cur.stride) extras.push(`stride: ${{cur.stride}}`);
        if (cur.units) extras.push(`units: ${{cur.units}}`);
        currentEl.textContent = [
          `index: ${{step + 1}} / ${{layers.length}}`,
          `name: ${{cur.name}}`,
          `type: ${{cur.kind}}`,
          `shape: ${{shapeText(cur.shape)}}`,
          `params: ${{cur.params}}`,
          `input source: real ir_spectra from .parquet`,
          ...extras
        ].join('\\n');
      }}

      drawSignal();
    }}

    function drawSeries(series, color, lineWidth, bounds, vmin, vmax) {{
      if (!series || !series.length) return;
      const min = vmin, max = vmax;
      const span = Math.max(1e-9, max - min);
      sctx.beginPath();
      for (let i = 0; i < series.length; i++) {{
        const x = bounds.left + (i / Math.max(1, series.length - 1)) * (bounds.right - bounds.left);
        const y = bounds.bottom - ((series[i] - min) / span) * (bounds.bottom - bounds.top);
        if (i === 0) sctx.moveTo(x, y); else sctx.lineTo(x, y);
      }}
      sctx.strokeStyle = color;
      sctx.lineWidth = lineWidth;
      sctx.stroke();
    }}

    function drawAxes(bounds, xmin, xmax, ymin, ymax) {{
      sctx.strokeStyle = '#334155';
      sctx.lineWidth = 1;
      sctx.beginPath();
      sctx.moveTo(bounds.left, bounds.bottom);
      sctx.lineTo(bounds.right, bounds.bottom);
      sctx.moveTo(bounds.left, bounds.bottom);
      sctx.lineTo(bounds.left, bounds.top);
      sctx.stroke();

      sctx.fillStyle = '#cbd5e1';
      sctx.font = '11px sans-serif';

      // X ticks
      const xt = 6;
      for (let i = 0; i <= xt; i++) {{
        const x = bounds.left + (i / xt) * (bounds.right - bounds.left);
        sctx.strokeStyle = '#253145';
        sctx.beginPath();
        sctx.moveTo(x, bounds.bottom);
        sctx.lineTo(x, bounds.bottom + 5);
        sctx.stroke();
        const val = Math.round(xmin + (i / xt) * (xmax - xmin));
        sctx.fillText(String(val), x - 10, bounds.bottom + 18);
      }}

      // Y ticks
      const yt = 5;
      for (let i = 0; i <= yt; i++) {{
        const y = bounds.bottom - (i / yt) * (bounds.bottom - bounds.top);
        sctx.strokeStyle = '#253145';
        sctx.beginPath();
        sctx.moveTo(bounds.left - 5, y);
        sctx.lineTo(bounds.left, y);
        sctx.stroke();
        const val = (ymin + (i / yt) * (ymax - ymin)).toFixed(2);
        sctx.fillText(String(val), 6, y + 3);
      }}

      sctx.fillStyle = '#e5e7eb';
      sctx.font = '12px sans-serif';
      sctx.fillText('X: Resampled index (0..599)', (bounds.left + bounds.right) / 2 - 80, signalCanvas.height - 6);

      sctx.save();
      sctx.translate(12, (bounds.top + bounds.bottom) / 2 + 36);
      sctx.rotate(-Math.PI / 2);
      sctx.fillText('Y: Intensity / Activation', 0, 0);
      sctx.restore();
    }}

    function drawSignal() {{
      const W = signalCanvas.width, H = signalCanvas.height;
      sctx.clearRect(0, 0, W, H);
      sctx.fillStyle = '#0b1220';
      sctx.fillRect(0, 0, W, H);
      sctx.strokeStyle = '#1f2a3d';
      sctx.strokeRect(0.5, 0.5, W - 1, H - 1);

      const bounds = {{ left: 62, right: W - 18, top: 16, bottom: H - 30 }};
      const cur = layers[step];
      const sig = cur ? layerSignals[cur.name] : null;
      const values = inputSignal.concat(sig || []);
      const ymin = Math.min(...values);
      const ymax = Math.max(...values);
      drawAxes(bounds, 0, 599, ymin, ymax);

      drawSeries(inputSignal, '#94a3b8', 1.4, bounds, ymin, ymax);
      drawSeries(sig, '#38bdf8', 2.0, bounds, ymin, ymax);

      // Molecule structure overlay from the same parquet sample.
      if (molImg.complete && molImageUrl) {{
        const mw = 220, mh = 140;
        const mx = W - mw - 16;
        const my = 14;
        sctx.fillStyle = 'rgba(10,16,30,0.82)';
        sctx.fillRect(mx - 4, my - 4, mw + 8, mh + 26);
        sctx.strokeStyle = '#334155';
        sctx.strokeRect(mx - 4, my - 4, mw + 8, mh + 26);
        sctx.drawImage(molImg, mx, my, mw, mh);
        sctx.fillStyle = '#cbd5e1';
        sctx.font = '11px monospace';
        sctx.fillText('SMILES: ' + (smiles || 'n/a'), mx, my + mh + 14);
      }}

      sctx.fillStyle = '#cbd5e1';
      sctx.font = '12px sans-serif';
      sctx.fillText('gray: real input spectrum (.parquet)', 18, 18);
      sctx.fillText('blue: activation at current layer', 18, 34);
    }}

    function setStep(v) {{
      step = Math.max(0, Math.min(Number(v), layers.length - 1));
      stepEl.value = String(step);
      draw();
    }}

    stepEl.addEventListener('input', e => setStep(e.target.value));
    prevBtn.addEventListener('click', () => setStep(Math.max(0, step - 1)));
    playBtn.addEventListener('click', () => {{
      if (playing) {{
        clearInterval(timer);
        timer = null;
        playing = false;
        playBtn.textContent = 'Play';
        return;
      }}
      playing = true;
      playBtn.textContent = 'Pause';
      timer = setInterval(() => {{
        if (step >= layers.length - 1) {{
          setStep(0);
        }} else {{
          setStep(step + 1);
        }}
      }}, 900);
    }});

    setStep(0);
  </script>
</body>
</html>
"""


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
    help="Output HTML path (default: <model_stem>_interactive.html in model folder)",
)
@click.option(
    "--ir_data_root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("data/raw"),
    show_default=True,
    help="Directory with parquet files that contain ir_spectra",
)
def main(model_path: Path, out_file: Path | None, ir_data_root: Path):
    model = load_model_compat(model_path)
    layers = extract_layers(model)
    input_len = int(model.input_shape[1])
    real_input, smiles = load_real_ir_input(ir_data_root, input_len)
    input_signal = _downsample(real_input[0, :, 0], n=240)
    layer_signals = compute_layer_signals(model, layers, real_input)
    output_raw = model.predict(real_input, verbose=0)[0]
    output_probs = [float(v) for v in np.asarray(output_raw).reshape(-1)]
    output_binary = [1 if v > 0.5 else 0 for v in output_probs]
    mol_image_url = smiles_to_data_url(smiles)
    model_name = str(model_path)
    spectrum_type = model_path.parent.parent.name if model_path.parent.parent else "unknown"
    html = render_html(
        model_name=model_name,
        layers=layers,
        input_signal=input_signal,
        layer_signals=layer_signals,
        smiles=smiles,
        mol_image_url=mol_image_url,
        spectrum_type=spectrum_type,
        output_probs=output_probs,
        output_binary=output_binary,
        functional_groups=FUNCTIONAL_GROUP_LABELS,
    )

    final_out = (
        out_file
        if out_file is not None
        else model_path.parent / f"{model_path.stem}_interactive.html"
    )
    final_out.parent.mkdir(parents=True, exist_ok=True)
    final_out.write_text(html, encoding="utf-8")
    print(f"Saved interactive HTML: {final_out}")


if __name__ == "__main__":
    main()
