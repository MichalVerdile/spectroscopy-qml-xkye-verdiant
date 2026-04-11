"""
Bloch-Sphären-Visualisierung für den Cantor-Quanten-Klassifikator.

Jedes IR-Spektrum wird als Punkt auf der Bloch-Sphäre dargestellt.
Die Farbe des Punktes kodiert die dominante funktionelle Gruppe (oder
eine andere auswählbare Eigenschaft).

Chemisch ähnliche Spektren sollten nach dem Training auf der Sphäre
zusammencluster — der trainierte QCNN hat den Quantenzustandsraum
entsprechend strukturiert.

Funktionen
----------
    compute_bloch_vectors(model, loader, device, n_samples)
        → (N, 3) Bloch-Koordinaten + (N, 37) Labels

    plot_bloch_sphere_interactive(bloch_vecs, labels, colorby, title)
        → Plotly-Figure (interaktiv, 3D-Rotation im Browser)

    plot_bloch_sphere_matplotlib(bloch_vecs, labels, colorby, title)
        → Matplotlib Figure (statisch, für Paper/PDF)

    plot_bloch_2d_projections(bloch_vecs, labels, title)
        → Matplotlib Figure mit XY, XZ, YZ Projektionen

    plot_all_levels(model, loader, device, out_dir)
        → Speichert Bloch-Sphären für alle Encoder-Level
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

# ── Klassen-Namen ─────────────────────────────────────────────────────────────
FG_NAMES: list[str] = [
    "Acid anhydride", "Acyl halide", "Alcohol", "Aldehyde", "Alkane",
    "Alkene", "Alkyne", "Amide", "Amine", "Arene", "Azo compound",
    "Carbamate", "Carboxylic acid", "Enamine", "Enol", "Ester", "Ether",
    "Haloalkane", "Hydrazine", "Hydrazone", "Imide", "Imine",
    "Isocyanate", "Isothiocyanate", "Ketone", "Nitrile", "Phenol",
    "Phosphine", "Sulfide", "Sulfonamide", "Sulfonate", "Sulfone",
    "Sulfonic acid", "Sulfoxide", "Thial", "Thioamide", "Thiol",
]

# 37 qualitativ unterscheidbare Farben (tab20 + tab20b + tab20c)
_COLORS_37 = [
    "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
    "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
    "#aec7e8","#ffbb78","#98df8a","#ff9896","#c5b0d5",
    "#c49c94","#f7b6d2","#c7c7c7","#dbdb8d","#9edae5",
    "#393b79","#5254a3","#6b6ecf","#9c9ede","#637939",
    "#8ca252","#b5cf6b","#cedb9c","#8b6d31","#bd9e39",
    "#e7ba52","#e7cb94","#843c39","#ad494a","#d6616b",
    "#e7969c","#7b4173",
]


# ── Bloch-Vektoren extrahieren ────────────────────────────────────────────────

@torch.no_grad()
def compute_bloch_vectors(
    model:    nn.Module,
    loader:   torch.utils.data.DataLoader,
    device:   torch.device,
    n_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Berechnet Bloch-Vektoren für alle (oder erste n_samples) Samples.

    Args:
        model:     CantorQuantumClassifier mit get_bloch_vectors()-Methode
        loader:    DataLoader der (spectra, labels) liefert
        device:    Ziel-Device
        n_samples: Falls angegeben: nur die ersten n_samples verarbeiten

    Returns:
        bloch_vecs : (N, 3) float32 — Bloch-Koordinaten in [−1,+1]³
        labels     : (N, 37) float32 — Ground-Truth Labels
    """
    model.eval()
    all_bv, all_labels = [], []
    collected = 0

    for spectra, labels in loader:
        if n_samples is not None and collected >= n_samples:
            break
        if n_samples is not None:
            remaining = n_samples - collected
            spectra = spectra[:remaining]
            labels  = labels[:remaining]

        spectra = spectra.to(device)
        bv = model.get_bloch_vectors(spectra)   # (batch, 3)
        all_bv.append(bv.cpu().numpy())
        all_labels.append(labels.numpy())
        collected += spectra.shape[0]

    return np.vstack(all_bv), np.vstack(all_labels)


# ── Interaktive Plotly 3D-Sphäre ──────────────────────────────────────────────

def plot_bloch_sphere_interactive(
    bloch_vecs:  np.ndarray,
    labels:      np.ndarray,
    colorby:     str | int = "dominant",
    title:       str = "Bloch-Sphäre — IR-Spektren",
    save_path:   Path | str | None = None,
    class_names: Sequence[str] = FG_NAMES,
) -> object:
    """
    Interaktive Plotly 3D-Scatter auf der Bloch-Sphäre.

    Jeder Punkt = ein IR-Spektrum, platziert auf der Bloch-Sphäre durch den
    Quantenzustand des Root-Qubits des vollen Spektrum-Encoders.

    Args:
        bloch_vecs:  (N, 3) Bloch-Koordinaten
        labels:      (N, 37) Binary-Label-Matrix
        colorby:     'dominant' — Farbe = häufigste vorhandene Klasse
                     'count'    — Farbe = Anzahl aktiver Labels
                     'entropy'  — Farbe = Label-Entropie
                     int (0–36) — Farbe = Vorhandensein dieser spezifischen Klasse
        title:       Plot-Titel
        save_path:   Falls angegeben: HTML-Datei speichern
        class_names: Liste der Klassennamen

    Returns:
        plotly.graph_objects.Figure
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly ist nicht installiert. `pip install plotly`")

    x, y, z = bloch_vecs[:, 0], bloch_vecs[:, 1], bloch_vecs[:, 2]

    # Farb-Werte berechnen
    if isinstance(colorby, int):
        color_vals = labels[:, colorby].astype(float)
        colorscale = [[0, "lightgray"], [1, _COLORS_37[colorby]]]
        colorbar_title = class_names[colorby] if colorby < len(class_names) else f"Klasse {colorby}"
        marker_color = color_vals
    elif colorby == "count":
        color_vals = labels.sum(axis=1).astype(float)
        colorscale = "Viridis"
        colorbar_title = "Anzahl aktiver Labels"
        marker_color = color_vals
    elif colorby == "entropy":
        # Binäre Entropie: H = -p log(p) - (1-p) log(1-p), gemittelt
        p = labels.mean(axis=1, keepdims=True).clip(1e-8, 1 - 1e-8)
        ent = -(p * np.log(p) + (1-p) * np.log(1-p))
        color_vals = ent.flatten()
        colorscale = "RdYlGn_r"
        colorbar_title = "Label-Entropie"
        marker_color = color_vals
    else:  # "dominant"
        dom_class = np.argmax(labels, axis=1)
        color_vals = dom_class
        colorscale = [[i / 36, _COLORS_37[i]] for i in range(37)]
        colorbar_title = "Dominante Gruppe"
        marker_color = color_vals

    # Sphären-Oberfläche (Referenz)
    theta = np.linspace(0, 2 * np.pi, 60)
    phi   = np.linspace(0, np.pi, 30)
    THETA, PHI = np.meshgrid(theta, phi)
    SX = np.sin(PHI) * np.cos(THETA)
    SY = np.sin(PHI) * np.sin(THETA)
    SZ = np.cos(PHI)

    sphere = go.Surface(
        x=SX, y=SY, z=SZ,
        opacity=0.08,
        colorscale=[[0, "lightblue"], [1, "lightblue"]],
        showscale=False,
        hoverinfo="none",
    )

    # Achsen-Pfeile (±X, ±Y, ±Z)
    axes = []
    for vec, label in [
        ([1,0,0], "+X"), ([-1,0,0], "−X"),
        ([0,1,0], "+Y"), ([0,-1,0], "−Y"),
        ([0,0,1], "+Z"), ([0,0,-1], "−Z"),
    ]:
        axes.append(go.Scatter3d(
            x=[0, 1.2*vec[0]], y=[0, 1.2*vec[1]], z=[0, 1.2*vec[2]],
            mode="lines+text",
            line=dict(color="gray", width=2),
            text=["", label],
            textposition="top center",
            textfont=dict(size=10),
            showlegend=False,
            hoverinfo="none",
        ))

    # Datenpunkte
    scatter = go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers",
        marker=dict(
            size=3,
            color=marker_color,
            colorscale=colorscale,
            colorbar=dict(title=colorbar_title, thickness=15),
            opacity=0.75,
        ),
        text=[
            f"Sample {i}<br>"
            f"Klassen: {', '.join(class_names[c] for c in np.where(labels[i])[0][:4])}"
            for i in range(len(x))
        ],
        hovertemplate="%{text}<br>r=(%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>",
    )

    fig = go.Figure(data=[sphere, *axes, scatter])
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=14)),
        scene=dict(
            xaxis_title="⟨X⟩ (Bloch-X)",
            yaxis_title="⟨Y⟩ (Bloch-Y)",
            zaxis_title="⟨Z⟩ (Bloch-Z)",
            aspectmode="cube",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        width=800, height=700,
    )

    if save_path is not None:
        fig.write_html(str(save_path))
        print(f"Gespeichert: {save_path}")

    return fig


# ── Statische Matplotlib 3D-Sphäre ────────────────────────────────────────────

def plot_bloch_sphere_matplotlib(
    bloch_vecs:  np.ndarray,
    labels:      np.ndarray,
    colorby:     str | int = "dominant",
    title:       str = "Bloch-Sphäre — IR-Spektren",
    save_path:   Path | str | None = None,
    class_names: Sequence[str] = FG_NAMES,
    figsize:     tuple[float, float] = (8, 7),
) -> object:
    """
    Statische Matplotlib 3D-Scatter auf der Bloch-Sphäre.

    Geeignet für Paper und PDF-Export.

    Args:
        bloch_vecs:  (N, 3) Bloch-Koordinaten
        labels:      (N, 37) Binary-Label-Matrix
        colorby:     'dominant', 'count', 'entropy', oder int (Klassen-Index)
        title:       Plot-Titel
        save_path:   Falls angegeben: PNG-Datei speichern (dpi=150)
        class_names: Liste der Klassennamen
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    x, y, z = bloch_vecs[:, 0], bloch_vecs[:, 1], bloch_vecs[:, 2]

    # Farb-Mapping
    if isinstance(colorby, int):
        c = [_COLORS_37[colorby] if labels[i, colorby] else "lightgray"
             for i in range(len(x))]
        ax.scatter(x, y, z, c=c, s=8, alpha=0.6)
        ax.set_title(f"{title}\nFarbe = {class_names[colorby] if colorby < len(class_names) else colorby}")
    elif colorby == "count":
        counts = labels.sum(axis=1)
        sc = ax.scatter(x, y, z, c=counts, cmap="viridis", s=8, alpha=0.6)
        plt.colorbar(sc, ax=ax, label="Anzahl aktiver Labels", shrink=0.6)
        ax.set_title(title)
    elif colorby == "entropy":
        p = labels.mean(axis=1).clip(1e-8, 1 - 1e-8)
        ent = -(p * np.log(p) + (1-p) * np.log(1-p))
        sc = ax.scatter(x, y, z, c=ent, cmap="RdYlGn_r", s=8, alpha=0.6)
        plt.colorbar(sc, ax=ax, label="Label-Entropie", shrink=0.6)
        ax.set_title(title)
    else:  # dominant
        dom = np.argmax(labels, axis=1)
        c = [_COLORS_37[d % 37] for d in dom]
        ax.scatter(x, y, z, c=c, s=8, alpha=0.6)
        ax.set_title(title)

    # Drahtgitter-Sphäre
    theta = np.linspace(0, 2*np.pi, 30)
    phi   = np.linspace(0, np.pi, 15)
    xs = np.outer(np.sin(phi), np.cos(theta))
    ys = np.outer(np.sin(phi), np.sin(theta))
    zs = np.outer(np.cos(phi), np.ones_like(theta))
    ax.plot_wireframe(xs, ys, zs, color="lightblue", alpha=0.15, linewidth=0.5)

    # Achsen
    for start, end, lbl in [
        ([0,0,0], [1.2,0,0], "⟨X⟩"),
        ([0,0,0], [0,1.2,0], "⟨Y⟩"),
        ([0,0,0], [0,0,1.2], "⟨Z⟩"),
    ]:
        ax.quiver(*start, *(np.array(end)-start),
                  color="gray", arrow_length_ratio=0.1, linewidth=1)
        ax.text(*end, lbl, fontsize=9)

    ax.set_xlim([-1.3, 1.3])
    ax.set_ylim([-1.3, 1.3])
    ax.set_zlim([-1.3, 1.3])
    ax.set_xlabel("⟨X⟩")
    ax.set_ylabel("⟨Y⟩")
    ax.set_zlabel("⟨Z⟩")

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Gespeichert: {save_path}")
    return fig


# ── 2D-Projektionen ────────────────────────────────────────────────────────────

def plot_bloch_2d_projections(
    bloch_vecs:  np.ndarray,
    labels:      np.ndarray,
    title:       str = "Bloch-Sphäre — 2D Projektionen",
    save_path:   Path | str | None = None,
    class_names: Sequence[str] = FG_NAMES,
    figsize:     tuple[float, float] = (13, 4),
) -> object:
    """
    Drei 2D-Projektionen: XY, XZ, YZ.

    Jeder Punkt wird mit der Farbe der dominanten Klasse eingefärbt.
    Die grau gestrichelte Linie zeigt den Einheitskreis.

    Args:
        bloch_vecs: (N, 3)
        labels:     (N, 37)
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    x, y, z = bloch_vecs[:, 0], bloch_vecs[:, 1], bloch_vecs[:, 2]
    dom = np.argmax(labels, axis=1)
    colors = [_COLORS_37[d % 37] for d in dom]

    projections = [
        (x, y, "⟨X⟩", "⟨Y⟩", "XY-Projektion"),
        (x, z, "⟨X⟩", "⟨Z⟩", "XZ-Projektion"),
        (y, z, "⟨Y⟩", "⟨Z⟩", "YZ-Projektion"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    theta = np.linspace(0, 2*np.pi, 300)
    cx, cy = np.cos(theta), np.sin(theta)

    for ax, (px, py, xl, yl, subtitle) in zip(axes, projections):
        ax.scatter(px, py, c=colors, s=6, alpha=0.55)
        ax.plot(cx, cy, ":", color="gray", linewidth=0.8, alpha=0.5)
        ax.axhline(0, color="gray", linewidth=0.5, alpha=0.4)
        ax.axvline(0, color="gray", linewidth=0.5, alpha=0.4)
        ax.set_xlabel(xl, fontsize=10)
        ax.set_ylabel(yl, fontsize=10)
        ax.set_title(subtitle, fontsize=10)
        ax.set_xlim([-1.15, 1.15])
        ax.set_ylim([-1.15, 1.15])
        ax.set_aspect("equal")
        ax.grid(False)

    # Legende (max. 10 Klassen mit den meisten Samples)
    class_counts = labels.sum(axis=0)
    top_classes = np.argsort(class_counts)[::-1][:10]
    patches = [
        mpatches.Patch(
            color=_COLORS_37[c % 37],
            label=class_names[c] if c < len(class_names) else f"Klasse {c}",
        )
        for c in top_classes
    ]
    fig.legend(handles=patches, loc="lower center", ncol=5,
               fontsize=7, bbox_to_anchor=(0.5, -0.08))

    fig.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Gespeichert: {save_path}")
    return fig


# ── Per-Klasse Bloch-Sphäre (Matplotlib) ─────────────────────────────────────

def plot_bloch_sphere_per_class(
    bloch_vecs:    np.ndarray,
    labels:        np.ndarray,
    class_indices: list[int],
    save_path:     Path | str | None = None,
    class_names:   Sequence[str] = FG_NAMES,
    ncols:         int = 4,
    cell_size:     float = 3.0,
) -> object:
    """
    Grid von Bloch-Sphären — eine pro Klasse.

    Jede Sphäre zeigt rot = Klasse vorhanden, blau = Klasse nicht vorhanden.
    Hilft zu verstehen, ob der Quantenzustand eine Klasse trennt.

    Args:
        class_indices: Indizes der darzustellenden Klassen (max. 12 empfohlen)
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    nrows = int(np.ceil(len(class_indices) / ncols))
    fig = plt.figure(figsize=(ncols * cell_size, nrows * cell_size))

    theta = np.linspace(0, 2*np.pi, 30)
    phi   = np.linspace(0, np.pi, 15)
    xs_s = np.outer(np.sin(phi), np.cos(theta))
    ys_s = np.outer(np.sin(phi), np.sin(theta))
    zs_s = np.outer(np.cos(phi), np.ones_like(theta))

    for idx, class_idx in enumerate(class_indices):
        ax = fig.add_subplot(nrows, ncols, idx + 1, projection="3d")

        mask_pos = labels[:, class_idx] == 1
        mask_neg = ~mask_pos

        x, y, z = bloch_vecs[:, 0], bloch_vecs[:, 1], bloch_vecs[:, 2]

        if mask_neg.any():
            ax.scatter(x[mask_neg], y[mask_neg], z[mask_neg],
                       c="steelblue", s=4, alpha=0.3, label="0")
        if mask_pos.any():
            ax.scatter(x[mask_pos], y[mask_pos], z[mask_pos],
                       c="crimson", s=8, alpha=0.7, label="1")

        ax.plot_wireframe(xs_s, ys_s, zs_s, color="lightblue",
                          alpha=0.1, linewidth=0.4)

        name = class_names[class_idx] if class_idx < len(class_names) else f"C{class_idx}"
        n_pos = int(mask_pos.sum())
        ax.set_title(f"{name}\n(n={n_pos})", fontsize=7, pad=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    fig.suptitle("Bloch-Sphäre per Klasse  (rot=vorhanden, blau=nicht)", fontsize=11)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Gespeichert: {save_path}")
    return fig


# ── Alle Visualisierungen auf einmal ──────────────────────────────────────────

def plot_all_levels(
    model:      nn.Module,
    loader:     torch.utils.data.DataLoader,
    device:     torch.device,
    out_dir:    Path | str,
    n_samples:  int = 500,
    save_html:  bool = True,
) -> None:
    """
    Erzeugt alle Bloch-Sphären-Visualisierungen für alle Encoder-Level.

    Erstellt in out_dir:
      bloch_full_interactive.html
      bloch_full_matplotlib.png
      bloch_full_2d_projections.png
      bloch_full_per_class.png
      bloch_half0.png  / bloch_half1.png
      bloch_quarter.png  (falls shared_quarters=True)

    Args:
        model:     CantorQuantumClassifier
        loader:    DataLoader
        device:    Device
        out_dir:   Ausgabeverzeichnis
        n_samples: Maximale Anzahl Samples
        save_html: Falls True: interaktiven Plotly-Plot speichern
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Lade Bloch-Vektoren (max. {n_samples} Samples) …")
    bloch_vecs, labels = compute_bloch_vectors(model, loader, device, n_samples)
    print(f"  {len(bloch_vecs)} Samples gesammelt")
    print(f"  |r| mean={np.linalg.norm(bloch_vecs, axis=1).mean():.4f}")

    # Interaktiv (HTML, Plotly)
    if save_html:
        fig_html = plot_bloch_sphere_interactive(
            bloch_vecs, labels,
            title="Cantor-Quanten IR-Klassifikator — Bloch-Sphäre (volles Spektrum)",
            save_path=out_dir / "bloch_full_interactive.html",
        )

    # Matplotlib 3D
    plot_bloch_sphere_matplotlib(
        bloch_vecs, labels,
        title="Bloch-Sphäre — Volles Spektrum",
        save_path=out_dir / "bloch_full_matplotlib.png",
    )

    # 2D-Projektionen
    plot_bloch_2d_projections(
        bloch_vecs, labels,
        title="Bloch-Sphäre 2D-Projektionen — Volles Spektrum",
        save_path=out_dir / "bloch_full_2d_projections.png",
    )

    # Per-Klasse (erste 12 nach Support)
    class_counts = labels.sum(axis=0)
    top12 = list(np.argsort(class_counts)[::-1][:12])
    plot_bloch_sphere_per_class(
        bloch_vecs, labels,
        class_indices=top12,
        save_path=out_dir / "bloch_full_per_class.png",
    )

    # Alle Encoder-Level (falls Methode vorhanden)
    if hasattr(model, "get_all_bloch_vectors"):
        print("\nLade Bloch-Vektoren für alle Encoder-Level …")
        model.eval()
        all_bv_list: dict[str, list] = {}
        label_list: list[np.ndarray] = []
        collected = 0

        with torch.no_grad():
            for spectra, labs in loader:
                if collected >= n_samples:
                    break
                remaining = n_samples - collected
                spectra = spectra[:remaining].to(device)
                labs    = labs[:remaining]
                bv_dict = model.get_all_bloch_vectors(spectra)
                for k, v in bv_dict.items():
                    all_bv_list.setdefault(k, []).append(v.cpu().numpy())
                label_list.append(labs.numpy())
                collected += spectra.shape[0]

        all_labels_np = np.vstack(label_list)
        for level_name, bv_parts in all_bv_list.items():
            bv_np = np.vstack(bv_parts)
            save_png = out_dir / f"bloch_{level_name}.png"
            plot_bloch_sphere_matplotlib(
                bv_np, all_labels_np,
                title=f"Bloch-Sphäre — {level_name}",
                save_path=save_png,
            )

    import matplotlib.pyplot as plt
    plt.close("all")
    print(f"\nAlle Visualisierungen gespeichert in: {out_dir}")


# ── Standalone-Main ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    import torch
    from torch.utils.data import DataLoader

    from spectroscopy_qml.ir.mps_encoder.data_loader import load_ir_data, prepare_dataloaders
    from spectroscopy_qml.ir.quantum_ir.model.classifier import CantorQuantumClassifier

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-dir",   default=Path("data/raw"), type=Path)
    parser.add_argument("--out-dir",    default=Path("results_bloch"), type=Path)
    parser.add_argument("--max-files",  default=3, type=int)
    parser.add_argument("--n-samples",  default=300, type=int)
    args = parser.parse_args()

    device = torch.device("cpu")

    X, y = load_ir_data(args.data_dir, target_length=1800,
                        max_files=args.max_files, apply_snv=True)
    _, _, test_loader = prepare_dataloaders(
        X, y, batch_size=16, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
        random_seed=42, num_workers=0, pin_memory=False,
    )

    model = CantorQuantumClassifier()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    plot_all_levels(model, test_loader, device, args.out_dir, n_samples=args.n_samples)
