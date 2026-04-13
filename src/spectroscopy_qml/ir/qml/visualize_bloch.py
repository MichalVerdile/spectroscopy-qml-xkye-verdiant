"""
Bloch sphere visualisation for the QCNNIRClassifier.

The Bloch sphere is not merely a diagram — it IS the quantum state space of a
single qubit.  A general single-qubit (mixed) state ρ corresponds to a point

    r = (⟨X⟩, ⟨Y⟩, ⟨Z⟩) ∈ ℝ³,   |r| ≤ 1

inside the Bloch sphere.  Pure states lie on the surface (|r| = 1); entangled
qubits in a larger system have |r| < 1 (mixed reduced state).

After training, every IR spectrum maps to a point on/inside the Bloch sphere via
the root qubit (qubit 10) of the QCNN.  The claim is that the model learns to
cluster chemically similar spectra together in this 3-D space.

Public API
----------
    compute_bloch_vectors(model, loader, device, n_samples) → (vecs, labels)
    plot_bloch_sphere(vecs, labels, ...)
    plot_bloch_sphere_per_class(vecs, labels, class_names, class_idx, ...)
    main()   # standalone: loads best model and saves all figures
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)

from spectroscopy_qml.ir.qml.model_qcnn import QCNNIRClassifier

# Functional-group names in the label order used by the data loader
FG_NAMES: list[str] = [
    "Acid anhydride", "Acyl halide", "Alcohol", "Aldehyde", "Alkane",
    "Alkene", "Alkyne", "Amide", "Amine", "Arene", "Azo compound",
    "Carbamate", "Carboxylic acid", "Enamine", "Enol", "Ester", "Ether",
    "Haloalkane", "Hydrazine", "Hydrazone", "Imide", "Imine",
    "Isocyanate", "Isothiocyanate", "Ketone", "Nitrile", "Phenol",
    "Phosphine", "Sulfide", "Sulfonamide", "Sulfonate", "Sulfone",
    "Sulfonic acid", "Sulfoxide", "Thial", "Thioamide", "Thiol",
]


# ── Data collection ───────────────────────────────────────────────────────────

@torch.no_grad()
def compute_bloch_vectors(
    model: QCNNIRClassifier,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    n_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run ``model.get_bloch_vectors`` over batches from *loader* and collect
    the Bloch coordinates together with the ground-truth multi-label vectors.

    Args:
        model:     Trained QCNNIRClassifier (must be in eval mode).
        loader:    DataLoader yielding (spectra, labels) pairs.
        device:    Torch device.
        n_samples: Maximum number of samples to process.  None = all.

    Returns:
        bloch_vecs : (N, 3) ndarray — (⟨X₁₀⟩, ⟨Y₁₀⟩, ⟨Z₁₀⟩) per spectrum
        labels     : (N, 37) ndarray — ground-truth multi-label binary matrix
    """
    model.eval()
    all_vecs:   list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    collected = 0

    for spectra, labels in loader:
        if n_samples is not None and collected >= n_samples:
            break
        remaining = n_samples - collected if n_samples is not None else len(spectra)
        spectra = spectra[:remaining].to(device)
        labels  = labels[:remaining]

        vecs = model.get_bloch_vectors(spectra)        # (batch, 3)
        all_vecs.append(vecs.cpu().numpy())
        all_labels.append(labels.numpy())
        collected += spectra.shape[0]
        print(f"  Bloch vectors: {collected} / {n_samples or '?'}", end="\r")

    print()
    return np.vstack(all_vecs), np.vstack(all_labels)


# ── Sphere wireframe helper ───────────────────────────────────────────────────

def _draw_bloch_wireframe(ax: "Axes3D", alpha: float = 0.08) -> None:
    """Draw unit-sphere wireframe and principal axes on *ax*."""
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, color="lightgray", alpha=alpha, linewidth=0.4)

    # Principal axes
    for vec, lbl in [([1,0,0],"|+x⟩"), ([-1,0,0],"|−x⟩"),
                     ([0,1,0],"|+y⟩"), ([0,-1,0],"|−y⟩"),
                     ([0,0,1],"|0⟩"),  ([0,0,-1],"|1⟩")]:
        ax.quiver(0, 0, 0, *[0.6*c for c in vec],
                  color="gray", alpha=0.5, linewidth=0.8)
        ax.text(*(1.1*c for c in vec), lbl, fontsize=7, color="gray",
                ha="center", va="center")


# ── Main visualisation functions ───────────────────────────────────────────────

def plot_bloch_sphere(
    bloch_vecs: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str] = FG_NAMES,
    save_path: Path | str | None = None,
    title: str = "Bloch Sphere — root qubit",
    colorby: str = "count",      # "count" | "entropy" | class index (int)
    figsize: tuple[float, float] = (9, 8),
) -> plt.Figure:
    """
    3-D scatter of all spectrum-state points on the Bloch sphere.

    Args:
        bloch_vecs  : (N, 3) — ⟨X⟩, ⟨Y⟩, ⟨Z⟩ per spectrum
        labels      : (N, 37) — binary multi-label ground truth
        class_names : ordered list of 37 functional-group names
        save_path   : if given, save the figure here (PNG/PDF)
        title       : figure title
        colorby     : how to colour points:
                        "count"   — number of functional groups (label sum)
                        "entropy" — label entropy
                        int i     — 1 if label i is present else 0
        figsize     : matplotlib figure size

    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=figsize)
    ax  = fig.add_subplot(111, projection="3d")

    _draw_bloch_wireframe(ax)

    rx, ry, rz = bloch_vecs[:, 0], bloch_vecs[:, 1], bloch_vecs[:, 2]
    norms = np.sqrt(rx**2 + ry**2 + rz**2)

    # Colour encoding
    if colorby == "count":
        c_vals  = labels.sum(axis=1).astype(float)
        c_label = "# functional groups"
        cmap    = "plasma"
    elif colorby == "entropy":
        eps     = 1e-7
        p       = labels.mean(axis=1).clip(eps, 1 - eps)
        c_vals  = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
        c_label = "label entropy"
        cmap    = "viridis"
    elif isinstance(colorby, int):
        c_vals  = labels[:, colorby].astype(float)
        c_label = f"has {class_names[colorby]}"
        cmap    = "RdYlGn"
    else:
        raise ValueError(f"Unknown colorby={colorby!r}")

    sc = ax.scatter(rx, ry, rz, c=c_vals, cmap=cmap, s=18, alpha=0.7,
                    edgecolors="none")
    plt.colorbar(sc, ax=ax, label=c_label, shrink=0.65, pad=0.1)

    ax.set_xlabel("⟨X₁₀⟩")
    ax.set_ylabel("⟨Y₁₀⟩")
    ax.set_zlabel("⟨Z₁₀⟩")
    ax.set_title(title, fontsize=11, pad=14)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)

    # Bloch vector norms as annotation
    fig.text(0.01, 0.01,
             f"N={len(bloch_vecs)}  |r| ∈ [{norms.min():.3f}, {norms.max():.3f}]  "
             f"mean|r|={norms.mean():.3f}",
             fontsize=8, color="gray")

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


def plot_bloch_sphere_per_class(
    bloch_vecs: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str] = FG_NAMES,
    class_indices: Sequence[int] | None = None,
    save_dir: Path | str | None = None,
    figsize: tuple[float, float] = (8, 7),
) -> list[plt.Figure]:
    """
    Generate one Bloch sphere plot per functional group, colouring points
    red (positive) vs blue (negative) for that group.

    Args:
        class_indices: which class indices to plot; None = all 37

    Returns:
        list of matplotlib Figure objects
    """
    if class_indices is None:
        class_indices = list(range(labels.shape[1]))
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    figs = []
    for idx in class_indices:
        pos_mask = labels[:, idx] == 1
        neg_mask = ~pos_mask
        n_pos = pos_mask.sum()
        n_neg = neg_mask.sum()
        name  = class_names[idx] if idx < len(class_names) else f"Class {idx}"

        fig = plt.figure(figsize=figsize)
        ax  = fig.add_subplot(111, projection="3d")
        _draw_bloch_wireframe(ax)

        if neg_mask.any():
            ax.scatter(
                bloch_vecs[neg_mask, 0], bloch_vecs[neg_mask, 1], bloch_vecs[neg_mask, 2],
                c="steelblue", s=12, alpha=0.4, label=f"No {name} (n={n_neg})"
            )
        if pos_mask.any():
            ax.scatter(
                bloch_vecs[pos_mask, 0], bloch_vecs[pos_mask, 1], bloch_vecs[pos_mask, 2],
                c="tomato", s=22, alpha=0.8, edgecolors="darkred", linewidths=0.3,
                label=f"{name} (n={n_pos})"
            )

        ax.set_xlabel("⟨X₁₀⟩"); ax.set_ylabel("⟨Y₁₀⟩"); ax.set_zlabel("⟨Z₁₀⟩")
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
        ax.set_title(f"Bloch sphere — {name}", fontsize=11)
        ax.legend(fontsize=8, loc="upper left")
        fig.tight_layout()

        if save_dir is not None:
            path = save_dir / f"bloch_{idx:02d}_{name.replace(' ', '_')}.png"
            fig.savefig(path, dpi=130, bbox_inches="tight")

        figs.append(fig)

    return figs


def plot_bloch_2d_projections(
    bloch_vecs: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str] = FG_NAMES,
    save_path: Path | str | None = None,
    title: str = "Bloch sphere — 2-D projections",
) -> plt.Figure:
    """
    Three 2-D projection plots (XY, XZ, YZ) of the Bloch sphere points,
    coloured by number of functional groups.
    """
    rx, ry, rz = bloch_vecs[:, 0], bloch_vecs[:, 1], bloch_vecs[:, 2]
    c_vals = labels.sum(axis=1).astype(float)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    pairs = [(rx, ry, "⟨X₁₀⟩", "⟨Y₁₀⟩"),
             (rx, rz, "⟨X₁₀⟩", "⟨Z₁₀⟩"),
             (ry, rz, "⟨Y₁₀⟩", "⟨Z₁₀⟩")]

    for ax, (a, b, xl, yl) in zip(axes, pairs):
        sc = ax.scatter(a, b, c=c_vals, cmap="plasma", s=8, alpha=0.6)
        # Unit circle
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), "gray", lw=0.6, alpha=0.4)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal")
        ax.axhline(0, color="gray", lw=0.4, alpha=0.5)
        ax.axvline(0, color="gray", lw=0.4, alpha=0.5)

    plt.colorbar(sc, ax=axes[-1], label="# functional groups", shrink=0.8)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


# ── Standalone entry point ────────────────────────────────────────────────────

def main() -> None:
    """
    Load the best-saved QCNN model and generate Bloch sphere figures.

    Usage (from repo root):
        python src/spectroscopy_qml/ir/qml/visualize_bloch.py
    """
    import sys
    sys.path.insert(0, "src")

    from spectroscopy_qml.ir.mps_encoder.data_loader import load_ir_data, prepare_dataloaders

    ROOT        = Path(__file__).resolve().parent
    BEST_MODEL  = ROOT / "models_qcnn" / "qcnn_best.pt"
    DATA_DIR    = Path("data/raw")
    OUT_DIR     = ROOT / "results_qcnn" / "bloch_plots"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not BEST_MODEL.exists():
        print(f"Model not found at {BEST_MODEL}. Run train_qcnn.py first.")
        return

    ckpt = torch.load(BEST_MODEL, map_location=device)
    use_cantor = ckpt.get("use_cantor", False)

    model = QCNNIRClassifier(use_cantor=use_cantor)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"Loaded model from {BEST_MODEL}  (epoch {ckpt['epoch']})")
    print(f"  val F1-micro={ckpt.get('val_f1_micro', '?'):.4f}  "
          f"F1-macro={ckpt.get('val_f1_macro', '?'):.4f}")

    # Load test data (small subset for visualisation)
    print("\nLoading data …")
    X, y = load_ir_data(DATA_DIR, target_length=1800, max_files=2, apply_snv=True)
    _, _, test_loader = prepare_dataloaders(X, y, batch_size=16,
                                            num_workers=0, pin_memory=False)

    print("Computing Bloch vectors …")
    bloch_vecs, labels = compute_bloch_vectors(
        model, test_loader, device, n_samples=300
    )
    print(f"  Got {len(bloch_vecs)} Bloch vectors, "
          f"|r| mean={np.linalg.norm(bloch_vecs, axis=1).mean():.4f}")

    # 3-D sphere plot (coloured by functional group count)
    plot_bloch_sphere(
        bloch_vecs, labels,
        save_path=OUT_DIR / "bloch_3d_count.png",
        title="Bloch Sphere — coloured by # functional groups",
        colorby="count",
    )

    # 2-D projections
    plot_bloch_2d_projections(
        bloch_vecs, labels,
        save_path=OUT_DIR / "bloch_2d_projections.png",
    )

    # Per-class sphere plots for the 5 most common functional groups
    label_counts  = labels.sum(axis=0)
    top5_indices  = label_counts.argsort()[-5:][::-1].tolist()
    print(f"\nTop-5 classes: {[FG_NAMES[i] for i in top5_indices]}")
    plot_bloch_sphere_per_class(
        bloch_vecs, labels,
        class_indices=top5_indices,
        save_dir=OUT_DIR,
    )

    print(f"\nAll figures saved to {OUT_DIR}")
    plt.show()


if __name__ == "__main__":
    main()
