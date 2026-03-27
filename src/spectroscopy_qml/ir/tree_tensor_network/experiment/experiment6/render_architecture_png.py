from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Circle


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "ttn_architecture_diagram.png"
OUTPUT_SVG = ROOT / "ttn_architecture_diagram.svg"

# ── Colour palette ────────────────────────────────────────────────────────────
BG      = "#F1F3F6"
BOX     = "#F8FBFF"
EDGE    = "#A5B6CC"
TEXT    = "#263445"
ARROW   = "#8FA3BC"
BLUE    = "#C6D9EE"
GREEN   = "#D7ECCB"
PURPLE  = "#E4D8F0"
TREE_BG = "#EEF3FB"
MERGE   = "#D8E5F4"
HIDDEN  = "#78D04E"
OUT_RED = "#FF6E63"
LINK    = "#D4DFEE"
POOL_C  = "#C8DDF5"


# ── Helpers ───────────────────────────────────────────────────────────────────
def rbox(ax, x, y, text, *, fc=BOX, ec=EDGE, fs=9.5, weight="normal",
         color=TEXT, lw=1.4, pad_x=0.15, pad_y=0.12, rs=0.10,
         min_w=0.0, min_h=0.0):
    """
    Draw a rounded text box centered at (x, y) with automatic width/height.
    Returns (patch, width, height).
    """
    txt = ax.text(
        x, y, text,
        ha="center", va="center",
        fontsize=fs, color=color, weight=weight,
        linespacing=1.05, zorder=5
    )

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = txt.get_window_extent(renderer=renderer)

    inv = ax.transData.inverted()
    bbox_data = bbox.transformed(inv)

    w = max(bbox_data.width + pad_x, min_w)
    h = max(bbox_data.height + pad_y, min_h)

    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={rs}",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=4
    )
    ax.add_patch(patch)
    txt.set_zorder(6)

    return patch, w, h


def rbox_fixed(ax, x, y, w, h, text, *, fc=BOX, ec=EDGE, fs=9.5, weight="normal",
               color=TEXT, lw=1.4, pad=0.10, rs=0.10):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad={pad},rounding_size={rs}",
        linewidth=lw, edgecolor=ec, facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center",
        fontsize=fs, color=color, weight=weight,
        linespacing=1.2,
    )
    return patch


def arr(ax, start, end, *, ms=13, lw=1.5, color=ARROW):
    ax.add_patch(FancyArrowPatch(
        start, end,
        arrowstyle="-|>",
        mutation_scale=ms,
        linewidth=lw,
        color=color,
        shrinkA=4, shrinkB=4,
    ))


def stack_cards(ax, x, y, w, h, *, n=3, dx=0.09, dy=0.07, fc=BLUE, ec="#91A8C4"):
    for i in range(n - 1, -1, -1):
        ax.add_patch(Rectangle(
            (x + i * dx, y + i * dy), w, h,
            linewidth=1.2, edgecolor=ec, facecolor=fc,
        ))


def draw_input_icon(ax, x, y, w, h):
    ax.add_patch(Rectangle(
        (x, y), w, h,
        linewidth=1.3, edgecolor="#6E7F95", facecolor="white"
    ))

    ax.plot([x + 0.10, x + 0.10], [y + 0.10, y + h - 0.10], color="#444", lw=1.2)
    ax.plot([x + 0.06, x + w - 0.06], [y + 0.14, y + 0.14], color="#444", lw=1.2)

    xs2 = [x + 0.12, x + 0.18, x + 0.25, x + 0.34, x + 0.43, x + 0.51, x + 0.62]
    ys2 = [y + 0.17, y + 0.29, y + 0.16, y + 0.46, y + 0.20, y + 0.40, y + 0.24]
    ax.plot(xs2, ys2, color="#333", lw=1.3)

    ax.add_patch(Rectangle(
        (x + 0.25, y + 0.52), 0.18, 0.28,
        lw=1.0, edgecolor="#FF8C8C", facecolor="none"
    ))


# ── TTN tree block ────────────────────────────────────────────────────────────
def draw_ttn_tree(ax, x, y, w, h):
    rbox_fixed(ax, x, y, w, h, "", fc=TREE_BG, ec=EDGE, lw=1.6, pad=0.04, rs=0.18)

    ax.text(
        x + w / 2, y + h + 0.22,
        "Hierarchical TTN Encoder",
        ha="center", va="center", fontsize=12, weight="bold", color=TEXT
    )

    ax.text(
        x + w / 2, y + h - 0.38,
        "Binary tree reduction over 56 segment states",
        ha="center", va="center", fontsize=10, weight="bold", color=TEXT
    )

    level_counts = [56, 28, 14, 7, 4, 2, 1]
    level_labels = ["56", "28", "14", "7", "4", "2", "1"]
    n_cols = len(level_counts)

    col_xs = [x + 0.55 + i * (w - 0.9) / (n_cols - 1) for i in range(n_cols)]

    top = y + h - 0.90
    bottom = y + 1.20
    max_shown = 8

    coords = []
    for col, cnt in enumerate(level_counts):
        shown = min(cnt, max_shown)

        if shown == 1:
            ys_col = [(top + bottom) / 2]
        else:
            step = (top - bottom) / (shown - 1)
            ys_col = [top - j * step for j in range(shown)]

        ax.text(
            col_xs[col], y + 0.82,
            level_labels[col] + " nodes",
            ha="center", va="center", fontsize=8.5, color="#5A6B7E"
        )
        coords.append([(col_xs[col], yy) for yy in ys_col])

    for col in range(n_cols - 1):
        left = coords[col]
        right = coords[col + 1]

        for i, (x1, y1) in enumerate(left):
            j = round(i * (len(right) - 1) / max(1, len(left) - 1))
            x2, y2 = right[j]

            ax.plot([x1 + 0.06, x2 - 0.06], [y1, y2],
                    color="#C3D2E6", lw=0.9, zorder=1)

            if j + 1 < len(right) and i % 2 == 0:
                x3, y3 = right[j + 1]
                ax.plot([x1 + 0.06, x3 - 0.06], [y1, y3],
                        color="#D4E0EE", lw=0.7, zorder=1)

    for nodes in coords:
        for nx, ny in nodes:
            s = 0.08
            ax.add_patch(Rectangle(
                (nx - s, ny - s), 2 * s, 2 * s,
                facecolor=MERGE, edgecolor="#8EA5C3", lw=0.9, zorder=2,
            ))

    ax.text(
        x + w / 2, y + 0.58,
        "RelaxedIsometricMerge per level",
        ha="center", va="center", fontsize=8.2, weight="bold", color=TEXT
    )
    ax.text(
        x + w / 2, y + 0.26,
        "outer product 64×64=4096  →  isometric projection 4096→64  +  residual  +  L2 norm",
        ha="center", va="center", fontsize=7.2, color="#5A6B7E"
    )


# ── Dense / MLP readout block ─────────────────────────────────────────────────
def draw_mlp_block(ax, x0, cy):
    ys = [cy + 1.10, cy + 0.55, cy, cy - 0.55, cy - 1.10]

    stub_x = x0
    for yy in ys:
        ax.add_patch(Rectangle(
            (stub_x - 0.10, yy - 0.10), 0.20, 0.20,
            facecolor="#BBC7D6", edgecolor="#8FA1B8", lw=0.9,
        ))

    rbox(ax, stub_x, cy + 2.10, "Flatten / concat\n(448 values)",
         fs=8.2, min_w=1.25, min_h=0.70)

    x1 = x0 + 1.55
    x2 = x0 + 3.15
    x3 = x0 + 4.75

    for y1 in ys:
        for y2 in ys:
            ax.plot([stub_x + 0.10, x1 - 0.14], [y1, y2], color=LINK, lw=0.7, zorder=1)
            ax.plot([x1 + 0.14, x2 - 0.14], [y1, y2], color=LINK, lw=0.7, zorder=1)
            ax.plot([x2 + 0.14, x3 - 0.14], [y1, y2], color=LINK, lw=0.7, zorder=1)

    r = 0.14
    for yy in ys:
        ax.add_patch(Circle((x1, yy), r, facecolor=HIDDEN, edgecolor="#607A99", lw=1.0, zorder=3))
        ax.add_patch(Circle((x2, yy), r, facecolor=HIDDEN, edgecolor="#607A99", lw=1.0, zorder=3))
        ax.add_patch(Circle((x3, yy), r, facecolor=OUT_RED, edgecolor="#607A99", lw=1.0, zorder=3))

    rbox(ax, x1, cy + 2.08, "fc_1\nLinear(448→256)\nGELU + Dropout(0.1)",
         fs=7.8, min_w=1.45, min_h=0.95)
    rbox(ax, x2, cy + 2.08, "fc_2\nLinear(256→37)\nOutput layer",
         fs=7.8, min_w=1.45, min_h=0.95)
    rbox(ax, x3, cy + 2.08, "OUTPUT\n(logits)\n37 labels",
         fs=7.8, min_w=1.45, min_h=0.95)

    for xc, lbl in [
        (stub_x, "Total: 448\nvalues"),
        (x1, "Total: 256\nneurons"),
        (x2, "Total: 37\nlogits"),
        (x3, "Total: 37\nlabels"),
    ]:
        ax.text(xc, cy - 1.72, lbl,
                ha="center", va="top", fontsize=7.8, color="#5A6B7E")

    for idx, yy in enumerate(ys[:3]):
        ax.text(x3 + 0.32, yy, str(idx), va="center", ha="left", fontsize=8.5, color=TEXT)
    ax.text(x3 + 0.32, ys[3], "…", va="center", ha="left", fontsize=10, color=TEXT)
    ax.text(x3 + 0.32, ys[-1], "36", va="center", ha="left", fontsize=8.5, color=TEXT)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    fig, ax = plt.subplots(figsize=(34, 10), dpi=200)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 34)
    ax.set_ylim(0, 10)
    ax.axis("off")

    CY = 5.2

    ax.text(
        17, 9.55,
        "TTN Functional Group Classifier — Experiment 6",
        ha="center", va="center", fontsize=19, weight="bold", color=TEXT
    )

    # INPUT
    draw_input_icon(ax, 0.20, CY - 0.70, 0.85, 1.30)
    ax.text(
        0.63, CY - 1.05,
        "INPUT\n(1800×1)\nIR spectrum",
        ha="center", va="top", fontsize=9, weight="bold", color=TEXT
    )

    # PREPROCESSING
    rbox(ax, 2.25, CY + 1.28, "Preprocessing\nSNV normalisation", fs=9, min_w=1.85, min_h=0.75)
    stack_cards(ax, 1.45, CY - 0.58, 1.10, 1.30, n=3, fc=BLUE)
    ax.text(2.00, CY - 0.95, "(1800×1)",
            ha="center", va="top", fontsize=8.5, color="#5A6B7E")

    # FEATURE MAP
    rbox(ax, 4.80, CY + 1.26,
         "SpectralDerivativeFeatureMap\nraw · first deriv · second deriv\nOutput: (1800×3)",
         fs=8.8, min_w=2.40, min_h=1.00)
    stack_cards(ax, 3.80, CY - 0.58, 1.10, 1.30, n=3, fc=GREEN, ec="#90B988")
    ax.text(4.35, CY - 0.95, "(1800×3)",
            ha="center", va="top", fontsize=8.5, color="#5A6B7E")

    # SEGMENTATION
    rbox(ax, 7.40, CY + 1.28,
         "Sliding-Window Segmentation\n56 windows × 64 points\nstride=32 (50 % overlap)",
         fs=8.5, min_w=2.20, min_h=0.85)

    # LEAF ENCODER
    rbox(ax, 7.40, CY - 0.15,
         "SegmentLeafEncoder\nflatten → LayerNorm\nLinear→GELU→Dropout(0.1)→Linear\nskip proj + LayerNorm + L2 norm\n(batch, 56, 64)",
         fs=8.0, fc="#F4F8FF", min_w=2.20, min_h=1.60)
    ax.text(7.40, CY - 1.35, "(56×64)",
            ha="center", va="top", fontsize=8.5, color="#5A6B7E")

    # POSITION EMBEDDING
    rbox(ax, 9.75, CY + 1.28,
         "Learnable Position Embedding\nEmbedding(56, 64)\nadded to leaf states",
         fs=8.5, min_w=2.00, min_h=0.85)

    # TTN
    TTN_X = 11.05
    TTN_W = 7.80
    TTN_H = 4.20
    TTN_Y = CY - TTN_H / 2
    draw_ttn_tree(ax, TTN_X, TTN_Y, TTN_W, TTN_H)

    # POOL
    POOL_X = 19.80
    rbox(ax, POOL_X + 0.85, CY + 0.05,
         "Multi-Scale\nPooling\nmean pool × 7 levels\n(batch, 7×64)",
         fs=8.5, fc=POOL_C, min_w=1.70, min_h=1.30)
    arr(ax, (18.85, CY), (POOL_X, CY))

    # FUSE
    FUSE_X = 21.95
    rbox(ax, FUSE_X + 0.75, CY + 0.02,
         "Readout Fusion\nconcat → (448)\nLayerNorm(448)",
         fs=8.5, min_w=1.50, min_h=0.88)
    arr(ax, (POOL_X + 1.70, CY), (FUSE_X, CY))

    # MLP
    MLP_X = 24.10
    draw_mlp_block(ax, MLP_X, CY)
    arr(ax, (FUSE_X + 1.50, CY), (MLP_X - 0.12, CY))

    # LEFT ARROWS
    arr(ax, (1.05, CY - 0.08), (1.32, CY - 0.08))
    arr(ax, (3.35, CY - 0.08), (3.63, CY - 0.08))
    arr(ax, (6.05, CY - 0.08), (6.33, CY - 0.08))
    arr(ax, (8.50, CY - 0.08), (8.78, CY - 0.08))
    arr(ax, (10.75, CY - 0.08), (11.08, CY - 0.08))

    # MULTI-SCALE ARROWS
    tree_right = TTN_X + TTN_W
    pool_left = POOL_X
    for frac in [0.88, 0.70, 0.52, 0.34, 0.16]:
        yy = TTN_Y + frac * TTN_H
        ax.annotate(
            "", xy=(pool_left, CY + 0.10), xytext=(tree_right, yy),
            arrowprops=dict(arrowstyle="-|>", color="#8FA3BC", lw=0.9)
        )

    # SEGMENT MASK
    rbox(ax, 7.40, CY - 2.38,
         "Segment Mask (56×64)\nmarks valid samples",
         fs=8, fc="#F0F4F8", ec="#B0BFCF", min_w=2.20, min_h=0.58)
    ax.annotate(
        "", xy=(7.40, CY - 0.95), xytext=(7.40, CY - 2.05),
        arrowprops=dict(arrowstyle="-|>", color=ARROW, lw=1.0, linestyle="dashed")
    )

    # BOTTOM
    ax.text(17, 1.30,
            "Readout path: 448 → 256 → 37",
            ha="center", va="center", fontsize=14, weight="bold", color=TEXT)
    ax.text(
        17, 0.86,
        "Config: input_dim=1800 · num_segments=56 · chi=64 · num_labels=37 · "
        "segment_window=64 · stride=32 · apply_snv=True",
        ha="center", va="center", fontsize=9, color="#607080"
    )
    ax.text(
        17, 0.48,
        "The TTN encoder hierarchically merges derivative-aware local segments via "
        "RelaxedIsometricMerge; multi-scale pooled states from all 7 levels are "
        "fused and mapped by an MLP to 37 functional-group logits.",
        ha="center", va="center", fontsize=7.8, color="#607080"
    )

    # LEGEND
    legend_items = [
        mpatches.Patch(facecolor=BLUE, edgecolor=EDGE, label="Preprocessed feature"),
        mpatches.Patch(facecolor=GREEN, edgecolor=EDGE, label="Derivative channels"),
        mpatches.Patch(facecolor=TREE_BG, edgecolor=EDGE, label="TTN encoder"),
        mpatches.Patch(facecolor=MERGE, edgecolor=EDGE, label="Merge node"),
        mpatches.Patch(facecolor=POOL_C, edgecolor=EDGE, label="Multi-scale pool"),
        mpatches.Patch(facecolor=HIDDEN, edgecolor=EDGE, label="Hidden neuron"),
        mpatches.Patch(facecolor=OUT_RED, edgecolor=EDGE, label="Output neuron"),
    ]
    ax.legend(
        handles=legend_items,
        loc="upper right",
        bbox_to_anchor=(0.998, 0.98),
        fontsize=8.5,
        framealpha=0.85,
        edgecolor=EDGE,
        facecolor=BOX,
        title="Legend",
        title_fontsize=9,
    )

    fig.savefig(OUTPUT, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(OUTPUT_SVG, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved → {OUTPUT}")
    print(f"Saved → {OUTPUT_SVG}")


if __name__ == "__main__":
    main()