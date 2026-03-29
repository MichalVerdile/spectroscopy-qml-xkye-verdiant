from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Circle


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "ttn_architecture_diagram.png"
OUTPUT_SVG = ROOT / "ttn_architecture_diagram.svg"
RUN_CONFIG = (
    ROOT
    / "results"
    / "full_dataset_run_mps_finetuned_winner_lr4e4_bs1024_20260328"
    / "run_config.json"
)

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


def compute_segment_count(
    input_dim: int,
    segment_window_size: int,
    segment_stride: int,
    segment_mode: str,
    segment_offset: int | None,
) -> int:
    window_size = min(segment_window_size, input_dim)
    max_start = max(0, input_dim - window_size)

    def build_starts(offset: int) -> list[int]:
        starts = list(range(offset, max_start + 1, segment_stride)) if offset <= max_start else []
        starts.extend([0, max_start])
        return [start for start in starts if 0 <= start <= max_start]

    starts = build_starts(0)
    if segment_mode == "dual_offset":
        effective_offset = segment_stride // 2 if segment_offset is None else segment_offset
        if effective_offset > 0:
            starts.extend(build_starts(effective_offset))
    return len(sorted(set(starts)))


def load_architecture_config() -> dict[str, int | float | bool | str | None]:
    config = json.loads(RUN_CONFIG.read_text())
    num_segments = compute_segment_count(
        input_dim=int(config["input_dim"]),
        segment_window_size=int(config["segment_window_size"]),
        segment_stride=int(config["segment_stride"]),
        segment_mode=str(config["segment_mode"]),
        segment_offset=config["segment_offset"],
    )

    level_counts = [num_segments]
    while level_counts[-1] > 1:
        level_counts.append((level_counts[-1] + 1) // 2)

    num_readout_scales = len(level_counts)
    readout_dim = num_readout_scales * int(config["chi"])
    readout_hidden_dim = max(128, 4 * int(config["chi"]), readout_dim // 2)
    leaf_input_dim = int(config["segment_window_size"]) * 3
    leaf_hidden_dim = config["leaf_hidden_dim"]
    if leaf_hidden_dim is None:
        leaf_hidden_dim = max(4 * int(config["chi"]), 2 * leaf_input_dim)

    config["num_segments"] = num_segments
    config["level_counts"] = level_counts
    config["num_readout_scales"] = num_readout_scales
    config["readout_dim"] = readout_dim
    config["readout_hidden_dim"] = readout_hidden_dim
    config["leaf_input_dim"] = leaf_input_dim
    config["leaf_hidden_dim_resolved"] = int(leaf_hidden_dim)
    return config


CFG = load_architecture_config()


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
        zorder=4,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center",
        fontsize=fs, color=color, weight=weight,
        linespacing=1.2,
        zorder=6,
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
        ha="center", va="center", fontsize=12, weight="bold", color=TEXT, zorder=6
    )

    ax.text(
        x + w / 2, y + h - 0.38,
        f"Binary tree reduction over {CFG['num_segments']} segment states",
        ha="center", va="center", fontsize=10, weight="bold", color=TEXT, zorder=6
    )

    level_counts = list(CFG["level_counts"])
    level_labels = [str(count) for count in level_counts]
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

        level_name = f"L{col}"
        ax.text(
            col_xs[col], y + h - 0.62,
            level_name,
            ha="center", va="center", fontsize=8.1, weight="bold", color="#5A6B7E", zorder=6
        )
        if col == 0:
            level_text = level_labels[col] + f" nodes\n(batch, {level_counts[col]}, {CFG['chi']})"
        else:
            prev_cnt = level_counts[col - 1]
            merge_count = prev_cnt // 2
            if prev_cnt % 2 == 0:
                merge_text = f"{merge_count}×(64×64)"
            else:
                merge_text = f"{merge_count}×(64×64) + carry"
            level_text = level_labels[col] + f" nodes\n{merge_text}"

        ax.text(
            col_xs[col], y + 0.82,
            level_text,
            ha="center", va="center", fontsize=7.7, color="#5A6B7E", zorder=6
        )
        coords.append([(col_xs[col], yy) for yy in ys_col])

    for col in range(n_cols - 1):
        left = coords[col]
        right = coords[col + 1]

        for i, (x1, y1) in enumerate(left):
            j = round(i * (len(right) - 1) / max(1, len(left) - 1))
            x2, y2 = right[j]

            ax.plot([x1 + 0.06, x2 - 0.06], [y1, y2],
                    color="#C3D2E6", lw=0.9, zorder=5)

            if j + 1 < len(right) and i % 2 == 0:
                x3, y3 = right[j + 1]
                ax.plot([x1 + 0.06, x3 - 0.06], [y1, y3],
                        color="#D4E0EE", lw=0.7, zorder=5)

    for nodes in coords:
        for nx, ny in nodes:
            s = 0.08
            ax.add_patch(Rectangle(
                (nx - s, ny - s), 2 * s, 2 * s,
                facecolor=MERGE, edgecolor="#8EA5C3", lw=0.9, zorder=6,
            ))

    ax.text(
        x + w / 2, y + 0.58,
        "Pairwise TTN merge per level",
        ha="center", va="center", fontsize=8.2, weight="bold", color=TEXT, zorder=6
    )
    ax.text(
        x + w / 2, y + 0.26,
        f"{CFG['chi']}×{CFG['chi']} outer product → proj. {CFG['chi']} + "
        f"res. {CFG['merge_residual_weight']:.2f} + L2 norm",
        ha="center", va="center", fontsize=7.0, color="#5A6B7E", zorder=6
    )
    ax.text(
        x + w / 2, y + 0.08,
        "models all left-right component interactions",
        ha="center", va="center", fontsize=6.9, color="#6C7C90", zorder=6
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

    rbox(ax, stub_x, cy + 2.10, f"Flatten / concat\n({CFG['readout_dim']} values)",
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

    fc1_text = (
        f"Fully connected 1\nFeature transformation\n"
        f"Linear({CFG['readout_dim']}→{CFG['readout_hidden_dim']})\nGELU"
    )
    if CFG["readout_dropout"] > 0:
        fc1_text += f" + Dropout({CFG['readout_dropout']:.1f})"
    rbox(ax, x1, cy + 2.08, fc1_text, fs=7.5, min_w=0.8, min_h=0.75)
    rbox(ax, x2, cy + 2.08,
         f"Fully connected 2\nOutput mapping\nLinear({CFG['readout_hidden_dim']}→{CFG['num_labels']})\nOutput layer",
         fs=7.1, min_w=0.8, min_h=0.75)
    rbox(
        ax,
        x3,
        cy + 2.08,
        f"OUTPUT\n{CFG['num_labels']} logits\nsigmoid → probabilities",
        fs=7.5,
        min_w=0.95,
        min_h=0.78,
    )

    for xc, lbl in [
        (stub_x, f"Total: {CFG['readout_dim']}\nvalues"),
        (x1, f"Total: {CFG['readout_hidden_dim']}\nneurons"),
        (x2, f"Total: {CFG['num_labels']}\nlogits"),
        (x3, f"Total: {CFG['num_labels']}\noutputs"),
    ]:
        ax.text(xc, cy - 1.72, lbl,
                ha="center", va="top", fontsize=7.8, color="#5A6B7E")

    for idx, yy in enumerate(ys[:3]):
        ax.text(x3 + 0.32, yy, str(idx), va="center", ha="left", fontsize=8.5, color=TEXT)
    ax.text(x3 + 0.32, ys[3], "…", va="center", ha="left", fontsize=10, color=TEXT)
    ax.text(x3 + 0.32, ys[-1], "36", va="center", ha="left", fontsize=8.5, color=TEXT)


def draw_leaf_encoder_block(ax, cx, cy):
    box_w = 7.2
    box_h = 3.9
    top_y = cy + 1.68
    box_bottom = top_y - box_h
    patch = FancyBboxPatch(
        (cx - box_w / 2, box_bottom), box_w, box_h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=1.4, edgecolor=EDGE, facecolor="#F4F8FF", zorder=4,
    )
    ax.add_patch(patch)

    ax.text(
        cx, top_y - 0.36,
        "SegmentLeafEncoder",
        ha="center", va="center", fontsize=10.0, weight="bold", color=TEXT, zorder=6,
    )
    ax.text(
        cx, top_y - 0.55,
        "per-segment MLP + skip",
        ha="center", va="center", fontsize=8.3, color="#5A6B7E", zorder=6,
    )

    x_in = cx - 3.15
    x_hidden = cx - 1.75
    x_main_out = cx - 0.35
    x_add = cx + 0.95
    x_norm = cx + 2.05
    x_final = cx + 3.35

    main_ys = [cy + 0.36, cy + 0.08, cy - 0.20, cy - 0.48]
    skip_ys = [cy - 0.98, cy - 1.28, cy - 1.58]
    merge_y = cy
    skip_merge_y = skip_ys[1]

    for y1 in main_ys:
        for y2 in main_ys:
            ax.plot([x_in + 0.12, x_hidden - 0.12], [y1, y2], color=LINK, lw=0.8, zorder=5)
            ax.plot([x_hidden + 0.12, x_main_out - 0.12], [y1, y2], color=LINK, lw=0.8, zorder=5)

    for y1 in skip_ys:
        for y2 in skip_ys:
            ax.plot([x_in + 0.12, x_main_out - 0.12], [y1, y2], color="#E6B58D", lw=0.85, zorder=5)

    for x_col, fill in [(x_in, BLUE), (x_hidden, HIDDEN), (x_main_out, MERGE)]:
        for yy in main_ys:
            ax.add_patch(Circle((x_col, yy), 0.11, facecolor=fill, edgecolor="#607A99", lw=1.1, zorder=6))

    for yy in skip_ys:
        ax.add_patch(Circle((x_in, yy), 0.10, facecolor="#F8D5B7", edgecolor="#A16634", lw=1.1, zorder=6))
        ax.add_patch(Circle((x_main_out, yy), 0.10, facecolor="#F8D5B7", edgecolor="#A16634", lw=1.1, zorder=6))

    ax.text(x_in, cy + 0.62, f"input\n64×3={CFG['leaf_input_dim']}\nflatten", ha="center", va="bottom",
            fontsize=8.0, color="#5A6B7E", zorder=6)
    ax.text(x_hidden, cy + 0.62, f"hidden\n{CFG['leaf_hidden_dim_resolved']}\nfeature expansion", ha="center", va="bottom",
            fontsize=8.0, color="#5A6B7E", zorder=6)
    ax.text(x_main_out, cy + 0.62, f"main\n{CFG['chi']}\ncompressed to {CFG['chi']}", ha="center", va="bottom",
            fontsize=8.0, color="#5A6B7E", zorder=6)
    ax.text(x_final, cy + 0.62, f"final\n{CFG['chi']}\nafter add+norm", ha="center", va="bottom",
            fontsize=8.0, color="#5A6B7E", zorder=6)

    ax.annotate(
        "",
        xy=(x_add - 0.14, merge_y),
        xytext=(x_main_out + 0.14, merge_y),
        arrowprops=dict(arrowstyle="-|>", color=ARROW, lw=1.4),
        zorder=5,
    )

    ax.annotate(
        "",
        xy=(x_add - 0.04, merge_y - 0.03),
        xytext=(x_main_out + 0.12, skip_merge_y),
        arrowprops=dict(arrowstyle="-|>", color="#D98A4E", lw=1.3),
        zorder=5,
    )
    ax.text(
        (x_in + x_main_out) / 2,
        skip_ys[-1] - 0.08,
        f"skip linear projection {CFG['leaf_input_dim']}→{CFG['chi']}",
        ha="center",
        va="top",
        fontsize=7.4,
        color="#A16634",
        zorder=6,
    )

    ax.add_patch(Circle((x_add, merge_y), 0.16, facecolor="white", edgecolor="#607A99", lw=1.2, zorder=7))
    ax.text(x_add, merge_y, "+", ha="center", va="center", fontsize=13, color=TEXT, weight="bold", zorder=8)

    ax.annotate(
        "",
        xy=(x_final - 0.14, merge_y),
        xytext=(x_add + 0.16, merge_y),
        arrowprops=dict(arrowstyle="-|>", color=ARROW, lw=1.4),
        zorder=5,
    )
    ax.text(
        (x_add + x_final) / 2,
        merge_y + 0.18,
        "LayerNorm + L2 norm",
        ha="center",
        va="center",
        fontsize=8.2,
        color="#5A6B7E",
        zorder=6,
    )

    for yy in main_ys:
        ax.add_patch(Circle((x_final, yy), 0.11, facecolor=MERGE, edgecolor="#607A99", lw=1.1, zorder=6))

    ax.text(
        x_hidden, cy - 0.74,
        f"GELU + Dropout({CFG['leaf_dropout']:.1f})",
        ha="center", va="center", fontsize=8.0, color="#5A6B7E", zorder=6,
    )
    ax.text(
        (x_add + x_final) / 2,
        merge_y - 0.20,
        f"Output: (batch, {CFG['num_segments']}, {CFG['chi']})",
        ha="center", va="center", fontsize=8.0, color="#5A6B7E", zorder=6,
    )
    return (cx - box_w / 2, cx + box_w / 2, box_bottom, top_y)


def draw_leaf_encoder_zoom(ax, x, y, w, h):
    rbox_fixed(ax, x, y, w, h, "", fc="#F4F8FF", ec=EDGE, lw=1.5, pad=0.05, rs=0.14)

    cx = x + w / 2
    top = y + h

    ax.text(
        cx,
        top - 0.30,
        "SegmentLeafEncoder Zoom-In",
        ha="center",
        va="center",
        fontsize=10.5,
        weight="bold",
        color=TEXT,
    )
    ax.text(
        cx,
        top - 0.62,
        f"flatten {CFG['leaf_input_dim']}  ->  hidden {CFG['leaf_hidden_dim_resolved']}  ->  output {CFG['chi']}",
        ha="center",
        va="center",
        fontsize=8.0,
        color="#5A6B7E",
    )

    x_in = x + 0.85
    x_hidden = x + 2.20
    x_main = x + 3.55
    x_add = x + 4.25
    x_out = x + 5.00

    main_ys = [y + 2.20, y + 1.75, y + 1.30, y + 0.85]
    skip_ys = [y + 0.42, y + 0.18]
    merge_y = y + 1.52

    for y1 in main_ys:
        for y2 in main_ys:
            ax.plot([x_in + 0.12, x_hidden - 0.12], [y1, y2], color=LINK, lw=0.8, zorder=2)
            ax.plot([x_hidden + 0.12, x_main - 0.12], [y1, y2], color=LINK, lw=0.8, zorder=2)

    for y1 in skip_ys:
        for y2 in skip_ys:
            ax.plot([x_in + 0.11, x_main - 0.11], [y1, y2], color="#E6B58D", lw=0.9, zorder=2)

    for col_x, fill, edge, radius, ys in [
        (x_in, BLUE, "#607A99", 0.10, main_ys),
        (x_hidden, HIDDEN, "#607A99", 0.10, main_ys),
        (x_main, MERGE, "#607A99", 0.10, main_ys),
        (x_in, "#F8D5B7", "#A16634", 0.09, skip_ys),
        (x_main, "#F8D5B7", "#A16634", 0.09, skip_ys),
    ]:
        for yy in ys:
            ax.add_patch(Circle((col_x, yy), radius, facecolor=fill, edgecolor=edge, lw=1.0, zorder=3))

    ax.annotate(
        "",
        xy=(x_add - 0.14, merge_y),
        xytext=(x_main + 0.12, merge_y),
        arrowprops=dict(arrowstyle="-|>", color=ARROW, lw=1.3),
        zorder=2,
    )
    ax.annotate(
        "",
        xy=(x_add - 0.05, merge_y - 0.03),
        xytext=(x_main + 0.10, y + 0.30),
        arrowprops=dict(arrowstyle="-|>", color="#D98A4E", lw=1.2),
        zorder=2,
    )

    ax.add_patch(Circle((x_add, merge_y), 0.14, facecolor="white", edgecolor="#607A99", lw=1.2, zorder=4))
    ax.text(x_add, merge_y, "+", ha="center", va="center", fontsize=12, weight="bold", color=TEXT, zorder=5)

    ax.annotate(
        "",
        xy=(x_out - 0.12, merge_y),
        xytext=(x_add + 0.14, merge_y),
        arrowprops=dict(arrowstyle="-|>", color=ARROW, lw=1.3),
        zorder=2,
    )
    for yy in main_ys:
        ax.add_patch(Circle((x_out, yy), 0.10, facecolor=MERGE, edgecolor="#607A99", lw=1.0, zorder=3))

    ax.text(x_in, top - 1.05, f"input\n{CFG['leaf_input_dim']}\nflatten", ha="center", va="top", fontsize=8.0, color="#5A6B7E")
    ax.text(x_hidden, top - 1.05, f"hidden\n{CFG['leaf_hidden_dim_resolved']}\nfeature expansion", ha="center", va="top", fontsize=8.0, color="#5A6B7E")
    ax.text(x_main, top - 1.05, f"main\n{CFG['chi']}\ncompressed to {CFG['chi']}", ha="center", va="top", fontsize=8.0, color="#5A6B7E")
    ax.text(x_out, top - 1.05, f"final\n{CFG['chi']}\nafter add+norm", ha="center", va="top", fontsize=8.0, color="#5A6B7E")

    ax.text((x_in + x_hidden) / 2, y + 2.48, f"flatten {CFG['leaf_input_dim']}", ha="center", va="bottom", fontsize=7.6, color="#5A6B7E")
    ax.text((x_hidden + x_main) / 2, y + 0.66, f"GELU + Dropout({CFG['leaf_dropout']:.1f})", ha="center", va="center", fontsize=7.6, color="#5A6B7E")
    ax.text(
        (x_in + x_main) / 2,
        y + 0.02,
        f"skip linear projection {CFG['leaf_input_dim']}→{CFG['chi']}",
        ha="center",
        va="bottom",
        fontsize=7.2,
        color="#A16634",
    )
    ax.text(x_add + 0.02, merge_y - 0.32, "main + skip", ha="center", va="center", fontsize=7.6, color="#5A6B7E")
    ax.text(cx, y + 0.40, "LayerNorm + L2 norm", ha="center", va="center", fontsize=7.8, color="#5A6B7E")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    fig, ax = plt.subplots(figsize=(38, 10), dpi=200)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 38)
    ax.set_ylim(0, 10)
    ax.axis("off")

    CY = 5.2
    TOP_BOX_Y = CY + 1.62
    PREPROCESS_X = 2.25
    FEATURE_X = 4.95
    LEAF_X = 9.10
    TOP_MIRROR_OFFSET = 1.95
    SEGMENT_X = LEAF_X
    POSITION_X = LEAF_X + TOP_MIRROR_OFFSET
    POSITION_BOX_Y = TOP_BOX_Y
    LEAF_CY = CY - 0.78

    ax.text(
        19, 9.55,
        "TTN Functional Group Classifier — Experiment 6 Winner Run",
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
    preprocessing_label = "Preprocessing\nSNV normalisation" if CFG["apply_snv"] else "Preprocessing\nraw spectrum"
    _, pre_w, pre_h = rbox(ax, PREPROCESS_X, TOP_BOX_Y, preprocessing_label, fs=9, min_w=1.85, min_h=0.75)
    stack_cards(ax, 1.45, CY - 0.58, 1.10, 1.30, n=3, fc=BLUE)
    ax.text(2.00, CY - 0.95, "(1800×1)",
            ha="center", va="top", fontsize=8.5, color="#5A6B7E")

    # FEATURE MAP
    _, feature_w, feature_h = rbox(
        ax,
        FEATURE_X,
        TOP_BOX_Y,
        "SpectralDerivativeFeatureMap\nraw · first deriv · second deriv\nOutput: (1800×3)",
        fs=8.8,
        min_w=2.40,
        min_h=1.00,
    )
    stack_cards(ax, 3.80, CY - 0.58, 1.10, 1.30, n=3, fc=GREEN, ec="#90B988")
    ax.text(4.35, CY - 0.95, "(1800×3)",
            ha="center", va="top", fontsize=8.5, color="#5A6B7E")
    arr(ax, (2.72, CY + 0.05), (3.78, CY + 0.05), lw=1.0)

    # SEGMENTATION
    SEGMENT_BOX_Y = TOP_BOX_Y + 0.22
    _, segment_w, segment_h = rbox(
        ax,
        SEGMENT_X,
        SEGMENT_BOX_Y,
        "Sliding-Window Segmentation\n"
        f"Input: ({CFG['input_dim']}×3)\n"
        f"Output: ({CFG['num_segments']}×{CFG['segment_window_size']}×3)\n"
        f"stride={CFG['segment_stride']} ({CFG['segment_mode']})",
        fs=8.3,
        min_w=2.20,
        min_h=1.05,
    )

    # LEAF ENCODER
    leaf_left, leaf_right, leaf_bottom, leaf_top = draw_leaf_encoder_block(ax, LEAF_X, LEAF_CY)

    # TOP-ROW BOX ARROWS
    arr(ax, (1.12, TOP_BOX_Y), (PREPROCESS_X - pre_w / 2, TOP_BOX_Y), lw=1.2)
    arr(ax, (FEATURE_X + feature_w / 2, TOP_BOX_Y), (SEGMENT_X - segment_w / 2, SEGMENT_BOX_Y), lw=1.2)
    ax.annotate(
        "",
        xy=(SEGMENT_X, leaf_top + 0.02),
        xytext=(SEGMENT_X, SEGMENT_BOX_Y - segment_h / 2),
        arrowprops=dict(arrowstyle="-|>", color=ARROW, lw=1.1),
    )

    # TTN
    TTN_X = 16.45
    TTN_W = 7.80
    TTN_H = 4.20
    TTN_Y = CY - TTN_H / 2

    # POSITION EMBEDDING
    POSITION_X = (leaf_right + TTN_X) / 2
    POSITION_BOX_Y = LEAF_CY + 0.58
    _, position_w, position_h = rbox(
        ax,
        POSITION_X,
        POSITION_BOX_Y,
        "Learnable Position Embedding\n"
        f"Input: (batch, {CFG['num_segments']}, {CFG['chi']})\n"
        f"Embedding({CFG['num_segments']}, {CFG['chi']})\n"
        "position vector + segment vector\n"
        f"Output: (batch, {CFG['num_segments']}, {CFG['chi']})",
        fs=7.6,
        min_w=1.80,
        min_h=1.34,
    )
    ax.text(
        POSITION_X,
        POSITION_BOX_Y - position_h / 2 - 0.28,
        "Example: s_v=[0.10,-0.20,0.40,...],  p_v=[0.03,0.01,...]\n"
        "s_p_v = s_v + p_v = [0.13,-0.19,0.40,...]\n"
        "similar in spirit to key-value vector addition",
        ha="center",
        va="top",
        fontsize=7.1,
        color="#5A6B7E",
        zorder=6,
    )
    draw_ttn_tree(ax, TTN_X, TTN_Y, TTN_W, TTN_H)
    arr(ax, (leaf_right + 0.10, POSITION_BOX_Y), (POSITION_X - position_w / 2 - 0.08, POSITION_BOX_Y), lw=1.3)
    arr(ax, (POSITION_X + position_w / 2 + 0.08, POSITION_BOX_Y), (TTN_X - 0.10, POSITION_BOX_Y), lw=1.3)

    # POOL
    POOL_X = TTN_X + TTN_W / 2 - 0.85
    POOL_CY = TTN_Y - 0.95
    _, pool_w, pool_h = rbox(
        ax, POOL_X + 0.85, POOL_CY,
        f"Multi-Scale\nPooling\nlevels L0-L{CFG['num_readout_scales'] - 1}\n"
        f"{CFG['num_readout_scales']} pooled {CFG['chi']}-d vectors\n"
        f"(batch, {CFG['num_readout_scales']}×{CFG['chi']})",
        fs=7.8, fc=POOL_C, min_w=1.45, min_h=1.10
    )

    # FUSE
    FUSE_X = 25.65
    rbox(ax, FUSE_X + 0.75, CY + 0.02,
         f"Readout Fusion\nconcat ({CFG['num_readout_scales']}×{CFG['chi']}) → {CFG['readout_dim']}\n"
         f"LayerNorm({CFG['readout_dim']})",
         fs=8.1, min_w=1.70, min_h=0.96)
    pool_right = POOL_X + 0.85 + pool_w / 2
    elbow_x = FUSE_X - 0.16
    ax.plot(
        [pool_right, elbow_x, elbow_x],
        [POOL_CY, POOL_CY, CY],
        color=ARROW, lw=1.15, zorder=5
    )
    ax.annotate(
        "",
        xy=(FUSE_X, CY),
        xytext=(elbow_x, CY),
        arrowprops=dict(arrowstyle="-|>", color=ARROW, lw=1.15),
    )

    # MLP
    MLP_X = 28.80
    draw_mlp_block(ax, MLP_X, CY)
    arr(ax, (FUSE_X + 1.00, CY), (MLP_X - 0.05, CY))

    # MULTI-SCALE ARROW
    tree_center_x = TTN_X + TTN_W / 2
    pool_top = POOL_CY + pool_h / 2
    pool_x = POOL_X + pool_w / 2
    ax.annotate(
        "",
        xy=(pool_x, pool_top),
        xytext=(tree_center_x, TTN_Y),
        arrowprops=dict(arrowstyle="-|>", color=ARROW, lw=1.0),
    )

    ax.text(
        TTN_X + TTN_W / 2, TTN_Y - 0.26,
        f"mean over all nodes at levels L0-L{CFG['num_readout_scales'] - 1}",
        ha="center", va="center", fontsize=7.4, color="#5A6B7E"
    )

    # BOTTOM
    ax.text(19, 1.30,
            f"Readout path: {CFG['readout_dim']} → {CFG['readout_hidden_dim']} → {CFG['num_labels']}",
            ha="center", va="center", fontsize=14, weight="bold", color=TEXT)
    ax.text(
        19, 0.86,
        f"Config: input_dim={CFG['input_dim']} · num_segments={CFG['num_segments']} · chi={CFG['chi']} · "
        f"num_labels={CFG['num_labels']} · segment_window={CFG['segment_window_size']} · "
        f"stride={CFG['segment_stride']} · leaf_dropout={CFG['leaf_dropout']:.1f} · "
        f"readout_dropout={CFG['readout_dropout']:.1f} · apply_snv={CFG['apply_snv']}",
        ha="center", va="center", fontsize=9, color="#607080"
    )
    ax.text(
        19, 0.48,
        "The TTN encoder hierarchically merges derivative-aware local segments via "
        "FastRelaxedIsometricMerge; multi-scale pooled states from all levels are "
        "fused and mapped by an MLP to final functional-group logits, which are "
        "converted by sigmoid into per-label probabilities for inference.",
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
