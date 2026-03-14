from __future__ import annotations

from pathlib import Path


WIDTH = 2300
HEIGHT = 840
BG = "#f3f4f6"
TEXT = "#1f2937"
MUTED = "#6b7280"
LINE = "#94a3b8"
BLUE = "#bcd3ea"
GREEN = "#cfe8c8"
PURPLE = "#d9d3f3"
GRAY = "#b8c0cc"
ORANGE = "#ff6b5b"
NODE = "#7acb5a"
STROKE = "#4b5563"


def rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = STROKE, rx: int = 0) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="2" />'
    )


def line(x1: float, y1: float, x2: float, y2: float, color: str = LINE, dashed: bool = False, arrow: bool = False) -> str:
    dash = ' stroke-dasharray="6 5"' if dashed else ""
    marker = ' marker-end="url(#arrow)"' if arrow else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
        f'stroke-width="2"{dash}{marker} />'
    )


def text(x: float, y: float, value: str, size: int = 22, weight: str = "600", anchor: str = "middle", fill: str = TEXT) -> str:
    return (
        f'<text x="{x}" y="{y}" fill="{fill}" font-size="{size}" font-family="Arial, Helvetica, sans-serif" '
        f'font-weight="{weight}" text-anchor="{anchor}">{value}</text>'
    )


def text_lines(x: float, y: float, lines: list[str], size: int = 18, weight: str = "500", anchor: str = "middle", fill: str = TEXT, line_gap: int = 24) -> str:
    parts = [
        f'<text x="{x}" y="{y}" fill="{fill}" font-size="{size}" font-family="Arial, Helvetica, sans-serif" '
        f'font-weight="{weight}" text-anchor="{anchor}">'
    ]
    for i, value in enumerate(lines):
        dy = 0 if i == 0 else line_gap
        parts.append(f'<tspan x="{x}" dy="{dy}">{value}</tspan>')
    parts.append("</text>")
    return "".join(parts)


def label_box(x: float, y: float, w: float, h: float, lines: list[str], fill: str = "#ffffff", size: int = 16) -> str:
    return (
        rect(x, y, w, h, fill, stroke=LINE, rx=10)
        + text_lines(x + w / 2, y + 24, lines, size=size, weight="700", line_gap=18)
    )


def circle(cx: float, cy: float, r: float, fill: str, stroke: str = STROKE) -> str:
    return f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="2" />'


def stack(x: float, y: float, w: float, h: float, layers: int, dx: float, dy: float, fill: str) -> str:
    parts: list[str] = []
    for i in range(layers):
        parts.append(
            rect(
                x + i * dx,
                y + (layers - 1 - i) * dy,
                w,
                h,
                fill,
                stroke=LINE,
            )
        )
    parts.append(rect(x + dx * 2, y + dy * 2, w, h, fill, stroke=STROKE))
    parts.append(
        rect(
            x + dx * 2 + w * 0.42,
            y + dy * 2 + h * 0.24,
            w * 0.18,
            h * 0.18,
            "none",
            stroke="#f87171",
            rx=2,
        )
    )
    return "".join(parts)


def tiny_spectrum(x: float, y: float, w: float, h: float) -> str:
    baseline = y + h - 18
    points = [
        (x + 12, baseline),
        (x + 20, baseline - 4),
        (x + 22, baseline),
        (x + 28, baseline - 26),
        (x + 30, baseline),
        (x + 42, baseline - 9),
        (x + 44, baseline),
        (x + 56, baseline - 4),
        (x + 58, baseline),
        (x + 70, baseline - 32),
        (x + 72, baseline),
        (x + 88, baseline - 12),
        (x + 90, baseline),
    ]
    poly = " ".join(f"{px},{py}" for px, py in points)
    return (
        rect(x, y, w, h, "#ffffff")
        + line(x + 10, baseline, x + w - 10, baseline, "#111827")
        + line(x + 18, y + 18, x + 18, baseline + 1, "#111827")
        + f'<polyline points="{poly}" fill="none" stroke="#111827" stroke-width="3" />'
        + rect(x + 38, y + 38, 30, 44, "none", stroke="#f87171", rx=2)
    )


def vector_stack(x: float, y: float, size: float, count: int, gap: float, fill: str) -> str:
    parts: list[str] = []
    for i in range(count):
        parts.append(rect(x, y + i * (size + gap), size, size, fill, stroke=LINE))
    return "".join(parts)


def dense_column(x: float, top: float, count: int, spacing: float, radius: float, fill: str) -> tuple[str, list[tuple[float, float]]]:
    parts: list[str] = []
    centers: list[tuple[float, float]] = []
    for i in range(count):
        cy = top + i * spacing
        centers.append((x, cy))
        parts.append(circle(x, cy, radius, fill))
    return "".join(parts), centers


def connect_columns(left: list[tuple[float, float]], right: list[tuple[float, float]]) -> str:
    parts: list[str] = []
    for x1, y1 in left:
        for x2, y2 in right:
            parts.append(line(x1 + 16, y1, x2 - 16, y2, color="#cbd5e1"))
    return "".join(parts)


def mps_chain(x: float, y: float, title: str) -> str:
    parts = [text(x + 140, y - 20, title, size=18, weight="700")]
    parts.append(text(x + 140, y + 8, "9 cores, physical_dim = 150, bond_dim = 128", size=14, weight="500", fill=MUTED))
    box_y = y + 34
    centers = []
    for i in range(9):
        bx = x + i * 30
        parts.append(rect(bx, box_y, 20, 20, PURPLE, stroke=STROKE, rx=3))
        centers.append((bx + 10, box_y + 10))
    for i in range(8):
        parts.append(line(centers[i][0] + 10, centers[i][1], centers[i + 1][0] - 10, centers[i + 1][1], color=STROKE))
    parts.append(text(x + 140, box_y + 40, "final state = 128", size=15, weight="700"))
    return "".join(parts)


def build_svg() -> str:
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        "<defs>",
        '<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">',
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#6b7280" />',
        "</marker>",
        "</defs>",
        rect(0, 0, WIDTH, HEIGHT, BG, stroke=BG),
        text(WIDTH / 2, 42, "MPS Functional Group Classifier", size=28, weight="700"),
    ]

    parts.append(tiny_spectrum(14, 220, 100, 146))
    parts.append(text_lines(64, 400, ["INPUT", "(1800 x 1)", "IR spectrum"], size=18, weight="700"))

    parts.append(line(124, 292, 172, 292, arrow=True))

    parts.append(stack(170, 242, 126, 114, layers=6, dx=12, dy=10, fill=BLUE))
    parts.append(label_box(146, 116, 220, 74, ["Reshape to sites", "(9 segments, each 200 values)"], size=16))
    parts.append(text(246, 438, "(9 x 200)", size=18, weight="600"))

    parts.append(line(356, 292, 408, 292, arrow=True))

    parts.append(stack(412, 252, 114, 104, layers=5, dx=10, dy=10, fill=GREEN))
    parts.append(label_box(392, 102, 220, 114, ["Shared LocalFeatureMap", "LayerNorm(200)", "Linear 200 -> 300 -> 150", "GELU, Dropout(0.05), Tanh"], size=15))
    parts.append(text(470, 438, "(9 x 150)", size=18, weight="600"))
    parts.append(text(470, 464, "Params: 105,850", size=16, weight="600", fill=MUTED))

    parts.append(line(540, 292, 608, 292, arrow=True))

    parts.append(rect(632, 176, 470, 300, "#eef2ff", stroke=LINE, rx=12))
    parts.append(label_box(734, 92, 270, 58, ["Bidirectional MPS Encoder"], fill="#eef2ff", size=18))
    parts.append(mps_chain(708, 226, "Forward MPS contraction"))
    parts.append(mps_chain(708, 370, "Backward MPS contraction"))
    parts.append(line(988, 278, 1060, 278, arrow=True))
    parts.append(line(988, 422, 1060, 422, arrow=True))
    parts.append(label_box(1068, 282, 120, 60, ["Concatenate", "128 + 128 = 256"], fill="#ffffff", size=14))
    parts.append(text(867, 510, "MPS encoder params: 39,393,408", size=18, weight="700", fill=MUTED))

    parts.append(line(1192, 312, 1252, 312, arrow=True))

    parts.append(vector_stack(1270, 192, 26, 7, 14, GRAY))
    for y in [206, 246, 286, 326, 366, 406]:
        parts.append(line(1296, y, 1388, 182 + (y - 206) * 1.15, dashed=True))
    parts.append(label_box(1224, 96, 150, 72, ["Flatten / concat vector", "(256 values)"], size=15))
    parts.append(text(1282, 540, "Total: 256 units", size=16, weight="700", fill=MUTED))

    fc1, fc1_nodes = dense_column(1470, 125, 7, 62, 15, NODE)
    fc2, fc2_nodes = dense_column(1700, 125, 7, 62, 15, NODE)
    fc3, fc3_nodes = dense_column(1930, 125, 7, 62, 15, NODE)
    out, out_nodes = dense_column(2170, 125, 7, 62, 15, ORANGE)
    parts.append(connect_columns([(1296, 206), (1296, 246), (1296, 286), (1296, 326), (1296, 366), (1296, 406)] + [(1296, 446)], fc1_nodes))
    parts.append(connect_columns(fc1_nodes, fc2_nodes))
    parts.append(connect_columns(fc2_nodes, fc3_nodes))
    parts.append(connect_columns(fc3_nodes, out_nodes))
    parts.append(fc1)
    parts.append(fc2)
    parts.append(fc3)
    parts.append(out)

    parts.append(label_box(1405, 34, 130, 64, ["proj_1", "Output projection", "Linear 256 -> 128"], size=14))
    parts.append(label_box(1635, 34, 130, 64, ["fc_1", "Fully-Connected", "Linear 128 -> 64"], size=14))
    parts.append(label_box(1865, 34, 130, 64, ["fc_2", "Output layer", "Linear 64 -> 37"], size=14))
    parts.append(label_box(2095, 34, 130, 64, ["OUTPUT", "(logits)", "37 labels"], size=14))
    parts.append(text(1470, 540, "Total: 128 units", size=16, weight="700", fill=MUTED))
    parts.append(text(1700, 540, "Total: 64 units", size=16, weight="700", fill=MUTED))
    parts.append(text(1930, 540, "Total: 37 logits", size=16, weight="700", fill=MUTED))
    parts.append(text(2170, 540, "Total: 37 labels", size=16, weight="700", fill=MUTED))

    parts.append(text(2200, 130, "0", size=16, weight="700", anchor="start"))
    parts.append(text(2200, 192, "1", size=16, weight="700", anchor="start"))
    parts.append(text(2200, 254, "2", size=16, weight="700", anchor="start"))
    parts.append(text(2200, 338, "...", size=16, weight="700", anchor="start"))
    parts.append(text(2200, 502, "36", size=16, weight="700", anchor="start"))

    parts.append(text(WIDTH / 2, 700, "Total parameters: 39,510,175", size=24, weight="700"))
    parts.append(text(WIDTH / 2, 732, "Current config: input_dim=1800, num_sites=9, physical_dim=150, bond_dim=128, num_classes=37", size=16, weight="500", fill=MUTED))
    parts.append(text(WIDTH / 2, 760, "Note: in the MPS block, the key capacity is the tensor core shape, not only classical neuron counts.", size=16, weight="500", fill=MUTED))
    parts.append("</svg>")
    return "".join(parts)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "src" / "spectroscopy_qml" / "ir" / "mps_encoder" / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    svg_path = out_dir / "mps_architecture.svg"
    svg_path.write_text(build_svg(), encoding="utf-8")
    print(svg_path)


if __name__ == "__main__":
    main()
