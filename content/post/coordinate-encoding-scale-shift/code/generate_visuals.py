from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).parents[1]
ENCODINGS = ["raw_pixels", "normalized", "quantized_32", "fourier"]
LABELS = {
    "raw_pixels": "raw pixels",
    "normalized": "normalized",
    "quantized_32": "32-bin",
    "fourier": "Fourier",
}
COLORS = {
    "in_domain": "#416f89",
    "translated_crop": "#9b6a2f",
}


def load_means() -> dict[tuple[str, str], float]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    with (ROOT / "data/results.csv").open() as f:
        for row in csv.DictReader(f):
            if row["model"] != "transformer":
                continue
            if row["scenario"] not in {"in_domain", "translated_crop"}:
                continue
            grouped[(row["encoding"], row["scenario"])].append(float(row["accuracy"]))
    return {key: sum(values) / len(values) for key, values in grouped.items()}


def bar(x: float, y: float, width: float, height: float, color: str) -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" fill="{color}" rx="4"/>'


def main() -> None:
    means = load_means()
    left, top, chart_w, chart_h = 120, 105, 880, 330
    floor = top + chart_h
    scale = chart_h
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 620">',
        "<title>Tiny transformer coordinate probe</title>",
        "<desc>In-domain and translated-crop accuracy for a two-point transformer classifier across coordinate encodings.</desc>",
        '<rect width="1200" height="620" fill="#fffdf8"/>',
        '<text x="70" y="62" font-size="34" font-family="sans-serif" fill="#1f252b">A tiny transformer does not remove the coordinate contract</text>',
        '<text x="70" y="95" font-size="18" font-family="sans-serif" fill="#65717a">Two point tokens, one TransformerEncoderLayer, five seeds</text>',
        f'<line x1="{left}" y1="{floor}" x2="{left + chart_w}" y2="{floor}" stroke="#ddd5c8" stroke-width="2"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{floor}" stroke="#ddd5c8" stroke-width="2"/>',
    ]
    for tick in [0.25, 0.50, 0.75, 1.00]:
        y = floor - tick * scale
        parts.append(f'<line x1="{left - 8}" y1="{y:.1f}" x2="{left + chart_w}" y2="{y:.1f}" stroke="#eee5d6" stroke-width="1"/>')
        parts.append(f'<text x="{left - 18}" y="{y + 6:.1f}" font-size="16" text-anchor="end" font-family="sans-serif" fill="#65717a">{tick:.2f}</text>')
    group_w = chart_w / len(ENCODINGS)
    bar_w = 56
    for i, enc in enumerate(ENCODINGS):
        base = left + i * group_w + 58
        for j, scenario in enumerate(["in_domain", "translated_crop"]):
            value = means[(enc, scenario)]
            h = value * scale
            x = base + j * (bar_w + 14)
            y = floor - h
            parts.append(bar(x, y, bar_w, h, COLORS[scenario]))
            parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 8:.1f}" font-size="15" text-anchor="middle" font-family="sans-serif" fill="#1f252b">{value:.3f}</text>')
        parts.append(f'<text x="{base + bar_w + 7:.1f}" y="{floor + 32}" font-size="18" text-anchor="middle" font-family="sans-serif" fill="#1f252b">{LABELS[enc]}</text>')
    parts += [
        '<rect x="790" y="438" width="20" height="20" fill="#416f89" rx="3"/>',
        '<text x="820" y="455" font-size="18" font-family="sans-serif" fill="#1f252b">in-domain</text>',
        '<rect x="790" y="472" width="20" height="20" fill="#9b6a2f" rx="3"/>',
        '<text x="820" y="489" font-size="18" font-family="sans-serif" fill="#1f252b">translated crop</text>',
        '<text x="120" y="548" font-size="18" font-family="sans-serif" fill="#65717a">The transformer learns normalized and quantized coordinates well in-domain, but the origin shift still exposes absolute-position dependence.</text>',
        "</svg>",
    ]
    (ROOT / "figures/figure-04-transformer.svg").write_text("\n".join(parts) + "\n")


if __name__ == "__main__":
    main()
