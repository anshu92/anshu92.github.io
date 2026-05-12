from __future__ import annotations

import html
import json
from collections import Counter
from pathlib import Path

from . import memory
from .models import RankedItem


def render_daily_assets(ranked: list[RankedItem], slug: str) -> list[Path]:
    return render_post_assets(ranked, slug)


def render_post_assets(ranked: list[RankedItem], slug: str) -> list[Path]:
    out_dir = memory.post_asset_dir(slug)
    paths = [
        _source_mix_svg(ranked, out_dir / "source-mix.svg"),
        _topic_mix_svg(ranked, out_dir / "topic-mix.svg"),
    ]
    (out_dir / "manifest.json").write_text(
        json.dumps({"assets": [p.name for p in paths]}, indent=2),
        encoding="utf-8",
    )
    return paths


def mermaid_for_ranked(ranked: list[RankedItem]) -> str:
    lines = ["flowchart LR", "  A[Sources] --> B[Normalize and dedupe]", "  B --> C[Rank and diversify]"]
    for idx, item in enumerate(ranked[:5], start=1):
        lines.append(f"  C --> I{idx}[{_mermaid_label(item.item.title)}]")
    return "\n".join(lines)


def _source_mix_svg(ranked: list[RankedItem], path: Path) -> Path:
    counts = Counter(r.item.source_kind for r in ranked)
    _bar_svg(path, "Source mix", counts)
    return path


def _topic_mix_svg(ranked: list[RankedItem], path: Path) -> Path:
    counts = Counter(track for r in ranked for track in r.topic_scores.tracks)
    _bar_svg(path, "Topic mix", counts)
    return path


def _bar_svg(path: Path, title: str, counts: Counter[str]) -> None:
    width, height = 720, 320
    labels = list(counts) or ["none"]
    max_value = max(counts.values() or [1])
    rows = []
    for idx, label in enumerate(labels):
        value = counts[label]
        y = 70 + idx * 54
        bar_width = int(460 * (value / max_value))
        rows.append(f'<text x="32" y="{y + 22}" font-size="18">{html.escape(label)}</text>')
        rows.append(f'<rect x="180" y="{y}" width="{bar_width}" height="30" fill="#2563eb" rx="4"/>')
        rows.append(f'<text x="{190 + bar_width}" y="{y + 22}" font-size="16">{value}</text>')
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" role="img">'
        f'<rect width="100%" height="100%" fill="#ffffff"/>'
        f'<text x="32" y="40" font-size="24" font-family="Arial" font-weight="700">{html.escape(title)}</text>'
        f'<g font-family="Arial" fill="#111827">{"".join(rows)}</g></svg>'
    )
    path.write_text(svg, encoding="utf-8")


def _mermaid_label(text: str) -> str:
    return "".join(ch for ch in text[:42] if ch.isalnum() or ch in " -_:").strip() or "item"
