"""Index existing posts, emit EditorialBrief + format pick."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

from . import config, formats, memory
from .config import art_cooldown, dry_run, force_format, format_cooldown, pillar_floor
from .models import EditorialBrief, Pillar, PostMeta
from .memory import _ROOT, load_json, save_json
from .openrouter_client import embed_text

LOG = logging.getLogger(__name__)

_TLDR = re.compile(r"^##\s*TL;DR", re.M)
_H2 = re.compile(r"^##\s+(.+)$", re.M)
_FOLLOW = re.compile(
    r"(in a future post|we'll revisit|open question|limitation|TODO:)",
    re.I,
)
_OPENER_STYLES = [
    "in_medias_res_scene",
    "contrarian_claim",
    "single_number",
    "reader_question",
    "concrete_artifact",
    "historical_rupture",
    "cold_open_dialogue",
]
_ART_DIRS = [
    "editorial_illustration",
    "isometric_diagram",
    "minimal_geometric",
    "risograph_print",
    "data_viz_portrait",
    "hand_drawn_schematic",
    "cyan_blueprint",
    "dark_technical_photo",
    "collage_cutout",
]
_DIAG = ["flowchart", "sequence", "state", "timeline", "block"]


def _read_front_matter(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n", raw, re.DOTALL)
    if not m:
        return {}, raw
    try:
        fm = yaml.safe_load(m.group(1)) or {}
    except Exception:
        fm = {}
    body = raw[m.end() :]
    if not isinstance(fm, dict):
        fm = {}
    return fm, body


def _classify_pillar(
    path: str, title: str, tags: list[str], cats: list[str], text: str
) -> Pillar:
    t = f"{title} {' '.join(tags)} {' '.join(cats)} {text[:2000]}".lower()
    if any(
        w in t
        for w in (
            "aec",
            "bim",
            "construction",
            "autodesk",
            "building",
            "cad",
            "civil",
        )
    ):
        return Pillar.aec
    if any(
        w in t
        for w in (
            "attention",
            "theory",
            "foundation",
            "deep-dive",
            "from code to theory",
        )
    ):
        return Pillar.foundations
    if any(
        w in t
        for w in (
            "inference",
            "gpu",
            "throughput",
            "kernel",
            "distributed",
            "ray",
            "vllm",
            "triton",
        )
    ):
        return Pillar.systems
    if any(
        w in t
        for w in (
            "rag",
            "agent",
            "eval",
            "prod",
            "orchestrat",
        )
    ):
        return Pillar.applied
    return Pillar.research


def _index_post(path: Path) -> Optional[PostMeta]:
    fm, body = _read_front_matter(path)
    if fm.get("draft") is True:
        return None
    title = str(fm.get("title", path.stem))
    date = str(fm.get("date", ""))[:10]
    tags = [str(x) for x in (fm.get("tags") or []) if x is not None]
    cats = [str(x) for x in (fm.get("categories") or []) if x is not None]
    h2s = [h.strip() for h in _H2.findall(body)][:30]
    tldr = ""
    if _TLDR.search(body):
        s = _TLDR.search(body)
        if s:
            end = body.find("##", s.end())
            tldr = body[s.end() : end if end > 0 else len(body)].strip()[:500]
    pillar = _classify_pillar(str(path), title, tags, cats, body)
    wc = len(body.split())
    return PostMeta(
        path=str(path.relative_to(_ROOT)),
        slug=path.stem,
        title=title,
        date=date,
        categories=cats,
        tags=tags,
        h2_titles=h2s,
        tldr=tldr,
        pillar=pillar,
        word_count=wc,
    )


def _embed_posts(metas: list[PostMeta]) -> None:
    if dry_run() or not config.openrouter_key() and not config.openai_key():
        return
    for m in metas[-30:]:  # incremental cap
        text = f"{m.title}\n{m.tldr}\n" + " ".join(m.h2_titles)
        v = embed_text(text[:8000])
        if v is not None:
            m.embedding = v


def _build_voice_guide(metas: list[PostMeta]) -> str:
    subst = [m for m in metas if m.word_count > 500][:5]
    if not subst:
        return "Clear, first-principles, concrete numbers, peer tone."
    lines = [f"Example titles: {', '.join(s.title for s in subst)}"]
    return "\n".join(lines)


def _ledger_row() -> dict[str, Any]:
    p = _ROOT / "cache" / "variety_ledger.json"
    if p.is_file():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {"entries": []}


def _last_n_formats(n: int) -> list[str]:
    e = _ledger_row().get("entries") or []
    return [str(x.get("format", "")) for x in e[-n:]]


def _pick_format(metas: list[PostMeta], brief: dict[str, float]) -> tuple[str, str, str, str, str]:
    """Returns format_name, opener, art, diagram, rationale."""
    counts: dict[str, int] = {}
    for f in _last_n_formats(10):
        counts[f] = counts.get(f, 0) + 1
    # Boost under-represented pillars from brief weights
    w = list(brief.items()) if brief else [("systems", 0.2), ("research", 0.2)]
    systems_boost = any(k == "systems" and v > 0.3 for k, v in w)

    candidates = list(formats.FORMATS.keys())
    fmt = "deep_dive"
    if (ff := force_format().strip()) and ff in formats.FORMATS:
        fmt = ff
    else:
        # Avoid repeating last format too often
        last = _last_n_formats(3)
        scored: list[tuple[int, str]] = []
        for c in candidates:
            score = 0
            if last and c == last[-1]:
                score -= 5
            if counts.get(c, 0) >= 3:
                score -= 3
            if c == "dialogue" and sum(1 for x in _last_n_formats(10) if x == "dialogue") >= 1:
                score -= 20
            if c == "opinion_with_receipts" and sum(1 for x in _last_n_formats(10) if x == "opinion_with_receipts") >= 1:
                score -= 20
            if systems_boost and c in (
                "under_the_hood",
                "benchmark_shootout",
                "field_notes",
            ):
                score += 3
            if not systems_boost and c == "deep_dive":
                score += 1
            scored.append((score, c))
        scored.sort(reverse=True)
        fmt = scored[0][1] if scored else "deep_dive"

    # Opener: differ from last 3
    ent = _ledger_row().get("entries") or []
    used_open = {str(x.get("opener")) for x in ent[-3:]}
    for o in _OPENER_STYLES:
        if o not in used_open and not (o == "cold_open_dialogue" and fmt != "dialogue"):
            opener = o
            break
    else:
        opener = "in_medias_res_scene"

    # Art direction
    used_art = {str(x.get("art")) for x in ent[-art_cooldown() :]}
    for a in _ART_DIRS:
        if a not in used_art:
            art = a
            break
    else:
        art = _ART_DIRS[0]

    # Diagram
    used_d = [str(x.get("diagram")) for x in ent[-6:]]
    from collections import Counter

    c = Counter(used_d)
    for d in _DIAG:
        if c.get(d, 0) == min(c.get(x, 0) for x in _DIAG):
            diagram = d
            break
    else:
        diagram = "flowchart"

    rationale = (
        f"Format {fmt} balances last 10 (counts {counts!s}) and systems_boost={systems_boost}."
    )
    return fmt, opener, art, diagram, rationale


def _pillar_weights(metas: list[PostMeta]) -> dict[str, float]:
    from collections import Counter

    recent = [m for m in metas if m.date][:20]
    if not recent:
        return {
            "research": 0.25,
            "systems": 0.25,
            "applied": 0.2,
            "foundations": 0.15,
            "aec": 0.15,
        }
    c = Counter()
    for m in recent[:10]:
        c[str(m.pillar)] += 1
    n = max(sum(c.values()), 1)
    floor = pillar_floor()
    out: dict[str, float] = {}
    for p in Pillar:
        share = c.get(p.value, 0) / n
        out[p.value] = max(floor, (1.0 - share) * 0.2 + (floor if share < floor else 0.0))
    s = sum(out.values()) or 1.0
    return {k: v / s for k, v in out.items()}


def run() -> EditorialBrief:
    memory.ensure_dirs()
    memory.try_restore_from_branch(
        "blogpipe-memory",
        "cache/covered_papers.json",
        "cache/post_index.json",
        "cache/variety_ledger.json",
    )
    post_dir = _ROOT / "content" / "post"
    metas: list[PostMeta] = []
    for path in sorted(post_dir.glob("*.md")):
        try:
            m = _index_post(path)
            if m:
                metas.append(m)
        except Exception as e:
            LOG.debug("curator skip %s: %s", path, e)
    metas.sort(key=lambda x: x.date, reverse=True)
    _embed_posts(metas)
    if metas and any(m.embedding is not None for m in metas):
        save_json(
            "post_index.json",
            {
                "updated": datetime.now(timezone.utc).isoformat(),
                "posts": [m.model_dump(mode="json") for m in metas],
            },
        )
    pweights = _pillar_weights(metas)
    recent = [
        {
            "slug": m.slug,
            "title": m.title,
            "pillar": str(m.pillar),
        }
        for m in metas[:20]
    ]
    follow: list[str] = []
    for m in metas:
        p = (m.tldr or "") + "\n" + "\n".join(m.h2_titles)
        for line in p.splitlines():
            if _FOLLOW.search(line) and line.strip() not in follow:
                follow.append(line.strip()[:200])
    follow = follow[:10]
    avoid: list[str] = []
    cp = load_json("covered_papers.json", {})
    for k, v in (cp or {}).get("by_url", {}).items():
        if isinstance(v, str) and "T-" in v:
            avoid.append(k[:200])
    voice = _build_voice_guide(metas)
    fmt, opener, art, diagram, fr = _pick_format(metas, pweights)
    brief = EditorialBrief(
        pillar_weights=pweights,
        recent_topics=recent,
        follow_up_candidates=follow,
        avoid_topics=avoid[:30],
        voice_guide=voice,
        suggested_series=None,
        format_name=fmt,
        format_rationale=fr,
        opener_hook=opener,
        art_direction=art,
        diagram_style=diagram,
        proactive_topic=follow[0] if follow else None,
    )
    memory.REPORTS.mkdir(parents=True, exist_ok=True)
    p = _ROOT / "reports" / "editorial_brief.json"
    p.write_text(brief.model_dump_json(indent=2), encoding="utf-8")
    LOG.info("curator: wrote editorial_brief.json format=%s", fmt)
    return brief


