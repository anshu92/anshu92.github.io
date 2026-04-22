"""Generate Hug draft markdown from EvidenceBundle + EditorialBrief + PostFormat."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from . import config, formats, lint, memory, openrouter_client
from .models import EditorialBrief, EvidenceBundle, Item
from .memory import _ROOT

LOG = logging.getLogger(__name__)


def _slugify(title: str) -> str:
    s = re.sub(r"[^\w\s-]", "", title.lower())
    s = re.sub(r"[-\s]+", "-", s).strip("-")
    return s[:80] or "post"


def _load_json(name: str) -> dict:
    p = _ROOT / "reports" / name
    return json.loads(p.read_text()) if p.is_file() else {}


def _resolve_cites(md: str, bundle: EvidenceBundle) -> str:
    bundle.register_ids()

    def repl(m: re.Match[str]) -> str:
        cid = m.group(1).strip()
        it = bundle.by_id.get(cid)
        if not it:
            return f"[missing cite: {cid}]"
        return f"[{it.title[:60]}]({it.url})"

    return re.sub(r"\[cite:\s*([^\]]+)\]", repl, md)


def _strip_unresolved(md: str) -> bool:
    return bool(re.search(r"\[missing cite:", md))


def build_prompt(
    bundle: EvidenceBundle,
    brief: EditorialBrief,
    fmt: formats.PostFormat,
) -> tuple[str, str]:
    sec = "\n".join(f"- ## {s}" for s in fmt.required_sections)
    ev = []
    ev.append(f"PRIMARY: {bundle.primary.title}\n{bundle.primary.abstract[:1500]}")
    for lab, items in (
        ("ANCESTORS", bundle.ancestors),
        ("COMPETITORS", bundle.competitors),
        ("FOLLOWUPS", bundle.followups),
    ):
        for it in items[:4]:
            ev.append(f"{lab} {it.title} | {it.url}\n{it.abstract[:400]}")
    ctx = "\n\n".join(ev)
    brief_json = brief.model_dump_json()
    system = (
        "You write Hugo markdown blog posts. Output ONLY the markdown body "
        "(no outer frontmatter — the tool will add it). "
        "Start with exactly one line: the one-sentence takeaway (no heading on that line). "
        "Then use ## headings exactly as listed in required sections (you may adjust the part after a colon to be specific). "
        "Use [cite: ID] where ID is one of: primary, or these exact ids: "
        + ", ".join(bundle.by_id.keys())
        + ". One mermaid code block (language mermaid) for the primary diagram. "
        "Include a markdown table with columns | Method | Metric | Baseline | under a results section. "
        "Include 'What did not work' with real failure content. "
        "Opener style: "
        + brief.opener_hook
        + ". Voice: "
        + brief.voice_guide
        + " "
        + fmt.voice_override
    )
    user = (
        f"REQUIRED SECTIONS:\n{sec}\n\n"
        f"EDITORIAL BRIEF:\n{brief_json}\n\n"
        f"EVIDENCE:\n{ctx}\n\n"
        f"RELATED POSTS (link with [title](/post/slug/) path only, no ref shortcode in output): "
        f"{brief.recent_topics[:5]}\n"
    )
    return system, user


def run() -> Path:
    memory.ensure_dirs()
    bundle = EvidenceBundle.model_validate_json(
        (_ROOT / "reports" / "evidence_bundle.json").read_text()
    )
    brief = EditorialBrief.model_validate_json(
        (_ROOT / "reports" / "editorial_brief.json").read_text()
    )
    fmt = formats.FORMATS.get(brief.format_name) or formats.FORMATS["deep_dive"]
    system, user = build_prompt(bundle, brief, fmt)
    body = ""
    if config.dry_run() or not config.llm_configured():
        body = _stub_body(bundle, brief, fmt)
    else:
        body = openrouter_client.llm_text(system, user)
        if not body.strip():
            body = _stub_body(bundle, brief, fmt)
    body = _resolve_cites(body, bundle)
    if _strip_unresolved(body):
        LOG.warning("draft: unresolved cites; attempting one repair")
        repair = openrouter_client.llm_text(
            "Fix [missing cite: ...] by replacing with paraphrase without external cite, or remove sentence.",
            body[:12000],
        )
        if repair.strip():
            body = repair
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    slug = _slugify(bundle.primary.title)
    title = bundle.primary.title[:200]
    takeaway = body.split("\n", 1)[0].strip()[:500]
    fm = f"""---
date: "{day}"
draft: true
title: "{title.replace(chr(34), "'")}"
description: "{takeaway[:160].replace(chr(34), "'")}"
categories: ["Machine Learning"]
tags: {json.dumps(bundle.primary.tags[:8] + ["blogpipe"])}
math: true
mermaid: true
one_sentence_takeaway: "{takeaway[:300].replace(chr(34), "'")}"
image: /img/posts/{slug}/hero.png
rubric_score: 0
---

"""
    full = fm + body.lstrip()
    if config.dry_run():
        out = _ROOT / "reports" / f"draft_preview_{day}-{slug}.md"
    else:
        out = _ROOT / "content" / "post" / f"{day}-{slug}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(full, encoding="utf-8")
    (_ROOT / "reports" / "draft_path.txt").write_text(str(out.relative_to(_ROOT)), encoding="utf-8")
    LOG.info("draft: wrote %s", out)
    return out


def _stub_body(
    bundle: EvidenceBundle, brief: EditorialBrief, fmt: formats.PostFormat
) -> str:
    """Deterministic placeholder for dry runs / no API key."""
    t = bundle.primary.title
    return (
        f"{t.split('.')[0][:120]}: a concise engineering view of the paper and its limits.\n\n"
        f"## TL;DR\n- Problem: training and deploying at scale.\n"
        f"- Takeaway: measure p95 before p50. [cite: {bundle.primary.id}]\n\n"
        f"## Why this matters\n"
        f"Teams shipping LLM features need predictable latency and cost. "
        f"This matters to platform engineers and applied researchers. "
        f"Constraint: fixed GPU budget. Number: 2x is not free. "
        f"You will learn where the paper helps and where it breaks.\n\n"
        + "\n".join(f"## {s}\n\nProse placeholder. [cite: {bundle.primary.id}]\n" for s in fmt.required_sections[1:6])
        + "\n\n## What did not work\n\nWe could not reproduce without the full training stack; "
        "that is a real failure mode for most teams.\n\n"
        "```mermaid\nflowchart LR\n  A[Input] --> B[Model]\n  B --> C[Output]\n```\n\n"
        "| Method | Metric | Baseline |\n|--------|--------|----------|\n"
        "| Paper | main | prior SOTA |\n\n"
        f"## Related posts on this site\n\nSee /post/ for prior notes.\n\n"
        f"## What to steal\n\n- Try ablations first.\n- Avoid hero metrics without baselines.\n"
        f"- Run a latency budget check.\n"
    )
