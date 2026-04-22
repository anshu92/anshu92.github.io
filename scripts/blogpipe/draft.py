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


def _cleanup_missing_cites(md: str) -> str:
    md = re.sub(r"\s*\[missing cite:\s*[^\]]+\]", "", md)
    md = re.sub(r"[ \t]+\n", "\n", md)
    return md


def _unwrap_markdown_fence(text: str) -> str:
    t = text.strip()
    m = re.match(r"^```(?:markdown|md)?\n([\s\S]*?)\n```$", t, re.I)
    return m.group(1).strip() if m else t


def _has_result_signal(text: str) -> bool:
    return bool(
        re.search(
            r"\d+\.?\d*\s*(%|x|×|ms|GB|tokens|params|pts|F1|BLEU|ROUGE)",
            text,
            re.I,
        )
    )


def _ensure_mermaid(body: str, title: str, brief: EditorialBrief) -> str:
    if "```mermaid" in body:
        return body
    LOG.info("draft: no mermaid block; injecting diagram")
    if config.dry_run() or not config.llm_configured():
        return (
            body.rstrip()
            + "\n\n```mermaid\nflowchart LR\n  A[Input] --> B[Core method] --> C[Output]\n```\n"
        )
    raw = openrouter_client.llm_text(
        "Output only a mermaid code block (```mermaid ... ```) — no explanation. "
        f"Style: {brief.diagram_style}. Show the key method or process flow.",
        f"Paper: {title}",
    )
    if raw.strip():
        body = body.rstrip() + "\n\n" + raw.strip() + "\n"
    return body


def _apply_result_first_takeaway(body: str, title: str) -> str:
    line = body.split("\n", 1)[0].strip() if body.strip() else ""
    if not line or _has_result_signal(line):
        return body
    if config.dry_run() or not config.llm_configured():
        return body
    fixed = openrouter_client.llm_text(
        "Rewrite as ONE sentence takeaway that names the key result with a number. "
        "Output only the sentence, no quotes.",
        f"Title: {title}\nCurrent first line: {line}",
    )
    if not fixed.strip() or not _has_result_signal(fixed):
        return body
    new_line = fixed.strip()[:500]
    if "\n" in body:
        return new_line + "\n" + body.split("\n", 1)[1]
    return new_line


def _looks_like_markdown_body(text: str) -> bool:
    t = _unwrap_markdown_fence(text)
    if "## " not in t:
        return False
    if len(t.split()) < 120:
        return False
    bad_starts = (
        "i've identified",
        "i identified",
        "here's the revised",
        "here is the revised",
        "i replaced",
        "i removed",
    )
    return not t.lower().startswith(bad_starts)


def build_prompt(
    bundle: EvidenceBundle,
    brief: EditorialBrief,
    fmt: formats.PostFormat,
) -> tuple[str, str]:
    bundle.register_ids()
    sec = "\n".join(f"- ## {s}" for s in fmt.required_sections)
    ev = []
    ev.append(f"PRIMARY: {bundle.primary.title}\n{bundle.primary.abstract[:1500]}")
    b_rows = []
    for r in bundle.benchmarks:
        u = f" {r.unit}" if r.unit else ""
        b_rows.append(
            f"- {r.name}: {r.value}{u} (baseline: {r.baseline or 'n/a'}) {r.notes or ''}"
        )
    if b_rows:
        ev.append(
            "BENCHMARK rows from evidence — use ONLY these numbers in the table and prose; "
            "do not invent other metrics:\n" + "\n".join(b_rows)
        )
    for lab, items in (
        ("ANCESTORS", bundle.ancestors),
        ("COMPETITORS", bundle.competitors),
        ("FOLLOWUPS", bundle.followups),
    ):
        for it in items[:4]:
            ev.append(f"{lab} {it.title} | {it.url}\n{it.abstract[:400]}")
    if bundle.planner_questions:
        ev.append(
            "RESEARCH_QUESTIONS (cover across sections where relevant):\n"
            + "\n".join(f"- {q}" for q in bundle.planner_questions[:12])
        )
    if bundle.section_evidence:
        for k, v in bundle.section_evidence.items():
            if (v or "").strip():
                ev.append(f"SECTION_EVIDENCE[{k}]:\n{v.strip()[:6000]}")
    if bundle.contradiction_notes:
        ev.append(
            "CONTRADICTIONS_AND_LIMITS (address honestly; do not bury):\n"
            + "\n".join(f"- {x}" for x in bundle.contradiction_notes[:8])
        )
    for it in bundle.enrichment_items[:12]:
        ev.append(
            f"ENRICHMENT {it.source} {it.title} | {it.url}\n{it.abstract[:500]}"
        )
    ctx = "\n\n".join(ev)
    brief_json = brief.model_dump_json()
    system = (
        "You write Hugo markdown blog posts. Output ONLY the markdown body "
        "(no outer frontmatter — the tool will add it). "
        "Start with exactly one line: the one-sentence takeaway with a key numeric result (no heading on that line). "
        "Then use ## headings exactly as listed in required sections (you may adjust the part after a colon to be specific). "
        "For the first section after TL;DR: include 2-3 sentences of plain-English intuition for the core algorithm or "
        "method before any equations or implementation details. "
        "Use [cite: ID] where ID is one of: primary, or these exact ids: "
        + ", ".join(bundle.by_id.keys())
        + ". One mermaid code block (language mermaid) for the primary diagram. "
        "Include a markdown table with columns | Method | Metric | Baseline | under a results section. "
        "If BENCHMARK rows are given in EVIDENCE, the table and prose must use only those values — do not invent numbers. "
        "If ENRICHMENT or SECTION_EVIDENCE blocks are present, use those URLs and snippets for web or repo claims; "
        "do not invent other external sources. "
        "'What did not work': cover at least one of — (a) a reproduction barrier (hardware, missing code, compute), "
        "(b) an assumption of the method that may break (model family, task type, scale), or (c) a failure mode or "
        "limitation the authors themselves mention. Do NOT describe ablations the authors ran as 'what did not work' — "
        "ablations that validate the method are not failures. "
        "In 'What to steal': begin with one sentence of explicit author opinion — start with 'What I find' or "
        "'The remarkable thing here is' — before the bullet list. "
        "For 'Where this shows up in AEC': either (a) name a specific construction, BIM, or design-tool application "
        "with one concrete sentence, or (b) write exactly: 'There is no direct AEC application for this paper.' — "
        "do not write generic sentences about 'implications for NLP and computer vision' or 'various applications'. "
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
    body = _unwrap_markdown_fence(body)
    body = _resolve_cites(body, bundle)
    if _strip_unresolved(body):
        LOG.warning("draft: unresolved cites; attempting one repair")
        repair = openrouter_client.llm_text(
            "Fix [missing cite: ...] markers by replacing them with paraphrase without external cite, "
            "or remove the affected sentence. Output only the full revised markdown body, no commentary "
            "and no code fences.",
            body[:12000],
        )
        if repair.strip() and _looks_like_markdown_body(repair):
            body = _unwrap_markdown_fence(repair)
        else:
            body = _cleanup_missing_cites(body)
    else:
        body = _cleanup_missing_cites(body)
    body = _ensure_mermaid(body, bundle.primary.title, brief)
    body = _apply_result_first_takeaway(body, bundle.primary.title)
    struct = lint.lint_structure(body)
    if struct:
        LOG.warning("draft: structural lint: %s", struct)
    (_ROOT / "reports" / "draft_lint.json").write_text(
        json.dumps({"structural": struct}, indent=2),
        encoding="utf-8",
    )
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
        f"2× clearer tradeoffs on the main latency metric: {t.split('.')[0][:100]} in practice.\n\n"
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
        f"## What to steal\n\nWhat I find: teams under-invest in baselines until a regression lands in production.\n\n"
        f"- Try ablations first.\n- Avoid hero metrics without baselines.\n"
        f"- Run a latency budget check.\n"
    )
