"""Generate Hug draft markdown from EvidenceBundle + EditorialBrief + PostFormat.

For section-level refinement with the same PR gate, prefer ``python -m blogpipe graph``.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from . import config, formats, lint, memory, openrouter_client
from .llm_chain import get_llm_usage
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
        max_tokens=1024,
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
        max_tokens=512,
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
    goals = fmt.goals_markdown()
    ev = []
    ev.append(f"PRIMARY: {bundle.primary.title}\n{bundle.primary.abstract[:900]}")
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
            ev.append(f"{lab} {it.title} | {it.url}\n{it.abstract[:260]}")
    if bundle.planner_questions:
        ev.append(
            "RESEARCH_QUESTIONS (cover across sections where relevant):\n"
            + "\n".join(f"- {q}" for q in bundle.planner_questions[:12])
        )
    if bundle.section_evidence:
        for k, v in bundle.section_evidence.items():
            if (v or "").strip():
                ev.append(f"SECTION_EVIDENCE[{k}]:\n{v.strip()[:800]}")
    if bundle.contradiction_notes:
        ev.append(
            "CONTRADICTIONS_AND_LIMITS (address honestly; do not bury):\n"
            + "\n".join(f"- {x}" for x in bundle.contradiction_notes[:8])
        )
    for it in bundle.enrichment_items[:5]:
        ev.append(
            f"ENRICHMENT {it.source} {it.title} | {it.url}\n{it.abstract[:260]}"
        )
    ctx = "\n\n".join(ev)
    if len(ctx) > 3500:
        ctx = ctx[:3500] + "\n\n[EVIDENCE truncated to budget]\n"
    brief_json = brief.model_dump_json()
    system = (
        "You write Hugo markdown blog posts for senior engineers and researchers. "
        "Output ONLY the markdown body (no outer frontmatter — the tool will add it). "
        "A great technical blog does three things at once: it teaches something real, earns trust, "
        "and respects the reader's time.\n"
        "\nTHE 10 PRINCIPLES — follow all:\n"
        "1. Start with a real problem the reader cares about. Not history. Not abstract theory.\n"
        "2. Be concrete. Replace vague wording with specific systems, numbers, constraints, and failure "
        "modes. 'We reduced p95 from 420 ms to 180 ms by batching requests and eliminating redundant "
        "serialization' beats 'we improved performance'.\n"
        "3. Clarity over cleverness. Flow should feel inevitable: context -> problem -> approach -> "
        "implementation -> results -> limits. Do not hide the problem.\n"
        "4. Teach through examples. Show before/after, inputs/outputs, edge cases, code that earns its "
        "place, diagrams that reduce cognitive load.\n"
        "5. Make tradeoffs explicit. Name the alternatives considered and what was given up "
        "(latency, cost, simplicity, accuracy, maintainability, data availability, team maturity).\n"
        "6. Show failures and limits. Name where the method breaks, the assumptions required, and when "
        "NOT to use it. Credibility lives here. Do NOT present only the success path.\n"
        "7. Back every claim (faster / cheaper / safer / more accurate / more robust) with a number, "
        "benchmark, cited source, or explicit comparison against a named baseline. No unsupported "
        "confidence.\n"
        "8. Tight narrative. Each section answers one question that leads to the next. Kill "
        "throat-clearing, repeated caveats, and long background sections that delay the payoff.\n"
        "9. Write for the reader's next action: a technique to try, a pitfall to avoid, a pattern to "
        "steal, or a different way to think about the system.\n"
        "10. Have a point of view. Argue why a design worked, why a common assumption is wrong, why a "
        "tradeoff is underrated, or why a result will not generalise.\n"
        "\nTHE FIVE-QUESTION TEST — a smart reader must be able to answer these after one skim:\n"
        "(a) What problem is being solved?\n"
        "(b) Why is it hard?\n"
        "(c) What was tried?\n"
        "(d) What worked and what did NOT?\n"
        "(e) What should I do with this?\n"
        "\nSTRUCTURE — no rigid template:\n"
        "- Invent your own ## H2 headings that fit THIS story. Headings must be descriptive and "
        "specific: '30% lower p95 at the cost of 2x memory', 'Where batched KV cache breaks'. "
        "Generic headings are banned: Introduction, Background, Overview, Results, Summary, Conclusion.\n"
        "- Use as many sections as the story needs. No fixed count, no fixed order. Each section "
        "should earn its place by answering one clear question.\n"
        "- Do NOT emit the CONTENT GOALS text as literal headings. They are objectives to cover, "
        "not titles.\n"
        "- First line of the body: a one-sentence takeaway with a concrete number (no heading on "
        "that line).\n"
        "- Short paragraphs. Descriptive sub-heads. Highlight key takeaways with a leading bold "
        "phrase or a one-sentence summary at the end of a section so the post is skimmable.\n"
        "\nHARD REQUIREMENTS:\n"
        "- Every external claim uses [cite: ID] where ID is 'primary' or one of these exact ids: "
        + ", ".join(bundle.by_id.keys())
        + ".\n"
        "- Exactly one ```mermaid``` code block that clarifies a mechanism, data flow, or comparison. "
        "A diagram must reduce cognitive load; if it needs a paragraph to decode, rewrite it.\n"
        "- One markdown table with | Method | Metric | Baseline | where numbers are compared. "
        "Numbers in the table MUST match the prose. If BENCHMARK rows are given in EVIDENCE, use "
        "ONLY those values — do not invent metrics.\n"
        "- If ENRICHMENT or SECTION_EVIDENCE blocks are present, use those URLs/snippets for external "
        "claims; do not invent other sources.\n"
        "- State explicit tool / model / library versions wherever relevant (e.g. 'PyTorch 2.4', "
        "'Transformers 4.40') so the post does not rot silently.\n"
        "- At least one sentence of honest limitation or failure mode: a reproduction barrier "
        "(hardware, missing code, compute), a broken assumption (model family, task type, scale), "
        "or a failure the authors themselves mention. Ablations that validate the method do NOT "
        "count as 'what did not work'.\n"
        "- At least one first-person sentence of opinion ('What I find is...', 'In my view...', "
        "'I think...', 'The remarkable thing here is...'). Follow the phrase with a real opinion, "
        "not a hedge.\n"
        "- Be honest about scope: if a result holds at one scale, one dataset, one team, say so. "
        "Do not oversell a local finding as a general law.\n"
        "- Close with a concrete next action the reader can take: a technique to try, a pitfall to "
        "avoid, an experiment to run, a pattern to steal.\n"
        "- If there is a concrete construction / BIM / design-tool angle, name it in one sentence. "
        "Otherwise omit — do NOT pad with generic 'implications for NLP / various applications'.\n"
        "\nRED FLAGS — avoid these; they mark weak posts:\n"
        "- Opening with a history lesson or 'In recent years...'.\n"
        "- Marketing vocabulary: 'leveraged', 'seamless', 'powerful', 'game-changing', 'unlock', "
        "'next-generation', 'cutting-edge', 'revolutionary', 'holistic', 'synergy'.\n"
        "- Hiding the actual problem past the midpoint.\n"
        "- Code with no explanation, or code pasted 'to look technical'.\n"
        "- Only the success path — no limitations.\n"
        "- Numbers in the prose that contradict the table.\n"
        "- Impressive-sounding prose that teaches nothing reusable.\n"
        "- Reading like an internal status update dressed up as an article.\n"
        "\nOpener style: "
        + brief.opener_hook
        + ". Voice: "
        + brief.voice_guide
        + " "
        + fmt.voice_override
    )
    user = (
        f"CONTENT GOALS (cover across the post, in your own words and your own headings):\n{goals}\n\n"
        f"EDITORIAL BRIEF:\n{brief_json}\n\n"
        f"EVIDENCE:\n{ctx}\n\n"
        f"RELATED POSTS (link with [title](/post/slug/) path only, no ref shortcode in output): "
        f"{brief.recent_topics[:5]}\n"
    )
    return system, user


def run() -> Path:
    memory.ensure_dirs()
    u0 = get_llm_usage()
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
        body = openrouter_client.llm_text(
            system, user, max_tokens=config.max_tokens_smart()
        )
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
            max_tokens=2048,
        )
        if repair.strip() and _looks_like_markdown_body(repair):
            body = _unwrap_markdown_fence(repair)
        else:
            body = _cleanup_missing_cites(body)
    else:
        body = _cleanup_missing_cites(body)
    body = _ensure_mermaid(body, bundle.primary.title, brief)
    body = _apply_result_first_takeaway(body, bundle.primary.title)
    struct = lint.structural_issues(body)
    if struct:
        LOG.warning("draft: structural lint: %s", struct)
    u = get_llm_usage()
    draft_calls_ok = u["ok"] - int(u0.get("ok", 0) or 0)
    draft_calls_fail = u["fail"] - int(u0.get("fail", 0) or 0)
    rep_draft = {
        "structural": struct,
        "stage": "draft",
        "stage_llm_ok": draft_calls_ok,
        "stage_llm_fail": draft_calls_fail,
    }
    (_ROOT / "reports" / "draft_lint.json").write_text(
        json.dumps(rep_draft, indent=2),
        encoding="utf-8",
    )
    print(
        f"stage=draft calls={draft_calls_ok}/{draft_calls_ok + max(0, draft_calls_fail)} "
        f"structural={len(struct)}",
        flush=True,
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
        f"## The tradeoff in one picture\n"
        f"Teams shipping LLM features need predictable latency and cost. "
        f"Constraint: fixed GPU budget. Number: 2x is not free. [cite: {bundle.primary.id}]\n\n"
        f"## How the method actually works\n"
        f"The core idea is a targeted change to the attention path, not a bigger model. [cite: {bundle.primary.id}]\n\n"
        "```mermaid\nflowchart LR\n  A[Input] --> B[Model]\n  B --> C[Output]\n```\n\n"
        "## Numbers against a named baseline\n\n"
        "| Method | Metric | Baseline |\n|--------|--------|----------|\n"
        "| Paper | main | prior SOTA |\n\n"
        "## Where this breaks in practice\n"
        f"We could not reproduce without the full training stack; that is a real failure mode for most teams. [cite: {bundle.primary.id}]\n\n"
        "## What I would steal from this\n"
        "What I find is: teams under-invest in baselines until a regression lands in production. "
        "The habit worth stealing is the latency budget check before any new decoder.\n"
    )
