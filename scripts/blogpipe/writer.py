from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from markdown_it import MarkdownIt

from . import memory
from .llm import LLMClient
from .models import EvidencePack, RankedItem, WriteResult

EVIDENCE_REF_RE = re.compile(r"\[E(\d+)\]")
NUMBER_RE = re.compile(r"(?<![\w-])\d+(?:\.\d+)?%?(?![\w-])")


def write_daily(pack: EvidencePack, *, llm: LLMClient | None = None, dry_run: bool = False) -> WriteResult:
    client = llm or LLMClient()
    today = datetime.now(timezone.utc).date().isoformat()
    title = f"Daily LLM, MLE, and AEC Technical Radar - {today}"
    body = _call_writer(client, _daily_system(), _daily_user(pack, title))
    return _validate_repair_and_publish(
        client=client,
        pack=pack,
        title=title,
        body=body,
        slug=f"{today}-research-radar",
        post_type="daily",
        dry_run=dry_run,
    )


def write_deep_dive(pack: EvidencePack, *, llm: LLMClient | None = None, dry_run: bool = False) -> WriteResult:
    client = llm or LLMClient()
    primary = pack.ranked_items[0].item
    today = datetime.now(timezone.utc).date().isoformat()
    title = f"{primary.title} - guided learning deep dive"
    slug = f"{today}-{memory.slugify(primary.title)}-guided-deep-dive"
    body = _call_writer(client, _deep_system(), _deep_user(pack, title))
    return _validate_repair_and_publish(
        client=client,
        pack=pack,
        title=title,
        body=body,
        slug=slug,
        post_type="deep_dive",
        dry_run=dry_run,
    )


def validate_body(body: str, pack: EvidencePack) -> list[str]:
    errors: list[str] = []
    evidence_ids = {chunk.evidence_id for chunk in pack.chunks}
    refs = {f"E{m.group(1)}" for m in EVIDENCE_REF_RE.finditer(body or "")}
    unknown = sorted(refs - evidence_ids)
    if unknown:
        errors.append("unknown_evidence_ids:" + ",".join(unknown))
    if not refs:
        errors.append("no_evidence_ids")
    for url in pack.urls():
        if url and url not in body:
            errors.append("missing_source_link:" + url)
    evidence_blob = pack.evidence_blob()
    for number in _meaningful_numbers(body):
        if number not in evidence_blob:
            errors.append("unsupported_number:" + number)
    if _copies_large_evidence_span(body, pack):
        errors.append("copied_large_evidence_span")
    try:
        MarkdownIt().parse(body)
    except Exception as exc:
        errors.append(f"markdown_parse_error:{exc}")
    return errors


def _call_writer(client: LLMClient, system: str, user: str) -> str:
    return _strip_fence(client.complete(system=system, user=user)).strip()


def _validate_repair_and_publish(
    *,
    client: LLMClient,
    pack: EvidencePack,
    title: str,
    body: str,
    slug: str,
    post_type: str,
    dry_run: bool,
) -> WriteResult:
    errors = validate_body(body, pack)
    repair_attempted = False
    if errors:
        repair_attempted = True
        repair_user = (
            "Fix this Markdown draft. Keep the same topic and source links. "
            "Use only evidence IDs from EVIDENCE_PACK. Remove unsupported numbers.\n\n"
            f"VALIDATOR_ERRORS:\n{json.dumps(errors, indent=2)}\n\n"
            f"EVIDENCE_PACK:\n{pack.as_prompt_json()}\n\n"
            f"DRAFT:\n{body}"
        )
        try:
            body = _call_writer(client, _repair_system(), repair_user)
            errors = validate_body(body, pack)
        except Exception as exc:
            errors.append(f"repair_failed:{exc}")
    if errors:
        report = memory.REPORTS / f"{slug}.blocked.json"
        memory.ensure_dirs()
        report.write_text(
            json.dumps(
                {"title": title, "slug": slug, "errors": errors, "body": body},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return WriteResult(ok=False, title=title, body=body, errors=errors, repair_attempted=repair_attempted)
    post = _frontmatter(title, post_type, pack) + "\n" + body.strip() + "\n"
    if dry_run:
        path = memory.REPORTS / f"{slug}.preview.md"
    else:
        path = memory.CONTENT_POST / f"{slug}.md"
    memory.ensure_dirs()
    path.write_text(post, encoding="utf-8")
    return WriteResult(ok=True, path=str(path.relative_to(memory.ROOT)), title=title, body=body, repair_attempted=repair_attempted)


def _frontmatter(title: str, post_type: str, pack: EvidencePack) -> str:
    tracks = sorted({track for ranked in pack.ranked_items for track in ranked.topic_scores.tracks})
    source_count = len(pack.ranked_items)
    paper_count = sum(1 for r in pack.ranked_items if r.item.source_kind == "paper")
    blog_count = sum(1 for r in pack.ranked_items if r.item.source_kind == "blog")
    return "\n".join(
        [
            "---",
            f'date: "{datetime.now(timezone.utc).date().isoformat()}"',
            "draft: false",
            f'title: "{_yaml_escape(title)}"',
            f"post_type: {post_type}",
            "categories: [\"Machine Learning\"]",
            "tags: [\"research-radar\", \"llm\", \"mle\", \"aec\"]",
            f"tracks: {json.dumps(tracks)}",
            f"source_count: {source_count}",
            f"paper_count: {paper_count}",
            f"blog_count: {blog_count}",
            "math: true",
            "mermaid: true",
            "---",
        ]
    )


def _daily_system() -> str:
    return (
        "You are writing Synaptic Radio, a technical ML research blog. Write polished Markdown, not JSON. "
        "Use the evidence pack only. The prose must be original and evidence-grounded. "
        "Every substantive item paragraph must include one or more evidence markers like [E1] and a source link. "
        "Do not invent numbers, benchmarks, authors, or claims. Target 1200-2000 words when evidence supports it."
    )


def _daily_user(pack: EvidencePack, title: str) -> str:
    return (
        f"TITLE: {title}\n\n"
        "Write a daily radar post with: what mattered today, top papers, top engineering blogs, "
        "cross-cutting patterns, caveats, and what to learn next. Include source URLs inline.\n\n"
        f"EVIDENCE_PACK:\n{pack.as_prompt_json()}"
    )


def _deep_system() -> str:
    return (
        "You write guided technical deep dives for ML engineers. Use only the evidence pack. "
        "Write original Markdown. Include evidence markers [E1] and source links for substantive claims. "
        "Include a Mermaid diagram and one code or pseudocode block only if supported by evidence. "
        "Do not invent numbers or implementation details. Target 2500-4500 words when evidence supports it."
    )


def _deep_user(pack: EvidencePack, title: str) -> str:
    return (
        f"TITLE: {title}\n\n"
        "Write sections for why it matters, prerequisites, problem framing, method walkthrough, "
        "experiments, visual intuition, reproduction notes, limits, and study questions.\n\n"
        f"EVIDENCE_PACK:\n{pack.as_prompt_json()}"
    )


def _repair_system() -> str:
    return (
        "Repair Markdown for factuality and source grounding. Return only Markdown body. "
        "Do not add unsupported facts. Remove claims that cannot be tied to evidence."
    )


def _strip_fence(text: str) -> str:
    stripped = (text or "").strip()
    m = re.match(r"^```(?:markdown|md)?\n([\s\S]*?)\n```$", stripped, re.I)
    return m.group(1).strip() if m else stripped


def _yaml_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _meaningful_numbers(body: str) -> list[str]:
    scan = re.sub(r"https?://\S+", " ", body or "")
    out: list[str] = []
    for match in NUMBER_RE.finditer(scan):
        token = match.group(0)
        if len(token) == 4 and token.startswith(("19", "20")):
            continue
        if token in {"1", "2", "3", "4", "5", "6", "7", "8"}:
            continue
        out.append(token)
    return out


def _copies_large_evidence_span(body: str, pack: EvidencePack) -> bool:
    compact_body = re.sub(r"\s+", " ", body or "")
    for chunk in pack.chunks:
        text = re.sub(r"\s+", " ", chunk.text or "").strip()
        if len(text) >= 300 and text[:300] in compact_body:
            return True
    return False
