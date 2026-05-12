from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from markdown_it import MarkdownIt

from . import assets, config, memory
from .llm import LLMClient
from .models import DailyOutline, EvidencePack, RankedItem, SelectionResult, WriteResult

LOG = logging.getLogger(__name__)

EVIDENCE_REF_RE = re.compile(r"\[E(\d+)\]")
NUMBER_RE = re.compile(r"(?<![\w-])\d+(?:\.\d+)?%?(?![\w-])")
HEADING_RE = re.compile(r"^#{2,3}\s+(.+?)\s*$", re.M)
URL_RE = re.compile(r"https?://[^\s<>\])\"]+")
NUMERIC_CLAIM_CUES = (
    "%",
    "percent",
    "x ",
    "ms",
    "s latency",
    "seconds",
    "tokens",
    "tasks",
    "examples",
    "samples",
    "parameters",
    "billion",
    "million",
    "accuracy",
    "score",
    "benchmark",
    "throughput",
    "latency",
    "faster",
    "slower",
    "lower",
    "higher",
    "reduced",
    "improved",
)
GENERIC_PHRASES = (
    "digital transformation",
    "stands at the precipice",
    "rapid evolution of artificial intelligence",
    "paving the way",
    "game-changer",
    "paramount",
    "crucial",
    "holistic strategy",
    "shape the future",
    "unlock value",
    "seamlessly",
    "revolutionize",
    "our vision at autodesk",
    "our strategic direction",
    "our path forward",
)
TECHNICAL_SPECIFICITY_CUES = (
    "algorithm",
    "architecture",
    "objective",
    "loss",
    "benchmark",
    "ablation",
    "latency",
    "throughput",
    "cache",
    "retrieval",
    "quantization",
    "calibration",
    "layout",
    "cadquery",
    "parametric",
    "grounding",
    "failure mode",
    "evaluation",
)
ENGINEERING_JUDGMENT_CUES = (
    "would test",
    "prototype",
    "blocker",
    "release gate",
    "adoption",
    "production",
    "tradeoff",
    "monitoring",
    "rollback",
    "latency budget",
    "failure category",
    "risk",
)
SYNTHESIS_CUES = (
    "compare",
    "contrast",
    "together",
    "whereas",
    "tradeoff",
    "across",
    "combined",
    "separates",
    "complements",
    "the common pattern",
)
FIRST_PERSON_AUTODESK_RE = re.compile(
    r"\b(as a principal machine learning engineer at autodesk|at autodesk,\s+(my|our)|our vision at autodesk|our strategic direction)\b",
    re.I,
)
DAILY_CONCEPTS: dict[str, tuple[str, ...]] = {
    "technical_thesis": ("thesis", "technical pattern", "framing", "claim"),
    "mechanism": ("method", "mechanism", "architecture", "pipeline", "model"),
    "math_or_objective": ("objective", "loss", "optimization", "metric", "measure", "score"),
    "experiments": ("experiment", "ablation", "benchmark", "evaluation", "evidence"),
    "limitations": ("limitation", "caveat", "failure", "risk", "tradeoff"),
    "impact": ("impact", "production", "deployment", "engineering", "practical"),
    "autodesk_relevance": ("autodesk", "aec", "document", "drawing", "sheet", "cad", "bim", "construction"),
}


def write_daily(
    pack: EvidencePack,
    *,
    outline: DailyOutline,
    selection: SelectionResult,
    llm: LLMClient | None = None,
    dry_run: bool = False,
) -> WriteResult:
    client = llm or LLMClient()
    today = datetime.now(timezone.utc).date().isoformat()
    title = outline.title or f"Research Radar: Autodesk MLE Brief - {today}"
    slug = f"{today}-research-radar"
    body = _generate_daily_body(
        client=client,
        pack=pack,
        outline=outline,
        selection=selection,
        title=title,
        slug=slug,
    )
    return _validate_repair_and_publish(
        client=client,
        pack=pack,
        outline=outline,
        selection=selection,
        title=title,
        body=body,
        slug=slug,
        post_type="daily",
        dry_run=dry_run,
    )


def write_deep_dive(pack: EvidencePack, *, llm: LLMClient | None = None, dry_run: bool = False) -> WriteResult:
    client = llm or LLMClient()
    primary = pack.ranked_items[0].item
    today = datetime.now(timezone.utc).date().isoformat()
    title = f"{primary.title} - guided learning deep dive"
    slug = f"{today}-{memory.slugify(primary.title)}-guided-deep-dive"
    body = _generate_deep_dive_body(client=client, pack=pack, title=title, slug=slug)
    return _validate_repair_and_publish(
        client=client,
        pack=pack,
        outline=None,
        selection=None,
        title=title,
        body=body,
        slug=slug,
        post_type="deep_dive",
        dry_run=dry_run,
    )


def validate_body(body: str, pack: EvidencePack, *, outline: DailyOutline | None = None) -> list[str]:
    errors: list[str] = []
    chunks_by_id = {chunk.evidence_id: chunk for chunk in pack.chunks}
    evidence_ids = set(chunks_by_id)
    refs = {f"E{m.group(1)}" for m in EVIDENCE_REF_RE.finditer(body or "")}
    unknown = sorted(refs - evidence_ids)
    if unknown:
        errors.append("unknown_evidence_ids:" + ",".join(unknown))
    if not refs:
        errors.append("no_evidence_ids")
    body_url_keys = _body_url_keys(body)
    for url in _required_urls_for_refs(refs, chunks_by_id):
        if url and not (_url_keys(url) & body_url_keys):
            errors.append("missing_source_link:" + url)
    evidence_blob = pack.evidence_blob()
    for number in sorted(set(_meaningful_numbers(body))):
        if number not in evidence_blob:
            errors.append("unsupported_number:" + number)
    if _copies_large_evidence_span(body, pack):
        errors.append("copied_large_evidence_span")
    if pack.kind == "daily":
        errors.extend(_validate_daily_technical_focus(body, pack, outline=outline))
    try:
        MarkdownIt().parse(body)
    except Exception as exc:
        errors.append(f"markdown_parse_error:{exc}")
    return errors


def _generate_daily_body(
    *,
    client: LLMClient,
    pack: EvidencePack,
    outline: DailyOutline,
    selection: SelectionResult,
    title: str,
    slug: str,
) -> str:
    def _fallback() -> str:
        return _call_writer(client, _daily_system(), _daily_user(pack, outline, selection, title), task="draft")

    if not isinstance(client, LLMClient):
        return _fallback()

    section_specs = [
        {
            "heading": section.heading,
            "intent": section.intent,
            "evidence_ids": list(section.evidence_ids),
            "word_budget": int(section.word_budget or 0),
        }
        for section in outline.sections
    ]
    required_calls = len(section_specs) + 1
    if not _has_call_budget(client, required_calls):
        return _fallback()
    try:
        drafted = _draft_sections(
            client=client,
            pack=pack,
            title=title,
            post_type="daily",
            section_specs=section_specs,
            outline=outline,
            selection=selection,
        )
        if not drafted:
            return _fallback()
        edited = _final_editor_pass(
            client=client,
            pack=pack,
            title=title,
            post_type="daily",
            section_drafts=drafted,
            slug=slug,
            outline=outline,
            selection=selection,
        )
        return edited or _fallback()
    except Exception as exc:  # noqa: BLE001
        LOG.warning("sectionwise daily writer failed; using fallback single pass: %s", exc)
        return _fallback()


def _generate_deep_dive_body(*, client: LLMClient, pack: EvidencePack, title: str, slug: str) -> str:
    def _fallback() -> str:
        return _call_writer(client, _deep_system(), _deep_user(pack, title), task="draft")

    if not isinstance(client, LLMClient):
        return _fallback()

    section_specs = _deep_dive_section_specs(pack)
    required_calls = len(section_specs) + 1
    if not _has_call_budget(client, required_calls):
        return _fallback()
    try:
        drafted = _draft_sections(
            client=client,
            pack=pack,
            title=title,
            post_type="deep_dive",
            section_specs=section_specs,
            outline=None,
            selection=None,
        )
        if not drafted:
            return _fallback()
        edited = _final_editor_pass(
            client=client,
            pack=pack,
            title=title,
            post_type="deep_dive",
            section_drafts=drafted,
            slug=slug,
            outline=None,
            selection=None,
        )
        return edited or _fallback()
    except Exception as exc:  # noqa: BLE001
        LOG.warning("sectionwise deep-dive writer failed; using fallback single pass: %s", exc)
        return _fallback()


def _draft_sections(
    *,
    client: LLMClient,
    pack: EvidencePack,
    title: str,
    post_type: str,
    section_specs: list[dict[str, object]],
    outline: DailyOutline | None,
    selection: SelectionResult | None,
) -> list[str]:
    drafted: list[str] = []
    for section in section_specs:
        heading = str(section.get("heading") or "").strip()
        if not heading:
            continue
        user = _section_user(
            title=title,
            post_type=post_type,
            section=section,
            pack=pack,
            outline=outline,
            selection=selection,
            drafted_so_far=drafted,
        )
        text = _call_writer(client, _section_system(post_type), user, task="draft_section")
        cleaned = _normalize_section_output(heading, text)
        drafted.append(cleaned)
    return drafted


def _deep_dive_section_specs(pack: EvidencePack) -> list[dict[str, object]]:
    order = [
        ("Technical thesis and motivation", "thesis framing and the practical engineering question", ("impact", "mechanism")),
        ("Method walkthrough and mechanism", "step-by-step mechanism with concrete evidence", ("mechanism", "experiment")),
        ("Objective or math interpretation", "objective, optimization, or metric interpretation", ("math_or_objective", "experiment")),
        ("Experiments and evidence", "benchmarks, ablations, and reported outcomes", ("experiment",)),
        ("Limits and failure modes", "limitations, caveats, and what could break", ("limitation", "impact")),
        ("Engineering implications and next actions", "how to apply or evaluate in production settings", ("impact", "limitation")),
    ]
    out: list[dict[str, object]] = []
    for heading, intent, evidence_types in order:
        evidence_ids = _evidence_ids_for_types(pack, evidence_types, limit=5)
        out.append(
            {
                "heading": heading,
                "intent": intent,
                "evidence_ids": evidence_ids,
                "word_budget": 520,
            }
        )
    return out


def _evidence_ids_for_types(pack: EvidencePack, evidence_types: tuple[str, ...], *, limit: int = 5) -> list[str]:
    out: list[str] = []
    for chunk in pack.chunks:
        et = (chunk.evidence_type or "").lower()
        if evidence_types and et not in {x.lower() for x in evidence_types}:
            continue
        out.append(chunk.evidence_id)
        if len(out) >= limit:
            return out
    for chunk in pack.chunks:
        if chunk.evidence_id not in out:
            out.append(chunk.evidence_id)
            if len(out) >= limit:
                break
    return out


def _section_system(post_type: str) -> str:
    scope = "daily research radar section" if post_type == "daily" else "deep-dive section"
    return (
        "Write one Markdown section only. "
        f"You are drafting a {scope}. "
        "Output a heading line '## ...' followed by concise technical prose. "
        "Use only supplied evidence and cite with [E#] plus source URL links for substantive claims. "
        "Open with a concrete claim, then cover mechanism, engineering implication, and a limitation or adoption blocker. "
        "Avoid corporate transformation language and broad industry setup unless it directly supports the mechanism. "
        "Do not invent numbers, tables, or results. No frontmatter, no JSON, no preamble."
    )


def _section_user(
    *,
    title: str,
    post_type: str,
    section: dict[str, object],
    pack: EvidencePack,
    outline: DailyOutline | None,
    selection: SelectionResult | None,
    drafted_so_far: list[str],
) -> str:
    heading = str(section.get("heading") or "").strip()
    intent = str(section.get("intent") or "").strip()
    word_budget = int(section.get("word_budget") or 0)
    evidence_ids = [str(x) for x in (section.get("evidence_ids") or []) if str(x).strip()]
    chunks = [chunk.model_dump(mode="json") for chunk in pack.chunks if chunk.evidence_id in set(evidence_ids)]
    if not chunks:
        chunks = [chunk.model_dump(mode="json") for chunk in pack.chunks[:8]]
    return (
        f"POST_TYPE: {post_type}\n"
        f"TITLE: {title}\n"
        f"HEADING: {heading}\n"
        f"INTENT: {intent}\n"
        f"TARGET_WORD_BUDGET: {word_budget}\n"
        f"EVIDENCE_IDS_TO_USE: {json.dumps(evidence_ids)}\n\n"
        f"DRAFTED_SECTIONS_SO_FAR:\n{json.dumps(drafted_so_far[-2:], ensure_ascii=False)}\n\n"
        f"OUTLINE:\n{outline.model_dump_json(indent=2) if outline is not None else '{}'}\n\n"
        f"SELECTION:\n{selection.model_dump_json(indent=2) if selection is not None else '{}'}\n\n"
        f"EVIDENCE_CHUNKS:\n{json.dumps(chunks, indent=2, ensure_ascii=False)}\n"
    )


def _normalize_section_output(expected_heading: str, text: str) -> str:
    cleaned = _strip_fence(text).strip()
    lines = cleaned.splitlines()
    if not lines:
        return f"## {expected_heading}\n"
    if not lines[0].startswith("## "):
        cleaned = f"## {expected_heading}\n\n{cleaned}"
    return cleaned.strip()


def _final_editor_pass(
    *,
    client: LLMClient,
    pack: EvidencePack,
    title: str,
    post_type: str,
    section_drafts: list[str],
    slug: str,
    outline: DailyOutline | None,
    selection: SelectionResult | None,
) -> str:
    user = _final_editor_user(
        pack=pack,
        title=title,
        post_type=post_type,
        section_drafts=section_drafts,
        slug=slug,
        outline=outline,
        selection=selection,
    )
    return _call_writer(client, _final_editor_system(post_type), user, task="editor")


def _final_editor_system(post_type: str) -> str:
    target = "daily research radar post" if post_type == "daily" else "deep-dive technical post"
    return (
        f"You are the final editor for a {target}. "
        "Merge section drafts into one cohesive Markdown body with smooth transitions. "
        "Preserve evidence markers [E#] and include source URL links for substantive claims. "
        "Keep section headings specific and non-generic. "
        "Ensure one mermaid diagram block is present and include the provided SVG image links in the body. "
        "Remove corporate strategy language, repeated hype adjectives, unsupported first-person Autodesk claims, "
        "and paper-by-paper abstract summaries that do not add engineering judgment. "
        "Do not invent claims, numbers, or references. "
        "If a numeric detail is not explicitly grounded in evidence, rewrite it qualitatively instead of guessing. "
        "Output Markdown only."
    )


def _final_editor_user(
    *,
    pack: EvidencePack,
    title: str,
    post_type: str,
    section_drafts: list[str],
    slug: str,
    outline: DailyOutline | None,
    selection: SelectionResult | None,
) -> str:
    image_paths = [
        f"/img/posts/{slug}/source-mix.svg",
        f"/img/posts/{slug}/topic-mix.svg",
    ]
    return (
        f"POST_TYPE: {post_type}\n"
        f"TITLE: {title}\n"
        "REQUIRED_VISUALS:\n"
        f"- mermaid diagram from this source graph if needed:\n```mermaid\n{assets.mermaid_for_ranked(pack.ranked_items)}\n```\n"
        + "\n".join(f"- embed image link: {path}" for path in image_paths)
        + "\n\n"
        f"OUTLINE:\n{outline.model_dump_json(indent=2) if outline is not None else '{}'}\n\n"
        f"SELECTION:\n{selection.model_dump_json(indent=2) if selection is not None else '{}'}\n\n"
        f"SECTION_DRAFTS:\n{json.dumps(section_drafts, indent=2, ensure_ascii=False)}\n\n"
        f"EVIDENCE_PACK:\n{pack.as_prompt_json()}\n"
    )


def _has_call_budget(client: LLMClient, required_calls: int) -> bool:
    if required_calls <= 0:
        return True
    if not isinstance(client, LLMClient):
        return True
    if client.cfg.max_calls <= 0:
        return False
    remaining = client.cfg.max_calls - client.usage.calls
    return remaining >= required_calls


def _call_writer(client: LLMClient, system: str, user: str, *, task: str | None = None) -> str:
    if isinstance(client, LLMClient):
        return _strip_fence(client.complete(system=system, user=user, task=task)).strip()
    return _strip_fence(client.complete(system=system, user=user)).strip()


def _validate_repair_and_publish(
    *,
    client: LLMClient,
    pack: EvidencePack,
    outline: DailyOutline | None,
    selection: SelectionResult | None,
    title: str,
    body: str,
    slug: str,
    post_type: str,
    dry_run: bool,
) -> WriteResult:
    _prepare_visual_assets(pack, slug)
    body = _ensure_visual_blocks(body, pack, slug)
    body, errors = _sanitize_then_validate(body, pack, outline=outline)
    repair_attempted = False
    if errors:
        repair_attempted = True
        repair_user = _repair_user(pack, body, errors, outline=outline, selection=selection)
        try:
            body = _call_writer(client, _repair_system(), repair_user, task="repair")
            body = _ensure_visual_blocks(body, pack, slug)
            body, errors = _sanitize_then_validate(body, pack, outline=outline)
        except Exception as exc:
            errors.append(f"repair_failed:{exc}")
    if errors:
        report = memory.REPORTS / f"{slug}.blocked.json"
        memory.ensure_dirs()
        rubric = _signal_rubric(body, pack) if pack.kind == "daily" else {}
        report.write_text(
            json.dumps(
                {"title": title, "slug": slug, "errors": errors, "signal_rubric": rubric, "body": body},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return WriteResult(ok=False, title=title, body=body, errors=errors, repair_attempted=repair_attempted)
    post = _frontmatter(title, post_type, pack, outline=outline, selection=selection, body=body) + "\n" + body.strip() + "\n"
    if dry_run:
        path = memory.REPORTS / f"{slug}.preview.md"
    else:
        path = memory.CONTENT_POST / f"{slug}.md"
    memory.ensure_dirs()
    path.write_text(post, encoding="utf-8")
    return WriteResult(ok=True, path=str(path.relative_to(memory.ROOT)), title=title, body=body, repair_attempted=repair_attempted)


def _frontmatter(
    title: str,
    post_type: str,
    pack: EvidencePack,
    *,
    outline: DailyOutline | None = None,
    selection: SelectionResult | None = None,
    body: str = "",
) -> str:
    tracks = sorted({track for ranked in pack.ranked_items for track in ranked.topic_scores.tracks})
    source_count = len(pack.ranked_items)
    paper_count = sum(1 for r in pack.ranked_items if r.item.source_kind == "paper")
    blog_count = sum(1 for r in pack.ranked_items if r.item.source_kind == "blog")
    tags = _frontmatter_tags(pack, outline=outline, selection=selection, body=body)
    return "\n".join(
        [
            "---",
            f'date: "{datetime.now(timezone.utc).date().isoformat()}"',
            "draft: true",
            f'title: "{_yaml_escape(title)}"',
            f"post_type: {post_type}",
            "categories: [\"Machine Learning\"]",
            f"tags: {json.dumps(tags)}",
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
        "Write from the practical point of view of a Principal Machine Learning Engineer at Autodesk evaluating research for "
        "AEC foundation models and 2D document intelligence. Do not claim to be employed by Autodesk. "
        "The post is paper-centered: explain mechanisms, math/objectives when evidence supports them, experiments, limits, and impact. "
        "Prefer deep synthesis over breadth: focus on 3-4 primary papers and use supporting items only when they sharpen the thesis. "
        "Use the evidence pack only. The prose must be original and evidence-grounded. "
        "Every substantive item paragraph must include one or more evidence markers like [E1] and a source link. "
        "Do not publish a single-source post: cite at least three distinct primary papers when available. "
        f"Do not invent numbers, benchmarks, authors, or claims. The daily post must be at least {config.daily_min_words()} words."
    )


def _daily_user(pack: EvidencePack, outline: DailyOutline, selection: SelectionResult, title: str) -> str:
    return (
        f"TITLE: {title}\n\n"
        "Write a paper-first technical blog post using the OUTLINE headings exactly as provided. "
        "Do not use fixed template headings such as 'Paper mechanisms', 'Math or objective details', or 'Why it matters' unless the outline uses them. "
        "Cover 3-4 primary papers deeply and mention supporting items briefly only when they strengthen a comparison. "
        "Cite at least three distinct primary papers when they exist in the evidence pack. "
        "For each item you discuss, answer what problem it attacks, what mechanism or objective it uses, what evidence supports it, "
        "what limitation or caveat is visible, and why it matters. Include source URLs inline for every cited evidence ID. "
        "Include at least one cross-paper comparison or tradeoff and at least one concrete adoption test for Autodesk/AEC 2D document systems. "
        "Make the Autodesk/AEC/2D-document implications concrete where supported by the evidence. "
        "Do not use first-person employment claims such as 'As a Principal MLE at Autodesk'; write from that practical viewpoint without claiming identity. "
        "Avoid exact numeric claims unless the number appears verbatim in the evidence text.\n\n"
        f"SELECTION:\n{selection.model_dump_json(indent=2)}\n\n"
        f"OUTLINE:\n{outline.model_dump_json(indent=2)}\n\n"
        f"EVIDENCE_CARDS:\n{json.dumps([card.model_dump(mode='json') for card in pack.evidence_cards], indent=2, ensure_ascii=False)}\n\n"
        f"EVIDENCE_PACK:\n{pack.as_prompt_json()}"
    )


def _deep_system() -> str:
    return (
        "You write guided technical deep dives for ML engineers. Use only the evidence pack. "
        "Center the method, math/objective interpretation, experiment analysis, reproduction notes, limits, and impact. "
        "Write original Markdown. Include evidence markers [E1] and source links for substantive claims. "
        "Include a Mermaid diagram and one code or pseudocode block only if supported by evidence. "
        "Do not invent numbers or implementation details. Target 2500-4500 words when evidence supports it."
    )


def _deep_user(pack: EvidencePack, title: str) -> str:
    return (
        f"TITLE: {title}\n\n"
        "Write sections for technical thesis, prerequisites, problem framing, method walkthrough, "
        "math or objective interpretation, experiments, reproduction notes, limits, impact, and study questions.\n\n"
        f"EVIDENCE_PACK:\n{pack.as_prompt_json()}"
    )


def _repair_system() -> str:
    return (
        "Repair Markdown for factuality and source grounding. Return only Markdown body. "
        "Keep the required daily technical sections when repairing daily posts. "
        "Every substantive paragraph must include valid evidence IDs such as [E1] and source URLs from the evidence pack. "
        "Do not add unsupported facts. Remove claims that cannot be tied to evidence. "
        "When a numeric detail is unsupported, replace it with qualitative wording instead of inventing a new number."
    )


def _repair_user(
    pack: EvidencePack,
    body: str,
    errors: list[str],
    *,
    outline: DailyOutline | None,
    selection: SelectionResult | None,
) -> str:
    chunks_by_id = {chunk.evidence_id: chunk for chunk in pack.chunks}
    refs = {f"E{m.group(1)}" for m in EVIDENCE_REF_RE.finditer(body or "")}
    cited_source_requirements = [
        {"evidence_id": evidence_id, "title": chunks_by_id[evidence_id].title, "url": chunks_by_id[evidence_id].url}
        for evidence_id in sorted(refs)
        if evidence_id in chunks_by_id
    ]
    section_contract = (
        "For daily posts, preserve the OUTLINE headings exactly and write from the practical viewpoint "
        "of a Principal MLE at Autodesk evaluating AEC foundation model and 2D-document relevance. "
        f"The daily post must be at least {config.daily_min_words()} words. Cite at least three primary papers "
        "when they exist in the evidence pack. Remove generic corporate prose and add concrete engineering judgment."
        if pack.kind == "daily"
        else "Preserve the requested deep-dive technical structure."
    )
    return (
        "Fix this Markdown draft. Return only the repaired Markdown body.\n\n"
        f"{section_contract}\n"
        "Hard requirements:\n"
        "- Use only evidence IDs present in EVIDENCE_PACK, formatted exactly like [E1].\n"
        "- For every evidence ID you cite, include the matching source URL inline in that paragraph.\n"
        "- You do not need to cover every item in EVIDENCE_PACK; omit weak items rather than inventing details.\n"
        "- Remove unsupported numbers and unsupported claims; prefer qualitative phrasing when uncertain.\n"
        "- Do not output JSON, explanations, or a validation report.\n\n"
        f"VALIDATOR_ERRORS:\n{json.dumps(_repair_safe_errors(errors), indent=2)}\n\n"
        f"CITED_SOURCE_URLS:\n{json.dumps(cited_source_requirements, indent=2, ensure_ascii=False)}\n\n"
        f"SELECTION:\n{selection.model_dump_json(indent=2) if selection is not None else '{}'}\n\n"
        f"OUTLINE:\n{outline.model_dump_json(indent=2) if outline is not None else '{}'}\n\n"
        f"EVIDENCE_PACK:\n{pack.as_prompt_json()}\n\n"
        f"DRAFT:\n{_repair_safe_draft(body, errors)}"
    )


def _repair_safe_errors(errors: list[str]) -> list[str]:
    safe: list[str] = []
    for error in errors:
        if error.startswith("unsupported_number:"):
            if "unsupported_number" not in safe:
                safe.append("unsupported_number")
            continue
        if error.startswith("missing_source_link:"):
            if "missing_source_link_for_cited_evidence" not in safe:
                safe.append("missing_source_link_for_cited_evidence")
            continue
        safe.append(error)
    return safe


def _repair_safe_draft(body: str, errors: list[str]) -> str:
    safe = body
    for error in errors:
        if error.startswith("unsupported_number:"):
            token = error.split(":", 1)[1]
            safe = safe.replace(token, "[qualitative wording only]")
    return safe


def _sanitize_then_validate(
    body: str,
    pack: EvidencePack,
    *,
    outline: DailyOutline | None,
) -> tuple[str, list[str]]:
    errors = validate_body(body, pack, outline=outline)
    if not any(error.startswith("unsupported_number:") for error in errors):
        return body, errors
    sanitized = _sanitize_unsupported_numbers(body, errors)
    if sanitized == body:
        return body, errors
    sanitized_errors = validate_body(sanitized, pack, outline=outline)
    return sanitized, sanitized_errors


def _sanitize_unsupported_numbers(body: str, errors: list[str]) -> str:
    tokens = sorted(
        {
            error.split(":", 1)[1]
            for error in errors
            if error.startswith("unsupported_number:") and ":" in error
        },
        key=len,
        reverse=True,
    )
    if not tokens:
        return body
    lines = (body or "").splitlines()
    cleaned: list[str] = []
    in_fence = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            cleaned.append(line)
            continue
        if in_fence or stripped.startswith("#"):
            cleaned.append(line)
            continue
        updated = line
        for token in tokens:
            if token not in updated:
                continue
            updated = _replace_numeric_token(updated, token)
        cleaned.append(updated)
    return "\n".join(cleaned)


def _replace_numeric_token(line: str, token: str) -> str:
    pattern = re.compile(rf"(?<![\w-]){re.escape(token)}(?![\w-])")
    replacement = _qualitative_number_replacement(line, token)
    return pattern.sub(replacement, line)


def _qualitative_number_replacement(line: str, token: str) -> str:
    lower = line.lower()
    if token.endswith("%"):
        if any(cue in lower for cue in ("latency", "runtime", "throughput", "accuracy", "score", "benchmark")):
            return "a reported margin"
        return "a measurable margin"
    if any(cue in lower for cue in ("benchmark", "task", "dataset", "example", "sample", "document", "drawing", "sheet")):
        return "multiple"
    if any(cue in lower for cue in ("parameter", "token", "layer", "step", "epoch")):
        return "many"
    return "several"


def _prepare_visual_assets(pack: EvidencePack, slug: str) -> None:
    try:
        assets.render_post_assets(pack.ranked_items, slug)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("asset render failed for slug %s: %s", slug, exc)


def _ensure_visual_blocks(body: str, pack: EvidencePack, slug: str) -> str:
    text = (body or "").strip()
    mermaid_src = assets.mermaid_for_ranked(pack.ranked_items)
    if "```mermaid" not in text:
        text = (
            text.rstrip()
            + "\n\n## Visual map\n```mermaid\n"
            + mermaid_src
            + "\n```\n"
        )
    source_img = f"/img/posts/{slug}/source-mix.svg"
    topic_img = f"/img/posts/{slug}/topic-mix.svg"
    missing_source = source_img not in text
    missing_topic = topic_img not in text
    if missing_source or missing_topic:
        lines = ["", "## Visual diagnostics"]
        if missing_source:
            lines.append(f"![Source mix chart]({source_img})")
        if missing_topic:
            lines.append(f"![Topic mix chart]({topic_img})")
        text = text.rstrip() + "\n\n" + "\n".join(lines) + "\n"
    return text.strip()


def _strip_fence(text: str) -> str:
    stripped = (text or "").strip()
    m = re.match(r"^```(?:markdown|md)?\n([\s\S]*?)\n```$", stripped, re.I)
    return m.group(1).strip() if m else stripped


def _yaml_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _frontmatter_tags(
    pack: EvidencePack,
    *,
    outline: DailyOutline | None,
    selection: SelectionResult | None,
    body: str,
) -> list[str]:
    blob = " ".join(
        [
            body or "",
            outline.model_dump_json() if outline is not None else "",
            selection.model_dump_json() if selection is not None else "",
            " ".join(" ".join(r.item.tags) for r in pack.ranked_items),
            " ".join(f"{r.item.title} {r.item.abstract_or_excerpt}" for r in pack.ranked_items),
        ]
    ).lower()
    tags = ["research-radar"]
    rules = (
        ("aec", ("aec", "bim", "ifc", "construction", "revit", "building controls")),
        ("document-ai", ("document", "drawing", "sheet", "plan", "pdf", "ocr", "layout")),
        ("foundation-models", ("foundation model", "pretraining", "multimodal", "vision-language")),
        ("multimodal", ("multimodal", "vision-language", "image", "visual")),
        ("cad", ("cad", "ifc", "bim", "revit", "scan-to-bim")),
        ("bim", ("bim", "ifc", "revit")),
        ("llm", ("llm", "language model", "agent", "rag", "transformer")),
        ("mle", ("serving", "evaluation", "monitoring", "pipeline", "latency", "throughput", "deployment")),
    )
    for tag, cues in rules:
        if any(cue in blob for cue in cues):
            tags.append(tag)
    return tags


def _required_urls_for_refs(refs: set[str], chunks_by_id: dict[str, object]) -> list[str]:
    urls = {
        chunk.url
        for evidence_id in refs
        if (chunk := chunks_by_id.get(evidence_id)) is not None and getattr(chunk, "url", "")
    }
    return sorted(urls)


def _body_url_keys(body: str) -> set[str]:
    return {key for url in URL_RE.findall(body or "") for key in _url_keys(_strip_url_punctuation(url))}


def _strip_url_punctuation(url: str) -> str:
    return (url or "").rstrip(".,;:")


def _url_keys(url: str) -> set[str]:
    raw = (url or "").strip()
    if not raw:
        return set()
    parts = urlsplit(raw)
    netloc = parts.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = parts.path.rstrip("/") or "/"
    keys = {f"{netloc}{path}".lower()}
    if netloc == "arxiv.org" and path.startswith("/abs/"):
        arxiv_id = path.rsplit("/", 1)[-1].lower()
        unversioned = re.sub(r"v\d+$", "", arxiv_id)
        keys.add(f"arxiv:{arxiv_id}")
        keys.add(f"arxiv:{unversioned}")
    return keys


def _meaningful_numbers(body: str) -> list[str]:
    scan = re.sub(r"https?://\S+", " ", body or "")
    out: list[str] = []
    for match in NUMBER_RE.finditer(scan):
        token = match.group(0)
        if len(token) == 4 and token.startswith(("19", "20")):
            continue
        if token in {"1", "2", "3", "4", "5", "6", "7", "8"}:
            continue
        if _looks_like_arxiv_identifier(token):
            continue
        if not _looks_like_numeric_claim(scan, match.start(), match.end(), token):
            continue
        out.append(token)
    return out


def _looks_like_arxiv_identifier(token: str) -> bool:
    return bool(re.fullmatch(r"\d{4}\.\d{4,5}", token or ""))


def _looks_like_numeric_claim(text: str, start: int, end: int, token: str) -> bool:
    if token.endswith("%"):
        return True
    window = text[max(0, start - 48) : min(len(text), end + 64)].lower()
    return any(cue in window for cue in NUMERIC_CLAIM_CUES)


def _copies_large_evidence_span(body: str, pack: EvidencePack) -> bool:
    compact_body = re.sub(r"\s+", " ", body or "")
    for chunk in pack.chunks:
        text = re.sub(r"\s+", " ", chunk.text or "").strip()
        if len(text) >= 300 and text[:300] in compact_body:
            return True
    return False


def _validate_daily_technical_focus(body: str, pack: EvidencePack, *, outline: DailyOutline | None) -> list[str]:
    errors: list[str] = []
    headings = [h.lower() for h in HEADING_RE.findall(body or "")]
    if outline is not None:
        errors.extend(_validate_outline_headings(body, outline))
    errors.extend(_validate_daily_coverage(body, pack))
    errors.extend(_validate_signal_quality(body, pack))
    if not any(r.item.source_kind == "paper" for r in pack.ranked_items):
        errors.append("daily_no_papers")
    if _looks_like_generic_roundup(headings):
        errors.append("generic_roundup_structure")
    lower = (body or "").lower()
    errors.extend(_validate_daily_concepts(lower))
    return errors


def _validate_outline_headings(body: str, outline: DailyOutline) -> list[str]:
    body_headings = {_normalize_heading(heading) for heading in HEADING_RE.findall(body or "")}
    errors: list[str] = []
    for section in outline.sections:
        expected = _normalize_heading(section.heading)
        if expected and expected not in body_headings:
            errors.append(f"missing_outline_section:{section.heading}")
    return errors


def _normalize_heading(heading: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", (heading or "").lower())).strip()


def _validate_daily_concepts(lower_body: str) -> list[str]:
    errors: list[str] = []
    for concept, cues in DAILY_CONCEPTS.items():
        if not any(cue in lower_body for cue in cues):
            errors.append(f"missing_daily_concept:{concept}")
    return errors


def _validate_daily_coverage(body: str, pack: EvidencePack) -> list[str]:
    errors: list[str] = []
    chunks_by_id = {chunk.evidence_id: chunk for chunk in pack.chunks}
    refs = {f"E{m.group(1)}" for m in EVIDENCE_REF_RE.finditer(body or "")}
    item_kind = {ranked.item.item_id: ranked.item.source_kind for ranked in pack.ranked_items}
    pack_item_ids = {ranked.item.item_id for ranked in pack.ranked_items}
    cited_item_ids = {
        chunks_by_id[evidence_id].item_id
        for evidence_id in refs
        if evidence_id in chunks_by_id and chunks_by_id[evidence_id].item_id in pack_item_ids
    }
    pack_paper_ids = {item_id for item_id, kind in item_kind.items() if kind == "paper"}
    cited_paper_ids = {item_id for item_id in cited_item_ids if item_kind.get(item_id) == "paper"}
    primary_ids = set(_primary_item_ids(pack))
    required_items = min(3, len(primary_ids or pack_item_ids))
    required_papers = min(3, len(pack_paper_ids & (primary_ids or pack_paper_ids)))
    cited_primary_ids = cited_item_ids & primary_ids if primary_ids else cited_item_ids
    if len(cited_primary_ids) < required_items:
        errors.append(f"insufficient_cited_primary_items:{len(cited_primary_ids)}/{required_items}")
    if required_papers and len(cited_paper_ids) < required_papers:
        errors.append(f"insufficient_cited_papers:{len(cited_paper_ids)}/{required_papers}")
    word_count = len(re.findall(r"\b[\w'-]+\b", body or ""))
    min_words = config.daily_min_words()
    if word_count < min_words:
        errors.append(f"daily_too_short:{word_count}/{min_words}")
    return errors


def _validate_signal_quality(body: str, pack: EvidencePack) -> list[str]:
    errors: list[str] = []
    lower = (body or "").lower()
    rubric = _signal_rubric(body, pack)
    threshold = config.min_signal_score()
    for name, score in rubric["scores"].items():
        if score < threshold:
            errors.append(f"low_signal:{name}:{score:.2f}/{threshold:.2f}")
    if rubric["generic_density"] > config.generic_phrase_max_density():
        errors.append(f"generic_phrase_density:{rubric['generic_density']:.4f}/{config.generic_phrase_max_density():.4f}")
    if FIRST_PERSON_AUTODESK_RE.search(body or ""):
        errors.append("first_person_autodesk_claim")
    if _paper_by_paper_without_synthesis(lower, pack):
        errors.append("paper_by_paper_summary_without_synthesis")
    return errors


def _signal_rubric(body: str, pack: EvidencePack) -> dict[str, object]:
    lower = (body or "").lower()
    words = max(1, len(re.findall(r"\b[\w'-]+\b", lower)))
    technical = _cue_score(lower, TECHNICAL_SPECIFICITY_CUES, target=8)
    judgment = _cue_score(lower, ENGINEERING_JUDGMENT_CUES, target=5)
    synthesis = _cue_score(lower, SYNTHESIS_CUES, target=4)
    primary_depth = _primary_depth_score(body, pack)
    generic_hits = sum(lower.count(phrase) for phrase in GENERIC_PHRASES)
    generic_density = generic_hits / words
    noise_control = 1.0 if generic_density <= config.generic_phrase_max_density() else 0.0
    return {
        "scores": {
            "technical_specificity": technical,
            "engineering_judgment": judgment,
            "synthesis": synthesis,
            "noise_control": noise_control,
            "primary_depth": primary_depth,
        },
        "generic_density": generic_density,
        "examples": _signal_failure_examples(body or ""),
    }


def _cue_score(lower_body: str, cues: tuple[str, ...], *, target: int) -> float:
    hits = sum(1 for cue in cues if cue in lower_body)
    return min(1.0, hits / max(1, target))


def _signal_failure_examples(body: str, *, limit: int = 3) -> list[str]:
    examples: list[str] = []
    sentences = re.split(r"(?<=[.!?])\s+", body or "")
    for sentence in sentences:
        lower = sentence.lower()
        if any(phrase in lower for phrase in GENERIC_PHRASES) or FIRST_PERSON_AUTODESK_RE.search(sentence):
            cleaned = re.sub(r"\s+", " ", sentence).strip()
            if cleaned:
                examples.append(cleaned[:240])
            if len(examples) >= limit:
                break
    return examples


def _primary_depth_score(body: str, pack: EvidencePack) -> float:
    primary_ids = _primary_item_ids(pack)
    if not primary_ids:
        return 1.0
    chunks_by_id = {chunk.evidence_id: chunk for chunk in pack.chunks}
    refs = {f"E{m.group(1)}" for m in EVIDENCE_REF_RE.finditer(body or "")}
    passed = 0
    for item_id in primary_ids:
        available_types = {chunk.evidence_type for chunk in pack.chunks if chunk.item_id == item_id}
        cited_types = {
            chunks_by_id[evidence_id].evidence_type
            for evidence_id in refs
            if evidence_id in chunks_by_id and chunks_by_id[evidence_id].item_id == item_id
        }
        required_secondary = available_types & {"limitation", "experiment", "math_or_objective"}
        if "mechanism" in cited_types and (not required_secondary or cited_types & required_secondary):
            passed += 1
    return passed / len(primary_ids)


def _primary_item_ids(pack: EvidencePack) -> list[str]:
    explicit = [
        ranked.item.item_id
        for ranked in pack.ranked_items
        if str(ranked.item.extra.get("selector_role", "")).lower() == "primary"
    ]
    if explicit:
        return explicit[: config.daily_primary_papers()]
    paper_ids = [ranked.item.item_id for ranked in pack.ranked_items if ranked.item.source_kind == "paper"]
    return paper_ids[: min(3, len(paper_ids))]


def _paper_by_paper_without_synthesis(lower_body: str, pack: EvidencePack) -> bool:
    if _cue_score(lower_body, SYNTHESIS_CUES, target=3) >= 1.0:
        return False
    cited_titles = sum(1 for ranked in pack.ranked_items if ranked.item.title.lower()[:24] in lower_body)
    return cited_titles >= 3


def _required_daily_item_count(item_count: int) -> int:
    if item_count >= 5:
        return 4
    if item_count >= 2:
        return 2
    return min(1, item_count)


def _looks_like_generic_roundup(headings: list[str]) -> bool:
    generic = {"top papers", "top engineering blogs", "what mattered today"}
    required = {"technical thesis", "paper mechanisms", "math or objective details"}
    return bool(generic & set(headings)) and not bool(required & set(headings))
