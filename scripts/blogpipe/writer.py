from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from markdown_it import MarkdownIt

from . import config, jsonish, memory
from .llm import LLMClient
from .models import DailyOutline, EvidencePack, RankedItem, SelectionResult, WriteResult

LOG = logging.getLogger(__name__)

EVIDENCE_REF_RE = re.compile(r"\[E(\d+)\]")
NUMBER_RE = re.compile(r"(?<![\w-])\d+(?:\.\d+)?%?(?![\w-])")
HEADING_RE = re.compile(r"^#{2,3}\s+(.+?)\s*$", re.M)
H1_RE = re.compile(r"^#\s+(.+?)\s*$", re.M)
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
ENGINEERING_LENS_CUES = (
    "benchmark",
    "failure mode",
    "release gate",
    "validation",
    "latency",
    "cost",
    "throughput",
    "deployment",
    "integration",
    "dependency",
    "prototype",
    "adoption test",
    "monitoring",
)
FIRST_PERSON_AUTODESK_RE = re.compile(
    r"\b(as a principal (?:machine learning engineer|mle).*?(?:at|for) autodesk|"
    r"at autodesk,\s+(?:my|our)|our (?:vision|strategic direction|roadmap|pipeline|pipelines|product|products)|"
    r"we (?:face|use|need|build|ship|own|have|could|should)|my focus)\b",
    re.I,
)
REPEATED_TOKEN_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9_-]{2,}),\1\b", re.I)
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
    errors.extend(_paragraph_source_errors(body, refs, chunks_by_id))
    evidence_blob = pack.evidence_blob()
    for number in sorted(set(_meaningful_numbers(body))):
        if number not in evidence_blob:
            errors.append("unsupported_number:" + number)
    if _copies_large_evidence_span(body, pack):
        errors.append("copied_large_evidence_span")
    if REPEATED_TOKEN_RE.search(body or ""):
        errors.append("repeated_token_artifact")
    if pack.kind == "daily":
        errors.extend(_validate_final_title(body))
    if "```mermaid" in (body or "").lower():
        errors.append("mermaid_diagram_present")
    if "## visual map" in (body or "").lower():
        errors.append("generic_visual_map_present")
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
    if not config.sectionwise_drafting_enabled():
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
    if not config.sectionwise_drafting_enabled():
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
        "Do not include Mermaid diagrams, generic visual maps, or visual diagnostics unless the source evidence directly supports a useful technical figure. "
        "Remove corporate strategy language, repeated hype adjectives, duplicated-token artifacts such as 'multiple,multiple', "
        "unsupported first-person Autodesk claims, first-person 'we/our/my' product-roadmap claims, and paper-by-paper abstract summaries that do not add engineering judgment. "
        "If supporting papers are discussed, label them as supporting rather than counting them as primary papers. "
        "Separate paper-supported claims from plausible transfers and open hypotheses; do not edit hypotheses into established findings. "
        "If two papers suggest a possible architecture together, present that as a proposed transfer hypothesis unless the evidence directly proves integration. "
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
    return (
        f"POST_TYPE: {post_type}\n"
        f"TITLE: {title}\n"
        "VISUAL_POLICY: Do not include Mermaid. Omit visuals unless they are a source-grounded mechanism diagram that adds technical signal.\n\n"
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
    body = _ensure_visual_blocks(body, pack, slug)
    quality_review = _llm_quality_review(client, body=body, pack=pack, outline=outline, selection=selection)
    body, errors = _sanitize_then_validate(body, pack, outline=outline)
    errors.extend(_llm_quality_errors(quality_review))
    repair_attempted = False
    if errors:
        repair_attempted = True
        if pack.kind == "daily" and outline is not None and _needs_full_daily_rewrite(errors):
            rewrite_user = _daily_rewrite_user(
                pack,
                body,
                errors,
                outline=outline,
                selection=selection,
                title=title,
                quality_review=quality_review,
            )
            try:
                body = _call_writer(client, _daily_rewrite_system(), rewrite_user, task="draft")
                body = _ensure_visual_blocks(body, pack, slug)
                quality_review = _llm_quality_review(client, body=body, pack=pack, outline=outline, selection=selection)
                body, errors = _sanitize_then_validate(body, pack, outline=outline)
                errors.extend(_llm_quality_errors(quality_review))
            except Exception as exc:
                errors.append(f"full_rewrite_failed:{exc}")
        if errors:
            repair_user = _repair_user(pack, body, errors, outline=outline, selection=selection, quality_review=quality_review)
            try:
                body = _call_writer(client, _repair_system(), repair_user, task="repair")
                body = _ensure_visual_blocks(body, pack, slug)
                quality_review = _llm_quality_review(client, body=body, pack=pack, outline=outline, selection=selection)
                body, errors = _sanitize_then_validate(body, pack, outline=outline)
                errors.extend(_llm_quality_errors(quality_review))
            except Exception as exc:
                errors.append(f"repair_failed:{exc}")
        if errors and pack.kind == "daily" and outline is not None and _needs_emergency_daily_fallback(errors):
            fallback_body = _deterministic_daily_body(pack, outline=outline, title=title)
            fallback_body = _ensure_visual_blocks(fallback_body, pack, slug)
            fallback_body, fallback_errors = _sanitize_then_validate(fallback_body, pack, outline=outline)
            if not fallback_errors:
                LOG.warning("daily writer using deterministic fallback after failed LLM repair")
                body = fallback_body
                errors = []
    if errors:
        report = memory.REPORTS / f"{slug}.blocked.json"
        memory.ensure_dirs()
        rubric = quality_review or (_signal_rubric(body, pack) if pack.kind == "daily" else {})
        report.write_text(
            json.dumps(
                {"title": title, "slug": slug, "errors": errors, "signal_rubric": rubric, "body": body},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return WriteResult(ok=False, title=title, body=body, errors=errors, repair_attempted=repair_attempted)
    final_title = _canonical_title_from_body(body) if pack.kind == "daily" else title
    post = _frontmatter(final_title, post_type, pack, outline=outline, selection=selection, body=body) + "\n" + body.strip() + "\n"
    if dry_run:
        path = memory.REPORTS / f"{slug}.preview.md"
    else:
        path = memory.CONTENT_POST / f"{slug}.md"
    memory.ensure_dirs()
    path.write_text(post, encoding="utf-8")
    return WriteResult(ok=True, path=str(path.relative_to(memory.ROOT)), title=final_title, body=body, repair_attempted=repair_attempted)


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
            "mermaid: false",
            "---",
        ]
    )


def _daily_system() -> str:
    return (
        "You are writing Synaptic Radio, a technical ML research blog. Write polished Markdown, not JSON. "
        "Start with exactly one Markdown H1 matching the requested title, then include every OUTLINE section as an exact Markdown H2. "
        "Write from the practical point of view of a Principal Machine Learning Engineer at Autodesk evaluating research for "
        "AEC foundation models and 2D document intelligence. Do not claim to be employed by Autodesk. "
        "The post is paper-centered: explain mechanisms, math/objectives when evidence supports them, experiments, limits, and impact. "
        "Keep the research depth, but make the article engineering-forward: benchmark design, failure modes, integration constraints, "
        "deployment dependencies, latency/cost tradeoffs, validation plans, and adoption tests should drive the practical judgment. "
        "Prefer deep synthesis over breadth: focus on 3-4 primary papers and use supporting items only when they sharpen the thesis. "
        "Use the evidence pack only. The prose must be original and evidence-grounded. "
        "Distinguish direct implication, plausible transfer, and open hypothesis when discussing AEC use. "
        "Every substantive item paragraph must include one or more evidence markers like [E1] and a source link. "
        "Do not publish a single-source post: cite at least three distinct primary papers when available. "
        f"Do not invent numbers, benchmarks, authors, or claims. The daily post must be at least {config.daily_min_words()} words."
    )


def _daily_user(pack: EvidencePack, outline: DailyOutline, selection: SelectionResult, title: str) -> str:
    return (
        f"TITLE: {title}\n\n"
        "Write a paper-first technical blog post using the OUTLINE headings exactly as provided. "
        f"The first line must be exactly '# {title}'. Each OUTLINE section heading must appear exactly once as '## <heading>'. "
        "Do not use fixed template headings such as 'Paper mechanisms', 'Math or objective details', or 'Why it matters' unless the outline uses them. "
        "Cover 3-4 primary papers deeply and mention supporting items briefly only when they strengthen a comparison. "
        "Cite at least three distinct primary papers when they exist in the evidence pack. "
        "For each item you discuss, answer what problem it attacks, what mechanism or objective it uses, what evidence supports it, "
        "what limitation or caveat is visible, and why it matters. Include source URLs inline for every cited evidence ID. "
        "Include at least one cross-paper comparison or tradeoff and at least one concrete adoption test for Autodesk/AEC 2D document systems. "
        "Every major section should connect the paper to at least one operational lens where evidence permits: benchmark design, failure mode, "
        "deployment constraint, latency/cost tradeoff, integration dependency, or prototype/validation test. "
        "The adoption section should read like an engineering decision memo, not a broad relevance section. "
        "Make the Autodesk/AEC/2D-document implications concrete where supported by the evidence; otherwise label them as plausible transfer or open hypothesis. "
        "Do not present a cross-paper stack, architecture, or workflow as proven unless the evidence cards directly support that integration. "
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
        "Do not include Mermaid diagrams or generic visual maps. Include code or pseudocode only if supported by evidence. "
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
    quality_review: dict[str, object] | None = None,
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
        "- For daily posts, start with exactly one Markdown H1 and keep every OUTLINE section heading exactly as a Markdown H2.\n"
        "- Use only evidence IDs present in EVIDENCE_PACK, formatted exactly like [E1].\n"
        "- For every evidence ID you cite, include the matching source URL inline in that paragraph.\n"
        "- You do not need to cover every item in EVIDENCE_PACK; omit weak items rather than inventing details.\n"
        "- Remove unsupported numbers and unsupported claims; prefer qualitative phrasing when uncertain.\n"
        "- Remove Mermaid diagrams, generic visual maps, stale source lists, duplicated-token artifacts, and first-person Autodesk employment/product claims.\n"
        "- Downgrade unsupported assertions into explicit plausible-transfer or open-hypothesis language.\n"
        "- Merge redundant same-paper sections unless the OUTLINE split is technically justified.\n"
        "- Mark supporting-paper mentions as supporting instead of presenting them as additional primaries.\n"
        "- Add experiment/evaluation detail where the evidence cards provide it.\n"
        "- Strengthen operational judgment with concrete benchmarks, failure modes, deployment constraints, integration dependencies, latency/cost tradeoffs, or validation tests when the evidence supports them.\n"
        "- If the body focus no longer matches the current headline, revise the H1 so the final title matches the actual article.\n"
        "- Downgrade phrases such as 'defines the reliability envelope', 'beyond prototype stage', and 'non-negotiable' unless the evidence clearly warrants them.\n"
        "- Remove late paper insertions that are not justified by the selected primary/supporting scope.\n"
        "- Do not output JSON, explanations, or a validation report.\n\n"
        f"VALIDATOR_ERRORS:\n{json.dumps(_repair_safe_errors(errors), indent=2)}\n\n"
        f"LLM_QUALITY_REVIEW:\n{json.dumps(quality_review or {}, indent=2, ensure_ascii=False)}\n\n"
        f"RELEVANT_EVIDENCE_CARDS:\n{json.dumps([card.model_dump(mode='json') for card in pack.evidence_cards], indent=2, ensure_ascii=False)}\n\n"
        f"CITED_SOURCE_URLS:\n{json.dumps(cited_source_requirements, indent=2, ensure_ascii=False)}\n\n"
        f"SELECTION:\n{selection.model_dump_json(indent=2) if selection is not None else '{}'}\n\n"
        f"OUTLINE:\n{outline.model_dump_json(indent=2) if outline is not None else '{}'}\n\n"
        f"EVIDENCE_PACK:\n{pack.as_prompt_json()}\n\n"
        f"DRAFT:\n{_repair_safe_draft(body, errors)}"
    )


def _needs_full_daily_rewrite(errors: list[str]) -> bool:
    severe_prefixes = (
        "missing_final_h1",
        "multiple_final_h1",
        "truncated_or_fallback_final_h1",
        "missing_outline_section:",
        "insufficient_cited_primary_items:",
        "insufficient_cited_papers:",
        "daily_too_short:",
        "llm_quality:failed",
    )
    return any(error.startswith(severe_prefixes) for error in errors)


def _needs_emergency_daily_fallback(errors: list[str]) -> bool:
    severe_prefixes = (
        "no_evidence_ids",
        "unknown_evidence_ids:",
        "missing_outline_section:",
        "insufficient_cited_primary_items:",
        "insufficient_cited_papers:",
        "daily_too_short:",
    )
    return any(error.startswith(severe_prefixes) for error in errors)


def _daily_rewrite_system() -> str:
    return (
        _daily_system()
        + " The previous draft was rejected as a fragment or structurally invalid. Rewrite the full article from scratch. "
        "Do not patch the fragment locally. Return only the complete Markdown body."
    )


def _daily_rewrite_user(
    pack: EvidencePack,
    body: str,
    errors: list[str],
    *,
    outline: DailyOutline,
    selection: SelectionResult | None,
    title: str,
    quality_review: dict[str, object] | None = None,
) -> str:
    headings = [section.heading for section in outline.sections]
    chunks_by_id = {chunk.evidence_id: chunk for chunk in pack.chunks}
    refs = {f"E{m.group(1)}" for m in EVIDENCE_REF_RE.finditer(body or "")}
    cited_source_requirements = [
        {"evidence_id": evidence_id, "title": chunks_by_id[evidence_id].title, "url": chunks_by_id[evidence_id].url}
        for evidence_id in sorted(refs)
        if evidence_id in chunks_by_id
    ]
    return (
        "Rewrite the rejected daily draft from scratch. Return only a complete Markdown article, not notes or JSON.\n\n"
        "Non-negotiable structure:\n"
        f"- First line exactly: # {title}\n"
        "- Include every required outline heading exactly once as a Markdown H2, in this order:\n"
        + "\n".join(f"  - ## {heading}" for heading in headings)
        + "\n"
        f"- Write at least {config.daily_min_words()} words; target 1400-1900 words if evidence supports it.\n"
        "- Cite at least three distinct primary papers when available in the evidence pack.\n"
        "- Every substantive paragraph must include evidence IDs such as [E1] and the matching source URL inline.\n"
        "- Cover mechanism/objective, experiments or benchmarks, limits, cross-paper tradeoffs, and Autodesk/AEC 2D-document adoption tests.\n"
        "- Treat AEC transfer as direct implication only when the evidence supports it; otherwise label it as plausible transfer or open hypothesis.\n"
        "- Do not preserve the rejected fragment unless a sentence is fully evidence-grounded and still fits the outline.\n\n"
        f"VALIDATOR_ERRORS:\n{json.dumps(_repair_safe_errors(errors), indent=2)}\n\n"
        f"LLM_QUALITY_REVIEW:\n{json.dumps(quality_review or {}, indent=2, ensure_ascii=False)}\n\n"
        f"SELECTION:\n{selection.model_dump_json(indent=2) if selection is not None else '{}'}\n\n"
        f"OUTLINE:\n{outline.model_dump_json(indent=2)}\n\n"
        f"RELEVANT_EVIDENCE_CARDS:\n{json.dumps([card.model_dump(mode='json') for card in pack.evidence_cards], indent=2, ensure_ascii=False)}\n\n"
        f"CITED_SOURCE_URLS:\n{json.dumps(cited_source_requirements, indent=2, ensure_ascii=False)}\n\n"
        f"EVIDENCE_PACK:\n{pack.as_prompt_json()}\n\n"
        f"REJECTED_DRAFT_FOR_DIAGNOSIS_ONLY:\n{_repair_safe_draft(body, errors)}"
    )


def _deterministic_daily_body(pack: EvidencePack, *, outline: DailyOutline, title: str) -> str:
    primary_ids = _primary_item_ids(pack)
    cards_by_item = {card.item_id: card for card in pack.evidence_cards}
    chunks_by_id = {chunk.evidence_id: chunk for chunk in pack.chunks}
    chunks_by_item: dict[str, list[object]] = {}
    for chunk in pack.chunks:
        chunks_by_item.setdefault(chunk.item_id, []).append(chunk)

    def card_for(item_id: str):
        return cards_by_item.get(item_id)

    def clean(text: str, fallback: str) -> str:
        value = re.sub(r"\s+", " ", (text or "").strip())
        if not value or value == "not found in evidence":
            return fallback
        return value.rstrip(".")

    def evidence_for_section(section) -> list[object]:
        out = [chunks_by_id[evidence_id] for evidence_id in section.evidence_ids if evidence_id in chunks_by_id]
        seen = {chunk.evidence_id for chunk in out}
        for item_id in [*section.focus_item_ids, *primary_ids]:
            for chunk in chunks_by_item.get(item_id, []):
                if chunk.evidence_id not in seen:
                    out.append(chunk)
                    seen.add(chunk.evidence_id)
                if len(out) >= 4:
                    return out
        for chunk in pack.chunks:
            if chunk.evidence_id not in seen:
                out.append(chunk)
                seen.add(chunk.evidence_id)
            if len(out) >= 4:
                return out
        return out

    def source_line(chunks: list[object]) -> str:
        refs = " ".join(f"[{chunk.evidence_id}]" for chunk in chunks if getattr(chunk, "evidence_id", ""))
        urls: list[str] = []
        for chunk in chunks:
            url = getattr(chunk, "url", "")
            if url and url not in urls:
                urls.append(url)
        label = "Source:" if len(urls) == 1 else "Sources:"
        return f"{refs} {label} {' '.join(urls)}"

    def paragraph_for_section(section, index: int) -> str:
        chunks = evidence_for_section(section)
        if not chunks:
            return ""
        focus_ids = list(section.focus_item_ids) or [getattr(chunk, "item_id", "") for chunk in chunks]
        cards = [card_for(item_id) for item_id in focus_ids if card_for(item_id)]
        if not cards:
            cards = [card for card in pack.evidence_cards if card.item_id in {getattr(chunk, "item_id", "") for chunk in chunks}]
        card = cards[0] if cards else None
        title_text = card.title if card is not None else getattr(chunks[0], "title", "the selected evidence")
        mechanism = clean(getattr(card, "mechanism", ""), "the mechanism is only partially specified in the available evidence")
        objective = clean(getattr(card, "math_or_objective", ""), "the objective is operational rather than fully formalized in the evidence")
        experiment = clean(getattr(card, "experiment", ""), "the available evidence points to evaluation needs rather than a complete benchmark description")
        limitation = clean(getattr(card, "limitation", ""), "the main limitation is that transfer to AEC document workflows remains an engineering hypothesis")
        impact = clean(getattr(card, "impact", ""), "the practical impact is strongest as a validation target for document intelligence systems")
        claim = clean(getattr(card, "paper_supported_claim", ""), "the paper-supported claim should be treated as bounded by the reported evidence")
        transfer = clean(getattr(card, "transfer_hypothesis", ""), "For Autodesk and AEC document workflows, this is best read as plausible transfer that needs direct validation")
        refs = source_line(chunks)
        intent = (section.intent or section.heading).lower()
        if "experiment" in intent or "benchmark" in intent or "evaluation" in intent:
            core = (
                f"For {title_text}, the evaluation question is the useful engineering object: {experiment}. "
                f"The claim to carry forward is that {claim}. The adoption test should separate retrieval errors, reasoning errors, latency regressions, "
                f"and document-specific failures such as OCR noise, title-block ambiguity, sheet references, and missing provenance. "
                f"The limitation is equally important: {limitation}. {refs}"
            )
        elif "objective" in intent or "math" in intent or "metric" in intent:
            core = (
                f"The objective view for {title_text} is deliberately conservative: {objective}. "
                f"That makes the paper useful for defining metrics rather than declaring a finished AEC system. "
                f"In a drawing, sheet, or BIM-linked workflow, the objective has to be converted into measurable checks for grounding, context reuse, "
                f"latency, provenance, and failure recovery. {refs}"
            )
        elif "synthesis" in intent or "tradeoff" in intent or "compare" in intent:
            names = ", ".join(card.title for card in cards[:3]) if cards else title_text
            core = (
                f"Across {names}, the synthesis is a tradeoff between mechanism quality, evaluation observability, and deployment cost. "
                f"One paper may make context handling or reasoning more capable, while another makes the failure boundary easier to measure. "
                f"For AEC document AI, the system should not merge these ideas into a claimed stack until each interface is tested: retrieval into context, "
                f"grounded answer generation, tool selection, and operator-visible uncertainty. {refs}"
            )
        elif "autodesk" in intent or "aec" in intent or "adoption" in intent or "document" in intent:
            core = (
                f"The Autodesk-facing implication is an adoption plan, not a product claim. {transfer}. "
                f"A practical prototype would use these papers to define gates for sheet retrieval, visual or textual grounding, CAD/BIM linkage, latency, "
                f"and regression monitoring. The evidence supports a focused validation path, while {limitation}. {refs}"
            )
        else:
            core = (
                f"{section.heading} matters because {title_text} gives a concrete technical object for the radar. "
                f"The mechanism signal is: {mechanism}. The paper-supported claim is: {claim}. "
                f"For AEC foundation models and 2D document intelligence, the useful reading is to turn that claim into a benchmarkable system boundary, "
                f"then test whether it survives drawings, sheets, plans, PDFs, OCR artifacts, layout dependencies, and project-specific terminology. {refs}"
            )
        follow_up = (
            " This section should be read as a conservative engineering brief. It identifies the mechanism, objective, experiment, limitation, and impact "
            "visible in the evidence, then converts them into validation work a team could run before relying on the method in production. "
            "The immediate next step is not broad adoption; it is a small benchmark slice with source-linked examples, explicit failure categories, and "
            "operational measurements for latency, throughput, monitoring, and reviewability. "
            "That benchmark should include positive and negative document cases, trace every answer to source evidence, and record which subsystem failed "
            "when the model misses a requirement."
        )
        return core + follow_up

    intro_chunks = []
    for item_id in primary_ids[:3]:
        intro_chunks.extend(chunks_by_item.get(item_id, [])[:1])
    if len(intro_chunks) < 3:
        intro_chunks.extend([chunk for chunk in pack.chunks if chunk not in intro_chunks][: 3 - len(intro_chunks)])
    intro = (
        f"# {title}\n\n"
        "The safest reading of this radar is that the selected papers define engineering boundaries for AEC foundation models and 2D document intelligence. "
        "The useful questions are concrete: what mechanism is being proposed, what objective or metric makes it measurable, what experiment supports it, "
        "what limitation remains, and what adoption test would expose failure before a team depends on it. "
        f"{source_line(intro_chunks)}\n"
    )
    sections = []
    for index, section in enumerate(outline.sections):
        paragraph = paragraph_for_section(section, index)
        if paragraph:
            sections.append(f"## {section.heading}\n\n{paragraph}")
    return intro + "\n\n".join(sections)


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
    body = _ensure_paragraph_source_links(body, pack)
    errors = validate_body(body, pack, outline=outline)
    if not any(error.startswith("unsupported_number:") for error in errors):
        return body, errors
    sanitized = _sanitize_unsupported_numbers(body, errors)
    if sanitized == body:
        return body, errors
    sanitized_errors = validate_body(sanitized, pack, outline=outline)
    return sanitized, sanitized_errors


def _ensure_paragraph_source_links(body: str, pack: EvidencePack) -> str:
    chunks_by_id = {chunk.evidence_id: chunk for chunk in pack.chunks}
    paragraphs = re.split(r"(\n\s*\n)", body or "")
    updated: list[str] = []
    changed = False
    for part in paragraphs:
        if not part or re.fullmatch(r"\n\s*\n", part):
            updated.append(part)
            continue
        refs = {f"E{match.group(1)}" for match in EVIDENCE_REF_RE.finditer(part)}
        if not refs:
            updated.append(part)
            continue
        paragraph_keys = _body_url_keys(part)
        missing_urls = []
        for evidence_id in sorted(refs):
            chunk = chunks_by_id.get(evidence_id)
            url = getattr(chunk, "url", "") if chunk is not None else ""
            if url and not (_url_keys(url) & paragraph_keys) and url not in missing_urls:
                missing_urls.append(url)
        if not missing_urls:
            updated.append(part)
            continue
        suffix = " Source:" if len(missing_urls) == 1 else " Sources:"
        updated.append(part.rstrip() + suffix + " " + " ".join(missing_urls))
        changed = True
    return "".join(updated) if changed else (body or "")


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


def _ensure_visual_blocks(body: str, pack: EvidencePack, slug: str) -> str:
    return (body or "").strip()


def _llm_quality_review(
    client: LLMClient,
    *,
    body: str,
    pack: EvidencePack,
    outline: DailyOutline | None,
    selection: SelectionResult | None,
) -> dict[str, object]:
    if not isinstance(client, LLMClient) or not _has_call_budget(client, 1):
        return {}
    try:
        raw = client.complete(
            system=_quality_review_system(),
            user=_quality_review_user(body=body, pack=pack, outline=outline, selection=selection),
            max_tokens=1600,
            task="quality_review",
        )
        return jsonish.loads_object(raw)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("quality review failed; using deterministic validators only: %s", exc)
        return {}


def _quality_review_system() -> str:
    return (
        "You are a quality review judge for a technical ML research blog. Return JSON only. "
        "Score and identify blocking issues. Use strict engineering-blog standards. "
        "If pass is false, include at least one specific machine-readable error code in errors."
    )


def _quality_review_user(
    *,
    body: str,
    pack: EvidencePack,
    outline: DailyOutline | None,
    selection: SelectionResult | None,
) -> str:
    return (
        "Review the draft for publication quality. Return JSON with this shape:\n"
        "{\n"
        '  "pass": true,\n'
        '  "scores": {"technical_specificity": 0.0, "engineering_judgment": 0.0, "synthesis": 0.0, "noise_control": 0.0, '
        '"primary_depth": 0.0, "evidence_discipline": 0.0, "section_nonredundancy": 0.0, "experiment_detail": 0.0},\n'
        '  "errors": ["machine_readable_error_code"],\n'
        '  "examples": ["quoted short failing text"],\n'
        '  "notes": "short explanation",\n'
        '  "top_editorial_failure": "one sentence on the most important problem"\n'
        "}\n\n"
        "Blocking criteria:\n"
        "- generic corporate prose, hype, or paper-by-paper abstract summaries without insight\n"
        "- first-person employment or product ownership claims involving Autodesk, including 'we', 'our', 'my', 'our roadmap', or 'our pipelines'\n"
        "- duplicated-token artifacts such as 'multiple,multiple'\n"
        "- missing source URL in the same paragraph as each cited evidence marker\n"
        "- Mermaid diagrams, generic visual maps, or stale visuals that mention non-discussed papers\n"
        "- intro says four primary papers but body materially adds more without marking them supporting\n"
        "- weak mechanism/objective/experiment/limitation coverage for primary papers\n"
        "- research exposition with weak engineering usefulness: no benchmark, blocker, deployment implication, integration constraint, validation test, or operational tradeoff\n"
        "- 'why it matters' or recommendation prose that remains generic rather than decision-oriented\n"
        "- repeated sections that cover the same mechanism or limitation with little new technical value\n"
        "- supporting paper introduced late or treated like a primary paper\n"
        "- experiment detail much weaker than the mechanism claims when experiments exist in the evidence cards\n"
        "- speculative Autodesk/AEC adoption prose that outruns paper-supported claims or transfer hypotheses\n"
        "- claims that fuse multiple papers into a proven stack/system without direct support in the evidence cards\n"
        "- synthesis claims whose strength exceeds the evidence, including phrases such as 'defines the reliability envelope', 'beyond prototype stage', or 'non-negotiable'\n"
        "- title/body mismatch: the H1 centers one paper or axis, but the body mainly argues a different throughline\n\n"
        f"OUTLINE:\n{outline.model_dump_json(indent=2) if outline is not None else '{}'}\n\n"
        f"SELECTION:\n{selection.model_dump_json(indent=2) if selection is not None else '{}'}\n\n"
        f"EVIDENCE_PACK:\n{pack.as_prompt_json()}\n\n"
        f"DRAFT:\n{body}\n"
    )


def _llm_quality_errors(review: dict[str, object]) -> list[str]:
    if not review:
        return []
    errors: list[str] = []
    scores = review.get("scores")
    threshold = config.min_signal_score()
    if isinstance(scores, dict):
        for name, value in scores.items():
            try:
                score = float(value)
            except (TypeError, ValueError):
                continue
            if score < threshold:
                errors.append(f"llm_low_signal:{name}:{score:.2f}/{threshold:.2f}")
    if review.get("pass") is False:
        raw_errors = review.get("errors")
        if isinstance(raw_errors, list) and raw_errors:
            errors.extend(f"llm_quality:{str(error)[:120]}" for error in raw_errors[:8])
    return errors


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
    blob = (body or "").lower()
    tags = ["research-radar"]
    rules = (
        ("aec", ("aec", "construction", "building workflow", "facility workflow")),
        ("document-ai", ("document intelligence", "drawing sheet", "sheet-level", "pdf translation", "ocr", "layout preservation")),
        ("foundation-models", ("foundation model", "foundation-model")),
        ("multimodal", ("multimodal", "vision-language", "visual grounding")),
        ("cad", ("cad", "cadquery", "programmatic cad", "scan-to-bim")),
        ("bim", ("bim", "ifc", "revit")),
        ("llm", ("llm", "language model", "agent", "rag", "transformer")),
        ("mle", ("serving", "evaluation", "monitoring", "pipeline", "latency", "throughput", "deployment")),
    )
    for tag, cues in rules:
        if any(cue in blob for cue in cues):
            tags.append(tag)
    return tags


def _validate_final_title(body: str) -> list[str]:
    titles = _h1_titles(body)
    if not titles:
        return ["missing_final_h1"]
    if len(titles) > 1:
        return [f"multiple_final_h1:{len(titles)}"]
    title = titles[0]
    errors: list[str] = []
    if _looks_like_truncated_fallback_title(title):
        errors.append("truncated_or_fallback_final_h1")
    return errors


def _canonical_title_from_body(body: str) -> str:
    titles = _h1_titles(body)
    return titles[0] if titles else ""


def _h1_titles(body: str) -> list[str]:
    return [re.sub(r"\s+", " ", match.group(1)).strip() for match in H1_RE.finditer(body or "")]


def _looks_like_truncated_fallback_title(title: str) -> bool:
    normalized = (title or "").strip()
    lower = normalized.lower()
    if lower.startswith("research radar: ") and len(normalized) >= 72 and not re.search(r"[.!?]$", normalized):
        return True
    if normalized.endswith(("...", "…")):
        return True
    if lower.startswith("research radar: benchcad: a comprehensive, industry-standard benchmark for programmati"):
        return True
    return False


def _required_urls_for_refs(refs: set[str], chunks_by_id: dict[str, object]) -> list[str]:
    urls = {
        chunk.url
        for evidence_id in refs
        if (chunk := chunks_by_id.get(evidence_id)) is not None and getattr(chunk, "url", "")
    }
    return sorted(urls)


def _body_url_keys(body: str) -> set[str]:
    return {key for url in URL_RE.findall(body or "") for key in _url_keys(_strip_url_punctuation(url))}


def _paragraph_source_errors(body: str, refs: set[str], chunks_by_id: dict[str, object]) -> list[str]:
    errors: list[str] = []
    for paragraph in re.split(r"\n\s*\n", body or ""):
        paragraph_refs = {f"E{m.group(1)}" for m in EVIDENCE_REF_RE.finditer(paragraph)}
        if not paragraph_refs:
            continue
        paragraph_keys = _body_url_keys(paragraph)
        for evidence_id in sorted(paragraph_refs & refs):
            chunk = chunks_by_id.get(evidence_id)
            url = getattr(chunk, "url", "") if chunk is not None else ""
            if url and not (_url_keys(url) & paragraph_keys):
                errors.append(f"missing_paragraph_source_link:{evidence_id}:{url}")
    return errors


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
    if FIRST_PERSON_AUTODESK_RE.search(body or ""):
        errors.append("first_person_autodesk_claim")
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
