from __future__ import annotations

import json
import logging
import re

from pydantic import ValidationError

from . import config
from . import jsonish
from .llm import LLMClient
from .models import DailyOutline, EvidencePack, SelectionResult

LOG = logging.getLogger(__name__)


class OutlineError(RuntimeError):
    pass


REQUIRED_INTENTS: dict[str, tuple[str, ...]] = {
    "technical_thesis": ("thesis", "angle", "framing"),
    "mechanism": ("mechanism", "method", "architecture", "pipeline"),
    "math_or_objective": ("math", "objective", "loss", "optimization", "metric"),
    "experiments": ("experiment", "evidence", "benchmark", "evaluation", "ablation"),
    "limitations": ("limit", "caveat", "failure", "risk", "tradeoff"),
    "impact": ("impact", "engineering", "production", "practical"),
    "autodesk_relevance": ("autodesk", "aec", "document", "drawing", "sheet", "cad", "bim"),
    "cross_paper_synthesis": ("compare", "contrast", "cross-paper", "synthesis", "tradeoff"),
}

GENERIC_HEADING_PATTERNS = (
    "navigating the future",
    "bridging the digital divide",
    "our path forward",
    "future of ai",
    "ai's blueprint",
    "what mattered today",
)


def generate_daily_outline(
    pack: EvidencePack,
    *,
    selection: SelectionResult,
    llm: LLMClient,
) -> DailyOutline:
    max_tokens = config.outline_max_tokens()
    parse_error = ""
    outline: DailyOutline | None = None
    try:
        outline = _parse_outline(
            _outline_complete(
                llm,
                system=_outline_system(),
                user=_outline_user(pack, selection),
                task="outline",
                max_tokens=max_tokens,
            )
        )
    except (OutlineError, RuntimeError) as exc:
        parse_error = str(exc)
    if outline is not None:
        errors = validate_outline(outline, pack)
        if not errors:
            LOG.info("outline path: primary")
            return outline
    else:
        errors = []

    # One repair pass keeps the LLM-driven structure but enforces required intents/schema.
    try:
        repaired = _parse_outline(
            _outline_complete(
                llm,
                system=_outline_repair_system(),
                user=_outline_repair_user(
                    pack,
                    selection,
                    outline=outline,
                    errors=errors,
                    parse_error=parse_error,
                ),
                task="outline_repair",
                max_tokens=max_tokens,
            )
        )
        repaired_errors = validate_outline(repaired, pack)
        if not repaired_errors:
            LOG.warning("outline path: repair")
            return repaired
    except (OutlineError, RuntimeError) as exc:
        LOG.warning("outline repair failed; using fallback if valid: %s", exc)

    # Deterministic fallback so daily runs still proceed during flaky outline outputs.
    fallback = _fallback_outline(pack, selection)
    fallback_errors = validate_outline(fallback, pack)
    if not fallback_errors:
        LOG.warning("outline path: fallback")
        return fallback
    raise OutlineError("outline_invalid:" + ",".join(fallback_errors))


def _outline_complete(
    llm: LLMClient,
    *,
    system: str,
    user: str,
    task: str,
    max_tokens: int,
) -> str:
    if isinstance(llm, LLMClient):
        return llm.complete(system=system, user=user, max_tokens=max_tokens, task=task)
    return llm.complete(system=system, user=user, max_tokens=max_tokens)


def validate_outline(outline: DailyOutline, pack: EvidencePack) -> list[str]:
    errors: list[str] = []
    if not outline.title.strip():
        errors.append("missing_outline_title")
    if not outline.angle.strip():
        errors.append("missing_outline_angle")
    headings = [section.heading.strip().lower() for section in outline.sections]
    if len(headings) < 4:
        errors.append("too_few_outline_sections")
    if len(headings) != len(set(headings)):
        errors.append("duplicate_outline_headings")
    for heading in headings:
        if any(pattern in heading for pattern in GENERIC_HEADING_PATTERNS):
            errors.append(f"generic_outline_heading:{heading}")
    known_evidence = {chunk.evidence_id for chunk in pack.chunks}
    for section in outline.sections:
        if not section.heading.strip():
            errors.append("blank_outline_heading")
        unknown = sorted(set(section.evidence_ids) - known_evidence)
        if unknown:
            errors.append("unknown_outline_evidence:" + ",".join(unknown))
    intents = " ".join(section.intent.lower() for section in outline.sections)
    for role, aliases in REQUIRED_INTENTS.items():
        if not any(alias in intents for alias in aliases):
            errors.append(f"missing_outline_intent:{role}")
    total_budget = sum(max(0, section.word_budget) for section in outline.sections)
    if total_budget and total_budget < config.daily_min_words():
        errors.append(f"outline_word_budget_too_short:{total_budget}/{config.daily_min_words()}")
    return errors


def _outline_system() -> str:
    return (
        "You are the Research Radar outline planner. Return JSON only. "
        "Create natural, non-template section headings for a technical blog from the point of view of a Principal MLE at Autodesk. "
        "The outline must be thesis-led, deep, and mechanism-first, not a broad roundup. "
        "It must cover thesis, mechanisms, math/objectives where supported, experiments, limitations, engineering impact, "
        "cross-paper synthesis, and relevance to AEC foundation models or 2D document workflows. "
        "Ban vague corporate headings such as 'Navigating the Future', 'Bridging the Digital Divide', and 'Our Path Forward'."
    )


def _outline_user(pack: EvidencePack, selection: SelectionResult) -> str:
    return (
        f"Create an outline for a daily post of at least {config.daily_min_words()} words. "
        "Do not use fixed headings such as 'Paper mechanisms' or 'Why it matters'. "
        "Use 3-4 deep primary-paper sections, one cross-paper comparison section, and one adoption section for Autodesk/AEC document workflows. "
        "Headings must name a technical object, mechanism, benchmark, or tradeoff. "
        "Return JSON in this exact shape:\n"
        "{\n"
        '  "title": "short post title",\n'
        '  "angle": "one sentence angle",\n'
        '  "sections": [{"heading": "natural heading", "intent": "role description", "evidence_ids": ["E1"], "word_budget": 220}],\n'
        '  "suggested_tags": ["tag"]\n'
        "}\n\n"
        f"SELECTION:\n{selection.model_dump_json(indent=2)}\n\n"
        f"EVIDENCE_CARDS:\n{json.dumps([card.model_dump(mode='json') for card in pack.evidence_cards], indent=2, ensure_ascii=False)}\n\n"
        f"EVIDENCE_PACK:\n{pack.as_prompt_json()}"
    )


def _outline_repair_system() -> str:
    return (
        "You repair an invalid research outline. Return JSON only in the exact DailyOutline schema. "
        "Keep concrete, non-template headings and ensure intents cover thesis, mechanism, math/objective, "
        "experiments, limitations, impact, cross-paper synthesis, and Autodesk/AEC/document relevance. "
        "Avoid corporate-strategy headings and name technical mechanisms or tradeoffs."
    )


def _outline_repair_user(
    pack: EvidencePack,
    selection: SelectionResult,
    *,
    outline: DailyOutline | None,
    errors: list[str],
    parse_error: str,
) -> str:
    return (
        f"MIN_WORDS: {config.daily_min_words()}\n\n"
        f"PARSE_ERROR: {parse_error or 'none'}\n"
        f"VALIDATION_ERRORS: {json.dumps(errors)}\n\n"
        f"PREVIOUS_OUTLINE:\n{outline.model_dump_json(indent=2) if outline is not None else '{}'}\n\n"
        "Fix the outline and return JSON with: title, angle, sections[{heading,intent,evidence_ids,word_budget}], suggested_tags. "
        "Include 3-4 deep primary-paper sections, one cross-paper comparison section, and one Autodesk/AEC adoption section.\n\n"
        f"SELECTION:\n{selection.model_dump_json(indent=2)}\n\n"
        f"EVIDENCE_CARDS:\n{json.dumps([card.model_dump(mode='json') for card in pack.evidence_cards], indent=2, ensure_ascii=False)}\n\n"
        f"EVIDENCE_PACK:\n{pack.as_prompt_json()}"
    )


def _fallback_outline(pack: EvidencePack, selection: SelectionResult) -> DailyOutline:
    title = "Research Radar: Autodesk AEC document intelligence update"
    if pack.ranked_items:
        title = f"Research Radar: {pack.ranked_items[0].item.title[:70]}"
    selected_ids = set(selection.selected_item_ids)
    selected_item_ids = [
        ranked.item.item_id
        for ranked in pack.ranked_items
        if ranked.item.item_id in selected_ids
    ] or [ranked.item.item_id for ranked in pack.ranked_items[:4]]
    chunks = [chunk for chunk in pack.chunks if chunk.item_id in set(selected_item_ids)] or list(pack.chunks)

    def pick_ids(evidence_types: tuple[str, ...], limit: int = 3) -> list[str]:
        out: list[str] = []
        wanted = {x.lower() for x in evidence_types}
        for chunk in chunks:
            et = (chunk.evidence_type or "").lower()
            if et in wanted and chunk.evidence_id not in out:
                out.append(chunk.evidence_id)
                if len(out) >= limit:
                    return out
        for chunk in chunks:
            if chunk.evidence_id not in out:
                out.append(chunk.evidence_id)
                if len(out) >= limit:
                    return out
        return out

    min_words = max(config.daily_min_words(), 900)
    per_section = max(140, min_words // 6)
    sections = [
        {
            "heading": "Document-model reliability starts with measurable failure boundaries",
            "intent": "technical thesis angle framing Autodesk AEC document relevance",
            "evidence_ids": pick_ids(("impact", "mechanism")),
            "word_budget": per_section,
        },
        {
            "heading": "Primary mechanisms worth testing against drawing workflows",
            "intent": "mechanism method architecture pipeline",
            "evidence_ids": pick_ids(("mechanism",)),
            "word_budget": per_section,
        },
        {
            "heading": "Objectives and metrics define the adoption boundary",
            "intent": "math objective loss optimization metric",
            "evidence_ids": pick_ids(("math_or_objective", "experiment")),
            "word_budget": per_section,
        },
        {
            "heading": "Evaluation evidence separates prototypes from product bets",
            "intent": "experiments evidence benchmark evaluation ablation",
            "evidence_ids": pick_ids(("experiment",)),
            "word_budget": per_section,
        },
        {
            "heading": "Cross-paper tradeoffs that change the engineering plan",
            "intent": "cross-paper synthesis compare contrast tradeoff limitations caveat failure risk",
            "evidence_ids": pick_ids(("limitation", "experiment")),
            "word_budget": per_section,
        },
        {
            "heading": "Adoption tests for Autodesk AEC document systems",
            "intent": "impact engineering production practical Autodesk AEC document drawing sheet CAD BIM",
            "evidence_ids": pick_ids(("impact", "limitation")),
            "word_budget": per_section,
        },
        {
            "heading": "Risks to validate before implementation",
            "intent": "limitations caveat failure risk tradeoff",
            "evidence_ids": pick_ids(("limitation",)),
            "word_budget": per_section,
        },
    ]
    return DailyOutline(
        title=title,
        angle="Recent work defines practical boundaries for Autodesk-facing AEC and 2D document ML systems.",
        sections=sections,
        suggested_tags=list(selection.suggested_tags),
    )


def _parse_outline(text: str) -> DailyOutline:
    try:
        return DailyOutline.model_validate(jsonish.loads_object(text))
    except (SyntaxError, ValidationError, ValueError, json.JSONDecodeError) as exc:
        raise OutlineError(f"outline_malformed:{exc}") from exc


def _json_payload(text: str) -> str:
    return jsonish.extract_object(text)
