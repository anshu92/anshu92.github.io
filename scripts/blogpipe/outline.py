from __future__ import annotations

import json
import re

from pydantic import ValidationError

from . import config
from .llm import LLMClient
from .models import DailyOutline, EvidencePack, SelectionResult


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
}


def generate_daily_outline(
    pack: EvidencePack,
    *,
    selection: SelectionResult,
    llm: LLMClient,
) -> DailyOutline:
    outline = _parse_outline(llm.complete(system=_outline_system(), user=_outline_user(pack, selection), max_tokens=2200))
    errors = validate_outline(outline, pack)
    if errors:
        raise OutlineError("outline_invalid:" + ",".join(errors))
    return outline


def validate_outline(outline: DailyOutline, pack: EvidencePack) -> list[str]:
    errors: list[str] = []
    if not outline.title.strip():
        errors.append("missing_outline_title")
    headings = [section.heading.strip().lower() for section in outline.sections]
    if len(headings) < 4:
        errors.append("too_few_outline_sections")
    if len(headings) != len(set(headings)):
        errors.append("duplicate_outline_headings")
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
        "The outline must cover thesis, mechanisms, math/objectives where supported, experiments, limitations, engineering impact, "
        "and relevance to AEC foundation models or 2D document workflows."
    )


def _outline_user(pack: EvidencePack, selection: SelectionResult) -> str:
    return (
        f"Create an outline for a daily post of at least {config.daily_min_words()} words. "
        "Do not use fixed headings such as 'Paper mechanisms' or 'Why it matters'. "
        "Return JSON in this exact shape:\n"
        "{\n"
        '  "title": "short post title",\n'
        '  "angle": "one sentence angle",\n'
        '  "sections": [{"heading": "natural heading", "intent": "role description", "evidence_ids": ["E1"], "word_budget": 220}],\n'
        '  "suggested_tags": ["tag"]\n'
        "}\n\n"
        f"SELECTION:\n{selection.model_dump_json(indent=2)}\n\n"
        f"EVIDENCE_PACK:\n{pack.as_prompt_json()}"
    )


def _parse_outline(text: str) -> DailyOutline:
    try:
        return DailyOutline.model_validate_json(_json_payload(text))
    except (ValidationError, ValueError, json.JSONDecodeError) as exc:
        raise OutlineError(f"outline_malformed:{exc}") from exc


def _json_payload(text: str) -> str:
    raw = (text or "").strip()
    fenced = re.match(r"^```(?:json)?\n([\s\S]*?)\n```$", raw, re.I)
    if fenced:
        raw = fenced.group(1).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        return raw[start : end + 1]
    return raw
