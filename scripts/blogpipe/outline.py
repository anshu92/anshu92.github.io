from __future__ import annotations

import json
import logging
import re

from pydantic import ValidationError

from . import config
from . import jsonish
from .llm import CompletionRejector, LLMClient
from .models import DailyOutline, EvidencePack, SelectionResult
from .topics import has_training_system_focus

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

TRAINING_HOWTO_INTENT_CUES: tuple[str, ...] = (
    "distributed training",
    "training stack",
    "fsdp",
    "deepspeed",
    "megatron",
    "sharding",
    "parallelism",
    "tensor parallel",
    "pipeline parallel",
    "sequence parallel",
    "activation checkpoint",
    "checkpointing",
    "microbatch",
    "communication",
    "all-reduce",
    "nccl",
    "profiling",
    "throughput",
    "gpu utilization",
    "data pipeline",
)

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
    reject = _outline_completion_rejector(pack)
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
                reject_completion=reject,
            )
        )
        if not validate_outline(outline, pack):
            LOG.info("outline path: primary")
            return outline
    except (OutlineError, RuntimeError) as exc:
        parse_error = str(exc)
    if outline is not None:
        errors = validate_outline(outline, pack)
    else:
        errors = []

    repair_error = ""
    repaired_errors: list[str] = []
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
                reject_completion=reject,
            )
        )
        repaired_errors = validate_outline(repaired, pack)
        if not repaired_errors:
            LOG.warning("outline path: repair")
            return repaired
        try:
            repaired_again = _parse_outline(
                _outline_complete(
                    llm,
                    system=_outline_repair_system(),
                    user=_outline_repair_user(
                        pack,
                        selection,
                        outline=repaired,
                        errors=repaired_errors,
                        parse_error="",
                    ),
                    task="outline_repair",
                    max_tokens=max_tokens,
                    reject_completion=reject,
                )
            )
            repaired_errors = validate_outline(repaired_again, pack)
            if not repaired_errors:
                LOG.warning("outline path: repair (second pass)")
                return repaired_again
            repaired = repaired_again
        except (OutlineError, RuntimeError) as exc:
            LOG.warning("outline second repair failed: %s", exc)
            repair_error = str(exc)
    except (OutlineError, RuntimeError) as exc:
        LOG.warning("outline repair failed: %s", exc)
        repair_error = str(exc)
        repaired_errors = []

    combined = [*errors, *repaired_errors]
    if parse_error:
        combined.append(f"outline_parse_failed:{parse_error}")
    if repair_error:
        combined.append(f"outline_repair_failed:{repair_error}")
    raise OutlineError("outline_invalid:" + ",".join(combined or ["outline_unusable"]))


def _outline_complete(
    llm: LLMClient,
    *,
    system: str,
    user: str,
    task: str,
    max_tokens: int,
    reject_completion: CompletionRejector | None = None,
) -> str:
    if isinstance(llm, LLMClient):
        return llm.complete(
            system=system,
            user=user,
            max_tokens=max_tokens,
            task=task,
            reject_completion=reject_completion,
        )
    return llm.complete(system=system, user=user, max_tokens=max_tokens)


def _outline_completion_rejector(pack: EvidencePack) -> CompletionRejector:
    def _reject(text: str) -> str | None:
        try:
            outline = _parse_outline(text)
        except OutlineError as exc:
            return str(exc)
        errors = validate_outline(outline, pack)
        if errors:
            return "outline_invalid:" + ",".join(errors[:8])
        return None

    return _reject


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
    evidence_to_item = {chunk.evidence_id: chunk.item_id for chunk in pack.chunks}
    selected_item_ids = {ranked.item.item_id for ranked in pack.ranked_items}
    primary_item_ids = {
        ranked.item.item_id
        for ranked in pack.ranked_items
        if str(ranked.item.extra.get("selector_role", "")).lower() == "primary"
    } or {
        ranked.item.item_id
        for ranked in pack.ranked_items
        if ranked.item.source_kind == "paper"
    }
    section_focus_counts: dict[str, int] = {}
    has_experiment_evidence = any(
        card.experiment and card.experiment != "not found in evidence"
        for card in pack.evidence_cards
        if card.role == "primary"
    )
    has_experiment_outline = False
    for section in outline.sections:
        if not section.heading.strip():
            errors.append("blank_outline_heading")
        unknown = sorted(set(section.evidence_ids) - known_evidence)
        if unknown:
            errors.append("unknown_outline_evidence:" + ",".join(unknown))
        referenced_items = {
            evidence_to_item[evidence_id]
            for evidence_id in section.evidence_ids
            if evidence_id in evidence_to_item
        }
        declared_focus = set(section.focus_item_ids)
        if declared_focus - selected_item_ids:
            errors.append("unknown_outline_focus:" + ",".join(sorted(declared_focus - selected_item_ids)))
        if referenced_items - selected_item_ids:
            errors.append("late_outline_item:" + ",".join(sorted(referenced_items - selected_item_ids)))
        focus_ids = declared_focus or referenced_items
        if section.section_role == "supporting" and focus_ids & primary_item_ids:
            errors.append("supporting_section_uses_primary_focus")
        if section.section_role == "primary" and focus_ids and not focus_ids <= primary_item_ids:
            errors.append("primary_section_uses_nonprimary_focus")
        if section.section_role == "primary":
            for item_id in focus_ids & primary_item_ids:
                section_focus_counts[item_id] = section_focus_counts.get(item_id, 0) + 1
        intent_blob = f"{section.intent} {section.heading}".lower()
        if any(alias in intent_blob for alias in REQUIRED_INTENTS["experiments"]):
            has_experiment_outline = True
    for item_id, count in sorted(section_focus_counts.items()):
        if count > 1:
            reasons = [
                section.split_reason.strip()
                for section in outline.sections
                if section.section_role == "primary" and item_id in (set(section.focus_item_ids) or {
                    evidence_to_item[evidence_id]
                    for evidence_id in section.evidence_ids
                    if evidence_id in evidence_to_item
                })
            ]
            if not reasons or any(not reason for reason in reasons):
                errors.append(f"duplicate_primary_outline_focus:{item_id}")
    if has_experiment_evidence and not has_experiment_outline:
        errors.append("missing_outline_experiment_detail_section")
    intents = " ".join(section.intent.lower() for section in outline.sections)
    for role, aliases in REQUIRED_INTENTS.items():
        if not any(alias in intents for alias in aliases):
            errors.append(f"missing_outline_intent:{role}")
    if _pack_has_training_system_focus(pack):
        training_outline_blob = " ".join(
            f"{section.heading} {section.intent}".lower()
            for section in outline.sections
        )
        if not any(cue in training_outline_blob for cue in TRAINING_HOWTO_INTENT_CUES):
            errors.append("missing_outline_training_howto")
    total_budget = sum(max(0, section.word_budget) for section in outline.sections)
    if total_budget and total_budget < config.daily_min_words():
        errors.append(f"outline_word_budget_too_short:{total_budget}/{config.daily_min_words()}")
    return errors


def _pack_has_training_system_focus(pack: EvidencePack) -> bool:
    parts = [
        f"{ranked.item.title} {ranked.item.abstract_or_excerpt} {' '.join(ranked.item.tags)}"
        for ranked in pack.ranked_items
    ]
    parts.extend(f"{card.title} {card.mechanism} {card.experiment} {card.impact}" for card in pack.evidence_cards)
    return has_training_system_focus(" ".join(parts))


def _outline_system() -> str:
    return (
        "You are the Research Radar outline planner. Return JSON only. "
        "Create natural, non-template section headings for a technical blog from the point of view of a Principal MLE. "
        "The outline must be thesis-led, deep, and mechanism-first, not a broad roundup. "
        "When the cluster is about scaled LLM training, include a concrete how-to section on training architecture, "
        "parallelism/sharding, memory and communication bottlenecks, checkpointing, profiling, or data-pipeline tradeoffs. "
        "It must cover thesis, mechanisms, math/objectives where supported, experiments, limitations, engineering impact, "
        "and cross-paper synthesis. Include an adoption or production-readiness section only when the selected cluster supports it. "
        "Ban vague corporate headings such as 'Navigating the Future', 'Bridging the Digital Divide', and 'Our Path Forward'. "
        "Normally use one deep primary section per primary paper. Split a primary paper across multiple sections only when each split has a distinct technical purpose and an explicit split_reason."
    )


def _outline_user(pack: EvidencePack, selection: SelectionResult) -> str:
    return (
        f"Create an outline for a daily post of at least {config.daily_min_words()} words. "
        "Do not use fixed headings such as 'Paper mechanisms' or 'Why it matters'. "
        "Use 3-4 deep primary-paper sections, one cross-paper comparison section, and one adoption or production-readiness section when warranted. "
        "Headings must name a technical object, mechanism, benchmark, or tradeoff. "
        "For scaled LLM training clusters, one heading should read like a training systems runbook: name the sharding/parallelism, "
        "memory, communication, checkpointing, profiling, data-pipeline, or throughput decision being taught. "
        "Return JSON in this exact shape:\n"
        "{\n"
        '  "title": "short post title",\n'
        '  "angle": "one sentence angle",\n'
        '  "sections": [{"heading": "natural heading", "intent": "role description", "evidence_ids": ["E1"], "word_budget": 220, '
        '"focus_item_ids": ["item-id"], "section_role": "primary|supporting|synthesis|adoption", "split_reason": ""}],\n'
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
        "experiments, limitations, impact, cross-paper synthesis, and production or adoption implications when supported. "
        "For scaled LLM training clusters, include training-how-to intent such as sharding/parallelism, activation checkpointing, "
        "communication bottlenecks, data-pipeline throughput, checkpoint recovery, profiling, or GPU utilization. "
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
        "Fix the outline and return JSON with: title, angle, sections[{heading,intent,evidence_ids,word_budget,focus_item_ids,section_role,split_reason}], suggested_tags. "
        "Include 3-4 deep primary-paper sections, one cross-paper comparison section, and one Autodesk/AEC adoption section. "
        "If VALIDATION_ERRORS contains missing_outline_training_howto, add a concrete training systems how-to section naming the sharding, "
        "parallelism, memory, communication, checkpointing, profiling, or data-pipeline decision supported by the evidence. "
        "Normally keep one primary paper to one deep section; if a primary paper is split, explain the technical split_reason. "
        "Supporting papers must stay in explicitly supporting sections or inside synthesis/adoption sections.\n\n"
        f"SELECTION:\n{selection.model_dump_json(indent=2)}\n\n"
        f"EVIDENCE_CARDS:\n{json.dumps([card.model_dump(mode='json') for card in pack.evidence_cards], indent=2, ensure_ascii=False)}\n\n"
        f"EVIDENCE_PACK:\n{pack.as_prompt_json()}"
    )


def _parse_outline(text: str) -> DailyOutline:
    try:
        return DailyOutline.model_validate(jsonish.loads_object(text))
    except (SyntaxError, ValidationError, ValueError, json.JSONDecodeError) as exc:
        raise OutlineError(f"outline_malformed:{exc}") from exc


def _json_payload(text: str) -> str:
    return jsonish.extract_object(text)
