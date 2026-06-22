from __future__ import annotations

import json
import logging
import re

from pydantic import ValidationError

from . import config
from . import jsonish
from .llm import CompletionRejector, LLMClient, RejectedCompletionTracker
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

STRUCTURAL_OUTLINE_ERROR_PREFIXES = (
    "missing_outline_title",
    "missing_outline_angle",
    "too_few_outline_sections",
    "duplicate_outline_headings",
    "generic_outline_heading:",
    "blank_outline_heading",
    "unknown_outline_evidence:",
    "unknown_outline_focus:",
    "late_outline_item:",
    "supporting_section_uses_primary_focus",
    "primary_section_uses_nonprimary_focus",
    "missing_outline_training_howto",
    "outline_word_budget_too_short:",
)


def generate_daily_outline(
    pack: EvidencePack,
    *,
    selection: SelectionResult,
    llm: LLMClient,
) -> DailyOutline:
    max_tokens = config.outline_max_tokens()
    primary_tracker = RejectedCompletionTracker()
    primary_reject = _outline_completion_rejector(pack, allow_close=True)
    strict_reject = _outline_completion_rejector(pack, allow_close=False)
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
                reject_completion=primary_reject,
                rejected_tracker=primary_tracker,
            )
        )
        errors = validate_outline(outline, pack)
        if not errors:
            LOG.info("outline path: primary")
            return outline
        if _outline_close_enough_for_repair(errors):
            LOG.info("outline path: primary close enough for repair (%d gaps)", len(errors))
        else:
            LOG.warning("outline path: primary returned %d validation gaps after chain", len(errors))
    except (OutlineError, RuntimeError) as exc:
        parse_error = str(exc)
        outline = None
        errors = []
    if outline is None:
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
                reject_completion=strict_reject,
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
                    reject_completion=strict_reject,
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

    fallback = _deterministic_outline(pack, selection)
    fallback_errors = validate_outline(fallback, pack)
    if not fallback_errors:
        LOG.warning("outline path: deterministic repair fallback")
        return fallback
    LOG.warning("outline deterministic repair fallback failed: %s", fallback_errors)

    combined = [*errors, *repaired_errors]
    combined.extend(f"deterministic_outline:{error}" for error in fallback_errors)
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
    rejected_tracker: RejectedCompletionTracker | None = None,
) -> str:
    if isinstance(llm, LLMClient):
        return llm.complete(
            system=system,
            user=user,
            max_tokens=max_tokens,
            task=task,
            reject_completion=reject_completion,
            rejected_tracker=rejected_tracker,
        )
    return llm.complete(system=system, user=user, max_tokens=max_tokens)


def _outline_close_enough_for_repair(errors: list[str]) -> bool:
    if not errors:
        return True
    if len(errors) > config.outline_repair_error_threshold():
        return False
    return not any(_is_structural_outline_error(error) for error in errors)


def _deterministic_outline(pack: EvidencePack, selection: SelectionResult) -> DailyOutline:
    evidence_by_item: dict[str, list[str]] = {}
    for chunk in pack.chunks:
        evidence_by_item.setdefault(chunk.item_id, []).append(chunk.evidence_id)
    selected_ids = [item_id for item_id in selection.selected_item_ids if item_id in evidence_by_item]
    if not selected_ids:
        selected_ids = [ranked.item.item_id for ranked in pack.ranked_items if ranked.item.item_id in evidence_by_item]

    primary_cards = [
        card
        for card in pack.evidence_cards
        if card.role == "primary" and card.item_id in selected_ids
    ]
    if not primary_cards:
        selected_set = set(selected_ids[: config.daily_primary_papers()])
        primary_cards = [card for card in pack.evidence_cards if card.item_id in selected_set]
    primary_cards = primary_cards[: max(2, config.daily_primary_papers())]
    support_cards = [card for card in pack.evidence_cards if card not in primary_cards]
    all_primary_evidence = _outline_evidence_ids(primary_cards, evidence_by_item)
    all_selected_evidence = _outline_evidence_ids([*primary_cards, *support_cards], evidence_by_item)
    training = _pack_has_training_system_focus(pack)

    title_object = "scaling AEC foundation-model training" if training else "solving AEC foundation-model reliability"
    title = f"Research Radar: {title_object.title()}"
    angle = (
        "The evidence is useful only when it becomes a concrete AEC foundation-model decision with measurable limits, "
        "failure modes, and release gates."
    )
    sections: list[dict[str, object]] = [
        {
            "heading": "The AEC foundation-model problem to solve",
            "intent": "technical thesis angle framing for the selected evidence cluster",
            "evidence_ids": all_primary_evidence[:4] or all_selected_evidence[:4],
            "word_budget": 260,
            "focus_item_ids": [],
            "section_role": "synthesis",
            "split_reason": "",
        }
    ]
    for index, card in enumerate(primary_cards, start=1):
        heading = _primary_outline_heading(card, training=training and index == 1)
        sections.append(
            {
                "heading": heading,
                "intent": _primary_outline_intent(card, training=training and index == 1),
                "evidence_ids": _outline_evidence_ids([card], evidence_by_item)[:3],
                "word_budget": 260,
                "focus_item_ids": [card.item_id],
                "section_role": "primary",
                "split_reason": "",
            }
        )
    sections.extend(
        [
            {
                "heading": "Benchmarks, failure modes, and debug signals",
                "intent": "experiments evidence benchmark evaluation ablation limitation caveat failure risk tradeoff",
                "evidence_ids": all_selected_evidence[:5],
                "word_budget": 260,
                "focus_item_ids": [],
                "section_role": "synthesis",
                "split_reason": "",
            },
            {
                "heading": "Mechanism tradeoffs across the evidence",
                "intent": "compare contrast cross-paper synthesis tradeoff across mechanisms objectives and limits",
                "evidence_ids": all_primary_evidence[:5] or all_selected_evidence[:5],
                "word_budget": 220,
                "focus_item_ids": [],
                "section_role": "synthesis",
                "split_reason": "",
            },
            {
                "heading": "Autodesk AEC adoption gate and next experiment",
                "intent": "impact engineering production practical Autodesk AEC document drawing sheet CAD BIM validation release gate",
                "evidence_ids": all_selected_evidence[:5],
                "word_budget": 260,
                "focus_item_ids": [],
                "section_role": "adoption",
                "split_reason": "",
            },
        ]
    )
    return DailyOutline(title=title, angle=angle, sections=sections, suggested_tags=selection.suggested_tags)


def _outline_evidence_ids(cards: list[object], evidence_by_item: dict[str, list[str]]) -> list[str]:
    out: list[str] = []
    for card in cards:
        item_id = getattr(card, "item_id", "")
        for evidence_id in evidence_by_item.get(item_id, []):
            if evidence_id not in out:
                out.append(evidence_id)
    return out


def _primary_outline_heading(card: object, *, training: bool) -> str:
    title = str(getattr(card, "title", "") or "Primary mechanism")
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9-]*", title)
    short = " ".join(words[:7]) or "Primary mechanism"
    if training:
        return f"Training pipeline decision from {short}"
    return f"Engineering decision from {short}"


def _primary_outline_intent(card: object, *, training: bool) -> str:
    base = (
        "mechanism method architecture pipeline math objective optimization metric "
        "experiment evidence benchmark evaluation limitation risk"
    )
    if training:
        return (
            base
            + " distributed training training stack FSDP sharding parallelism activation checkpointing "
            "communication profiling throughput data pipeline GPU utilization"
        )
    return base


def _is_structural_outline_error(error: str) -> bool:
    return any(error == prefix or error.startswith(prefix) for prefix in STRUCTURAL_OUTLINE_ERROR_PREFIXES)


def _outline_completion_rejector(pack: EvidencePack, *, allow_close: bool) -> CompletionRejector:
    def _reject(text: str) -> str | None:
        try:
            outline = _parse_outline(text)
        except OutlineError as exc:
            return str(exc)
        errors = validate_outline(outline, pack)
        if not errors:
            return None
        if allow_close and _outline_close_enough_for_repair(errors):
            return None
        return "outline_invalid:" + ",".join(errors[:8])

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
        "Create natural, non-template section headings for a technical blog from the point of view of a Principal MLE in AEC at Autodesk. "
        "The outline must be problem-led, deep, and mechanism-first, not a broad roundup or literature survey. "
        "Frame the post around building AEC foundation models: scaling training pipelines, data/evaluation loops, drawing/document/BIM grounding, "
        "failure diagnosis, and concrete engineering decisions. "
        "When the cluster is about scaled LLM training, include a concrete how-to section on training architecture, "
        "parallelism/sharding, memory and communication bottlenecks, checkpointing, profiling, or data-pipeline tradeoffs. "
        "It must cover thesis, mechanisms, math/objectives where supported, experiments, limitations, engineering impact, "
        "and cross-paper synthesis. Include an adoption or production-readiness section only when the selected cluster supports it. "
        "Ban vague corporate headings such as 'Navigating the Future', 'Bridging the Digital Divide', and 'Our Path Forward'. "
        "Normally use one deep primary section per primary evidence item. Split one item across multiple sections only when each split has a distinct technical purpose and an explicit split_reason."
    )


def _outline_user(pack: EvidencePack, selection: SelectionResult) -> str:
    return (
        f"Create an outline for a daily post of at least {config.daily_min_words()} words. "
        "Do not use fixed headings such as 'Paper mechanisms' or 'Why it matters'. "
        "Use the evidence to write a problem-solving memo, not a paper roundup. "
        "Use 3-4 deep primary evidence sections, one comparison section, and one adoption or production-readiness section when warranted. "
        "Headings must name a concrete problem, technical object, mechanism, benchmark, bottleneck, or tradeoff. "
        "Prefer titles such as 'Diagnosing data-loader stalls before scaling FSDP' over titles that merely name papers. "
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
        "Include 3-4 deep primary evidence sections, one cross-evidence comparison section, and one Autodesk/AEC adoption section. "
        "If VALIDATION_ERRORS contains missing_outline_training_howto, add a concrete training systems how-to section naming the sharding, "
        "parallelism, memory, communication, checkpointing, profiling, or data-pipeline decision supported by the evidence. "
        "Normally keep one primary evidence item to one deep section; if one item is split, explain the technical split_reason. "
        "Supporting items must stay in explicitly supporting sections or inside synthesis/adoption sections.\n\n"
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
