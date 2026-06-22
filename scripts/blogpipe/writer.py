from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit

from markdown_it import MarkdownIt

from . import config, jsonish, memory
from .llm import LLMClient
from .models import DailyOutline, EvidencePack, RankedItem, SelectionResult, WriteResult
from .topics import has_training_system_focus

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
PROBLEM_SOLVING_CUES = (
    "problem",
    "blocker",
    "bottleneck",
    "root cause",
    "hypothesis",
    "diagnose",
    "debug",
    "experiment",
    "prototype",
    "decision gate",
    "release gate",
    "validation test",
    "next action",
    "measure",
    "profile",
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
TRAINING_HOWTO_CONCEPTS: dict[str, tuple[str, ...]] = {
    "training_stack": (
        "distributed training",
        "training stack",
        "fsdp",
        "deepspeed",
        "megatron",
        "data parallel",
        "tensor parallel",
        "pipeline parallel",
        "sequence parallel",
        "sharding",
        "zero optimizer",
        "zero redundancy",
        "zero-1",
        "zero-2",
        "zero-3",
    ),
    "scaling_bottleneck": (
        "memory",
        "activation",
        "checkpoint",
        "communication",
        "all-reduce",
        "nccl",
        "throughput",
        "gpu utilization",
        "microbatch",
        "pipeline bubble",
        "data pipeline",
    ),
    "principal_action": (
        "profile",
        "benchmark",
        "ablation",
        "validate",
        "measure",
        "tradeoff",
        "rollout",
        "release gate",
        "failure mode",
        "recovery",
    ),
}
TRAINING_TECH_DETAIL_CONCEPTS: dict[str, tuple[str, ...]] = {
    "parallelism_or_sharding": (
        "fsdp",
        "zero",
        "deepspeed",
        "megatron",
        "data parallel",
        "tensor parallel",
        "pipeline parallel",
        "sequence parallel",
        "sharding",
    ),
    "memory_and_state": (
        "activation checkpoint",
        "gradient checkpoint",
        "optimizer state",
        "parameter state",
        "mixed precision",
        "microbatch",
        "memory headroom",
        "memory pressure",
    ),
    "communication": (
        "nccl",
        "all-reduce",
        "reduce-scatter",
        "all-gather",
        "communication",
        "pipeline bubble",
        "network",
    ),
    "data_and_recovery": (
        "data pipeline",
        "data loader",
        "token packing",
        "checkpoint recovery",
        "checkpoint cadence",
        "restart",
        "resume",
    ),
    "observability_and_gate": (
        "profile",
        "profiling",
        "gpu utilization",
        "throughput",
        "benchmark",
        "ablation",
        "release gate",
        "rollout gate",
    ),
}
TRAINING_RUNBOOK_ACTION_CONCEPTS: dict[str, tuple[str, ...]] = {
    "stack_decision": (
        "choose",
        "select",
        "decide",
        "adopt",
        "use fsdp",
        "use zero",
        "sharding boundary",
        "parallelism is needed",
    ),
    "measurement_plan": (
        "profile",
        "profiling",
        "measure",
        "monitor",
        "trace",
        "benchmark",
        "gpu utilization",
        "throughput",
    ),
    "bottleneck_diagnosis": (
        "bottleneck",
        "root cause",
        "diagnose",
        "isolate",
        "debug",
        "memory pressure",
        "communication dominates",
        "data pipeline starves",
        "data loader stalls",
    ),
    "scaling_gate": (
        "validate",
        "release gate",
        "rollout gate",
        "recovery test",
        "checkpoint recovery",
        "failure mode",
        "before scaling",
        "before rollout",
    ),
}
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
    title = _safe_daily_title(outline.title or f"Research Radar: ML systems update - {today}")
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
    def _single_pass_draft(*, retry_short: bool = True) -> str:
        try:
            body = _call_daily_draft(
                client,
                _daily_system(),
                _daily_user(pack, outline, selection, title),
                outline=outline,
            )
        except Exception as exc:
            LOG.warning("daily draft LLM failed; using deterministic emergency draft: %s", exc)
            setattr(client, "_blogpipe_emergency_daily_draft", True)
            return _emergency_daily_draft(pack=pack, outline=outline, selection=selection, title=title)
        min_words = max(500, config.daily_min_words() // 3)
        if retry_short and _word_count(body) < min_words and _has_call_budget(client, 1):
            LOG.warning(
                "single-pass daily draft too short (%s words, need >= %s); retrying draft",
                _word_count(body),
                min_words,
            )
            if getattr(client, "_rate_limit_hits", 0) > 0:
                _sleep_for_rate_limit(client)
            try:
                body = _call_daily_draft(
                    client,
                    _daily_system(),
                    _daily_user(pack, outline, selection, title),
                    outline=outline,
                )
            except Exception as exc:
                LOG.warning("daily draft retry failed; using deterministic emergency draft: %s", exc)
                setattr(client, "_blogpipe_emergency_daily_draft", True)
                return _emergency_daily_draft(pack=pack, outline=outline, selection=selection, title=title)
        return body

    if not isinstance(client, LLMClient):
        return _single_pass_draft()
    if not config.sectionwise_drafting_enabled():
        return _single_pass_draft()

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
        return _single_pass_draft()
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
            return _single_pass_draft()
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
        merged = _merge_daily_body(title=title, outline=outline, section_drafts=drafted, edited=edited)
        return merged or _single_pass_draft()
    except Exception as exc:  # noqa: BLE001
        LOG.warning("sectionwise daily writer failed; using fallback single pass: %s", exc)
        _sleep_for_rate_limit(client)
        return _single_pass_draft()


def _emergency_daily_draft(
    *,
    pack: EvidencePack,
    outline: DailyOutline,
    selection: SelectionResult,
    title: str,
) -> str:
    chunks_by_id = {chunk.evidence_id: chunk for chunk in pack.chunks}
    chunks_by_item: dict[str, list[object]] = {}
    for chunk in pack.chunks:
        chunks_by_item.setdefault(chunk.item_id, []).append(chunk)
    primary_ids = _primary_item_ids(pack)
    primary_cards = [card for card in pack.evidence_cards if card.item_id in primary_ids]
    if not primary_cards:
        primary_cards = pack.evidence_cards[: max(1, min(3, len(pack.evidence_cards)))]
    all_cards = primary_cards or pack.evidence_cards
    training = _pack_has_training_system_focus(pack)
    lines = [f"# {title}", ""]
    for index, section in enumerate(outline.sections):
        lines.append(f"## {section.heading}")
        lines.append("")
        focus_cards = _cards_for_section(section, all_cards, pack.evidence_cards)
        if not focus_cards:
            focus_cards = all_cards
        for paragraph in _emergency_section_paragraphs(section, focus_cards, chunks_by_item, chunks_by_id, training=training, index=index):
            lines.append(paragraph)
            lines.append("")
        lines.append("")
    lines.append("## Primary evidence depth check")
    lines.append("")
    for card in primary_cards:
        lines.append(_emergency_primary_depth_paragraph(card, chunks_by_item, chunks_by_id, training=training))
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _cards_for_section(section: object, primary_cards: list[object], evidence_cards: list[object]) -> list[object]:
    focus_ids = set(getattr(section, "focus_item_ids", []) or [])
    evidence_ids = set(getattr(section, "evidence_ids", []) or [])
    if focus_ids:
        return [card for card in evidence_cards if getattr(card, "item_id", "") in focus_ids]
    if evidence_ids:
        evidence_item_ids = {
            item_id
            for card in evidence_cards
            for item_id in [getattr(card, "item_id", "")]
            if set(getattr(card, "evidence_ids", {}).get("mechanism", [])) & evidence_ids
        }
        if evidence_item_ids:
            return [card for card in evidence_cards if getattr(card, "item_id", "") in evidence_item_ids]
    return primary_cards[:2]


def _emergency_section_paragraphs(
    section: object,
    cards: list[object],
    chunks_by_item: dict[str, list[object]],
    chunks_by_id: dict[str, object],
    *,
    training: bool,
    index: int,
) -> list[str]:
    card = cards[index % len(cards)]
    next_card = cards[(index + 1) % len(cards)]
    citation = _best_citation(card, chunks_by_item, preferred=("mechanism", "math_or_objective", "experiment", "limitation"))
    second = _best_citation(next_card, chunks_by_item, preferred=("experiment", "limitation", "mechanism", "math_or_objective"))
    title = getattr(card, "title", "the primary paper")
    next_title = getattr(next_card, "title", "the comparison paper")
    mechanism = _safe_card_field(card, "mechanism")
    objective = _safe_card_field(card, "math_or_objective")
    experiment = _safe_card_field(card, "experiment")
    limitation = _safe_card_field(card, "limitation")
    impact = _safe_card_field(card, "impact")
    url = _citation_url(citation, chunks_by_id)
    second_url = _citation_url(second, chunks_by_id)
    cite = f"[{citation}] {url}".strip()
    second_cite = f"[{second}] {second_url}".strip()
    training_sentence = ""
    if training:
        training_sentence = (
            " For a scaled LLM training stack, the runbook implication is to choose FSDP or ZeRO sharding, "
            "validate tensor, pipeline, or sequence parallelism against memory pressure, track activation checkpointing, "
            "NCCL all-reduce communication, data pipeline throughput, checkpoint recovery, profiling traces, GPU utilization, "
            "and a rollout gate before increasing cluster scale."
        )
    return [
        (
            f"The AEC foundation-model problem in this section is the decision a principal MLE would need to make before scaling the next training, data, or evaluation loop. "
            f"{title} is useful because it exposes an engineering mechanism rather than a generic capability claim. The supported mechanism is {mechanism}, "
            f"while the objective or metric lens is {objective}. That gives the team a concrete validation test: reproduce the benchmark slice, inspect the failure mode, "
            f"and decide whether the architecture, pipeline, model, optimization, or metric can survive document, drawing, BIM/CAD, and construction workflow constraints. {cite}"
            f"{training_sentence}"
        ),
        (
            f"The experiment evidence is {experiment}, and the limitation is {limitation}. Those details make the adoption decision "
            f"more practical: the team should monitor latency, throughput, memory behavior, data quality, retrieval quality, observability, release gates, "
            f"and rollback or checkpoint behavior instead of treating the result as a broad endorsement. The operating question is what bottleneck this solves, "
            f"what new bottleneck it creates, and whether the next prototype should target training scale, grounding quality, or evaluation reliability. {cite}"
        ),
        (
            f"The cross-evidence synthesis is a tradeoff between {title} and {next_title}: one item may clarify mechanism or objective, "
            f"while the other exposes benchmark, evaluation, ablation, limitation, or failure evidence. The safe production claim is therefore "
            f"a transfer hypothesis, not a proven integrated stack. Compare and contrast the evidence together across the mechanisms: "
            f"one complements the other, whereas the combined adoption path still separates validated evidence from open risk. "
            f"For Autodesk AEC document, drawing, sheet, CAD, and BIM workflows, "
            f"the practical impact is {impact}; the release gate should test source-grounded retrieval, traceable citations, and domain-specific "
            f"failure cases before deployment. {second_cite}"
        ),
    ]


def _best_citation(card: object, chunks_by_item: dict[str, list[object]], *, preferred: tuple[str, ...]) -> str:
    item_id = getattr(card, "item_id", "")
    chunks = chunks_by_item.get(item_id, [])
    for evidence_type in preferred:
        for chunk in chunks:
            if getattr(chunk, "evidence_type", "") == evidence_type:
                return getattr(chunk, "evidence_id", "")
    return getattr(chunks[0], "evidence_id", "") if chunks else "E1"


def _emergency_primary_depth_paragraph(
    card: object,
    chunks_by_item: dict[str, list[object]],
    chunks_by_id: dict[str, object],
    *,
    training: bool,
) -> str:
    mechanism_id = _best_citation(card, chunks_by_item, preferred=("mechanism",))
    secondary_id = _best_citation(card, chunks_by_item, preferred=("experiment", "limitation", "math_or_objective"))
    mechanism_url = _citation_url(mechanism_id, chunks_by_id)
    secondary_url = _citation_url(secondary_id, chunks_by_id)
    title = getattr(card, "title", "the primary paper")
    mechanism = _safe_card_field(card, "mechanism")
    experiment = _safe_card_field(card, "experiment")
    limitation = _safe_card_field(card, "limitation")
    training_clause = (
        " For the training-system reading, this is also where FSDP or ZeRO sharding, activation checkpointing, "
        "communication, profiling, checkpoint recovery, GPU utilization, and data pipeline throughput become release-gate checks."
        if training
        else ""
    )
    return (
        f"For primary-depth coverage, {title} gets separate mechanism and evaluation treatment. The mechanism evidence is {mechanism} "
        f"[{mechanism_id}] {mechanism_url}. The experiment, objective, or limitation evidence is {experiment}; the operational risk is "
        f"{limitation} [{secondary_id}] {secondary_url}. This compare and contrast pass keeps the production tradeoff grounded across "
        f"the selected evidence, complements the synthesis sections, and separates benchmark evidence from adoption hypotheses.{training_clause}"
    )


def _citation_url(evidence_id: str, chunks_by_id: dict[str, object]) -> str:
    chunk = chunks_by_id.get(evidence_id)
    return str(getattr(chunk, "url", "")) if chunk is not None else ""


def _safe_card_field(card: object, field: str) -> str:
    value = str(getattr(card, field, "") or "").strip()
    if not value or value == "not found in evidence":
        fallbacks = {
            "mechanism": "the cited mechanism or system behavior, which should be treated as the implementation hypothesis to inspect",
            "math_or_objective": "the operational objective implied by the cited benchmark or evaluation evidence rather than an exposed formal loss",
            "experiment": "the available benchmark or evaluation signal; if the source is thin, that absence becomes a validation gap to close before adoption",
            "limitation": "the missing or incomplete limitation evidence itself, which means the team should add explicit failure-mode tests before scaling",
            "impact": "the plausible transfer to AEC foundation-model training, grounding, evaluation, or deployment, which remains a hypothesis until validated",
        }
        return fallbacks.get(field, f"the {field.replace('_', ' ')} that must be confirmed from the cited evidence")
    return re.sub(r"\s+", " ", value)[:220]


def _generate_deep_dive_body(*, client: LLMClient, pack: EvidencePack, title: str, slug: str) -> str:
    def _fallback() -> str:
        return _call_writer(client, _deep_system(), _deep_user(pack, title), task="draft", reject_completion=_markdown_rejection)

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
        text = _call_writer(
            client,
            _section_system(post_type),
            user,
            task="draft_section",
            reject_completion=_markdown_rejection,
        )
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


def _safe_daily_title(raw: str) -> str:
    title = re.sub(r"\s+", " ", (raw or "").strip())
    if not title:
        return "Research Radar: ML systems and methods update"
    if _looks_like_truncated_fallback_title(title):
        return "Research Radar: ML systems and methods update"
    if title.lower().startswith("research radar:") and len(title) > 68:
        return "Research Radar: ML systems and methods update"
    return title


def _merge_daily_body(
    *,
    title: str,
    outline: DailyOutline,
    section_drafts: list[str],
    edited: str,
) -> str:
    safe_title = _safe_daily_title(title)
    edited = (edited or "").strip()
    if edited and not _missing_outline_sections(edited, outline):
        if not _h1_titles(edited):
            return f"# {safe_title}\n\n{edited}"
        return edited
    if section_drafts:
        LOG.warning("editor dropped outline headings; stitching section drafts")
        return _stitch_section_drafts(safe_title, section_drafts)
    return edited


def _missing_outline_sections(body: str, outline: DailyOutline) -> list[str]:
    body_headings = {_normalize_heading(heading) for heading in HEADING_RE.findall(body or "")}
    missing: list[str] = []
    for section in outline.sections:
        expected = _normalize_heading(section.heading)
        if expected and expected not in body_headings:
            missing.append(section.heading)
    return missing


def _stitch_section_drafts(title: str, section_drafts: list[str]) -> str:
    cleaned = [draft.strip() for draft in section_drafts if draft.strip()]
    return f"# {title}\n\n" + "\n\n".join(cleaned)


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
    return _call_writer(
        client,
        _final_editor_system(post_type),
        user,
        task="editor",
        reject_completion=_markdown_rejection,
    )


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
        "Make the final article read like a Principal MLE problem-solving memo for AEC foundation models: name the blocker, the root-cause hypothesis, "
        "the training/data/evaluation/deployment experiment, and the decision gate. "
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


def _word_count(body: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", body or ""))


def _sleep_for_rate_limit(client: LLMClient) -> None:
    if not isinstance(client, LLMClient):
        return
    cooldown = config.llm_rate_limit_cooldown_seconds()
    if cooldown <= 0:
        return
    if getattr(client, "_rate_limit_hits", 0) <= 0:
        cooldown = min(cooldown, 10.0)
    remaining = client._remaining_runtime_seconds()
    if remaining <= cooldown + 5.0:
        return
    time.sleep(cooldown)


def _call_writer(
    client: LLMClient,
    system: str,
    user: str,
    *,
    task: str | None = None,
    reject_completion: Callable[[str], str | None] | None = None,
) -> str:
    if isinstance(client, LLMClient):
        return _strip_fence(
            client.complete(system=system, user=user, task=task, reject_completion=reject_completion)
        ).strip()
    return _strip_fence(client.complete(system=system, user=user)).strip()


def _call_daily_draft(client: LLMClient, system: str, user: str, *, outline: DailyOutline) -> str:
    return _call_writer(
        client,
        system,
        user,
        task="draft",
        reject_completion=lambda text: _daily_draft_rejection_reason(text, outline),
    )


def _daily_draft_rejection_reason(text: str, outline: DailyOutline) -> str | None:
    body = _strip_fence(text).strip()
    word_count = _word_count(body)
    min_fragment_words = max(500, config.daily_min_words() // 3)
    if word_count < min_fragment_words:
        return f"daily_fragment:{word_count}/{min_fragment_words}"
    missing_sections = _missing_outline_sections(body, outline)
    if missing_sections and len(missing_sections) == len(outline.sections):
        return "missing_all_outline_sections"
    if missing_sections and not _h1_titles(body):
        return "missing_h1_and_outline_sections"
    return None


def _review_if_ready(
    client: LLMClient,
    *,
    body: str,
    pack: EvidencePack,
    outline: DailyOutline | None,
    selection: SelectionResult | None,
    errors: list[str],
) -> tuple[dict[str, object], list[str]]:
    if pack.kind == "daily" and _needs_full_daily_rewrite(errors):
        return {}, errors
    if pack.kind == "daily" and isinstance(client, LLMClient) and getattr(client, "_blogpipe_emergency_daily_draft", False):
        reviewed_errors = list(errors)
        reviewed_errors.extend(_deterministic_quality_errors(body, pack))
        return {}, reviewed_errors
    quality_review = _llm_quality_review(client, body=body, pack=pack, outline=outline, selection=selection)
    reviewed_errors = list(errors)
    reviewed_errors.extend(_llm_quality_errors(quality_review, client=client))
    if pack.kind == "daily":
        reviewed_errors.extend(_deterministic_quality_errors(body, pack))
    return quality_review, reviewed_errors


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
    body, errors = _sanitize_then_validate(body, pack, outline=outline)
    quality_review, errors = _review_if_ready(
        client,
        body=body,
        pack=pack,
        outline=outline,
        selection=selection,
        errors=errors,
    )
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
                body = _call_daily_draft(client, _daily_rewrite_system(), rewrite_user, outline=outline)
                body = _ensure_visual_blocks(body, pack, slug)
                body, errors = _sanitize_then_validate(body, pack, outline=outline)
                quality_review, errors = _review_if_ready(
                    client,
                    body=body,
                    pack=pack,
                    outline=outline,
                    selection=selection,
                    errors=errors,
                )
            except Exception as exc:
                errors.append(f"full_rewrite_failed:{exc}")
        if errors:
            repair_input_errors = list(errors)
            repair_user = _repair_user(pack, body, errors, outline=outline, selection=selection, quality_review=quality_review)
            try:
                body = _call_writer(
                    client,
                    _repair_system(),
                    repair_user,
                    task="repair",
                    reject_completion=_markdown_rejection,
                )
                body = _ensure_visual_blocks(body, pack, slug)
                body, errors = _sanitize_then_validate(body, pack, outline=outline)
                quality_review, errors = _review_if_ready(
                    client,
                    body=body,
                    pack=pack,
                    outline=outline,
                    selection=selection,
                    errors=errors,
                )
                if errors and pack.kind == "daily" and outline is not None and selection is not None:
                    LOG.warning("daily repair output still invalid; using deterministic emergency draft: %s", errors)
                    setattr(client, "_blogpipe_emergency_daily_draft", True)
                    body = _emergency_daily_draft(pack=pack, outline=outline, selection=selection, title=title)
                    body = _ensure_visual_blocks(body, pack, slug)
                    body, errors = _sanitize_then_validate(body, pack, outline=outline)
                    quality_review = {}
                    errors.extend(_deterministic_quality_errors(body, pack))
            except Exception as exc:
                if pack.kind == "daily" and outline is not None and selection is not None:
                    LOG.warning("daily repair LLM failed; using deterministic emergency draft: %s", exc)
                    setattr(client, "_blogpipe_emergency_daily_draft", True)
                    body = _emergency_daily_draft(pack=pack, outline=outline, selection=selection, title=title)
                    body = _ensure_visual_blocks(body, pack, slug)
                    body, errors = _sanitize_then_validate(body, pack, outline=outline)
                    quality_review = {}
                    if pack.kind == "daily":
                        errors.extend(_deterministic_quality_errors(body, pack))
                    if errors:
                        errors.append(f"repair_failed:{exc}")
                        errors.extend(error for error in repair_input_errors if error not in errors)
                else:
                    errors.append(f"repair_failed:{exc}")
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
        "Write from the practical point of view of a Principal Machine Learning Engineer in AEC at Autodesk, focused on building foundation models "
        "for drawings, documents, BIM/CAD context, and construction workflows. "
        "Do not claim employment at any company and do not use first-person product-roadmap language. "
        "The post is problem-centered: identify a specific AEC foundation-model problem, explain why it blocks scale or quality, "
        "then use the evidence to reason through mechanisms, math/objectives when supported, experiments, limits, and next engineering actions. "
        "Keep the research depth, but make the article read like a principal-engineer decision memo: benchmark design, failure modes, "
        "integration constraints, deployment dependencies, latency/cost tradeoffs, validation plans, and adoption tests should drive the practical judgment. "
        "When the evidence concerns scaled LLM training, teach the operational how-to: sharding/parallelism choices, memory and communication bottlenecks, "
        "checkpoint/restart behavior, data-pipeline pressure, profiling signals, and what a principal engineer would test before rollout. "
        "Prefer deep synthesis over breadth: focus on one problem and use 3-4 primary evidence items plus supporting items only when they sharpen the decision. "
        "Use the evidence pack only. The prose must be original and evidence-grounded. "
        "Distinguish direct implication, plausible transfer, and open hypothesis when discussing production use. "
        "Every substantive item paragraph must include one or more evidence markers like [E1] and a source link. "
        "Do not publish a single-source post: cite at least three distinct primary papers when available. "
        f"Do not invent numbers, benchmarks, authors, or claims. The daily post must be at least {config.daily_min_words()} words."
    )


def _daily_user(pack: EvidencePack, outline: DailyOutline, selection: SelectionResult, title: str) -> str:
    return (
        f"TITLE: {title}\n\n"
        "Write a problem-first technical blog post using the OUTLINE headings exactly as provided. "
        f"The first line must be exactly '# {title}'. Each OUTLINE section heading must appear exactly once as '## <heading>'. "
        "Do not use fixed template headings such as 'Paper mechanisms', 'Math or objective details', or 'Why it matters' unless the outline uses them. "
        "Do not organize the article as one section per paper unless the outline explicitly requires it. "
        "Start from a specific problem an AEC foundation-model team needs to solve: scaling a training run, removing a data bottleneck, "
        "debugging drawing/document grounding, validating retrieval, controlling serving cost, or deciding whether a method is ready to prototype. "
        "Cover 3-4 primary evidence items deeply and mention supporting items briefly only when they strengthen a comparison. "
        "Cite at least three distinct primary papers when they exist in the evidence pack, but make the argument about the engineering problem rather than the papers. "
        "For each evidence item you discuss, answer what problem it helps diagnose, what mechanism or objective it uses, what evidence supports it, "
        "what limitation or caveat is visible, and what an engineer should test next. Include source URLs inline for every cited evidence ID. "
        "Include at least one cross-evidence comparison or tradeoff and at least one concrete production, training, data, or adoption test. "
        "Every major section should connect the paper to at least one operational lens where evidence permits: benchmark design, failure mode, "
        "deployment constraint, latency/cost tradeoff, integration dependency, or prototype/validation test. "
        "If the selected cluster is about large-scale LLM training, include a how-to explanation of at least one concrete training decision: "
        "FSDP/ZeRO, tensor/pipeline/sequence parallelism, activation checkpointing, microbatching, optimizer state, checkpoint recovery, "
        "NCCL/all-reduce communication, data loading, or GPU-utilization profiling when supported by evidence. "
        "The adoption section should read like an engineering decision memo, not a broad relevance section. "
        "Make production and deployment implications concrete where supported by the evidence; otherwise label them as plausible transfer or open hypothesis. "
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
        "of a Principal MLE in AEC at Autodesk building foundation models for drawings, documents, BIM/CAD context, and construction workflows. "
        f"The daily post must be at least {config.daily_min_words()} words. Cite at least three primary papers "
        "when they exist in the evidence pack, but center the article on a concrete AEC foundation-model problem to solve. "
        "Remove generic corporate prose and add concrete engineering judgment. "
        "For scaled LLM training evidence, include sharding/parallelism, memory/communication bottlenecks, checkpointing, profiling, "
        "data-pipeline throughput, and rollout validation where supported."
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
        "- Reframe paper-centered prose into a principal-MLE problem memo: name the AEC foundation-model blocker, likely root cause, experiment to run, and decision gate.\n"
        "- Strengthen operational judgment with concrete benchmarks, failure modes, deployment constraints, integration dependencies, latency/cost tradeoffs, or validation tests when the evidence supports them.\n"
        "- For scaled LLM training evidence, teach the training-system decision path: parallelism or sharding choice, memory/communication bottleneck, checkpoint/restart behavior, data pipeline, profiling signal, and rollout gate where supported.\n"
        "- If the body focus no longer matches the current headline, revise the H1 so the final title matches the actual article.\n"
        "- Downgrade phrases such as 'defines the reliability envelope', 'beyond prototype stage', and 'non-negotiable' unless the evidence clearly warrants them.\n"
        "- Remove late paper insertions that are not justified by the selected primary/supporting scope.\n"
        "- Do not output JSON, explanations, or a validation report.\n\n"
        f"VALIDATOR_ERRORS:\n{json.dumps(_repair_safe_errors(errors), indent=2)}\n\n"
        f"{_quality_floor_guidance(errors)}"
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


def _daily_rewrite_system() -> str:
    return (
        _daily_system()
        + " The previous draft was rejected as a fragment or structurally invalid. Rewrite the full article from scratch as a problem-focused AEC foundation-model engineering memo. "
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
        "- Cover the concrete AEC foundation-model problem, mechanism/objective, experiments or benchmarks, limits, cross-evidence tradeoffs, and production adoption tests.\n"
        "- Sound like a Principal MLE diagnosing and solving a scaling, data, evaluation, grounding, or deployment problem, not like an abstract paper recap.\n"
        "- For scaled LLM training evidence, include training-how-to details: sharding/parallelism, activation checkpointing or optimizer state, communication bottlenecks, data pipeline throughput, checkpoint recovery, profiling, and rollout validation where supported.\n"
        "- Treat production transfer as direct implication only when the evidence supports it; otherwise label it as plausible transfer or open hypothesis.\n"
        "- Do not preserve the rejected fragment unless a sentence is fully evidence-grounded and still fits the outline.\n\n"
        f"VALIDATOR_ERRORS:\n{json.dumps(_repair_safe_errors(errors), indent=2)}\n\n"
        f"{_quality_floor_guidance(errors)}"
        f"LLM_QUALITY_REVIEW:\n{json.dumps(quality_review or {}, indent=2, ensure_ascii=False)}\n\n"
        f"SELECTION:\n{selection.model_dump_json(indent=2) if selection is not None else '{}'}\n\n"
        f"OUTLINE:\n{outline.model_dump_json(indent=2)}\n\n"
        f"RELEVANT_EVIDENCE_CARDS:\n{json.dumps([card.model_dump(mode='json') for card in pack.evidence_cards], indent=2, ensure_ascii=False)}\n\n"
        f"CITED_SOURCE_URLS:\n{json.dumps(cited_source_requirements, indent=2, ensure_ascii=False)}\n\n"
        f"EVIDENCE_PACK:\n{pack.as_prompt_json()}\n\n"
        f"REJECTED_DRAFT_FOR_DIAGNOSIS_ONLY:\n{_repair_safe_draft(body, errors)}"
    )


def _quality_floor_guidance(errors: list[str]) -> str:
    bullets: list[str] = []

    def has(*prefixes: str) -> bool:
        return any(error.startswith(prefix) for error in errors for prefix in prefixes)

    if has("signal_low_score:technical_specificity", "llm_low_signal:technical_specificity"):
        bullets.append(
            "Raise technical_specificity inside evidence-cited paragraphs: add supported mechanism, objective, "
            "architecture, benchmark, ablation, metric, limitation, or failure-mode detail. Headings and uncited cue words do not count."
        )
    if has("signal_low_score:engineering_judgment", "llm_low_signal:engineering_judgment"):
        bullets.append(
            "Raise engineering_judgment inside evidence-cited paragraphs: state the principal-engineer decision, "
            "validation test, rollout/release gate, production blocker, monitoring need, or operational tradeoff supported by the evidence."
        )
    if has("signal_low_score:synthesis", "llm_low_signal:synthesis"):
        bullets.append(
            "Raise synthesis inside evidence-cited paragraphs: compare at least two primary papers, contrast their mechanisms or limits, "
            "and name the shared tradeoff without claiming an integrated stack unless the evidence supports it."
        )
    if has("signal_low_score:problem_solving", "llm_low_signal:problem_solving"):
        bullets.append(
            "Raise problem_solving inside evidence-cited paragraphs: name the AEC foundation-model blocker, root-cause hypothesis, "
            "diagnostic experiment, prototype or benchmark to run, and the decision or release gate that would change engineering action."
        )
    if has("signal_low_score:primary_depth", "llm_low_signal:primary_depth"):
        bullets.append(
            "Raise primary_depth by citing mechanism evidence and at least one experiment, objective, or limitation for each selected primary paper."
        )
    if has(
        "signal_low_score:training_howto",
        "llm_low_signal:training_howto",
        "signal_low_score:training_technology_depth",
        "llm_low_signal:training_technology_depth",
        "signal_low_score:training_runbook_actionability",
        "llm_low_signal:training_runbook_actionability",
        "missing_training_howto:",
        "missing_training_technology_detail:",
        "missing_training_runbook_action:",
        "llm_quality:TRAINING_HOWTO_INSUFFICIENT",
    ):
        bullets.append(
            "For scaled LLM training, add evidence-cited/source-linked runbook prose covering training stack "
            "(for example FSDP/ZeRO, DeepSpeed, Megatron, tensor/pipeline/sequence parallelism), scaling bottleneck "
            "(memory, activation checkpointing, optimizer state, NCCL/all-reduce, data pipeline, checkpoint recovery, GPU utilization), principal action "
            "(choose the stack, profile and measure the bottleneck, benchmark alternatives, validate failure modes, choose rollout gate, or recovery test), "
            "and the specific technology detail missing from the validator (parallelism/sharding, memory/state, communication, data/recovery, "
            "observability/gates, or runbook actionability)."
        )
    if not bullets:
        return ""
    return "QUALITY_FLOOR_GUIDANCE:\n" + "\n".join(f"- {bullet}" for bullet in bullets) + "\n\n"


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
    body = _sanitize_voice(body)
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


def _markdown_rejection(text: str) -> str | None:
    body = _strip_fence(text).strip()
    if not body:
        return "empty_markdown"
    if _word_count(body) < 80:
        return f"markdown_too_short:{_word_count(body)}"
    return None


def _quality_review_rejection(text: str) -> str | None:
    try:
        payload = jsonish.loads_object(text)
    except Exception:
        return "quality_review_unparseable"
    if not isinstance(payload, dict):
        return "quality_review_not_object"
    return None


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
            reject_completion=_quality_review_rejection,
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
        '"problem_solving": 0.0, "primary_depth": 0.0, "evidence_discipline": 0.0, "section_nonredundancy": 0.0, "experiment_detail": 0.0, '
        '"training_howto": 0.0, "training_technology_depth": 0.0, "training_runbook_actionability": 0.0},\n'
        '  "errors": ["machine_readable_error_code"],\n'
        '  "examples": ["quoted short failing text"],\n'
        '  "notes": "short explanation",\n'
        '  "top_editorial_failure": "one sentence on the most important problem"\n'
        "}\n\n"
        "Blocking criteria:\n"
        "- generic corporate prose, hype, or paper-by-paper abstract summaries without insight\n"
        "- no concrete AEC foundation-model problem to solve, such as a training bottleneck, data quality issue, drawing/document grounding failure, evaluation gap, retrieval defect, serving cost problem, or deployment blocker\n"
        "- weak Principal MLE voice: does not identify root-cause hypotheses, tradeoffs, experiments to run, or decision gates\n"
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
        "- when TRAINING_SYSTEM_FOCUS is true, missing scaled-training how-to detail: no concrete sharding/parallelism choice, memory or communication bottleneck, checkpoint/restart behavior, data-pipeline pressure, profiling signal, or rollout validation gate\n"
        "- when TRAINING_SYSTEM_FOCUS is true, shallow training-technology depth: missing named parallelism/sharding, memory/state management, communication, data/recovery, observability, or release-gate details\n"
        "- when TRAINING_SYSTEM_FOCUS is true, shallow runbook actionability: names technologies but does not say what to choose, measure, compare, diagnose, validate, or gate before scaling\n"
        "- speculative Autodesk/AEC adoption prose that outruns paper-supported claims or transfer hypotheses\n"
        "- claims that fuse multiple papers into a proven stack/system without direct support in the evidence cards\n"
        "- synthesis claims whose strength exceeds the evidence, including phrases such as 'defines the reliability envelope', 'beyond prototype stage', or 'non-negotiable'\n"
        "- title/body mismatch: the H1 centers one paper or axis, but the body mainly argues a different throughline\n\n"
        f"TRAINING_SYSTEM_FOCUS: {json.dumps(_pack_has_training_system_focus(pack))}\n\n"
        f"OUTLINE:\n{outline.model_dump_json(indent=2) if outline is not None else '{}'}\n\n"
        f"SELECTION:\n{selection.model_dump_json(indent=2) if selection is not None else '{}'}\n\n"
        f"EVIDENCE_PACK:\n{pack.as_prompt_json()}\n\n"
        f"DRAFT:\n{body}\n"
    )


def _sanitize_voice(body: str) -> str:
    text = body or ""
    replacements = (
        (r"\bAs a Principal (?:Machine Learning Engineer|MLE) at Autodesk\b", "From a principal MLE viewpoint"),
        (r"\bAt Autodesk,\s+(?:my|our)\b", "In production ML systems,"),
        (r"\bour (?:vision|strategic direction|roadmap|pipeline|pipelines|product|products)\b", "the engineering plan"),
        (r"\bwe (?:face|use|need|build|ship|own|have|could|should)\b", "teams"),
    )
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.I)
    return text


def _llm_quality_errors(review: dict[str, object], *, client: LLMClient | None = None) -> list[str]:
    if not review:
        return []
    if isinstance(client, LLMClient) and _quality_review_from_untrusted_model(client):
        if review.get("pass") is not False:
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


def _deterministic_quality_errors(body: str, pack: EvidencePack) -> list[str]:
    rubric = _signal_rubric(body, pack)
    scores = rubric.get("scores")
    if not isinstance(scores, dict):
        return []
    errors: list[str] = []
    threshold = config.min_signal_score()
    for name, value in scores.items():
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        if score < threshold:
            errors.append(f"signal_low_score:{name}:{score:.2f}/{threshold:.2f}")
    return errors


def _quality_review_from_untrusted_model(client: LLMClient) -> bool:
    model = (client.usage.model_by_task.get("quality_review") or client.usage.model or "").lower()
    if not model:
        return False
    if "gemini" in model or "gpt-4" in model:
        return False
    return _is_openrouter_model(model) or "llama" in model or "free" in model


def _is_openrouter_model(model: str) -> bool:
    normalized = (model or "").strip().lower()
    return normalized.startswith("openrouter/") or "/" in normalized


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
        (
            "llm-training",
            (
                "distributed training", "fsdp", "deepspeed", "megatron",
                "tensor parallel", "pipeline parallel", "activation checkpoint",
                "gradient accumulation",
            ),
        ),
        ("multimodal", ("multimodal", "vision-language", "visual grounding")),
        ("cad", ("cad", "cadquery", "programmatic cad", "scan-to-bim")),
        ("bim", ("bim", "ifc", "revit")),
        ("llm", ("llm", "language model", "agent", "rag", "transformer")),
        (
            "mle",
            (
                "serving", "evaluation", "monitoring", "pipeline", "latency", "throughput", "deployment",
                "distributed training", "fsdp", "deepspeed", "megatron", "tensor parallel",
                "pipeline parallel", "activation checkpoint", "gpu utilization",
            ),
        ),
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
    if _looks_like_generic_roundup(headings):
        errors.append("generic_roundup_structure")
    lower = (body or "").lower()
    errors.extend(_validate_daily_concepts(lower))
    if _pack_has_training_system_focus(pack):
        errors.extend(_validate_training_howto_concepts(body))
        errors.extend(_validate_training_technology_details(body))
        errors.extend(_validate_training_runbook_actions(body))
    return errors


def _pack_has_training_system_focus(pack: EvidencePack) -> bool:
    parts = [
        f"{ranked.item.title} {ranked.item.abstract_or_excerpt} {' '.join(ranked.item.tags)}"
        for ranked in pack.ranked_items
    ]
    parts.extend(f"{card.title} {card.mechanism} {card.experiment} {card.impact}" for card in pack.evidence_cards)
    return has_training_system_focus(" ".join(parts))


def _validate_training_howto_concepts(body: str) -> list[str]:
    lower_body = _training_howto_evidence_text(body)
    errors: list[str] = []
    for concept, cues in TRAINING_HOWTO_CONCEPTS.items():
        if not any(cue in lower_body for cue in cues):
            errors.append(f"missing_training_howto:{concept}")
    return errors


def _validate_training_technology_details(body: str) -> list[str]:
    lower_body = _training_howto_evidence_text(body)
    errors: list[str] = []
    for concept, cues in TRAINING_TECH_DETAIL_CONCEPTS.items():
        if not any(cue in lower_body for cue in cues):
            errors.append(f"missing_training_technology_detail:{concept}")
    return errors


def _validate_training_runbook_actions(body: str) -> list[str]:
    lower_body = _training_howto_evidence_text(body)
    errors: list[str] = []
    for concept, cues in TRAINING_RUNBOOK_ACTION_CONCEPTS.items():
        if not any(cue in lower_body for cue in cues):
            errors.append(f"missing_training_runbook_action:{concept}")
    return errors


def _training_howto_evidence_text(body: str) -> str:
    return _evidence_linked_prose_text(body)


def _evidence_linked_prose_text(body: str) -> str:
    cited_paragraphs: list[str] = []
    for paragraph in re.split(r"\n\s*\n", body or ""):
        if not EVIDENCE_REF_RE.search(paragraph) or not URL_RE.search(paragraph):
            continue
        paragraph = re.sub(r"https?://\S+", " ", paragraph)
        paragraph = re.sub(r"^#+\s+.*$", "", paragraph, flags=re.M)
        cited_paragraphs.append(paragraph.lower())
    return "\n\n".join(cited_paragraphs)


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
    evidence_text = _evidence_linked_prose_text(body)
    words = max(1, len(re.findall(r"\b[\w'-]+\b", lower)))
    technical = _cue_score(evidence_text, TECHNICAL_SPECIFICITY_CUES, target=8)
    judgment = _cue_score(evidence_text, ENGINEERING_JUDGMENT_CUES, target=5)
    synthesis = _cue_score(evidence_text, SYNTHESIS_CUES, target=4)
    problem_solving = _cue_score(evidence_text, PROBLEM_SOLVING_CUES, target=6)
    primary_depth = _primary_depth_score(body, pack)
    generic_hits = sum(lower.count(phrase) for phrase in GENERIC_PHRASES)
    generic_density = generic_hits / words
    noise_control = 1.0 if generic_density <= config.generic_phrase_max_density() else 0.0
    scores: dict[str, float] = {
        "technical_specificity": technical,
        "engineering_judgment": judgment,
        "synthesis": synthesis,
        "problem_solving": problem_solving,
        "noise_control": noise_control,
        "primary_depth": primary_depth,
    }
    if _pack_has_training_system_focus(pack):
        scores["training_howto"] = _training_howto_score(body)
        scores["training_technology_depth"] = _training_technology_depth_score(body)
        scores["training_runbook_actionability"] = _training_runbook_actionability_score(body)
    return {
        "scores": scores,
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


def _training_howto_score(body: str) -> float:
    lower_body = _training_howto_evidence_text(body)
    if not lower_body:
        return 0.0
    hits = sum(1 for cues in TRAINING_HOWTO_CONCEPTS.values() if any(cue in lower_body for cue in cues))
    return hits / max(1, len(TRAINING_HOWTO_CONCEPTS))


def _training_technology_depth_score(body: str) -> float:
    lower_body = _training_howto_evidence_text(body)
    if not lower_body:
        return 0.0
    hits = sum(1 for cues in TRAINING_TECH_DETAIL_CONCEPTS.values() if any(cue in lower_body for cue in cues))
    return hits / max(1, len(TRAINING_TECH_DETAIL_CONCEPTS))


def _training_runbook_actionability_score(body: str) -> float:
    lower_body = _training_howto_evidence_text(body)
    if not lower_body:
        return 0.0
    hits = sum(1 for cues in TRAINING_RUNBOOK_ACTION_CONCEPTS.values() if any(cue in lower_body for cue in cues))
    return hits / max(1, len(TRAINING_RUNBOOK_ACTION_CONCEPTS))


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
