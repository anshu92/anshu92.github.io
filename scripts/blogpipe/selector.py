from __future__ import annotations

import json
import logging
import re

from pydantic import ValidationError

from . import config
from . import jsonish
from .llm import LLMClient
from .models import RankedItem, SelectionResult


class SelectionError(RuntimeError):
    pass


LOG = logging.getLogger(__name__)


def select_daily_items(ranked: list[RankedItem], *, llm: LLMClient) -> tuple[list[RankedItem], SelectionResult]:
    # Let the selector reason over all available paper titles for the run.
    # This intentionally avoids pre-filtering by deterministic score buckets.
    papers = [r for r in ranked if r.item.source_kind == "paper"]
    candidates = papers or list(ranked)
    if not candidates:
        raise SelectionError("selector_no_candidates")
    raw = _call_selector(llm, candidates)
    try:
        result = _parse_selection(raw)
    except SelectionError as exc:
        recovered = _salvage_selection(raw, candidates)
        if recovered is None:
            raise
        LOG.warning("selector parse failed (%s); using salvaged selected_item_ids", exc)
        result = recovered
    if not result.selected_item_ids:
        raise SelectionError("selector_empty")
    selected = _apply_selection(candidates, result)
    if not selected:
        raise SelectionError("selector_no_valid_items")
    return selected, result


def _call_selector(llm: LLMClient, candidates: list[RankedItem]) -> str:
    max_tokens = config.selector_max_tokens()
    reject = _selector_completion_rejector(candidates)
    if isinstance(llm, LLMClient):
        return llm.complete(
            system=_selector_system(),
            user=_selector_user(candidates),
            max_tokens=max_tokens,
            task="selector",
            reject_completion=reject,
        )
    return llm.complete(system=_selector_system(), user=_selector_user(candidates), max_tokens=max_tokens)


def _selector_completion_rejector(candidates: list[RankedItem]):
    def _reject(text: str) -> str | None:
        try:
            result = _parse_selection(text)
        except SelectionError:
            if _salvage_selection(text, candidates) is None:
                return "selector_unparseable"
            return None
        if not result.selected_item_ids:
            return "selector_empty"
        return None

    return _reject


def _selector_system() -> str:
    return (
        "You are the Research Radar selector. Return JSON only. "
        "You select papers for a Principal Machine Learning Engineer who reads research for technical judgment, "
        "not for domain-only filtering. Prefer one coherent thesis cluster over broad topic diversity. "
        "Apply this strict priority order when choosing the daily cluster:\n"
        "1. ML engineering, with scaled LLM training how-to first: FSDP/ZeRO, DeepSpeed, Megatron, tensor/pipeline/sequence parallelism, "
        "activation checkpointing, optimizer state, NCCL/all-reduce communication, data pipelines, checkpointing, profiling, GPU utilization, "
        "serving optimizations, GPU/CUDA kernels, JAX/PyTorch/HuggingFace, observability, and substantive technical engineering blogs.\n"
        "2. ML applied research: strong methods with clear benchmarks, ablations, and deployment or evaluation lessons.\n"
        "3. ML theory: only when unusually interesting, surprising, or clearly relevant to systems or training practice.\n"
        "4. AEC topics: BIM/CAD/construction/document workflows when they are the best fit after 1-3.\n"
        "5. General popular ML stories: only when tier-1, technically substantive, and not a shallow roundup.\n"
        "Do not default to AEC just because the reader works at Autodesk. Use AEC as one transfer lens, not the primary filter."
    )


def _selector_user(candidates: list[RankedItem]) -> str:
    payload = []
    for r in candidates:
        item = r.item
        payload.append(
            {
                "item_id": item.item_id,
                "title": item.title,
                "source_kind": item.source_kind,
                "source_name": item.source_name,
                "published_at": item.published_at.isoformat() if item.published_at else "",
                "abstract": item.abstract_or_excerpt[:700],
            }
        )
    primary = config.daily_primary_papers()
    supporting = config.daily_supporting_items()
    return (
        f"Select {primary} primary papers and up to {supporting} supporting mentions for today's post. "
        "Primary papers must form a coherent technical thesis cluster, not a loose roundup. "
        "Follow the priority order: ML engineering > applied research > interesting theory > AEC > popular ML. "
        "Score each selected item from 0.0 to 1.0 on engineering value, applied-research value, theory interest, "
        "AEC transfer value, experiment strength, engineering actionability, and novelty relative to prior radar posts. "
        "Also score training_howto_value: whether the item can teach a principal engineer a concrete scaled-training decision, "
        "such as sharding/parallelism, activation checkpointing, communication, optimizer state, data loading, profiling, or checkpoint recovery. "
        "Prioritize items with mechanisms, objectives, evaluation design, deployment tradeoffs, training runbook lessons, or systems lessons. "
        "Return JSON in this exact shape:\n"
        "{\n"
        '  "selected_item_ids": ["item-id"],\n'
        '  "items": [{"item_id": "item-id", "role": "primary|supporting", '
        '"relevance_label": "ml_engineering|applied_research|ml_theory|aec|popular_ml|supporting_blog", '
        '"scores": {"engineering_value": 0.0, "applied_research_value": 0.0, "theory_interest": 0.0, '
        '"aec_transfer_value": 0.0, "experiment_strength": 0.0, '
        '"engineering_actionability": 0.0, "training_howto_value": 0.0, "novelty": 0.0}, '
        '"reason": "short reason", "suggested_tags": ["tag"]}],\n'
        '  "suggested_tags": ["tag"]\n'
        "}\n\n"
        f"CANDIDATES:\n{json.dumps(payload, indent=2, ensure_ascii=False)}"
    )


def _parse_selection(text: str) -> SelectionResult:
    try:
        return SelectionResult.model_validate(_normalize_selection_payload(jsonish.loads_object(text)))
    except (SyntaxError, ValidationError, ValueError, json.JSONDecodeError) as exc:
        raise SelectionError(f"selector_malformed:{exc}") from exc


def _json_payload(text: str) -> str:
    return jsonish.extract_object(text)


def _normalize_selection_payload(data: object) -> object:
    if not isinstance(data, dict):
        return data
    normalized = dict(data)
    cleaned_items = []
    for raw_item in data.get("items") or []:
        if not isinstance(raw_item, dict):
            continue
        item = dict(raw_item)
        scores = raw_item.get("scores")
        if isinstance(scores, dict):
            clean_scores: dict[str, float] = {}
            for key, value in scores.items():
                try:
                    clean_scores[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue
            item["scores"] = clean_scores
        cleaned_items.append(item)
    normalized["items"] = cleaned_items
    return normalized


def _salvage_selection(raw: str, candidates: list[RankedItem]) -> SelectionResult | None:
    text = (raw or "").strip()
    if not text:
        return None
    structured = _salvage_structured_selection(text, candidates)
    if structured is not None:
        return structured
    match = re.search(
        r'"selected_item_ids"\s*:\s*\[([\s\S]*?)(?:\]|"items"\s*:|"suggested_tags"\s*:|\}\s*$)',
        text,
        re.I,
    )
    if not match:
        return None
    candidate_ids = {r.item.item_id for r in candidates}
    found = re.findall(r'"([^"\n]{1,200})"', match.group(1))
    selected_ids: list[str] = []
    seen: set[str] = set()
    for item_id in found:
        normalized = item_id.strip()
        if not normalized or normalized in seen or normalized not in candidate_ids:
            continue
        selected_ids.append(normalized)
        seen.add(normalized)
    if not selected_ids:
        return None
    return SelectionResult(
        selected_item_ids=selected_ids,
        items=[{"item_id": item_id, "role": "primary"} for item_id in selected_ids],
        suggested_tags=[],
    )


def _salvage_structured_selection(raw: str, candidates: list[RankedItem]) -> SelectionResult | None:
    try:
        data = jsonish.loads_object(raw)
    except (SyntaxError, ValueError, json.JSONDecodeError):
        return None
    candidate_ids = {r.item.item_id for r in candidates}
    selected_ids = [
        item_id
        for item_id in _as_string_list(data.get("selected_item_ids"))
        if item_id in candidate_ids
    ]
    items = []
    for raw_item in data.get("items") or []:
        if not isinstance(raw_item, dict):
            continue
        item_id = str(raw_item.get("item_id") or "").strip()
        if not item_id or item_id not in candidate_ids:
            continue
        item = {
            "item_id": item_id,
            "role": str(raw_item.get("role") or "primary"),
            "relevance_label": str(raw_item.get("relevance_label") or ""),
            "reason": str(raw_item.get("reason") or ""),
            "suggested_tags": _as_string_list(raw_item.get("suggested_tags")),
        }
        scores = raw_item.get("scores")
        if isinstance(scores, dict):
            item["scores"] = {
                str(key): float(value)
                for key, value in scores.items()
                if isinstance(value, (int, float))
            }
        items.append(item)
    for item_id in selected_ids:
        if not any(item["item_id"] == item_id for item in items):
            items.append({"item_id": item_id, "role": "primary"})
    if not selected_ids:
        selected_ids = [item["item_id"] for item in items]
    if not selected_ids:
        return None
    return SelectionResult.model_validate(
        {
            "selected_item_ids": selected_ids,
            "items": items,
            "suggested_tags": _as_string_list(data.get("suggested_tags")),
        }
    )


def _as_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _apply_selection(candidates: list[RankedItem], selection: SelectionResult) -> list[RankedItem]:
    by_id = {r.item.item_id: r for r in candidates}
    reasons = {item.item_id: item for item in selection.items}
    primary: list[RankedItem] = []
    supporting: list[RankedItem] = []
    seen: set[str] = set()
    for item_id in selection.selected_item_ids:
        ranked_item = by_id.get(item_id)
        if not ranked_item or item_id in seen:
            continue
        _annotate(ranked_item, selection, reasons)
        if _selection_role(reasons.get(item_id)) == "supporting":
            supporting.append(ranked_item)
        else:
            supporting_count = len(supporting)
            if len(primary) < config.daily_primary_papers():
                primary.append(ranked_item)
            elif supporting_count < config.daily_supporting_items():
                ranked_item.item.extra["selector_role"] = "supporting"
                supporting.append(ranked_item)
        seen.add(item_id)
    if not primary and not supporting:
        return []
    if len(primary) < config.daily_primary_papers():
        selected_ids = {r.item.item_id for r in [*primary, *supporting]}
        for ranked_item in candidates:
            if ranked_item.item.item_id in selected_ids:
                continue
            ranked_item.item.extra["selector_role"] = "primary"
            primary.append(ranked_item)
            selected_ids.add(ranked_item.item.item_id)
            if len(primary) >= config.daily_primary_papers():
                break
    return [*primary[: config.daily_primary_papers()], *supporting[: config.daily_supporting_items()]]


def _annotate(ranked_item: RankedItem, selection: SelectionResult, reasons: dict[str, object]) -> None:
    reason = reasons.get(ranked_item.item.item_id)
    if reason is not None:
        ranked_item.item.extra["selector_role"] = _selection_role(reason)
        ranked_item.item.extra["selector_relevance"] = getattr(reason, "relevance_label", "")
        ranked_item.item.extra["selector_reason"] = getattr(reason, "reason", "")
        ranked_item.item.extra["selector_scores"] = getattr(reason, "scores", {})
        ranked_item.item.extra["selector_tags"] = getattr(reason, "suggested_tags", [])
    else:
        ranked_item.item.extra["selector_role"] = "primary"
    ranked_item.item.extra["selector_suggested_tags"] = selection.suggested_tags


def _selection_role(reason: object | None) -> str:
    role = str(getattr(reason, "role", "") or "").strip().lower()
    if role in {"primary", "supporting"}:
        return role
    label = str(getattr(reason, "relevance_label", "") or "").strip().lower()
    return "supporting" if label == "supporting_blog" else "primary"
