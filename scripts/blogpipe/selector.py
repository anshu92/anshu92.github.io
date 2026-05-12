from __future__ import annotations

import json
import logging
import re

from pydantic import ValidationError

from . import config
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
    if isinstance(llm, LLMClient):
        return llm.complete(
            system=_selector_system(),
            user=_selector_user(candidates),
            max_tokens=max_tokens,
            task="selector",
        )
    return llm.complete(system=_selector_system(), user=_selector_user(candidates), max_tokens=max_tokens)


def _selector_system() -> str:
    return (
        "You are the Research Radar selector. Return JSON only. "
        "You select papers for a Principal Machine Learning Engineer at Autodesk working on AEC foundation models, "
        "especially 2D documents: drawings, sheets, plans, PDFs, OCR, layout understanding, construction documents, "
        "CAD/BIM/IFC conversion, and scan-to-BIM adjacent workflows. "
        "Prefer one coherent thesis cluster over broad topic diversity. "
        "Use strong bias toward direct AEC/2D-document relevance, but include adjacent ML systems, evaluation, "
        "multimodal, and foundation-model papers when direct matches are sparse. "
        "Classify each pick as primary or supporting. Primary papers get deep treatment; supporting items are brief context."
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
        "Score each selected item from 0.0 to 1.0 on direct AEC/2D-document relevance, transferable mechanism, "
        "experiment strength, engineering actionability, and novelty relative to prior radar posts. "
        "Prioritize items with mechanisms, objectives, evaluation design, deployment tradeoffs, or document/CAD transfer value. "
        "Return JSON in this exact shape:\n"
        "{\n"
        '  "selected_item_ids": ["item-id"],\n'
        '  "items": [{"item_id": "item-id", "role": "primary|supporting", '
        '"relevance_label": "direct_aec_2d|aec_adjacent|ml_engineering|supporting_blog", '
        '"scores": {"aec_document_relevance": 0.0, "transferable_mechanism": 0.0, "experiment_strength": 0.0, '
        '"engineering_actionability": 0.0, "novelty": 0.0}, '
        '"reason": "short reason", "suggested_tags": ["tag"]}],\n'
        '  "suggested_tags": ["tag"]\n'
        "}\n\n"
        f"CANDIDATES:\n{json.dumps(payload, indent=2, ensure_ascii=False)}"
    )


def _parse_selection(text: str) -> SelectionResult:
    try:
        return SelectionResult.model_validate_json(_json_payload(text))
    except (ValidationError, ValueError, json.JSONDecodeError) as exc:
        raise SelectionError(f"selector_malformed:{exc}") from exc


def _json_payload(text: str) -> str:
    raw = (text or "").strip()
    if raw.startswith("```"):
        parts = raw.split("\n", 1)
        raw = parts[1].strip() if len(parts) == 2 else raw.lstrip("`").strip()
        if raw.endswith("```"):
            raw = raw[:-3].strip()
    fenced = re.match(r"^```(?:json)?\n([\s\S]*?)\n```$", raw, re.I)
    if fenced:
        raw = fenced.group(1).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        return raw[start : end + 1]
    return raw


def _salvage_selection(raw: str, candidates: list[RankedItem]) -> SelectionResult | None:
    text = (raw or "").strip()
    if not text:
        return None
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
