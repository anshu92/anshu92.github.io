from __future__ import annotations

import json
import re

from pydantic import ValidationError

from . import config
from .llm import LLMClient
from .models import RankedItem, SelectionResult


class SelectionError(RuntimeError):
    pass


def select_daily_items(ranked: list[RankedItem], *, llm: LLMClient) -> tuple[list[RankedItem], SelectionResult]:
    # Let the selector reason over all available paper titles for the run.
    # This intentionally avoids pre-filtering by deterministic score buckets.
    papers = [r for r in ranked if r.item.source_kind == "paper"]
    candidates = papers or list(ranked)
    if not candidates:
        raise SelectionError("selector_no_candidates")
    result = _parse_selection(_call_selector(llm, candidates))
    if not result.selected_item_ids:
        raise SelectionError("selector_empty")
    selected = _apply_selection(candidates, result)
    if not selected:
        raise SelectionError("selector_no_valid_items")
    return selected, result


def _call_selector(llm: LLMClient, candidates: list[RankedItem]) -> str:
    if isinstance(llm, LLMClient):
        return llm.complete(system=_selector_system(), user=_selector_user(candidates), max_tokens=2200, task="selector")
    return llm.complete(system=_selector_system(), user=_selector_user(candidates), max_tokens=2200)


def _selector_system() -> str:
    return (
        "You are the Research Radar selector. Return JSON only. "
        "You select papers for a Principal Machine Learning Engineer at Autodesk working on AEC foundation models, "
        "especially 2D documents: drawings, sheets, plans, PDFs, OCR, layout understanding, construction documents, "
        "CAD/BIM/IFC conversion, and scan-to-BIM adjacent workflows. "
        "Use strong bias toward direct AEC/2D-document relevance, but include adjacent ML systems, evaluation, "
        "multimodal, and foundation-model papers when direct matches are sparse. "
        "You will receive all available paper titles for this run."
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
            }
        )
    return (
        "Select 5-8 items for today's post. Prefer at least 4 papers when enough relevant papers exist and at most 2 source blogs. "
        "Prioritize relevance from titles first; avoid broad generic picks when more Autodesk/AEC-relevant titles exist. "
        "Return JSON in this exact shape:\n"
        "{\n"
        '  "selected_item_ids": ["item-id"],\n'
        '  "items": [{"item_id": "item-id", "relevance_label": "direct_aec_2d|aec_adjacent|ml_engineering|supporting_blog", '
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
    fenced = re.match(r"^```(?:json)?\n([\s\S]*?)\n```$", raw, re.I)
    if fenced:
        raw = fenced.group(1).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        return raw[start : end + 1]
    return raw


def _apply_selection(candidates: list[RankedItem], selection: SelectionResult) -> list[RankedItem]:
    by_id = {r.item.item_id: r for r in candidates}
    reasons = {item.item_id: item for item in selection.items}
    picked: list[RankedItem] = []
    seen: set[str] = set()
    for item_id in selection.selected_item_ids:
        ranked_item = by_id.get(item_id)
        if not ranked_item or item_id in seen:
            continue
        _annotate(ranked_item, selection, reasons)
        picked.append(ranked_item)
        seen.add(item_id)
    if not picked:
        return []
    out = list(picked)
    need = max(5, min(8, config.min_papers()))
    if len(out) < need:
        selected_ids = {r.item.item_id for r in out}
        for ranked_item in candidates:
            if ranked_item.item.item_id in selected_ids:
                continue
            out.append(ranked_item)
            if len(out) >= need:
                break
    return out[:8]


def _annotate(ranked_item: RankedItem, selection: SelectionResult, reasons: dict[str, object]) -> None:
    reason = reasons.get(ranked_item.item.item_id)
    if reason is not None:
        ranked_item.item.extra["selector_relevance"] = getattr(reason, "relevance_label", "")
        ranked_item.item.extra["selector_reason"] = getattr(reason, "reason", "")
        ranked_item.item.extra["selector_tags"] = getattr(reason, "suggested_tags", [])
    ranked_item.item.extra["selector_suggested_tags"] = selection.suggested_tags
