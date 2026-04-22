"""Rank candidates using EditorialBrief + memory."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path

from . import config, memory, openrouter_client
from .llm_chain import reset_llm_usage
from .models import EditorialBrief, Item, Pillar, RankResult
from .memory import _ROOT, load_json, save_json

LOG = logging.getLogger(__name__)


def _covered() -> set[str]:
    c = load_json("covered_papers.json", {})
    urls: set[str] = set()
    for k in (c or {}).get("urls", []):
        urls.add(k)
    for k in (c or {}).get("by_url", {}):
        urls.add(k)
    return urls


def _brief() -> EditorialBrief:
    p = _ROOT / "reports" / "editorial_brief.json"
    if p.is_file():
        return EditorialBrief.model_validate_json(p.read_text())
    return EditorialBrief()


def _load_harvested() -> list[Item]:
    p = _ROOT / "reports" / "harvested.json"
    if not p.is_file():
        return []
    data = json.loads(p.read_text())
    return [Item.model_validate(x) for x in data.get("items", [])]


def _filter_avoid(items: list[Item], avoid: list[str]) -> list[Item]:
    if not avoid:
        return items
    out: list[Item] = []
    for it in items:
        blob = f"{it.title} {it.abstract}".lower()
        if any(
            a.lower() in blob for a in avoid if len(a) > 5
        ):
            continue
        out.append(it)
    return out or items


def _heuristic_score(it: Item, brief: EditorialBrief) -> float:
    w = brief.pillar_weights or {}
    base = 1.0 * float(
        w.get(
            it.pillar.value
            if isinstance(it.pillar, Pillar)
            else str(it.pillar),
            0.2,
        )
    )
    if "llm" in it.title.lower() or "language model" in it.abstract.lower()[:200]:
        base += 0.1
    if it.source == "huggingface_daily_papers":
        base += 0.15
    return base


def _llm_rank(candidates: list[Item], brief: EditorialBrief) -> tuple[Item, str, list[Item]]:
    if len(candidates) == 1:
        return candidates[0], "single candidate", []
    if config.dry_run() or not config.llm_configured():
        return (
            candidates[0],
            "dry_run" if config.dry_run() else "no LLM key",
            candidates[1:6],
        )
    system = (
        "You are an editorial ranker. Pick ONE best item for a technical blog. "
        "Return JSON only: { \"pick_index\": 0, \"reasoning\": \"...\" }"
        f" Pillar weights: {brief.pillar_weights}. "
        "Principal ML Engineer at Autodesk: prefer reproducible, engineering, "
        "AEC when relevant, LLM when core."
    )
    lines: list[str] = []
    for i, c in enumerate(candidates[:20]):
        lines.append(
            f"{i}. [{c.pillar}] {c.title} | {c.source} | {c.url}\n  {c.abstract[:400]}"
        )
    user = "Candidates:\n" + "\n".join(lines)
    try:
        raw = openrouter_client.llm_text(system, user, max_tokens=1536)
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            raise ValueError("no json")
        j = json.loads(m.group(0))
        idx = int(j.get("pick_index", 0))
        reason = str(j.get("reasoning", ""))
    except Exception as e:
        LOG.warning("llm rank fallback: %s", e)
        return candidates[0], "heuristic", candidates[1:6]
    idx = max(0, min(idx, len(candidates) - 1))
    primary = candidates[idx]
    rest = [c for j, c in enumerate(candidates) if j != idx][:5]
    return primary, reason, rest


def run() -> RankResult:
    reset_llm_usage()
    items = _load_harvested()
    brief = _brief()
    covered = _covered()
    items = [
        it
        for it in items
        if it.url not in covered
        and not any(sha in it.id for sha in ("placeholder",))
    ]
    items = _filter_avoid(items, brief.avoid_topics)
    if not items:
        LOG.warning("rank: no items after filter; using proactive topic path")
        # Synthetic item from follow-up
        if brief.proactive_topic:
            items = [
                Item(
                    id="proactive_" + hashlib.sha1(brief.proactive_topic.encode()).hexdigest()[:12],
                    title=f"Engineering follow-up: {brief.proactive_topic[:80]}",
                    url="https://anshu92.github.io/",
                    abstract=brief.proactive_topic,
                    source="proactive",
                    tags=["follow-up", "systems"],
                    pillar=Pillar.systems,
                )
            ]
        else:
            items = [
                Item(
                    id="fallback_1",
                    title="Scalable LLM inference: memory and latency",
                    url="https://anshu92.github.io/",
                    abstract="A systems post when harvest is empty.",
                    source="fallback",
                    tags=["inference", "llm"],
                    pillar=Pillar.systems,
                )
            ]
    # Sort by heuristic then take top 20 for LLM
    items = sorted(
        items,
        key=lambda x: -_heuristic_score(x, brief),
    )[:20]
    primary, reason, rejected = _llm_rank(items, brief)
    rr = RankResult(
        primary=primary,
        score=1.0,
        rejected=rejected,
        reasoning=reason,
    )
    p = _ROOT / "reports" / "rank_result.json"
    p.write_text(rr.model_dump_json(indent=2), encoding="utf-8")
    LOG.info("rank: primary %s", primary.title[:60])
    return rr
