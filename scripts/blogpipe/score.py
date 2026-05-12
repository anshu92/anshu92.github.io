from __future__ import annotations

import math
import re
from datetime import datetime, timezone

from .models import RankedItem, SourceItem, TopicScores
from .topics import TRACKS, keyword_hits


def rank_items(
    items: list[SourceItem],
    *,
    now: datetime | None = None,
    limit: int = 50,
    max_age_hours: int | None = 72,
) -> list[RankedItem]:
    current = now or datetime.now(timezone.utc)
    recent = _recent_items(items, current, max_age_hours=max_age_hours)
    scored = [_score_one(item, current) for item in recent]
    gated = [r for r in scored if _passes_gate(r)]
    gated.sort(key=lambda r: r.daily_score, reverse=True)
    return _diversify(gated[:limit])


def daily_shortlist(ranked: list[RankedItem], *, minimum: int = 5, maximum: int = 8) -> list[RankedItem]:
    strong = [r for r in ranked if r.daily_score >= 0.68]
    if len(strong) < minimum:
        strong = [r for r in ranked if r.daily_score >= 0.50] or ranked
    return strong[:maximum]


def deep_dive_shortlist(ranked: list[RankedItem], *, maximum: int = 1) -> list[RankedItem]:
    return [r for r in ranked if r.deep_dive_score >= 0.78][:maximum]


def _score_one(item: SourceItem, now: datetime) -> RankedItem:
    blob = f"{item.title}\n{item.abstract_or_excerpt}\n{item.body_text}\n{' '.join(item.tags)}"
    scores = _topic_scores(blob)
    source_quality = min(1.0, max(0.0, 1.1 - (item.source_tier * 0.18)))
    freshness = _freshness(item, now)
    depth = _technical_depth(blob, item)
    novelty = _novelty(blob)
    practical = _practical_signal(blob, item)
    evidence = min(1.0, len(blob) / 1200.0)
    daily = (
        0.25 * scores.best
        + 0.20 * source_quality
        + 0.15 * freshness
        + 0.15 * depth
        + 0.10 * novelty
        + 0.10 * practical
        + 0.05 * evidence
    )
    deep = (
        0.20 * scores.best
        + 0.15 * source_quality
        + 0.05 * freshness
        + 0.30 * depth
        + 0.20 * novelty
        + 0.10 * max(practical, _reproducibility(blob, item))
    )
    quality = {
        "freshness": round(freshness, 3),
        "technical_depth": round(depth, 3),
        "novelty": round(novelty, 3),
        "practical_impact": round(practical, 3),
        "evidence_quality": round(evidence, 3),
        "source_quality": round(source_quality, 3),
    }
    citations = {"citation_count": int(item.extra.get("citation_count", 0) or 0)}
    reason = ", ".join(scores.matched_keywords[:8]) or "source quality/freshness"
    return RankedItem(
        item=item,
        topic_scores=scores,
        daily_score=round(daily, 4),
        deep_dive_score=round(deep, 4),
        quality_signals=quality,
        citation_signals=citations,
        rank_reason=reason,
    )


def _topic_scores(text: str) -> TopicScores:
    hits_by_track: dict[str, list[str]] = {}
    for track in TRACKS:
        hits = keyword_hits(text, track.keywords)
        if track.required_any and not keyword_hits(text, track.required_any):
            hits = []
        hits_by_track[track.name] = hits
    all_hits = sorted({kw for hits in hits_by_track.values() for kw in hits})

    def score(name: str) -> float:
        hits = hits_by_track[name]
        return min(1.0, len(hits) / 5.0)

    return TopicScores(
        llm=score("llm"),
        mle=score("mle"),
        aec=score("aec"),
        matched_keywords=all_hits,
    )


def _passes_gate(ranked: RankedItem) -> bool:
    if ranked.topic_scores.best < 0.20:
        return False
    if ranked.item.source_kind == "blog" and ranked.item.source_tier > 2:
        return False
    return True


def _freshness(item: SourceItem, now: datetime) -> float:
    published = item.published_at or item.updated_at
    if not published:
        return 0.45
    if published.tzinfo is None:
        published = published.replace(tzinfo=timezone.utc)
    age_days = max(0.0, (now - published).total_seconds() / 86400.0)
    return math.exp(-age_days / 5.0)


def _recent_items(
    items: list[SourceItem],
    now: datetime,
    *,
    max_age_hours: int | None,
) -> list[SourceItem]:
    if max_age_hours is None:
        return items
    out: list[SourceItem] = []
    for item in items:
        published = item.published_at or item.updated_at
        if not published:
            continue
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        age_hours = (now - published).total_seconds() / 3600.0
        if 0 <= age_hours <= float(max_age_hours):
            out.append(item)
    return out


def _technical_depth(text: str, item: SourceItem) -> float:
    cues = (
        "ablation", "benchmark", "dataset", "architecture", "objective", "loss",
        "throughput", "latency", "implementation", "reproduce", "appendix",
        "table", "equation", "code", "open source", "github",
    )
    cue_hits = sum(1 for cue in cues if cue in text.lower())
    length_score = min(1.0, len(text) / 3500.0)
    paper_bonus = 0.15 if item.source_kind == "paper" else 0.0
    return min(1.0, (cue_hits / 8.0) * 0.65 + length_score * 0.25 + paper_bonus)


def _novelty(text: str) -> float:
    cues = ("we propose", "we introduce", "new", "novel", "first", "state-of-the-art", "sota")
    return min(1.0, 0.25 + 0.15 * sum(1 for cue in cues if cue in text.lower()))


def _practical_signal(text: str, item: SourceItem) -> float:
    cues = ("production", "serving", "latency", "cost", "deployment", "code", "github", "api", "pipeline")
    return min(1.0, 0.10 * sum(1 for cue in cues if cue in text.lower()) + (0.15 if item.source_kind == "blog" else 0.0))


def _reproducibility(text: str, item: SourceItem) -> float:
    cues = ("github", "code", "dataset", "appendix", "reproduce", "implementation", "open-source")
    return min(1.0, 0.14 * sum(1 for cue in cues if cue in text.lower()))


def _diversify(ranked: list[RankedItem]) -> list[RankedItem]:
    selected: list[RankedItem] = []
    remaining = list(ranked)
    while remaining:
        best = max(remaining, key=lambda r: r.daily_score - _redundancy(r, selected))
        selected.append(best)
        remaining.remove(best)
    return selected


def _tokens(item: SourceItem) -> set[str]:
    return set(re.findall(r"[a-z0-9]{3,}", f"{item.title} {item.abstract_or_excerpt}".lower()))


def _redundancy(candidate: RankedItem, selected: list[RankedItem]) -> float:
    if not selected:
        return 0.0
    ct = _tokens(candidate.item)
    if not ct:
        return 0.0
    worst = 0.0
    for item in selected:
        st = _tokens(item.item)
        if st:
            worst = max(worst, len(ct & st) / len(ct | st))
    return 0.18 * worst
