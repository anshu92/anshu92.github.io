from __future__ import annotations

import math
import re
import logging
from datetime import datetime, timezone
from collections import Counter

from . import config
from .models import RankedItem, SourceItem, TopicScores
from .topics import TRACKS, keyword_hits

LOG = logging.getLogger(__name__)


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
    if not gated and scored:
        # Daily feeds can be sparse when upstream sources are rate-limited (e.g., arXiv 429s).
        # Fall back to a softer gate so selector/outliner can still operate on best available items.
        gated = [r for r in scored if _passes_relaxed_gate(r)]
        if gated:
            LOG.warning(
                "rank: strict gate filtered all %d candidates; using relaxed fallback (%d)",
                len(scored),
                len(gated),
            )
    gated.sort(key=lambda r: r.daily_score, reverse=True)
    return _diversify(gated[:limit])


def daily_shortlist(
    ranked: list[RankedItem],
    *,
    minimum: int = 5,
    maximum: int = 8,
    min_papers: int | None = None,
    max_blogs: int | None = None,
) -> list[RankedItem]:
    min_papers = config.min_papers() if min_papers is None else min_papers
    max_blogs = config.max_blogs() if max_blogs is None else max_blogs
    strong = [r for r in ranked if r.daily_score >= 0.68]
    medium = [r for r in ranked if r.daily_score >= 0.50]
    required_papers = min(min_papers, sum(1 for r in ranked if r.item.source_kind == "paper"), maximum)
    if _shortlist_pool_is_sufficient(strong, minimum=minimum, required_papers=required_papers):
        pool = strong
    elif _shortlist_pool_is_sufficient(medium, minimum=minimum, required_papers=required_papers):
        pool = medium
    else:
        pool = ranked
    return _paper_first_shortlist(pool, minimum=minimum, maximum=maximum, min_papers=min_papers, max_blogs=max_blogs)


def deep_dive_shortlist(ranked: list[RankedItem], *, maximum: int = 1) -> list[RankedItem]:
    return [r for r in ranked if r.deep_dive_score >= 0.78][:maximum]


def _score_one(item: SourceItem, now: datetime) -> RankedItem:
    item = item.normalized()
    blob = f"{item.title}\n{item.abstract_or_excerpt}\n{item.body_text}\n{' '.join(item.tags)}"
    scores = _topic_scores(blob)
    source_quality = min(1.0, max(0.0, 1.1 - (item.source_tier * 0.18)))
    freshness = _freshness(item, now)
    depth = _technical_depth(blob, item)
    novelty = _novelty(blob)
    practical = _practical_signal(blob, item)
    evidence = min(1.0, len(blob) / 1200.0)
    kind_prior = 0.06 if item.source_kind == "paper" else -0.05
    daily = (
        0.25 * scores.best
        + 0.20 * source_quality
        + 0.15 * freshness
        + 0.15 * depth
        + 0.10 * novelty
        + 0.10 * practical
        + 0.05 * evidence
        + kind_prior
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
        daily_score=round(max(0.0, min(1.0, daily)), 4),
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
    if ranked.item.source_kind == "blog":
        if ranked.quality_signals.get("technical_depth", 0) < 0.28 or ranked.quality_signals.get("practical_impact", 0) < 0.25:
            return False
    return True


def _passes_relaxed_gate(ranked: RankedItem) -> bool:
    """Fallback gate for sparse days when strict topical filtering yields zero candidates."""
    if ranked.item.source_kind == "blog" and ranked.item.source_tier > 3:
        return False
    topic = float(ranked.topic_scores.best)
    depth = float(ranked.quality_signals.get("technical_depth", 0.0))
    practical = float(ranked.quality_signals.get("practical_impact", 0.0))
    if topic >= 0.08:
        return True
    if depth >= 0.45:
        return True
    return ranked.item.source_kind == "paper" and depth >= 0.30 and practical >= 0.10


def _shortlist_pool_is_sufficient(pool: list[RankedItem], *, minimum: int, required_papers: int) -> bool:
    if len(pool) < minimum:
        return False
    return sum(1 for r in pool if r.item.source_kind == "paper") >= required_papers


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
        "table", "equation", "code", "open source", "github", "algorithm",
        "theorem", "bound", "gradient", "complexity", "training", "inference",
        "evaluation", "experiment", "failure mode", "profiling", "memory layout",
        "bim", "ifc", "cad", "digital twin", "hvac", "building controls",
        "graph extraction", "facility operations",
    )
    cue_hits = sum(1 for cue in cues if cue in text.lower())
    length_score = min(1.0, len(text) / 3500.0)
    paper_bonus = 0.15 if item.source_kind == "paper" else 0.0
    return min(1.0, (cue_hits / 8.0) * 0.65 + length_score * 0.25 + paper_bonus)


def _novelty(text: str) -> float:
    cues = ("we propose", "we introduce", "new", "novel", "first", "state-of-the-art", "sota")
    return min(1.0, 0.25 + 0.15 * sum(1 for cue in cues if cue in text.lower()))


def _practical_signal(text: str, item: SourceItem) -> float:
    cues = (
        "production", "serving", "latency", "cost", "deployment", "code", "github", "api", "pipeline",
        "facility", "operations", "building controls", "revit", "monitoring",
    )
    return min(1.0, 0.10 * sum(1 for cue in cues if cue in text.lower()) + (0.05 if item.source_kind == "blog" else 0.0))


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


def _paper_first_shortlist(
    ranked: list[RankedItem],
    *,
    minimum: int,
    maximum: int,
    min_papers: int,
    max_blogs: int,
) -> list[RankedItem]:
    candidates = list(ranked)
    target_len = min(maximum, max(minimum, min(len(candidates), maximum)))
    paper_target = min(min_papers, sum(1 for r in candidates if r.item.source_kind == "paper"), maximum)
    selected: list[RankedItem] = []
    selected_ids: set[str] = set()

    def add_best(pool: list[RankedItem], *, force_paper: bool = False, relaxed: bool = False) -> bool:
        options = [
            r
            for r in pool
            if r.item.item_id not in selected_ids
            and (not force_paper or r.item.source_kind == "paper")
            and (relaxed or _can_add_to_shortlist(r, selected, max_blogs=max_blogs))
        ]
        if not options:
            return False
        best = max(options, key=lambda r: r.daily_score - _redundancy(r, selected) - _shortlist_penalty(r, selected))
        selected.append(best)
        selected_ids.add(best.item.item_id)
        return True

    while sum(1 for r in selected if r.item.source_kind == "paper") < paper_target and len(selected) < maximum:
        if not add_best(candidates, force_paper=True):
            if not add_best(candidates, force_paper=True, relaxed=True):
                break
    while len(selected) < target_len:
        if not add_best(candidates):
            if not add_best(candidates, relaxed=True):
                break
    return selected[:maximum]


def _can_add_to_shortlist(candidate: RankedItem, selected: list[RankedItem], *, max_blogs: int) -> bool:
    if candidate.item.source_kind == "blog":
        if sum(1 for r in selected if r.item.source_kind == "blog") >= max_blogs:
            return False
    families = Counter(_source_family(r.item) for r in selected)
    profiles = Counter(_search_profile(r.item) for r in selected)
    clusters = Counter(_topic_cluster(r) for r in selected)
    return (
        families[_source_family(candidate.item)] < 2
        and profiles[_search_profile(candidate.item)] < 2
        and clusters[_topic_cluster(candidate)] < 2
    )


def _shortlist_penalty(candidate: RankedItem, selected: list[RankedItem]) -> float:
    penalty = 0.0
    for item in selected:
        if _source_family(candidate.item) == _source_family(item.item):
            penalty += 0.04
        if _search_profile(candidate.item) == _search_profile(item.item):
            penalty += 0.06
        if _topic_cluster(candidate) == _topic_cluster(item):
            penalty += 0.04
    return penalty


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
    metadata_penalty = 0.0
    if any(_source_family(candidate.item) == _source_family(item.item) for item in selected):
        metadata_penalty += 0.04
    if any(_search_profile(candidate.item) == _search_profile(item.item) for item in selected):
        metadata_penalty += 0.06
    if any(_topic_cluster(candidate) == _topic_cluster(item) for item in selected):
        metadata_penalty += 0.04
    return 0.18 * worst + metadata_penalty


def _search_profile(item: SourceItem) -> str:
    return str(item.extra.get("search_profile") or item.source_name or item.venue_or_blog or "unknown")


def _source_family(item: SourceItem) -> str:
    profile = _search_profile(item)
    if item.source_name == "arxiv":
        return f"arxiv:{profile}"
    if item.source_name == "openreview":
        return profile
    return item.source_name or item.venue_or_blog or profile


def _topic_cluster(ranked: RankedItem) -> str:
    return ranked.topic_scores.tracks[0] if ranked.topic_scores.tracks else "ML"
