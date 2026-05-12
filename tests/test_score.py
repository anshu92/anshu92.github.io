from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import TypeAdapter

from blogpipe.models import RankedItem, SourceItem, TopicScores
from blogpipe.score import daily_shortlist, rank_items


def test_rank_items_scores_tracks_and_aec_gate():
    data = json.loads(Path("tests/fixtures/source_items.json").read_text())
    items = TypeAdapter(list[SourceItem]).validate_python(data["items"])
    ranked = rank_items(items, now=datetime(2026, 5, 11, tzinfo=timezone.utc))
    assert ranked
    assert ranked[0].daily_score >= ranked[-1].daily_score
    assert any("AEC" in r.topic_scores.tracks for r in ranked)
    assert len(daily_shortlist(ranked)) >= 5


def test_aec_requires_domain_keyword():
    item = SourceItem(
        canonical_url="https://example.com/generic",
        source_kind="paper",
        source_name="generic",
        source_tier=1,
        title="Language model architecture for spatial mood boards",
        published_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
        abstract_or_excerpt="A generic language model post about architecture and generative imagery.",
    )
    ranked = rank_items([item], now=datetime(2026, 5, 11, tzinfo=timezone.utc))
    assert ranked[0].topic_scores.aec == 0


def test_rank_items_drops_stale_and_undated_items():
    now = datetime(2026, 5, 11, tzinfo=timezone.utc)
    stale = SourceItem(
        canonical_url="https://example.com/stale",
        source_kind="paper",
        source_name="example",
        source_tier=1,
        title="Stale LLM benchmark",
        published_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
        abstract_or_excerpt="A language model benchmark with latency and throughput.",
    )
    undated = SourceItem(
        canonical_url="https://example.com/undated",
        source_kind="paper",
        source_name="example",
        source_tier=1,
        title="Undated LLM benchmark",
        abstract_or_excerpt="A language model benchmark with latency and throughput.",
    )
    fresh = SourceItem(
        canonical_url="https://example.com/fresh",
        source_kind="paper",
        source_name="example",
        source_tier=1,
        title="Fresh LLM benchmark",
        published_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
        abstract_or_excerpt="A language model benchmark with latency and throughput.",
    )
    ranked = rank_items([stale, undated, fresh], now=now, max_age_hours=72)
    assert [r.item.title for r in ranked] == ["Fresh LLM benchmark"]


def test_daily_shortlist_prefers_papers_and_diverse_profiles():
    now = datetime(2026, 5, 11, tzinfo=timezone.utc)

    def item(idx: int, *, kind: str = "paper", profile: str = "llm_methods", track: str = "llm") -> SourceItem:
        topic = {
            "llm": "language model inference benchmark with architecture ablation latency throughput",
            "mle": "evaluation benchmark reproducibility monitoring dataset pipeline deployment",
            "aec": "BIM IFC CAD digital twin HVAC building controls benchmark deployment",
        }[track]
        return SourceItem(
            canonical_url=f"https://example.com/{kind}/{idx}",
            source_kind=kind,
            source_name="arxiv" if kind == "paper" else "pytorch",
            source_tier=1,
            title=f"{kind.title()} {idx} {topic}",
            published_at=now,
            abstract_or_excerpt=f"We propose a method with objective, benchmark, ablation, limitation, and practical impact. {topic}",
            tags=[kind, track, profile],
            extra={"search_profile": profile},
        )

    items = [
        item(1, profile="llm_methods", track="llm"),
        item(2, profile="llm_methods", track="llm"),
        item(3, profile="llm_systems", track="mle"),
        item(4, profile="mle_eval", track="mle"),
        item(5, profile="aec_ai", track="aec"),
        item(6, profile="multimodal_geometry", track="llm"),
        item(7, kind="blog", profile="blog:pytorch", track="mle"),
        item(8, kind="blog", profile="blog:pytorch", track="mle"),
    ]
    shortlist = daily_shortlist(rank_items(items, now=now, max_age_hours=72), maximum=6, min_papers=4, max_blogs=1)
    assert sum(1 for r in shortlist if r.item.source_kind == "paper") >= 4
    assert sum(1 for r in shortlist if r.item.source_kind == "blog") <= 1
    profiles = [r.item.extra.get("search_profile") for r in shortlist]
    assert max(profiles.count(profile) for profile in set(profiles)) <= 2


def test_daily_shortlist_falls_back_when_high_score_pool_has_too_few_papers():
    now = datetime(2026, 5, 11, tzinfo=timezone.utc)

    def ranked(idx: int, *, kind: str, profile: str, score: float) -> RankedItem:
        return RankedItem(
            item=SourceItem(
                item_id=f"{kind}:{idx}",
                canonical_url=f"https://example.com/{kind}/{idx}",
                source_kind=kind,
                source_name="arxiv" if kind == "paper" else "openai",
                source_tier=1,
                title=f"{kind} {idx} language model benchmark architecture objective",
                published_at=now,
                abstract_or_excerpt="A method with objective, benchmark, ablation, limitation, and deployment impact.",
                extra={"search_profile": profile},
            ),
            topic_scores=TopicScores(llm=0.6, mle=0.2, matched_keywords=["language model", "benchmark"]),
            daily_score=score,
            deep_dive_score=score,
            quality_signals={"technical_depth": 0.7, "practical_impact": 0.4},
        )

    items = [
        ranked(1, kind="blog", profile="blog:openai", score=0.7),
        ranked(2, kind="paper", profile="llm_methods", score=0.48),
        ranked(3, kind="paper", profile="llm_systems", score=0.47),
        ranked(4, kind="paper", profile="mle_eval", score=0.46),
        ranked(5, kind="paper", profile="multimodal_geometry", score=0.45),
        ranked(6, kind="paper", profile="aec_ai", score=0.44),
    ]
    shortlist = daily_shortlist(items, minimum=5, maximum=6, min_papers=4, max_blogs=1)
    assert len(shortlist) >= 5
    assert sum(1 for r in shortlist if r.item.source_kind == "paper") >= 4
    assert sum(1 for r in shortlist if r.item.source_kind == "blog") <= 1
