from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import TypeAdapter

from blogpipe.models import SourceItem
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
