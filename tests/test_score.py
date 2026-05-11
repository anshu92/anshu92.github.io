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
        source_kind="blog",
        source_name="generic",
        source_tier=3,
        title="AI for architecture mood boards",
        abstract_or_excerpt="A generic post about architecture and generative imagery.",
    )
    ranked = rank_items([item], now=datetime(2026, 5, 11, tzinfo=timezone.utc))
    assert ranked[0].topic_scores.aec == 0
