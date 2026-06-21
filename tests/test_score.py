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
    assert any("AEC" in r.topic_scores.tracks or "ML-Engineering" in r.topic_scores.tracks for r in ranked)
    assert len(daily_shortlist(ranked)) >= 4


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


def test_ml_engineering_outranks_aec_with_same_depth():
    now = datetime(2026, 5, 11, tzinfo=timezone.utc)
    engineering = SourceItem(
        canonical_url="https://example.com/engineering",
        source_kind="paper",
        source_name="arxiv",
        source_tier=1,
        title="FlashAttention-3 serving kernel for distributed PyTorch inference",
        published_at=now,
        abstract_or_excerpt=(
            "We optimize GPU CUDA kernels, KV cache memory layout, throughput, latency, quantization, "
            "and distributed training for HuggingFace-scale LLM serving."
        ),
        extra={"search_profile": "ml_engineering"},
    )
    aec = SourceItem(
        canonical_url="https://example.com/aec",
        source_kind="paper",
        source_name="arxiv",
        source_tier=1,
        title="BIM IFC CAD digital twin benchmark for construction facility workflows",
        published_at=now,
        abstract_or_excerpt=(
            "We benchmark BIM IFC CAD digital twin HVAC building controls and scan-to-bim deployment "
            "for construction facility operations."
        ),
        extra={"search_profile": "aec_ai"},
    )
    ranked = rank_items([aec, engineering], now=now, max_age_hours=72)
    assert ranked[0].item.title.startswith("FlashAttention")
    assert ranked[0].topic_scores.priority_track == "ml_engineering"


def test_scaled_training_howto_signal_boosts_ranked_items():
    now = datetime(2026, 5, 11, tzinfo=timezone.utc)
    training = SourceItem(
        canonical_url="https://example.com/training",
        source_kind="paper",
        source_name="arxiv",
        source_tier=1,
        title="FSDP tensor parallel training runbook for trillion-token LLMs",
        published_at=now,
        abstract_or_excerpt=(
            "We study distributed training with FSDP sharding, tensor parallelism, activation checkpointing, "
            "NCCL all-reduce communication, microbatch scheduling, data pipeline throughput, GPU utilization, "
            "checkpoint recovery, profiling, and benchmark ablations."
        ),
    )
    generic = SourceItem(
        canonical_url="https://example.com/generic-serving",
        source_kind="paper",
        source_name="arxiv",
        source_tier=1,
        title="Generic language model serving benchmark",
        published_at=now,
        abstract_or_excerpt="A language model benchmark with architecture, latency, throughput, and deployment notes.",
    )
    ranked = rank_items([generic, training], now=now, max_age_hours=72)
    assert ranked[0].item.item_id == training.stable_id()
    assert ranked[0].quality_signals["training_howto"] >= 0.7
    assert ranked[0].topic_scores.priority_track == "ml_engineering"


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


def test_rank_items_keeps_recently_updated_papers_with_old_publication_date():
    now = datetime(2026, 5, 11, tzinfo=timezone.utc)
    updated = SourceItem(
        canonical_url="https://openreview.net/forum?id=recent-update",
        source_kind="paper",
        source_name="openreview",
        source_tier=1,
        title="Recently updated evaluation benchmark",
        published_at=datetime(2025, 5, 1, tzinfo=timezone.utc),
        updated_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
        abstract_or_excerpt="A language model evaluation benchmark with dataset monitoring and deployment signals.",
    )
    ranked = rank_items([updated], now=now, max_age_hours=72)
    assert [r.item.title for r in ranked] == ["Recently updated evaluation benchmark"]
    assert ranked[0].quality_signals["freshness"] > 0.75


def test_rank_items_relaxes_gate_when_strict_gate_filters_everything():
    now = datetime(2026, 5, 11, tzinfo=timezone.utc)
    items = [
        SourceItem(
            canonical_url="https://example.com/paper/1",
            source_kind="paper",
            source_name="example",
            source_tier=2,
            title="Sparse graph structure under unknown assumptions",
            published_at=now,
            abstract_or_excerpt=(
                "We study choices, assumptions, setup, and findings for a narrow graph task "
                "without quantitative evaluation or production claims."
            ),
            tags=["research"],
            extra={"search_profile": "fallback_test"},
        ),
    ]
    ranked = rank_items(items, now=now, max_age_hours=72)
    assert ranked
    assert ranked[0].topic_scores.best < 0.20


def test_daily_shortlist_prefers_papers_and_diverse_profiles():
    now = datetime(2026, 5, 11, tzinfo=timezone.utc)

    def item(idx: int, *, kind: str = "paper", profile: str = "llm_methods", track: str = "applied_research") -> SourceItem:
        topic = {
            "applied_research": "language model inference benchmark with architecture ablation latency throughput",
            "ml_engineering": "pytorch jax huggingface kernel cuda quantization distributed training serving throughput latency",
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
        item(1, profile="llm_methods", track="applied_research"),
        item(2, profile="llm_methods", track="applied_research"),
        item(3, profile="ml_engineering", track="ml_engineering"),
        item(4, profile="mle_eval", track="ml_engineering"),
        item(5, profile="aec_ai", track="aec"),
        item(6, profile="multimodal_geometry", track="applied_research"),
        item(7, kind="blog", profile="blog:pytorch", track="ml_engineering"),
        item(8, kind="blog", profile="blog:pytorch", track="ml_engineering"),
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
            topic_scores=TopicScores(
                ml_engineering=0.6,
                applied_research=0.2,
                matched_keywords=["language model", "benchmark"],
            ),
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
