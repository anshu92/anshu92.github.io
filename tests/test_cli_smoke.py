from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from blogpipe import memory, store
from blogpipe.cli import main
from blogpipe.llm import LLMClient
from blogpipe.models import DailyOutline, SelectionResult, SourceItem
from blogpipe.pipeline import _augment_ranked_with_store_papers, _plan_agent_run, write_daily
from blogpipe.score import rank_items


def test_run_with_fixtures_and_fake_llm(monkeypatch, tmp_path):
    fake = Path("tests/fixtures/fake_daily.md").read_text()
    selector = Path("tests/fixtures/fake_selector.json").read_text()
    outline = Path("tests/fixtures/fake_outline.json").read_text()
    monkeypatch.setattr(memory, "ROOT", tmp_path)
    monkeypatch.setattr(memory, "DATA", tmp_path / "radar-data")
    monkeypatch.setattr(memory, "DAILY_DATA", tmp_path / "radar-data" / "daily")
    monkeypatch.setattr(memory, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(memory, "CONTENT_POST", tmp_path / "content" / "post")
    monkeypatch.setattr(memory, "STATIC_POSTS", tmp_path / "static" / "img" / "posts")
    monkeypatch.setenv("BLOGPIPE_REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("BLOGPIPE_FAKE_SELECTOR_RESPONSE", selector)
    monkeypatch.setenv("BLOGPIPE_FAKE_OUTLINE_RESPONSE", outline)
    monkeypatch.setenv("BLOGPIPE_FAKE_LLM_RESPONSE", fake)
    code = main(
        [
            "run",
            "--fixtures",
            "tests/fixtures",
            "--dry-run",
            "--db",
            str(tmp_path / "items.sqlite"),
            "--max-deep-dives",
            "0",
        ]
    )
    assert code == 0
    assert (tmp_path / "reports" / "run_report.json").is_file()
    assert (tmp_path / "radar-data" / "daily").is_dir()
    report = json.loads((tmp_path / "reports" / "run_report.json").read_text(encoding="utf-8"))
    assert report["agent_plan"]["initial"]["daily_required"] is True
    assert report["agent_plan"]["final"]["allowed_deep_dives"] == 0


def test_run_with_invalid_daily_writes_blocked_report_without_failing(monkeypatch, tmp_path):
    selector = Path("tests/fixtures/fake_selector.json").read_text()
    outline = Path("tests/fixtures/fake_outline.json").read_text()
    monkeypatch.setattr(memory, "ROOT", tmp_path)
    monkeypatch.setattr(memory, "DATA", tmp_path / "radar-data")
    monkeypatch.setattr(memory, "DAILY_DATA", tmp_path / "radar-data" / "daily")
    monkeypatch.setattr(memory, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(memory, "CONTENT_POST", tmp_path / "content" / "post")
    monkeypatch.setattr(memory, "STATIC_POSTS", tmp_path / "static" / "img" / "posts")
    monkeypatch.setenv("BLOGPIPE_REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("BLOGPIPE_FAKE_SELECTOR_RESPONSE", selector)
    monkeypatch.setenv("BLOGPIPE_FAKE_OUTLINE_RESPONSE", outline)
    monkeypatch.setenv("BLOGPIPE_FAKE_LLM_RESPONSE", "No citations or source links.")
    code = main(
        [
            "run",
            "--fixtures",
            "tests/fixtures",
            "--db",
            str(tmp_path / "items.sqlite"),
            "--max-deep-dives",
            "0",
        ]
    )
    assert code == 0
    assert list((tmp_path / "reports").glob("*.blocked.json"))
    assert not list((tmp_path / "content" / "post").glob("*.md"))


def test_run_with_training_fixtures_generates_howto_preview(monkeypatch, tmp_path):
    items = _training_fixture_items()
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    (fixtures / "source_items.json").write_text(
        json.dumps({"items": [item.model_dump(mode="json") for item in items]}, indent=2),
        encoding="utf-8",
    )
    selector = {
        "selected_item_ids": [item.item_id for item in items],
        "items": [
            {
                "item_id": item.item_id,
                "role": "primary",
                "relevance_label": "ml_engineering",
                "scores": {
                    "engineering_value": 0.95,
                    "experiment_strength": 0.82,
                    "engineering_actionability": 0.94,
                    "training_howto_value": 0.96,
                    "novelty": 0.7,
                },
                "reason": "Concrete scaled LLM training runbook evidence.",
                "suggested_tags": ["llm-training", "mle"],
            }
            for item in items
        ],
        "suggested_tags": ["llm-training", "mle"],
    }
    outline = {
        "title": "Research Radar: Scaled LLM training runbooks",
        "angle": "Principal engineers need training-stack decisions, not generic production prose.",
        "sections": [
            {
                "heading": "Training thesis",
                "intent": "technical thesis angle framing",
                "evidence_ids": ["E1", "E2", "E3"],
                "word_budget": 80,
            },
            {
                "heading": "FSDP sharding and parallelism runbook",
                "intent": "mechanism method architecture distributed training sharding parallelism",
                "evidence_ids": ["E1"],
                "word_budget": 80,
            },
            {
                "heading": "Objective and profiling metrics",
                "intent": "math objective metric optimization throughput profiling",
                "evidence_ids": ["E1", "E2"],
                "word_budget": 80,
            },
            {
                "heading": "Benchmarks and failure modes",
                "intent": "experiments evidence benchmark evaluation limitation failure risk",
                "evidence_ids": ["E2", "E3"],
                "word_budget": 80,
            },
            {
                "heading": "Principal rollout decision",
                "intent": "cross-paper synthesis compare contrast tradeoff impact engineering production practical Autodesk AEC document validation release gate",
                "evidence_ids": ["E1", "E2", "E3"],
                "word_budget": 80,
            },
        ],
        "suggested_tags": ["llm-training", "mle"],
    }
    fake_body = _training_fake_daily_body()
    monkeypatch.setattr(memory, "ROOT", tmp_path)
    monkeypatch.setattr(memory, "DATA", tmp_path / "radar-data")
    monkeypatch.setattr(memory, "DAILY_DATA", tmp_path / "radar-data" / "daily")
    monkeypatch.setattr(memory, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(memory, "CONTENT_POST", tmp_path / "content" / "post")
    monkeypatch.setattr(memory, "STATIC_POSTS", tmp_path / "static" / "img" / "posts")
    monkeypatch.setenv("BLOGPIPE_REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("BLOGPIPE_DAILY_MIN_WORDS", "300")
    monkeypatch.setenv("BLOGPIPE_FAKE_SELECTOR_RESPONSE", json.dumps(selector))
    monkeypatch.setenv("BLOGPIPE_FAKE_OUTLINE_RESPONSE", json.dumps(outline))
    monkeypatch.setenv("BLOGPIPE_FAKE_LLM_RESPONSE", fake_body)

    code = main(
        [
            "run",
            "--fixtures",
            str(fixtures),
            "--dry-run",
            "--db",
            str(tmp_path / "items.sqlite"),
            "--max-deep-dives",
            "0",
        ]
    )

    assert code == 0
    previews = list((tmp_path / "reports").glob("*.preview.md"))
    assert len(previews) == 1
    rendered = previews[0].read_text(encoding="utf-8")
    assert '"llm-training"' in rendered
    assert "FSDP sharding" in rendered
    assert "tensor parallelism" in rendered
    assert "NCCL all-reduce" in rendered
    assert "checkpoint recovery" in rendered
    assert "release gate" in rendered
    report = json.loads((tmp_path / "reports" / "run_report.json").read_text(encoding="utf-8"))
    assert report["daily"]["ok"] is True
    assert report["deep_dives"] == []


def test_write_daily_blocks_before_llm_when_ranked_papers_are_insufficient(monkeypatch, tmp_path):
    class FailIfCalledLLM:
        class Usage:
            __dict__ = {"calls": 0}

        usage = Usage()

        def complete(self, **kwargs):
            raise AssertionError("LLM should not run when there are not enough ranked papers")

    monkeypatch.setattr(memory, "ROOT", tmp_path)
    monkeypatch.setattr(memory, "DATA", tmp_path / "radar-data")
    monkeypatch.setattr(memory, "DAILY_DATA", tmp_path / "radar-data" / "daily")
    monkeypatch.setattr(memory, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(memory, "CONTENT_POST", tmp_path / "content" / "post")
    monkeypatch.setattr(memory, "STATIC_POSTS", tmp_path / "static" / "img" / "posts")

    result = write_daily(ranked=[], llm=FailIfCalledLLM(), dry_run=True)
    assert not result.ok
    assert result.errors == ["insufficient_ranked_papers:0/3"]
    assert list((tmp_path / "reports").glob("*.blocked.json"))


def test_daily_rank_fallback_recovers_recent_store_papers(tmp_path):
    db = tmp_path / "items.sqlite"
    now = datetime.now(timezone.utc)
    items = [_recent_paper(idx, now=now) for idx in range(3)]
    with store.connect(db) as conn:
        store.upsert_items(conn, items)

    ranked = _augment_ranked_with_store_papers(
        [],
        required_papers=3,
        db=str(db),
        max_age_hours=72,
    )

    assert sum(1 for item in ranked if item.item.source_kind == "paper") >= 3


def test_daily_rank_fallback_avoids_duplicate_recent_papers(tmp_path):
    db = tmp_path / "items.sqlite"
    now = datetime.now(timezone.utc)
    items = [_recent_paper(idx, now=now) for idx in range(3)]
    with store.connect(db) as conn:
        store.upsert_items(conn, items)

    already_ranked = rank_items([items[0]], now=now, max_age_hours=72)
    ranked = _augment_ranked_with_store_papers(
        already_ranked,
        required_papers=3,
        db=str(db),
        max_age_hours=72,
    )
    item_ids = [item.item.item_id for item in ranked]

    assert sum(1 for item in ranked if item.item.source_kind == "paper") >= 3
    assert len(item_ids) == len(set(item_ids))


def test_write_daily_blocks_instead_of_crashing_when_writer_runtime_expires(monkeypatch, tmp_path):
    now = datetime.now(timezone.utc)
    ranked = rank_items([_recent_paper(idx, now=now) for idx in range(3)], now=now, max_age_hours=72)

    class FakeLLM:
        class Usage:
            __dict__ = {"calls": 0}

        usage = Usage()

    monkeypatch.setattr(memory, "ROOT", tmp_path)
    monkeypatch.setattr(memory, "DATA", tmp_path / "radar-data")
    monkeypatch.setattr(memory, "DAILY_DATA", tmp_path / "radar-data" / "daily")
    monkeypatch.setattr(memory, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(memory, "CONTENT_POST", tmp_path / "content" / "post")
    monkeypatch.setattr(memory, "STATIC_POSTS", tmp_path / "static" / "img" / "posts")
    monkeypatch.setattr(
        "blogpipe.pipeline.selector.select_daily_items",
        lambda ranked, llm: (
            ranked,
            SelectionResult(selected_item_ids=[item.item.item_id for item in ranked]),
        ),
    )
    monkeypatch.setattr(
        "blogpipe.pipeline.outline_mod.generate_daily_outline",
        lambda pack, selection, llm: DailyOutline(title="Research Radar: Test", angle="budget guard", sections=[]),
    )
    monkeypatch.setattr(
        "blogpipe.pipeline.writer.write_daily",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("BLOGPIPE_LLM_MAX_RUNTIME_SECONDS reached")),
    )

    result = write_daily(ranked=ranked, llm=FakeLLM(), dry_run=True)

    assert not result.ok
    assert result.errors == ["daily_writer_failed:BLOGPIPE_LLM_MAX_RUNTIME_SECONDS reached"]
    assert list((tmp_path / "reports").glob("*.blocked.json"))


def test_agent_run_plan_skips_optional_deep_dives_when_runtime_budget_is_low(monkeypatch):
    monkeypatch.setenv("BLOGPIPE_LLM_MAX_RUNTIME_SECONDS", "1200")
    monkeypatch.setenv("BLOGPIPE_AGENT_DEEP_DIVE_MIN_BUDGET_SECONDS", "420")
    llm = LLMClient()
    monkeypatch.setattr("blogpipe.llm.time.monotonic", lambda: llm.started_at + 1000.0)

    plan = _plan_agent_run(llm, requested_deep_dives=1)

    assert plan.daily_required is True
    assert plan.requested_deep_dives == 1
    assert plan.allowed_deep_dives == 0
    assert plan.rationale == "skip_optional_deep_dives_to_preserve_github_actions_budget"


def test_agent_run_plan_allows_deep_dives_when_runtime_budget_remains(monkeypatch):
    monkeypatch.setenv("BLOGPIPE_LLM_MAX_RUNTIME_SECONDS", "1200")
    monkeypatch.setenv("BLOGPIPE_AGENT_DEEP_DIVE_MIN_BUDGET_SECONDS", "420")
    llm = LLMClient()
    monkeypatch.setattr("blogpipe.llm.time.monotonic", lambda: llm.started_at + 100.0)

    plan = _plan_agent_run(llm, requested_deep_dives=2)

    assert plan.allowed_deep_dives == 2
    assert plan.rationale == "runtime_budget_allows_optional_deep_dives"


def _recent_paper(idx: int, *, now: datetime) -> SourceItem:
    return SourceItem(
        canonical_url=f"https://arxiv.org/abs/2605.10{idx:03d}",
        source_kind="paper",
        source_name="arxiv",
        source_tier=1,
        title=f"Benchmarking CAD deployment paper {idx}",
        published_at=now,
        abstract_or_excerpt=(
            "We propose a language model benchmark with objective design, implementation detail, "
            "failure mode analysis, latency measurements, deployment constraints, and ablation evidence "
            "for CAD and document intelligence workflows."
        ),
        extra={"search_profile": "fallback_test"},
    )


def _training_fixture_items() -> list[SourceItem]:
    now = datetime(2026, 5, 11, tzinfo=timezone.utc)
    return [
        SourceItem(
            item_id="train-fsdp",
            canonical_url="https://example.com/train-fsdp",
            source_kind="paper",
            source_name="arxiv",
            source_tier=1,
            title="FSDP sharding runbooks for scaled LLM training",
            published_at=now,
            abstract_or_excerpt=(
                "We propose a distributed training method using FSDP sharding, tensor parallelism, activation checkpointing, "
                "microbatch scheduling, and optimizer state partitioning to improve scaled LLM training throughput under memory pressure."
            ),
            tags=["distributed training", "fsdp", "tensor parallel"],
        ),
        SourceItem(
            item_id="train-nccl",
            canonical_url="https://example.com/train-nccl",
            source_kind="paper",
            source_name="arxiv",
            source_tier=1,
            title="NCCL profiling for LLM training communication bottlenecks",
            published_at=now,
            abstract_or_excerpt=(
                "The evaluation reports benchmark ablations for NCCL all-reduce communication, GPU utilization, data pipeline throughput, "
                "checkpoint recovery, profiling, and scaling efficiency in large-scale training."
            ),
            tags=["distributed training", "nccl", "profiling"],
        ),
        SourceItem(
            item_id="train-data",
            canonical_url="https://example.com/train-data",
            source_kind="paper",
            source_name="arxiv",
            source_tier=1,
            title="Data pipeline and checkpoint recovery for reliable LLM pretraining",
            published_at=now,
            abstract_or_excerpt=(
                "The limitation is that data loader stalls, workload balance, checkpoint cadence, and restart behavior create failure modes "
                "that a principal engineer must validate before a production training rollout."
            ),
            tags=["distributed training", "data pipeline", "checkpointing"],
        ),
    ]


def _training_fake_daily_body() -> str:
    return """
# Research Radar: Scaled LLM training runbooks

## Training thesis
The thesis is that scaled LLM training papers should be read as runbooks for resource allocation, not as generic model-quality updates. Across the three papers, the common pattern is a compare-and-tradeoff loop: sharding, communication, and data-pipeline choices must be evaluated together before the next training run. The evidence frames the problem as training throughput under memory and communication pressure, so the practical question is what stack decision a principal engineer would change. The mechanism, objective, benchmark, limitation, impact, and Autodesk/AEC transfer lens all point to the same claim: training reliability depends on making the parallelism and recovery plan explicit [E1] [E2] [E3]. Sources: https://example.com/train-fsdp https://example.com/train-nccl https://example.com/train-data

## FSDP sharding and parallelism runbook
The concrete training stack starts with FSDP sharding for optimizer and parameter state, then uses tensor parallelism where a single layer no longer fits cleanly on one device. Activation checkpointing trades recomputation for memory headroom, while microbatch scheduling controls how much work each device sees before synchronization. That is the how-to decision path: choose the sharding boundary, decide whether tensor parallelism is needed, then check whether activation memory or communication is the current blocker [E1]. Source: https://example.com/train-fsdp

## Objective and profiling metrics
The objective should be treated as an operational optimization problem rather than a single model score. A principal MLE would measure throughput, GPU utilization, data pipeline stalls, NCCL all-reduce time, checkpoint cadence, and recovery time after interruption. Those metrics decide whether to tune microbatch size, revise token packing, move data loading work, or change checkpoint frequency. The evidence supports this profiling view because it names benchmark ablations around utilization, recovery, and scaling efficiency [E1] [E2]. Sources: https://example.com/train-fsdp https://example.com/train-nccl

## Benchmarks and failure modes
The benchmark section should become a validation matrix. Test one run where memory pressure is dominant, one where all-reduce communication dominates, one where the data pipeline starves accelerators, and one restart path that exercises checkpoint recovery. The limitation is that workload balance and checkpoint cadence can turn a nominally efficient setup into a fragile production run. The failure mode to watch is not just lower throughput; it is a training job that cannot resume cleanly or wastes expensive GPU time while waiting on input data [E2] [E3]. Sources: https://example.com/train-nccl https://example.com/train-data

## Principal rollout decision
The rollout decision is to treat this stack as a staged adoption, not a default architecture. First profile a representative training slice, then benchmark FSDP-only against FSDP plus tensor parallelism, then validate checkpoint recovery and data loader pressure before scaling. For Autodesk or AEC document-model work, the transfer is an open hypothesis: better training infrastructure can shorten iteration loops for document models, but only if the same release gate tracks utilization, cost, restart behavior, and quality regression risk [E1] [E2] [E3]. Sources: https://example.com/train-fsdp https://example.com/train-nccl https://example.com/train-data
"""
