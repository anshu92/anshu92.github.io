from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from blogpipe import memory, store
from blogpipe.cli import main
from blogpipe.models import DailyOutline, SelectionResult, SourceItem
from blogpipe.pipeline import _augment_ranked_with_store_papers, write_daily
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
