from __future__ import annotations

from pathlib import Path

from blogpipe import memory
from blogpipe.cli import main
from blogpipe.pipeline import write_daily


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
