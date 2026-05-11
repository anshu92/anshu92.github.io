from __future__ import annotations

from pathlib import Path

from blogpipe import memory
from blogpipe.cli import main


def test_run_with_fixtures_and_fake_llm(monkeypatch, tmp_path):
    fake = Path("tests/fixtures/fake_daily.md").read_text()
    monkeypatch.setattr(memory, "ROOT", tmp_path)
    monkeypatch.setattr(memory, "DATA", tmp_path / "data")
    monkeypatch.setattr(memory, "DAILY_DATA", tmp_path / "data" / "daily")
    monkeypatch.setattr(memory, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(memory, "CONTENT_POST", tmp_path / "content" / "post")
    monkeypatch.setattr(memory, "STATIC_POSTS", tmp_path / "static" / "img" / "posts")
    monkeypatch.setenv("BLOGPIPE_REPO_ROOT", str(tmp_path))
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
    assert (tmp_path / "data" / "daily").is_dir()
