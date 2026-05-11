from __future__ import annotations

import json
from pathlib import Path

from pydantic import TypeAdapter

from blogpipe.evidence import build_daily_pack
from blogpipe.llm import LLMClient
from blogpipe import memory
from blogpipe.models import SourceItem
from blogpipe.score import daily_shortlist, rank_items
from blogpipe.writer import validate_body, write_daily


def _pack():
    data = json.loads(Path("tests/fixtures/source_items.json").read_text())
    items = TypeAdapter(list[SourceItem]).validate_python(data["items"])
    ranked = daily_shortlist(rank_items(items))
    return build_daily_pack(ranked)


def _patch_root(monkeypatch, tmp_path):
    monkeypatch.setattr(memory, "ROOT", tmp_path)
    monkeypatch.setattr(memory, "DATA", tmp_path / "data")
    monkeypatch.setattr(memory, "DAILY_DATA", tmp_path / "data" / "daily")
    monkeypatch.setattr(memory, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(memory, "CONTENT_POST", tmp_path / "content" / "post")
    monkeypatch.setattr(memory, "STATIC_POSTS", tmp_path / "static" / "img" / "posts")


def test_valid_llm_daily_post_passes(monkeypatch, tmp_path):
    pack = _pack()
    fake = Path("tests/fixtures/fake_daily.md").read_text()
    _patch_root(monkeypatch, tmp_path)
    monkeypatch.setenv("BLOGPIPE_FAKE_LLM_RESPONSE", fake)
    result = write_daily(pack, llm=LLMClient(), dry_run=True)
    assert result.ok, result.errors
    assert result.path.startswith("reports/")


def test_unsupported_number_fails():
    pack = _pack()
    body = Path("tests/fixtures/fake_daily.md").read_text() + "\n\nInvented result: 99.9% faster. [E1]"
    errors = validate_body(body, pack)
    assert "unsupported_number:99.9%" in errors


def test_missing_source_link_fails():
    pack = _pack()
    body = "Grounded claim [E1]. Source: https://arxiv.org/abs/2605.00001"
    errors = validate_body(body, pack)
    assert any(e.startswith("missing_source_link:") for e in errors)


def test_repair_prompt_receives_validator_errors(monkeypatch, tmp_path):
    pack = _pack()
    bad = "Unsupported 99.9% claim [E1]. Source: https://arxiv.org/abs/2605.00001"
    good = Path("tests/fixtures/fake_daily.md").read_text()
    calls = []

    class FakeLLM:
        def complete(self, *, system, user, max_tokens=None):
            calls.append(user)
            return bad if len(calls) == 1 else good

    _patch_root(monkeypatch, tmp_path)
    result = write_daily(pack, llm=FakeLLM(), dry_run=True)
    assert result.ok
    assert result.repair_attempted
    assert "VALIDATOR_ERRORS" in calls[1]


def test_blocked_post_is_not_published_after_failed_repair(monkeypatch, tmp_path):
    pack = _pack()

    class BadLLM:
        def complete(self, *, system, user, max_tokens=None):
            return "Unsupported 99.9% claim [E1]. Source: https://arxiv.org/abs/2605.00001"

    _patch_root(monkeypatch, tmp_path)
    result = write_daily(pack, llm=BadLLM(), dry_run=False)
    assert not result.ok
    assert not list((tmp_path / "content" / "post").glob("*.md"))
    assert list((tmp_path / "reports").glob("*.blocked.json"))
