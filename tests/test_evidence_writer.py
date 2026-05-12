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
    ranked = daily_shortlist(rank_items(items, max_age_hours=None))
    return build_daily_pack(ranked)


def _patch_root(monkeypatch, tmp_path):
    monkeypatch.setattr(memory, "ROOT", tmp_path)
    monkeypatch.setattr(memory, "DATA", tmp_path / "radar-data")
    monkeypatch.setattr(memory, "DAILY_DATA", tmp_path / "radar-data" / "daily")
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
    assert "draft: true" in (tmp_path / result.path).read_text()


def test_unsupported_number_fails():
    pack = _pack()
    body = Path("tests/fixtures/fake_daily.md").read_text() + "\n\nInvented result: 99.9% faster. [E1]"
    errors = validate_body(body, pack)
    assert "unsupported_number:99.9%" in errors


def test_plain_reference_numbers_are_not_metric_claims():
    pack = _pack()
    body = Path("tests/fixtures/fake_daily.md").read_text() + "\n\nRelated citation marker [49] is not a metric. [E1]"
    errors = validate_body(body, pack)
    assert "unsupported_number:49" not in errors


def test_arxiv_ids_are_not_metric_claims():
    pack = _pack()
    body = Path("tests/fixtures/fake_daily.md").read_text() + "\n\nArXiv identifier 2605.10195 is not a metric claim. [E1]"
    errors = validate_body(body, pack)
    assert "unsupported_number:2605.10195" not in errors


def test_missing_source_link_fails():
    pack = _pack()
    body = "Grounded claim [E1] without the source URL."
    errors = validate_body(body, pack)
    assert any(e.startswith("missing_source_link:") for e in errors)


def test_uncited_sources_do_not_require_links():
    pack = _pack()
    first = pack.ranked_items[0].item
    pack.ranked_items = [pack.ranked_items[0]]
    pack.chunks = [chunk for chunk in pack.chunks if chunk.item_id == first.item_id]
    body = """
## Technical thesis
Mechanism, objective, experiment, limitation, and impact are covered [E1]. Source: https://arxiv.org/abs/2605.00001

## Paper mechanisms
The method uses a cache-aware mechanism [E1]. Source: https://arxiv.org/abs/2605.00001

## Math or objective details
The objective is operational rather than formal in this evidence [E1]. Source: https://arxiv.org/abs/2605.00001

## Experiments and limits
The benchmark and limitation are described in the evidence [E1]. Source: https://arxiv.org/abs/2605.00001

## Why it matters
The impact is practical for inference systems [E1]. Source: https://arxiv.org/abs/2605.00001
"""
    errors = validate_body(body, pack)
    assert not any(error.startswith("missing_source_link:") for error in errors)


def test_arxiv_source_link_accepts_scheme_and_version_variants():
    pack = _pack()
    first = pack.ranked_items[0].item
    first.canonical_url = "http://arxiv.org/abs/2605.00001v1"
    pack.ranked_items = [pack.ranked_items[0]]
    pack.chunks = [chunk for chunk in pack.chunks if chunk.item_id == first.item_id]
    body = """
## Technical thesis
Mechanism, objective, experiment, limitation, and impact are covered [E1]. Source: https://arxiv.org/abs/2605.00001

## Paper mechanisms
The method uses a cache-aware mechanism [E1]. Source: https://arxiv.org/abs/2605.00001

## Math or objective details
The objective is operational rather than formal in this evidence [E1]. Source: https://arxiv.org/abs/2605.00001

## Experiments and limits
The benchmark and limitation are described in the evidence [E1]. Source: https://arxiv.org/abs/2605.00001

## Why it matters
The impact is practical for inference systems [E1]. Source: https://arxiv.org/abs/2605.00001
"""
    assert not any(error.startswith("missing_source_link:") for error in validate_body(body, pack))


def test_evidence_chunks_are_typed():
    pack = _pack()
    types = {chunk.evidence_type for chunk in pack.chunks}
    assert "mechanism" in types
    assert "experiment" in types
    assert "impact" in types


def test_generic_roundup_without_required_sections_fails():
    pack = _pack()
    body = """
## What mattered today

Generic roundup text with a source [E1]. Source: https://arxiv.org/abs/2605.00001
"""
    errors = validate_body(body, pack)
    assert "generic_roundup_structure" in errors or any(e.startswith("missing_daily_section:") for e in errors)


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
    assert "CITED_SOURCE_URLS" in calls[1]
    assert "Technical thesis" in calls[1]
    assert "99.9%" not in calls[1]


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
