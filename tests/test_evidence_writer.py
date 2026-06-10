from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import TypeAdapter

from blogpipe.evidence import build_daily_pack
from blogpipe import evidence
from blogpipe.llm import LLMClient
from blogpipe import memory
from blogpipe.models import DailyOutline, SelectionResult, SourceItem
from blogpipe.score import daily_shortlist, rank_items
from blogpipe.writer import (
    _canonical_title_from_body,
    _llm_quality_errors,
    _validate_final_title,
    validate_body,
    write_daily,
    write_deep_dive,
)


def _pack():
    data = json.loads(Path("tests/fixtures/source_items.json").read_text())
    items = TypeAdapter(list[SourceItem]).validate_python(data["items"])
    ranked = daily_shortlist(rank_items(items, max_age_hours=None))
    return build_daily_pack(ranked)


def _selection() -> SelectionResult:
    return SelectionResult.model_validate_json(Path("tests/fixtures/fake_selector.json").read_text())


def _outline() -> DailyOutline:
    return DailyOutline.model_validate_json(Path("tests/fixtures/fake_outline.json").read_text())


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
    result = write_daily(pack, outline=_outline(), selection=_selection(), llm=LLMClient(), dry_run=True)
    assert result.ok, result.errors
    assert result.path.startswith("reports/")
    rendered = (tmp_path / result.path).read_text()
    assert "draft: true" in rendered
    assert 'title: "Research Radar: Evidence Boundaries for AEC Foundation Models"' in rendered
    assert result.title == "Research Radar: Evidence Boundaries for AEC Foundation Models"


def test_final_h1_is_canonical_title():
    body = "# Final engineering title\n\n## Mechanism\nGrounded."
    assert _canonical_title_from_body(body) == "Final engineering title"
    assert _validate_final_title(body) == []


def test_missing_or_multiple_final_h1_fails():
    assert _validate_final_title("## No headline") == ["missing_final_h1"]
    assert _validate_final_title("# One\n\n# Two") == ["multiple_final_h1:2"]


def test_truncated_fallback_style_h1_fails():
    body = "# Research Radar: BenchCAD: A Comprehensive, Industry-Standard Benchmark for Programmati"
    assert _validate_final_title(body) == ["truncated_or_fallback_final_h1"]


def test_unsupported_number_fails():
    pack = _pack()
    body = Path("tests/fixtures/fake_daily.md").read_text() + "\n\nInvented result: 99.9% faster. [E1]"
    errors = validate_body(body, pack, outline=_outline())
    assert "unsupported_number:99.9%" in errors


def test_plain_reference_numbers_are_not_metric_claims():
    pack = _pack()
    body = Path("tests/fixtures/fake_daily.md").read_text() + "\n\nRelated citation marker [49] is not a metric. [E1]"
    errors = validate_body(body, pack, outline=_outline())
    assert "unsupported_number:49" not in errors


def test_arxiv_ids_are_not_metric_claims():
    pack = _pack()
    body = Path("tests/fixtures/fake_daily.md").read_text() + "\n\nArXiv identifier 2605.10195 is not a metric claim. [E1]"
    errors = validate_body(body, pack, outline=_outline())
    assert "unsupported_number:2605.10195" not in errors


def test_missing_source_link_fails():
    pack = _pack()
    body = "Grounded claim [E1] without the source URL."
    errors = validate_body(body, pack, outline=_outline())
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
    errors = validate_body(body, pack, outline=_outline())
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
    assert len(types) >= 2


def test_daily_pack_builds_structured_evidence_cards():
    pack = _pack()
    assert pack.evidence_cards
    first = pack.evidence_cards[0]
    assert first.role == "primary"
    assert first.problem
    assert first.mechanism != "not found in evidence"
    assert first.evidence_ids["mechanism"]
    assert first.paper_supported_claim != "not found in evidence"
    assert first.paper_supported_limit
    assert first.open_research_question.startswith("Open question:")
    assert any(card.math_or_objective == "not found in evidence" for card in pack.evidence_cards)
    assert any("AEC" in card.transfer_hypothesis or "document" in card.transfer_hypothesis for card in pack.evidence_cards)


def test_generic_roundup_without_required_sections_fails():
    pack = _pack()
    body = """
## What mattered today

Generic roundup text with a source [E1]. Source: https://arxiv.org/abs/2605.00001
"""
    errors = validate_body(body, pack, outline=_outline())
    assert "generic_roundup_structure" in errors or any(e.startswith("missing_daily_concept:") for e in errors)


def test_generic_corporate_prose_fails_llm_quality_review(monkeypatch):
    monkeypatch.setenv("BLOGPIPE_MIN_SIGNAL_SCORE", "0.75")
    review = {
        "pass": False,
        "scores": {
            "technical_specificity": 0.41,
            "engineering_judgment": 0.33,
            "synthesis": 0.29,
            "noise_control": 0.22,
            "primary_depth": 0.38,
        },
        "errors": ["generic corporate prose", "insufficient engineering judgment"],
        "examples": ["paving the way for a holistic strategy"],
        "notes": "Too abstract to publish.",
    }
    errors = _llm_quality_errors(review)
    assert "llm_low_signal:technical_specificity:0.41/0.75" in errors
    assert "llm_low_signal:engineering_judgment:0.33/0.75" in errors
    assert "llm_quality:generic corporate prose" in errors


def test_paper_by_paper_summary_without_synthesis_fails_llm_quality_review(monkeypatch):
    monkeypatch.setenv("BLOGPIPE_MIN_SIGNAL_SCORE", "0.75")
    review = {
        "pass": False,
        "scores": {
            "technical_specificity": 0.80,
            "engineering_judgment": 0.77,
            "synthesis": 0.44,
            "noise_control": 0.88,
            "primary_depth": 0.73,
        },
        "errors": ["paper-by-paper abstract summaries without cross-paper insight"],
        "examples": ["The sections above list each paper in sequence."],
        "notes": "Needs synthesis rather than serial summaries.",
    }
    errors = _llm_quality_errors(review)
    assert "llm_low_signal:synthesis:0.44/0.75" in errors
    assert "llm_low_signal:primary_depth:0.73/0.75" in errors
    assert any("paper-by-paper abstract summaries" in error for error in errors)


def test_evidence_discipline_and_redundancy_scores_fail_llm_quality_review(monkeypatch):
    monkeypatch.setenv("BLOGPIPE_MIN_SIGNAL_SCORE", "0.75")
    review = {
        "pass": False,
        "scores": {
            "technical_specificity": 0.88,
            "engineering_judgment": 0.80,
            "synthesis": 0.78,
            "noise_control": 0.92,
            "primary_depth": 0.81,
            "evidence_discipline": 0.42,
            "section_nonredundancy": 0.55,
            "experiment_detail": 0.51,
        },
        "errors": ["overstated_transfer_claim", "duplicate_primary_section"],
        "examples": ["creates a principled stack for robust AEC intelligence"],
        "notes": "Claims outrun evidence.",
        "top_editorial_failure": "Speculative transfer is phrased as a demonstrated system.",
    }
    errors = _llm_quality_errors(review)
    assert "llm_low_signal:evidence_discipline:0.42/0.75" in errors
    assert "llm_low_signal:section_nonredundancy:0.55/0.75" in errors
    assert "llm_low_signal:experiment_detail:0.51/0.75" in errors
    assert "llm_quality:overstated_transfer_claim" in errors


def test_engineering_focus_and_title_alignment_fail_llm_quality_review(monkeypatch):
    monkeypatch.setenv("BLOGPIPE_MIN_SIGNAL_SCORE", "0.75")
    review = {
        "pass": False,
        "scores": {
            "technical_specificity": 0.88,
            "engineering_judgment": 0.46,
            "synthesis": 0.81,
            "noise_control": 0.91,
            "primary_depth": 0.84,
            "evidence_discipline": 0.82,
            "section_nonredundancy": 0.86,
            "experiment_detail": 0.80,
        },
        "errors": ["weak_engineering_focus", "title_body_mismatch", "claim_strength_exceeds_evidence"],
        "examples": ["This defines the reliability envelope for production AEC systems."],
        "notes": "Mechanisms are explained, but the engineering decisions stay vague.",
        "top_editorial_failure": "The article promises an operational brief but closes with generic research synthesis.",
    }
    errors = _llm_quality_errors(review)
    assert "llm_low_signal:engineering_judgment:0.46/0.75" in errors
    assert "llm_quality:weak_engineering_focus" in errors
    assert "llm_quality:title_body_mismatch" in errors
    assert "llm_quality:claim_strength_exceeds_evidence" in errors


def test_empty_quality_failure_is_advisory_without_low_scores(monkeypatch):
    monkeypatch.setenv("BLOGPIPE_MIN_SIGNAL_SCORE", "0.75")
    review = {
        "pass": False,
        "scores": {
            "technical_specificity": 0.91,
            "engineering_judgment": 0.88,
            "synthesis": 0.84,
            "noise_control": 0.93,
            "primary_depth": 0.89,
        },
        "errors": [],
        "examples": [],
        "notes": "No actionable blocker provided.",
    }
    assert _llm_quality_errors(review) == []


def test_first_person_autodesk_claim_fails():
    pack = _pack()
    body = (
        Path("tests/fixtures/fake_daily.md").read_text()
        + "\n\nAs a Principal Machine Learning Engineer at Autodesk, my focus is this claim [E1]. "
        "Source: https://arxiv.org/abs/2605.00001\n"
    )
    assert "first_person_autodesk_claim" in validate_body(body, pack, outline=_outline())


def test_single_source_daily_post_fails_even_with_required_sections():
    pack = _pack()
    body = """
## Technical thesis
One item is discussed as the whole radar, which is not enough coverage for a daily technical post [E5].
Source: https://www.research.autodesk.com/blog/bim-digital-twin-controls

## Paper mechanisms
The post describes a workflow mechanism but does not cite any paper evidence [E5].
Source: https://www.research.autodesk.com/blog/bim-digital-twin-controls

## Math or objective details
No paper objective is discussed from the available evidence [E5].
Source: https://www.research.autodesk.com/blog/bim-digital-twin-controls

## Experiments and limits
The limitation is that this is a short single-source discussion [E5].
Source: https://www.research.autodesk.com/blog/bim-digital-twin-controls

## Why it matters
The practical impact is described without connecting the broader paper set [E5].
Source: https://www.research.autodesk.com/blog/bim-digital-twin-controls
"""
    errors = validate_body(body, pack, outline=_outline())
    assert any(error.startswith("insufficient_cited_primary_items:") for error in errors)
    assert any(error.startswith("insufficient_cited_papers:") for error in errors)
    assert any(error.startswith("daily_too_short:") for error in errors)



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
    result = write_daily(pack, outline=_outline(), selection=_selection(), llm=FakeLLM(), dry_run=True)
    assert result.ok
    assert result.repair_attempted
    assert "VALIDATOR_ERRORS" in calls[1]
    assert "LLM_QUALITY_REVIEW" in calls[1]
    assert "RELEVANT_EVIDENCE_CARDS" in calls[1]
    assert "CITED_SOURCE_URLS" in calls[1]
    assert "OUTLINE" in calls[1]
    assert "Document-model reliability starts with measurable failure boundaries" in calls[1]
    assert "99.9%" not in calls[1]


def test_write_daily_auto_sanitizes_unsupported_numbers_with_source_links(monkeypatch, tmp_path):
    pack = _pack()
    outline = _outline()
    selection = _selection()
    body = (
        Path("tests/fixtures/fake_daily.md").read_text()
        + "\n\nThe batch spans 106 tasks across 17 benchmarks and a 900 document slice [E1]. "
        "Source: https://arxiv.org/abs/2605.00001\n"
    )

    class StaticLLM:
        def complete(self, *, system, user, max_tokens=None):
            return body

    _patch_root(monkeypatch, tmp_path)
    result = write_daily(pack, outline=outline, selection=selection, llm=StaticLLM(), dry_run=True)
    assert result.ok, result.errors
    assert "106" not in result.body
    assert "17" not in result.body
    assert "900" not in result.body
    assert "multiple tasks" in result.body or "several tasks" in result.body or "many tasks" in result.body


def test_write_daily_auto_adds_missing_paragraph_source_links(monkeypatch, tmp_path):
    pack = _pack()
    outline = _outline()
    selection = _selection()
    body = re.sub(r"\s+Sources?: https?://[^\n]+", "", Path("tests/fixtures/fake_daily.md").read_text())

    class StaticLLM:
        def complete(self, *, system, user, max_tokens=None):
            return body

    _patch_root(monkeypatch, tmp_path)
    result = write_daily(pack, outline=outline, selection=selection, llm=StaticLLM(), dry_run=True)
    assert result.ok, result.errors
    assert "Source:" in result.body or "Sources:" in result.body
    assert not any(error.startswith("missing_paragraph_source_link:") for error in result.errors)


def test_invalid_daily_blocks_after_failed_repair(monkeypatch, tmp_path):
    pack = _pack()

    class BadLLM:
        def complete(self, *, system, user, max_tokens=None):
            return "Unsupported 99.9% claim [E1]. Source: https://arxiv.org/abs/2605.00001"

    _patch_root(monkeypatch, tmp_path)
    result = write_daily(pack, outline=_outline(), selection=_selection(), llm=BadLLM(), dry_run=False)
    assert not result.ok
    assert result.errors
    assert not list((tmp_path / "content" / "post").glob("*.md"))
    assert list((tmp_path / "reports").glob("*.blocked.json"))


def test_daily_sectionwise_writer_and_editor_with_visual_embeds(monkeypatch, tmp_path):
    pack = _pack()
    outline = _outline()
    selection = _selection()
    _patch_root(monkeypatch, tmp_path)
    section_outputs = []
    for section in outline.sections:
        section_outputs.append(
            f"## {section.heading}\n"
            "Grounded claim [E1]. Source: https://arxiv.org/abs/2605.00001\n"
        )
    final_body = Path("tests/fixtures/fake_daily.md").read_text()
    monkeypatch.setenv("BLOGPIPE_LLM_MAX_CALLS", "20")
    monkeypatch.setenv("BLOGPIPE_SECTIONWISE_DRAFTING", "1")
    monkeypatch.setenv("BLOGPIPE_FAKE_LLM_RESPONSES", json.dumps([*section_outputs, final_body]))
    monkeypatch.setenv(
        "BLOGPIPE_FAKE_QUALITY_RESPONSE",
        json.dumps(
            {
                "pass": True,
                "scores": {
                    "technical_specificity": 0.91,
                    "engineering_judgment": 0.92,
                    "synthesis": 0.90,
                    "noise_control": 0.95,
                    "primary_depth": 0.93,
                },
                "errors": [],
                "examples": [],
                "notes": "publishable",
            }
        ),
    )
    llm = LLMClient()
    result = write_daily(pack, outline=outline, selection=selection, llm=llm, dry_run=True)
    assert result.ok, result.errors
    assert llm.usage.calls >= len(outline.sections) + 1
    assert "```mermaid" not in result.body
    assert "/img/posts/" not in result.body
    assert "source-mix.svg" not in result.body
    assert "topic-mix.svg" not in result.body


def test_deep_dive_sectionwise_writer_and_editor_with_visual_embeds(monkeypatch, tmp_path):
    ranked = _pack().ranked_items[0]
    pack = evidence.build_deep_dive_pack(ranked)
    _patch_root(monkeypatch, tmp_path)
    section_outputs = [
        "## Technical thesis and motivation\nGrounded claim [E1]. Source: https://arxiv.org/abs/2605.00001",
        "## Method walkthrough and mechanism\nGrounded claim [E1]. Source: https://arxiv.org/abs/2605.00001",
        "## Objective or math interpretation\nGrounded claim [E1]. Source: https://arxiv.org/abs/2605.00001",
        "## Experiments and evidence\nGrounded claim [E1]. Source: https://arxiv.org/abs/2605.00001",
        "## Limits and failure modes\nGrounded claim [E1]. Source: https://arxiv.org/abs/2605.00001",
        "## Engineering implications and next actions\nGrounded claim [E1]. Source: https://arxiv.org/abs/2605.00001",
    ]
    final_body = (
        "## Technical thesis and motivation\nGrounded claim [E1]. Source: https://arxiv.org/abs/2605.00001\n\n"
        "## Method walkthrough and mechanism\nGrounded claim [E1]. Source: https://arxiv.org/abs/2605.00001\n\n"
        "## Objective or math interpretation\nGrounded claim [E1]. Source: https://arxiv.org/abs/2605.00001\n\n"
        "## Experiments and evidence\nGrounded claim [E1]. Source: https://arxiv.org/abs/2605.00001\n\n"
        "## Limits and failure modes\nGrounded claim [E1]. Source: https://arxiv.org/abs/2605.00001\n\n"
        "## Engineering implications and next actions\nGrounded claim [E1]. Source: https://arxiv.org/abs/2605.00001\n"
    )
    monkeypatch.setenv("BLOGPIPE_LLM_MAX_CALLS", "20")
    monkeypatch.setenv("BLOGPIPE_SECTIONWISE_DRAFTING", "1")
    monkeypatch.setenv("BLOGPIPE_FAKE_LLM_RESPONSES", json.dumps([*section_outputs, final_body]))
    monkeypatch.setenv(
        "BLOGPIPE_FAKE_QUALITY_RESPONSE",
        json.dumps(
            {
                "pass": True,
                "scores": {
                    "technical_specificity": 0.91,
                    "engineering_judgment": 0.92,
                    "synthesis": 0.90,
                    "noise_control": 0.95,
                    "primary_depth": 0.93,
                },
                "errors": [],
                "examples": [],
                "notes": "publishable",
            }
        ),
    )
    llm = LLMClient()
    result = write_deep_dive(pack, llm=llm, dry_run=True)
    assert result.ok, result.errors
    assert llm.usage.calls >= 7
    assert "```mermaid" not in result.body
    assert "source-mix.svg" not in result.body
    assert "topic-mix.svg" not in result.body
