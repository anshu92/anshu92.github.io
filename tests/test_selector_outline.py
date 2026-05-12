from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from blogpipe.evidence import build_daily_pack
from blogpipe.models import DailyOutline, EvidencePack, RankedItem, SelectionResult, SourceItem, TopicScores
from blogpipe.outline import OutlineError, generate_daily_outline, validate_outline
from blogpipe.selector import SelectionError, select_daily_items
from blogpipe.writer import _frontmatter


class FakeLLM:
    def __init__(self, text: str):
        self.text = text

    def complete(self, *, system, user, max_tokens=None):
        return self.text


class RecordingLLM(FakeLLM):
    def __init__(self, text: str):
        super().__init__(text)
        self.last_user = ""

    def complete(self, *, system, user, max_tokens=None):
        self.last_user = user
        return self.text


def _fixture_ranked() -> list[RankedItem]:
    data = json.loads(Path("tests/fixtures/source_items.json").read_text())
    items = TypeAdapter(list[SourceItem]).validate_python(data["items"])
    return [
        RankedItem(
            item=item.normalized(),
            topic_scores=TopicScores(llm=0.5, mle=0.5, aec=0.4, matched_keywords=["language model"]),
            daily_score=0.8,
            deep_dive_score=0.7,
            quality_signals={"technical_depth": 0.7, "practical_impact": 0.5},
        )
        for item in items
    ]


def _ranked_item(item_id: str, *, text: str, score: float = 0.7) -> RankedItem:
    return RankedItem(
        item=SourceItem(
            item_id=item_id,
            canonical_url=f"https://example.com/{item_id}",
            source_kind="paper",
            source_name="arxiv",
            source_tier=1,
            title=text,
            published_at=datetime(2026, 5, 12, tzinfo=timezone.utc),
            abstract_or_excerpt=text,
            tags=["paper"],
        ),
        topic_scores=TopicScores(llm=0.5, mle=0.5, aec=0.5, matched_keywords=["benchmark"]),
        daily_score=score,
        deep_dive_score=score,
        quality_signals={"technical_depth": 0.7, "practical_impact": 0.5},
    )


def test_selector_prefers_aec_2d_document_candidates():
    ranked = [
        _ranked_item("generic", text="Generic language model benchmark with serving latency", score=0.9),
        _ranked_item("aec-doc", text="AEC drawing sheet PDF layout OCR foundation model benchmark", score=0.7),
        _ranked_item("rag", text="RAG evaluation pipeline benchmark for document retrieval", score=0.69),
        _ranked_item("agent", text="Agent tool use evaluation benchmark and failure modes", score=0.68),
        _ranked_item("cad", text="CAD BIM IFC construction document multimodal model", score=0.67),
        _ranked_item("systems", text="Distributed inference throughput monitoring deployment", score=0.66),
    ]
    fake = {
        "selected_item_ids": ["aec-doc", "cad", "rag", "agent", "systems"],
        "items": [
            {"item_id": "aec-doc", "relevance_label": "direct_aec_2d", "reason": "direct", "suggested_tags": ["aec", "document-ai"]},
            {"item_id": "cad", "relevance_label": "direct_aec_2d", "reason": "direct", "suggested_tags": ["cad"]},
            {"item_id": "rag", "relevance_label": "aec_adjacent", "reason": "adjacent", "suggested_tags": ["mle"]},
            {"item_id": "agent", "relevance_label": "ml_engineering", "reason": "adjacent", "suggested_tags": ["llm"]},
            {"item_id": "systems", "relevance_label": "ml_engineering", "reason": "adjacent", "suggested_tags": ["mle"]},
        ],
        "suggested_tags": ["aec", "document-ai"],
    }
    selected, result = select_daily_items(ranked, llm=FakeLLM(json.dumps(fake)))
    selected_ids = {r.item.item_id for r in selected}
    assert "aec-doc" in selected_ids
    assert "generic" not in selected_ids
    assert result.items[0].relevance_label == "direct_aec_2d"


def test_selector_malformed_json_blocks_publication():
    with pytest.raises(SelectionError):
        select_daily_items(_fixture_ranked(), llm=FakeLLM("not json"))


def test_selector_uses_all_paper_titles_without_score_fields(monkeypatch):
    monkeypatch.setenv("BLOGPIPE_SELECTOR_CANDIDATES", "2")
    ranked = [
        _ranked_item("p1", text="Generic serving benchmark"),
        _ranked_item("p2", text="Another systems benchmark"),
        _ranked_item("p3", text="AEC drawing PDF OCR foundation model"),
    ]
    fake = {
        "selected_item_ids": ["p3", "p2", "p1"],
        "items": [{"item_id": "p3", "relevance_label": "direct_aec_2d", "reason": "best fit", "suggested_tags": ["aec"]}],
        "suggested_tags": ["aec"],
    }
    llm = RecordingLLM(json.dumps(fake))
    selected, _result = select_daily_items(ranked, llm=llm)
    assert selected[0].item.item_id == "p3"
    assert '"item_id": "p3"' in llm.last_user
    assert "daily_score" not in llm.last_user
    assert "rank_reason" not in llm.last_user


def test_outline_accepts_generated_headings_and_required_intents():
    ranked = _fixture_ranked()
    pack = build_daily_pack(ranked[:5])
    outline = DailyOutline.model_validate_json(Path("tests/fixtures/fake_outline.json").read_text())
    assert validate_outline(outline, pack) == []


def test_outline_missing_required_intent_fails():
    ranked = _fixture_ranked()
    pack = build_daily_pack(ranked[:5])
    outline = DailyOutline(
        title="Thin outline",
        sections=[{"heading": "Only mechanisms", "intent": "mechanism method architecture", "evidence_ids": ["E1"], "word_budget": 1200}],
    )
    errors = validate_outline(outline, pack)
    assert "missing_outline_intent:autodesk_relevance" in errors
    assert any(error.startswith("missing_outline_intent:") for error in errors)


def test_generate_outline_malformed_json_blocks_publication():
    selection = SelectionResult(selected_item_ids=["arxiv:2605.00001"])
    with pytest.raises(OutlineError):
        generate_daily_outline(build_daily_pack(_fixture_ranked()[:5]), selection=selection, llm=FakeLLM("[]"))


def test_frontmatter_tags_are_dynamic():
    ranked = [
        _ranked_item("plain", text="Language model serving latency benchmark for monitoring production systems", score=0.8)
    ]
    pack = EvidencePack(kind="daily", ranked_items=ranked, chunks=[])
    outline = DailyOutline(title="Plain ML systems")
    frontmatter = _frontmatter("Plain ML systems", "daily", pack, outline=outline, selection=None, body="serving latency monitoring")
    assert 'tags: ["research-radar", "llm", "mle"]' in frontmatter
    assert '"aec"' not in frontmatter
