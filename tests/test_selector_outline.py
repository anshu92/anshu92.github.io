from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from blogpipe.evidence import build_daily_pack
from blogpipe.llm import LLMClient
from blogpipe.models import DailyOutline, EvidencePack, RankedItem, SelectionResult, SourceItem, TopicScores
from blogpipe.outline import OutlineError, generate_daily_outline, validate_outline
from blogpipe.selector import SelectionError, _selector_system, _selector_user, select_daily_items
from blogpipe.writer import _frontmatter


class FakeLLM:
    def __init__(self, text: str):
        self.text = text

    def complete(self, *, system, user, max_tokens=None):
        return self.text


class RaisingLLM:
    def complete(self, *, system, user, max_tokens=None):
        raise RuntimeError("http_status:404")


class RecordingLLM(FakeLLM):
    def __init__(self, text: str):
        super().__init__(text)
        self.last_user = ""

    def complete(self, *, system, user, max_tokens=None):
        self.last_user = user
        return self.text


class TaskRecordingLLM(LLMClient):
    def __init__(self, responses: list[str]):
        super().__init__()
        self.responses = list(responses)
        self.tasks: list[str | None] = []

    def complete(self, *, system, user, max_tokens=None, task=None):
        self.tasks.append(task)
        assert self.responses
        return self.responses.pop(0)


def _fixture_ranked() -> list[RankedItem]:
    data = json.loads(Path("tests/fixtures/source_items.json").read_text())
    items = TypeAdapter(list[SourceItem]).validate_python(data["items"])
    return [
        RankedItem(
            item=item.normalized(),
            topic_scores=TopicScores(
                ml_engineering=0.5,
                applied_research=0.5,
                aec=0.4,
                matched_keywords=["language model"],
            ),
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
        topic_scores=TopicScores(
            ml_engineering=0.5,
            applied_research=0.5,
            aec=0.5,
            matched_keywords=["benchmark"],
        ),
        daily_score=score,
        deep_dive_score=score,
        quality_signals={"technical_depth": 0.7, "practical_impact": 0.5},
    )


def test_selector_applies_structured_priority_labels():
    ranked = [
        _ranked_item("generic", text="Generic language model benchmark with serving latency", score=0.9),
        _ranked_item("engineering", text="PyTorch CUDA kernel optimization for distributed LLM serving", score=0.88),
        _ranked_item("applied", text="RAG evaluation pipeline benchmark for document retrieval", score=0.69),
        _ranked_item("agent", text="Agent tool use evaluation benchmark and failure modes", score=0.68),
        _ranked_item("aec-doc", text="AEC drawing sheet PDF layout OCR foundation model benchmark", score=0.67),
        _ranked_item("systems", text="Distributed inference throughput monitoring deployment", score=0.66),
    ]
    fake = {
        "selected_item_ids": ["engineering", "systems", "applied", "agent", "aec-doc"],
        "items": [
            {"item_id": "engineering", "role": "primary", "relevance_label": "ml_engineering", "reason": "systems", "suggested_tags": ["mle"]},
            {"item_id": "systems", "role": "primary", "relevance_label": "ml_engineering", "reason": "serving", "suggested_tags": ["mle"]},
            {"item_id": "applied", "role": "primary", "relevance_label": "applied_research", "reason": "eval", "suggested_tags": ["llm"]},
            {"item_id": "agent", "role": "primary", "relevance_label": "applied_research", "reason": "agents", "suggested_tags": ["llm"]},
            {"item_id": "aec-doc", "role": "supporting", "relevance_label": "aec", "reason": "aec lens", "suggested_tags": ["aec"]},
        ],
        "suggested_tags": ["mle", "llm"],
    }
    selected, result = select_daily_items(ranked, llm=FakeLLM(json.dumps(fake)))
    selected_ids = {r.item.item_id for r in selected}
    assert "engineering" in selected_ids
    assert "generic" not in selected_ids
    assert result.items[0].relevance_label == "ml_engineering"
    assert sum(1 for r in selected if r.item.extra.get("selector_role") == "primary") == 4
    assert sum(1 for r in selected if r.item.extra.get("selector_role") == "supporting") <= 2


def test_selector_prompt_prioritizes_scaled_training_howto():
    ranked = [
        _ranked_item(
            "train",
            text=(
                "FSDP tensor parallel activation checkpointing all-reduce NCCL "
                "data pipeline throughput benchmark for distributed LLM training"
            ),
        )
    ]
    system = _selector_system()
    user = _selector_user(ranked)
    assert "scaled LLM training how-to first" in system
    assert "FSDP/ZeRO" in system
    assert "NCCL/all-reduce" in system
    assert "training_howto_value" in user
    assert "sharding/parallelism" in user


def test_selector_malformed_json_blocks_publication():
    with pytest.raises(SelectionError):
        select_daily_items(_fixture_ranked(), llm=FakeLLM("not json"))


def test_selector_salvages_selected_ids_from_truncated_json():
    ranked = [
        _ranked_item("p1", text="Generic serving benchmark"),
        _ranked_item("p2", text="AEC drawing PDF OCR foundation model"),
        _ranked_item("p3", text="Another systems benchmark"),
    ]
    raw = """```json
{
  "selected_item_ids": ["p2", "p1",
  "items": [
"""
    selected, result = select_daily_items(ranked, llm=FakeLLM(raw))
    assert [r.item.item_id for r in selected[:2]] == ["p2", "p1"]
    assert result.selected_item_ids[:2] == ["p2", "p1"]
    assert result.items[0].role == "primary"


def test_selector_recovers_truncated_object_with_scores():
    ranked = [
        _ranked_item("p1", text="Generic serving benchmark"),
        _ranked_item("p2", text="AEC drawing PDF OCR foundation model"),
        _ranked_item("p3", text="Document layout benchmark with evaluation"),
    ]
    raw = """{
  "selected_item_ids": ["p2", "p3"],
  "items": [
    {
      "item_id": "p2",
      "role": "primary",
      "relevance_label": "aec",
      "scores": {
        "aec_transfer_value": 0.95,
        "transferable_mechanism": 0.8,
        "experiment_strength": 0.7,
        "engineering_actionability": 0.9,
        "novelty": 0.9
      }
"""
    selected, result = select_daily_items(ranked, llm=FakeLLM(raw))
    assert [item.item.item_id for item in selected[:2]] == ["p2", "p3"]
    assert result.items[0].role == "primary"
    assert result.items[0].scores["novelty"] == 0.9


def test_selector_accepts_pythonish_dict_output():
    ranked = [
        _ranked_item("p1", text="Generic serving benchmark"),
        _ranked_item("p2", text="AEC drawing PDF OCR foundation model"),
    ]
    raw = "{'selected_item_ids': ['p2'], 'items': [{'item_id': 'p2', 'role': 'primary'}], 'suggested_tags': ['aec']}"
    selected, result = select_daily_items(ranked, llm=FakeLLM(raw))
    assert selected[0].item.item_id == "p2"
    assert result.suggested_tags == ["aec"]


def test_selector_ignores_misplaced_nonnumeric_score_fields():
    ranked = [
        _ranked_item("p1", text="Generic serving benchmark"),
        _ranked_item("p2", text="AEC drawing PDF OCR foundation model"),
    ]
    raw = json.dumps(
        {
            "selected_item_ids": ["p2"],
            "items": [
                {
                    "item_id": "p2",
                    "role": "primary",
                    "relevance_label": "aec",
                    "scores": {"aec_transfer_value": 0.9, "relevance_label": "aec"},
                }
            ],
        }
    )
    selected, result = select_daily_items(ranked, llm=FakeLLM(raw))
    assert selected[0].item.item_id == "p2"
    assert result.items[0].scores == {"aec_transfer_value": 0.9}
    assert result.items[0].relevance_label == "aec"


def test_selector_uses_all_paper_titles_without_score_fields(monkeypatch):
    monkeypatch.setenv("BLOGPIPE_SELECTOR_CANDIDATES", "2")
    ranked = [
        _ranked_item("p1", text="Generic serving benchmark"),
        _ranked_item("p2", text="Another systems benchmark"),
        _ranked_item("p3", text="AEC drawing PDF OCR foundation model"),
    ]
    fake = {
        "selected_item_ids": ["p3", "p2", "p1"],
        "items": [{"item_id": "p3", "role": "primary", "relevance_label": "ml_engineering", "reason": "best fit", "suggested_tags": ["mle"]}],
        "suggested_tags": ["mle"],
    }
    llm = RecordingLLM(json.dumps(fake))
    selected, _result = select_daily_items(ranked, llm=llm)
    assert selected[0].item.item_id == "p3"
    assert '"item_id": "p3"' in llm.last_user
    assert "daily_score" not in llm.last_user
    assert "rank_reason" not in llm.last_user


def test_selector_uses_selector_task_for_llm_client():
    ranked = [
        _ranked_item("p1", text="Generic serving benchmark"),
        _ranked_item("p2", text="AEC drawing PDF OCR foundation model"),
    ]
    fake = {
        "selected_item_ids": ["p2", "p1"],
        "items": [{"item_id": "p2", "role": "primary", "relevance_label": "ml_engineering", "reason": "best fit", "suggested_tags": ["mle"]}],
        "suggested_tags": ["aec"],
    }
    llm = TaskRecordingLLM([json.dumps(fake)])
    select_daily_items(ranked, llm=llm)
    assert llm.tasks == ["selector"]


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


def test_outline_rejects_generic_corporate_headings():
    ranked = _fixture_ranked()
    pack = build_daily_pack(ranked[:5])
    outline = DailyOutline(
        title="Generic",
        angle="Generic angle",
        sections=[
            {"heading": "Navigating the Future of AI", "intent": "technical thesis angle framing", "evidence_ids": ["E1"], "word_budget": 300},
            {"heading": "Mechanisms", "intent": "mechanism method architecture pipeline", "evidence_ids": ["E1"], "word_budget": 300},
            {"heading": "Objectives", "intent": "math objective metric optimization", "evidence_ids": ["E1"], "word_budget": 300},
            {"heading": "Experiments", "intent": "experiments evidence benchmark evaluation", "evidence_ids": ["E1"], "word_budget": 300},
            {"heading": "Limits", "intent": "limitations caveat failure risk tradeoff", "evidence_ids": ["E1"], "word_budget": 300},
            {"heading": "Comparison", "intent": "cross-paper synthesis compare contrast tradeoff", "evidence_ids": ["E1"], "word_budget": 300},
            {"heading": "Adoption", "intent": "impact engineering production practical Autodesk AEC document", "evidence_ids": ["E1"], "word_budget": 300},
        ],
    )
    assert any(error.startswith("generic_outline_heading:") for error in validate_outline(outline, pack))


def test_outline_rejects_duplicate_primary_focus_without_split_reason():
    ranked = _fixture_ranked()
    pack = build_daily_pack(ranked[:5])
    first_item_id = pack.ranked_items[0].item.item_id
    outline = DailyOutline(
        title="Duplicate primary focus",
        angle="Two overlapping sections should not pass.",
        sections=[
            {"heading": "Thesis", "intent": "technical thesis angle framing Autodesk AEC document relevance", "evidence_ids": ["E1"], "word_budget": 220},
            {"heading": "Mechanism one", "intent": "mechanism method architecture pipeline", "evidence_ids": ["E1"], "word_budget": 220, "focus_item_ids": [first_item_id], "section_role": "primary"},
            {"heading": "Mechanism two", "intent": "math objective metric optimization experiments evidence benchmark evaluation limitations caveat failure risk", "evidence_ids": ["E1"], "word_budget": 220, "focus_item_ids": [first_item_id], "section_role": "primary"},
            {"heading": "Comparison", "intent": "cross-paper synthesis compare contrast tradeoff", "evidence_ids": ["E1"], "word_budget": 220},
            {"heading": "Adoption", "intent": "impact engineering production practical Autodesk AEC document", "evidence_ids": ["E1"], "word_budget": 220},
        ],
    )
    assert f"duplicate_primary_outline_focus:{first_item_id}" in validate_outline(outline, pack)


def test_outline_rejects_primary_section_with_nonprimary_focus():
    ranked = _fixture_ranked()
    ranked[0].item.extra["selector_role"] = "primary"
    ranked[1].item.extra["selector_role"] = "supporting"
    pack = build_daily_pack(ranked[:5])
    supporting_item_id = pack.ranked_items[1].item.item_id
    outline = DailyOutline(
        title="Scope drift",
        angle="Supporting paper should not be elevated.",
        sections=[
            {"heading": "Thesis", "intent": "technical thesis angle framing Autodesk AEC document relevance", "evidence_ids": ["E1"], "word_budget": 220},
            {"heading": "Primary", "intent": "mechanism method architecture pipeline", "evidence_ids": ["E1"], "word_budget": 220, "focus_item_ids": [supporting_item_id], "section_role": "primary"},
            {"heading": "Objectives", "intent": "math objective metric optimization", "evidence_ids": ["E1"], "word_budget": 220},
            {"heading": "Experiments", "intent": "experiments evidence benchmark evaluation limitations caveat failure risk", "evidence_ids": ["E1"], "word_budget": 220},
            {"heading": "Adoption", "intent": "cross-paper synthesis compare contrast tradeoff impact engineering production practical Autodesk AEC document", "evidence_ids": ["E1"], "word_budget": 220},
        ],
    )
    assert "primary_section_uses_nonprimary_focus" in validate_outline(outline, pack)


def test_training_focused_outline_requires_howto_intent():
    ranked = [
        _ranked_item(
            "training",
            text=(
                "FSDP tensor parallel distributed training with activation checkpointing, "
                "NCCL all-reduce communication, microbatch throughput, checkpointing, "
                "data pipeline pressure, GPU utilization, profiling, and benchmark ablations."
            ),
        )
    ]
    pack = build_daily_pack(ranked)
    sections = [
        {"heading": "Thesis", "intent": "technical thesis angle framing Autodesk AEC document relevance", "evidence_ids": ["E1"], "word_budget": 220},
        {"heading": "Mechanism", "intent": "mechanism method architecture pipeline", "evidence_ids": ["E1"], "word_budget": 220},
        {"heading": "Objective", "intent": "math objective metric optimization", "evidence_ids": ["E1"], "word_budget": 220},
        {"heading": "Experiment", "intent": "experiments evidence benchmark evaluation ablation", "evidence_ids": ["E1"], "word_budget": 220},
        {"heading": "Limits", "intent": "limitations caveat failure risk tradeoff", "evidence_ids": ["E1"], "word_budget": 220},
        {"heading": "Impact", "intent": "cross-paper synthesis compare contrast impact engineering production practical", "evidence_ids": ["E1"], "word_budget": 220},
    ]
    outline = DailyOutline(title="Training systems", angle="Training stack decisions matter.", sections=sections)
    assert "missing_outline_training_howto" in validate_outline(outline, pack)

    sections[1] = {
        "heading": "FSDP sharding and NCCL profiling as the training runbook",
        "intent": "mechanism method architecture pipeline distributed training sharding communication nccl profiling throughput",
        "evidence_ids": ["E1"],
        "word_budget": 220,
    }
    repaired = DailyOutline(title="Training systems", angle="Training stack decisions matter.", sections=sections)
    assert "missing_outline_training_howto" not in validate_outline(repaired, pack)


def test_generate_outline_malformed_json_raises():
    selection = SelectionResult(selected_item_ids=["arxiv:2605.00001"])
    pack = build_daily_pack(_fixture_ranked()[:5])
    with pytest.raises(OutlineError, match="outline_invalid"):
        generate_daily_outline(pack, selection=selection, llm=FakeLLM("[]"))


def test_generate_outline_accepts_pythonish_dict_output():
    selection = SelectionResult(selected_item_ids=["arxiv:2605.00001"])
    pack = build_daily_pack(_fixture_ranked()[:5])
    raw = "{'title': 'Pythonish outline', 'angle': 'AEC document systems need measurable evidence.', 'sections': ["
    raw += "{'heading': 'Thesis boundary', 'intent': 'technical thesis angle framing Autodesk AEC document relevance', 'evidence_ids': ['E1'], 'word_budget': 300},"
    raw += "{'heading': 'Mechanism boundary', 'intent': 'mechanism method architecture pipeline', 'evidence_ids': ['E1'], 'word_budget': 300},"
    raw += "{'heading': 'Objective boundary', 'intent': 'math objective metric optimization', 'evidence_ids': ['E1'], 'word_budget': 300},"
    raw += "{'heading': 'Experiment boundary', 'intent': 'experiments evidence benchmark evaluation ablation', 'evidence_ids': ['E1'], 'word_budget': 300},"
    raw += "{'heading': 'Cross paper boundary', 'intent': 'cross-paper synthesis compare contrast tradeoff limitations caveat failure risk', 'evidence_ids': ['E1'], 'word_budget': 300},"
    raw += "{'heading': 'Adoption boundary', 'intent': 'impact engineering production practical Autodesk AEC document', 'evidence_ids': ['E1'], 'word_budget': 300}"
    raw += "], 'suggested_tags': ['document-ai']}"
    outline = generate_daily_outline(pack, selection=selection, llm=FakeLLM(raw))
    assert outline.title == "Pythonish outline"
    assert validate_outline(outline, pack) == []


def test_generate_outline_llm_failure_raises():
    selection = SelectionResult(selected_item_ids=["arxiv:2605.00001"])
    pack = build_daily_pack(_fixture_ranked()[:5])
    with pytest.raises(OutlineError, match="outline_invalid"):
        generate_daily_outline(pack, selection=selection, llm=RaisingLLM())


def test_generate_outline_uses_outline_then_repair_tasks():
    ranked = _fixture_ranked()
    pack = build_daily_pack(ranked[:5])
    selection = SelectionResult(selected_item_ids=[ranked[0].item.item_id])
    valid_outline = Path("tests/fixtures/fake_outline.json").read_text()
    llm = TaskRecordingLLM(["[]", valid_outline])
    outline = generate_daily_outline(pack, selection=selection, llm=llm)
    assert isinstance(outline, DailyOutline)
    assert llm.tasks == ["outline", "outline_repair"]


def test_frontmatter_tags_are_dynamic():
    ranked = [
        _ranked_item("plain", text="Language model serving latency benchmark for monitoring production systems", score=0.8)
    ]
    pack = EvidencePack(kind="daily", ranked_items=ranked, chunks=[])
    outline = DailyOutline(title="Plain ML systems")
    frontmatter = _frontmatter(
        "Plain ML systems",
        "daily",
        pack,
        outline=outline,
        selection=None,
        body="language model serving latency monitoring",
    )
    assert 'tags: ["research-radar", "llm", "mle"]' in frontmatter
    assert '"aec"' not in frontmatter


def test_frontmatter_tags_llm_training_when_body_supports_it():
    ranked = [
        _ranked_item(
            "train",
            text="FSDP tensor parallel activation checkpointing benchmark for distributed training",
            score=0.8,
        )
    ]
    pack = EvidencePack(kind="daily", ranked_items=ranked, chunks=[])
    frontmatter = _frontmatter(
        "Training systems",
        "daily",
        pack,
        outline=None,
        selection=None,
        body="FSDP tensor parallel distributed training uses activation checkpointing and profiling.",
    )
    assert '"llm-training"' in frontmatter
    assert '"mle"' in frontmatter


def test_frontmatter_tags_ignore_metadata_without_body_support():
    ranked = [
        _ranked_item("cad", text="AEC BIM CAD foundation model benchmark", score=0.8)
    ]
    ranked[0].item.tags = ["paper", "aec", "bim", "cad", "foundation-models"]
    pack = EvidencePack(kind="daily", ranked_items=ranked, chunks=[])
    outline = DailyOutline(title="Metadata-only tag drift", suggested_tags=["aec", "bim", "cad", "foundation-models"])
    selection = SelectionResult(suggested_tags=["aec", "bim", "cad", "foundation-models"])
    frontmatter = _frontmatter("Metadata-only tag drift", "daily", pack, outline=outline, selection=selection, body="Latency and monitoring matter for deployment.")
    assert 'tags: ["research-radar", "mle"]' in frontmatter
    assert '"aec"' not in frontmatter
    assert '"bim"' not in frontmatter
    assert '"cad"' not in frontmatter
    assert '"foundation-models"' not in frontmatter


def test_frontmatter_specialized_tags_require_body_support():
    ranked = [_ranked_item("doc", text="Generic benchmark", score=0.8)]
    pack = EvidencePack(kind="daily", ranked_items=ranked, chunks=[])
    body = (
        "This document intelligence pipeline evaluates drawing sheet layout preservation for AEC workflows. "
        "A BIM-aware CAD prototype connects CadQuery evidence to deployment validation. "
        "The multimodal foundation model framing remains explicit."
    )
    frontmatter = _frontmatter("Supported tags", "daily", pack, outline=None, selection=None, body=body)
    assert '"aec"' in frontmatter
    assert '"document-ai"' in frontmatter
    assert '"cad"' in frontmatter
    assert '"bim"' in frontmatter
    assert '"foundation-models"' in frontmatter
    assert '"multimodal"' in frontmatter


def test_non_openrouter_endpoint_can_fall_back_to_openrouter(monkeypatch):
    monkeypatch.setenv("BLOGPIPE_LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")
    monkeypatch.setenv("BLOGPIPE_LLM_API_KEY", "gemini-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    monkeypatch.delenv("BLOGPIPE_LLM_MODEL", raising=False)
    monkeypatch.delenv("BLOGPIPE_MODEL", raising=False)
    monkeypatch.setenv(
        "BLOGPIPE_LLM_CHAIN_OUTLINE",
        "gemini-3.5-flash,gemini-3.1-flash-lite,openrouter/free",
    )
    client = LLMClient()
    chain = client._model_chain("outline")
    assert chain[:3] == ["gemini-3.5-flash", "gemini-3.1-flash-lite", "openrouter/free"]
    assert "qwen/qwen3-next-80b-a3b-instruct:free" in chain
    assert client._endpoint_for_model("gemini-3.5-flash") == (
        "https://generativelanguage.googleapis.com/v1beta/openai",
        "gemini-key",
    )
    assert client._endpoint_for_model("openrouter/free") == ("https://openrouter.ai/api/v1", "openrouter-key")
