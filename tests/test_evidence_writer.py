from __future__ import annotations

import json
import re
from datetime import datetime, timezone
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
    _daily_draft_rejection_reason,
    _deterministic_quality_errors,
    _daily_rewrite_user,
    _emergency_daily_draft,
    _llm_quality_errors,
    _quality_floor_guidance,
    _quality_review_user,
    _repair_user,
    _review_if_ready,
    _signal_rubric,
    _validate_final_title,
    _validate_training_howto_concepts,
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


def _training_pack():
    now = datetime(2026, 5, 11, tzinfo=timezone.utc)
    item = SourceItem(
        canonical_url="https://example.com/fsdp-runbook",
        source_kind="paper",
        source_name="arxiv",
        source_tier=1,
        title="FSDP tensor parallel training runbook for scaled LLMs",
        published_at=now,
        abstract_or_excerpt=(
            "The problem is scaled LLM training throughput under memory and communication pressure. "
            "We propose a distributed training method using FSDP sharding, tensor parallelism, activation checkpointing, "
            "NCCL all-reduce communication, microbatch scheduling, and data pipeline throughput controls. "
            "The evaluation reports benchmark ablations for GPU utilization, checkpoint recovery, profiling, and scaling efficiency. "
            "The limitation is that workload balance, checkpoint cadence, and data loader stalls create production risks."
        ),
        tags=["distributed training", "fsdp", "tensor parallel"],
    )
    return build_daily_pack(rank_items([item], now=now, max_age_hours=72))


def _training_outline() -> DailyOutline:
    return DailyOutline(
        title="Research Radar: Scaled LLM training runbooks",
        angle="Principal engineers need training-stack decisions, not generic production prose.",
        sections=[
            {"heading": "Training thesis", "intent": "technical thesis angle framing", "evidence_ids": ["E1"], "word_budget": 80},
            {"heading": "FSDP sharding and parallelism runbook", "intent": "mechanism method architecture distributed training sharding parallelism", "evidence_ids": ["E1"], "word_budget": 80},
            {"heading": "Objective and profiling metrics", "intent": "math objective metric optimization throughput profiling", "evidence_ids": ["E1"], "word_budget": 80},
            {"heading": "Benchmarks and failure modes", "intent": "experiments evidence benchmark evaluation limitation failure risk", "evidence_ids": ["E1"], "word_budget": 80},
            {"heading": "Principal rollout decision", "intent": "impact engineering production practical Autodesk AEC document validation release gate", "evidence_ids": ["E1"], "word_budget": 80},
        ],
    )


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


def test_daily_draft_rejects_obvious_fragments():
    outline = _outline()
    assert _daily_draft_rejection_reason("Short fragment [E1].", outline) == "daily_fragment:3/500"
    assert _daily_draft_rejection_reason(Path("tests/fixtures/fake_daily.md").read_text(), outline) is None


def test_training_howto_concepts_require_stack_bottleneck_and_action():
    thin = "This training post says production matters but never names the technical runbook."
    errors = _validate_training_howto_concepts(thin)
    assert "missing_training_howto:training_stack" in errors
    assert "missing_training_howto:scaling_bottleneck" in errors
    assert "missing_training_howto:principal_action" in errors

    rich = (
        "FSDP sharding and tensor parallelism define the training stack. "
        "Memory pressure, NCCL all-reduce communication, checkpoint recovery, and data pipeline throughput "
        "are the scaling bottlenecks. A principal engineer would profile GPU utilization, benchmark microbatch choices, "
        "validate failure modes, and set a rollout release gate [E1]. Source: https://example.com/fsdp-runbook"
    )
    assert _validate_training_howto_concepts(rich) == []


def test_training_howto_concepts_ignore_uncited_decorative_terms():
    decorative = """
# FSDP tensor parallel checkpoint recovery release gate

FSDP sharding, tensor parallelism, memory pressure, NCCL all-reduce, checkpoint recovery, profiling, and release gates sound concrete here, but this paragraph is not evidence-cited.

The cited paragraph says only that training systems need practical engineering judgment [E1]. Source: https://example.com/fsdp-runbook
"""
    errors = _validate_training_howto_concepts(decorative)
    assert "missing_training_howto:training_stack" in errors
    assert "missing_training_howto:scaling_bottleneck" in errors
    assert "missing_training_howto:principal_action" in errors


def test_training_focused_daily_body_requires_howto_details(monkeypatch):
    monkeypatch.setenv("BLOGPIPE_DAILY_MIN_WORDS", "300")
    pack = _training_pack()
    outline = _training_outline()
    source = "https://example.com/fsdp-runbook"
    body = f"""
# Research Radar: Scaled LLM training runbooks

## Training thesis
The thesis is that training research should be read as an engineering decision record, not a broad production story. The evidence describes a scaled training problem with a mechanism, objective, benchmark, limitation, impact, and practical production relevance, but this draft intentionally keeps the guidance generic. A principal engineer would need more than a statement that the method is useful; the article must explain what operational choice changes and why the claim is evidence-grounded [E1]. Source: {source}

## FSDP sharding and parallelism runbook
This section says the mechanism improves a training pipeline, but it avoids naming the actual stack decision. It does not identify the layout, the execution mode, or the state that moves between devices. That makes the section too thin for a training how-to because a reader cannot convert the paper into an implementation review, an approval plan, or a risk register [E1]. Source: {source}

## Objective and profiling metrics
The objective is framed as better practical efficiency, with metrics that should support optimization and evaluation. The draft mentions measurement and score design, but it still hides the technical levers that would let a team reproduce the result. Without explicit stack choices and resource bottlenecks, the section reads like a generic benchmark note rather than a decision guide [E1]. Source: {source}

## Benchmarks and failure modes
The experiment and benchmark evidence should tell the team what to test, where the limitation sits, and what failure mode could break a launch. This paragraph deliberately stays abstract: it says there are risks, caveats, and tradeoffs, but it does not teach a concrete scaling diagnosis. The article therefore lacks the principal-level specificity needed for adoption [E1]. Source: {source}

## Principal rollout decision
For Autodesk or AEC document workflows, the impact would be indirect: training infrastructure quality changes whether a model iteration can be produced, evaluated, and deployed on schedule. That relevance is practical, but this draft still does not give the training runbook details a principal MLE would expect before approving the approach [E1]. Source: {source}
"""
    errors = validate_body(body, pack, outline=outline)
    assert "missing_training_howto:training_stack" in errors


def test_training_focused_daily_body_passes_with_runbook_details(monkeypatch):
    monkeypatch.setenv("BLOGPIPE_DAILY_MIN_WORDS", "300")
    pack = _training_pack()
    outline = _training_outline()
    source = "https://example.com/fsdp-runbook"
    body = f"""
# Research Radar: Scaled LLM training runbooks

## Training thesis
The thesis is that scaled LLM training papers should be read as runbooks for resource allocation, not as generic model-quality updates. The evidence frames the problem as training throughput under memory and communication pressure, so the practical question is what stack decision a principal engineer would change before the next training run. The mechanism, objective, benchmark, limitation, impact, and Autodesk/AEC transfer lens all point to the same claim: training reliability depends on making the parallelism and recovery plan explicit [E1]. Source: {source}

## FSDP sharding and parallelism runbook
The concrete training stack starts with FSDP sharding for optimizer and parameter state, then uses tensor parallelism where a single layer no longer fits cleanly on one device. Activation checkpointing trades recomputation for memory headroom, while microbatch scheduling controls how much work each device sees before synchronization. That is the how-to decision path: choose the sharding boundary, decide whether tensor parallelism is needed, then check whether activation memory or communication is the current blocker [E1]. Source: {source}

## Objective and profiling metrics
The objective should be treated as an operational optimization problem rather than a single model score. A principal MLE would measure throughput, GPU utilization, data pipeline stalls, NCCL all-reduce time, checkpoint cadence, and recovery time after interruption. Those metrics decide whether to tune microbatch size, revise token packing, move data loading work, or change checkpoint frequency. The evidence supports this profiling view because it names benchmark ablations around utilization, recovery, and scaling efficiency [E1]. Source: {source}

## Benchmarks and failure modes
The benchmark section should become a validation matrix. Test one run where memory pressure is dominant, one where all-reduce communication dominates, one where the data pipeline starves accelerators, and one restart path that exercises checkpoint recovery. The limitation is that workload balance and checkpoint cadence can turn a nominally efficient setup into a fragile production run. The failure mode to watch is not just lower throughput; it is a training job that cannot resume cleanly or wastes expensive GPU time while waiting on input data [E1]. Source: {source}

## Principal rollout decision
The rollout decision is to treat this stack as a staged adoption, not a default architecture. First profile a representative training slice, then benchmark FSDP-only against FSDP plus tensor parallelism, then validate checkpoint recovery and data loader pressure before scaling. For Autodesk or AEC document-model work, the transfer is an open hypothesis: better training infrastructure can shorten iteration loops for document models, but only if the same release gate tracks utilization, cost, restart behavior, and quality regression risk [E1]. Source: {source}
"""
    assert validate_body(body, pack, outline=outline) == []


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


def test_quality_floor_guidance_handles_llm_low_signal_training_howto():
    guidance = _quality_floor_guidance(
        [
            "llm_low_signal:engineering_judgment:0.62/0.75",
            "llm_low_signal:training_howto:0.45/0.75",
            "llm_quality:TRAINING_HOWTO_INSUFFICIENT",
        ]
    )
    assert "QUALITY_FLOOR_GUIDANCE" in guidance
    assert "Raise engineering_judgment" in guidance
    assert "FSDP/ZeRO" in guidance
    assert "checkpoint recovery" in guidance


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


def test_quality_review_prompt_exposes_training_howto_focus():
    prompt = _quality_review_user(
        body="Thin training draft.",
        pack=_training_pack(),
        outline=_training_outline(),
        selection=SelectionResult(),
    )
    assert "TRAINING_SYSTEM_FOCUS: true" in prompt
    assert '"training_howto": 0.0' in prompt
    assert "scaled-training how-to detail" in prompt


def test_signal_rubric_scores_training_howto_from_cited_prose():
    body = (
        "FSDP sharding and tensor parallelism define the training stack. "
        "Memory pressure, NCCL all-reduce communication, checkpoint recovery, and data pipeline throughput "
        "are the scaling bottlenecks. A principal engineer would profile GPU utilization, benchmark microbatch choices, "
        "validate failure modes, and set a rollout release gate [E1]. Source: https://example.com/fsdp-runbook"
    )
    scores = _signal_rubric(body, _training_pack())["scores"]
    assert scores["training_howto"] == 1.0


def test_signal_rubric_ignores_uncited_decorative_quality_cues():
    body = """
# Benchmark architecture objective ablation release gate compare tradeoff across together

This uncited paragraph says algorithm, architecture, objective, benchmark, ablation, throughput, latency, cache, retrieval, quantization, failure mode, release gate, production, tradeoff, monitoring, compare, across, together, and contrast.

The cited paragraph says only that the item matters for engineering readers [E1]. Source: https://arxiv.org/abs/2605.00001
"""
    scores = _signal_rubric(body, _pack())["scores"]
    assert scores["technical_specificity"] == 0.0
    assert scores["engineering_judgment"] == 0.0
    assert scores["synthesis"] == 0.0


def test_missing_quality_review_falls_back_to_deterministic_signal_errors(monkeypatch):
    monkeypatch.setenv("BLOGPIPE_MIN_SIGNAL_SCORE", "0.75")
    body = (
        "This draft is coherent but generic. It says training systems matter and that teams should be careful. "
        "It does not explain mechanisms, concrete choices, operating limits, or a practical runbook [E1]. "
        "Source: https://example.com/fsdp-runbook"
    )
    quality_review, errors = _review_if_ready(
        object(),  # type: ignore[arg-type]
        body=body,
        pack=_training_pack(),
        outline=_training_outline(),
        selection=SelectionResult(),
        errors=[],
    )
    assert quality_review == {}
    assert _deterministic_quality_errors(body, _training_pack())
    assert "signal_low_score:technical_specificity:0.00/0.75" in errors
    assert "signal_low_score:training_howto:0.00/0.75" in errors


def test_permissive_quality_review_cannot_override_signal_floor(monkeypatch):
    monkeypatch.setenv("BLOGPIPE_MIN_SIGNAL_SCORE", "0.75")
    monkeypatch.setenv(
        "BLOGPIPE_FAKE_QUALITY_RESPONSE",
        json.dumps(
            {
                "pass": True,
                "scores": {
                    "technical_specificity": 0.95,
                    "engineering_judgment": 0.94,
                    "synthesis": 0.93,
                    "noise_control": 0.96,
                    "primary_depth": 0.92,
                    "training_howto": 0.91,
                },
                "errors": [],
                "examples": [],
                "notes": "Looks publishable.",
            }
        ),
    )
    body = (
        "This draft is coherent but generic. It says training systems matter and that teams should be careful. "
        "It does not explain mechanisms, concrete choices, operating limits, or a practical runbook [E1]. "
        "Source: https://example.com/fsdp-runbook"
    )
    quality_review, errors = _review_if_ready(
        LLMClient(),
        body=body,
        pack=_training_pack(),
        outline=_training_outline(),
        selection=SelectionResult(),
        errors=[],
    )
    assert quality_review["pass"] is True
    assert not any(error.startswith("llm_low_signal:") for error in errors)
    assert "signal_low_score:technical_specificity:0.00/0.75" in errors
    assert "signal_low_score:training_howto:0.00/0.75" in errors


def test_repair_prompt_explains_quality_floor_errors():
    errors = [
        "signal_low_score:technical_specificity:0.00/0.75",
        "signal_low_score:synthesis:0.25/0.75",
        "signal_low_score:training_howto:0.33/0.75",
        "missing_training_howto:training_stack",
    ]
    prompt = _repair_user(
        _training_pack(),
        "Thin training draft [E1]. Source: https://example.com/fsdp-runbook",
        errors,
        outline=_training_outline(),
        selection=SelectionResult(),
    )
    assert "QUALITY_FLOOR_GUIDANCE" in prompt
    assert "Headings and uncited cue words do not count" in prompt
    assert "compare at least two primary papers" in prompt
    assert "FSDP/ZeRO" in prompt
    assert "NCCL/all-reduce" in prompt
    assert "profile, benchmark, validate failure modes" in prompt


def test_full_rewrite_prompt_explains_quality_floor_errors():
    errors = [
        "missing_final_h1",
        "signal_low_score:engineering_judgment:0.20/0.75",
        "signal_low_score:training_howto:0.00/0.75",
    ]
    prompt = _daily_rewrite_user(
        _training_pack(),
        "Fragment.",
        errors,
        outline=_training_outline(),
        selection=SelectionResult(),
        title="Research Radar: Scaled LLM training runbooks",
    )
    assert "QUALITY_FLOOR_GUIDANCE" in prompt
    assert "principal-engineer decision" in prompt
    assert "evidence-cited/source-linked runbook prose" in prompt
    assert "checkpoint recovery" in prompt


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
    good = Path("tests/fixtures/fake_daily.md").read_text()
    bad = good + "\n\nUnsupported extra claim [E999]. Source: https://arxiv.org/abs/2605.00001\n"
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


def test_emergency_daily_draft_passes_deterministic_validation():
    pack = _training_pack()
    outline = _training_outline()
    body = _emergency_daily_draft(pack=pack, outline=outline, selection=_selection(), title=outline.title)
    assert validate_body(body, pack, outline=outline) == []
    assert _deterministic_quality_errors(body, pack) == []


def test_write_daily_uses_emergency_draft_when_llm_draft_fails(monkeypatch, tmp_path):
    pack = _training_pack()
    outline = _training_outline()

    class FailingLLM(LLMClient):
        def complete(self, *, system, user, max_tokens=None, task=None, reject_completion=None, rejected_tracker=None):
            raise RuntimeError(f"forced_failure:{task}")

    _patch_root(monkeypatch, tmp_path)
    result = write_daily(pack, outline=outline, selection=_selection(), llm=FailingLLM(), dry_run=True)
    assert result.ok, result.errors
    assert "FSDP" in result.body
    assert "QUALITY_FLOOR_GUIDANCE" not in result.body
    assert validate_body(result.body, pack, outline=outline) == []


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


def test_invalid_daily_uses_emergency_draft_after_failed_repair(monkeypatch, tmp_path):
    pack = _pack()

    class BadLLM:
        def complete(self, *, system, user, max_tokens=None):
            return "Unsupported 99.9% claim [E1]. Source: https://arxiv.org/abs/2605.00001"

    _patch_root(monkeypatch, tmp_path)
    result = write_daily(pack, outline=_outline(), selection=_selection(), llm=BadLLM(), dry_run=False)
    assert result.ok, result.errors
    assert result.repair_attempted
    assert list((tmp_path / "content" / "post").glob("*.md"))
    assert not list((tmp_path / "reports").glob("*.blocked.json"))


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
