from __future__ import annotations

from datetime import datetime, timezone

import pytest

from blogpipe import curriculum, memory
from blogpipe.models import CurriculumNode, RankedItem, SourceItem, TopicScores, WriteResult


def _ranked(item_id: str, text: str, *, track: str = "applied_research") -> RankedItem:
    scores = {
        "ml_engineering": 0.2,
        "applied_research": 0.2,
        "ml_theory": 0.0,
        "aec": 0.0,
        "popular_ml": 0.0,
    }
    scores[track] = 0.8
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
        topic_scores=TopicScores(**scores, priority_track=track, matched_keywords=text.lower().split()),
        daily_score=0.8,
        deep_dive_score=0.7,
        quality_signals={"technical_depth": 0.8, "practical_impact": 0.6},
    )


def test_tree_validation_rejects_vague_topic_nodes():
    vague = CurriculumNode(
        id="bad-transformers",
        problem_statement="Transformers?",
        why_it_matters="Too vague.",
        concepts_to_teach=["attention"],
        engineering_questions=["What is it?"],
        evidence_requirements=["conceptual explanation"],
    )
    with pytest.raises(ValueError, match="vague_topic|problem_too_short"):
        curriculum.validate_tree([vague])


def test_scheduler_selects_first_problem_without_ranked_input(monkeypatch, tmp_path):
    monkeypatch.setattr(memory, "DATA", tmp_path / "radar-data")
    monkeypatch.delenv("BLOGPIPE_CURRICULUM_NODE", raising=False)
    plan = curriculum.choose_next_problem()
    assert plan.node.id == "matmul-from-scalar-operations"
    assert plan.selected_by == "next_uncompleted_problem"


def test_scheduler_ignores_search_results_when_choosing_problem(monkeypatch, tmp_path):
    monkeypatch.setattr(memory, "DATA", tmp_path / "radar-data")
    ranked = [_ranked("frontier", "frontier transformer benchmark agent architecture")]
    plan = curriculum.plan_curriculum(ranked)
    assert plan.node.id == "matmul-from-scalar-operations"


def test_curriculum_state_advances_after_success(monkeypatch, tmp_path):
    monkeypatch.setattr(memory, "DATA", tmp_path / "radar-data")
    first = curriculum.choose_next_problem()
    curriculum.record_completion(first, WriteResult(ok=True, path="content/post/post.md", title="Post"))
    second = curriculum.choose_next_problem()
    assert "matmul-from-scalar-operations" in second.completed_node_ids
    assert second.node.id == "matrices-as-linear-maps"


def test_forced_problem_override(monkeypatch, tmp_path):
    monkeypatch.setattr(memory, "DATA", tmp_path / "radar-data")
    monkeypatch.setenv("BLOGPIPE_CURRICULUM_NODE", "gpu-matmul-tiling")
    plan = curriculum.choose_next_problem()
    assert plan.node.id == "gpu-matmul-tiling"
    assert plan.selected_by == "forced_env"


def test_forced_unknown_problem_blocks(monkeypatch, tmp_path):
    monkeypatch.setattr(memory, "DATA", tmp_path / "radar-data")
    monkeypatch.setenv("BLOGPIPE_CURRICULUM_NODE", "not-a-node")
    with pytest.raises(ValueError, match="unknown_curriculum_node:not-a-node"):
        curriculum.choose_next_problem()


def test_research_planner_generates_targeted_queries(monkeypatch, tmp_path):
    monkeypatch.setattr(memory, "DATA", tmp_path / "radar-data")
    monkeypatch.setenv("BLOGPIPE_CURRICULUM_NODE", "gpu-matmul-tiling")
    plan = curriculum.choose_next_problem()
    research_plan = curriculum.build_research_plan(plan)
    query_blob = " ".join(research_plan.queries).lower()
    assert "tiled matrix multiplication" in query_blob
    assert "tensor cores" in query_blob
    assert research_plan.problem_id == "gpu-matmul-tiling"


def test_evidence_curation_cannot_substitute_different_problem(monkeypatch, tmp_path):
    monkeypatch.setattr(memory, "DATA", tmp_path / "radar-data")
    plan = curriculum.choose_next_problem()
    ranked = [_ranked("rag", "RAG evaluation benchmark for document agents with retrieval monitoring")]
    curation = curriculum.curate_evidence(ranked, plan)
    assert not curation.sufficient
    assert curation.problem_id == "matmul-from-scalar-operations"
    assert any("primary" in gap or "matrix" in gap.lower() or "evidence" in gap.lower() for gap in curation.gaps)


def test_evidence_curation_accepts_problem_matching_items(monkeypatch, tmp_path):
    monkeypatch.setattr(memory, "DATA", tmp_path / "radar-data")
    plan = curriculum.choose_next_problem()
    ranked = [
        _ranked(
            "matmul-core",
            "Matrix multiplication computes each output with row-column dot product multiply-accumulate operations. "
            "The tutorial includes an implementation algorithm and numerical precision limitation.",
        ),
        _ranked(
            "matmul-impl",
            "A matrix multiplication implementation uses loops over scalar multiplication and accumulation, with code and indexing caveats.",
        ),
        _ranked(
            "matmul-caveat",
            "Matrix multiplication preserves linear combinations but can suffer floating point precision and rounding failure modes.",
        ),
    ]
    curation = curriculum.curate_evidence(ranked, plan)
    assert curation.sufficient
    selected, selection = curriculum.apply_curation(ranked, curation)
    assert len(selected) >= 2
    assert selection.selected_item_ids == [item.item.item_id for item in selected]
    assert selected[0].item.extra["curriculum_problem_id"] == "matmul-from-scalar-operations"


def test_tree_growth_proposals_are_review_gated(monkeypatch, tmp_path):
    monkeypatch.setattr(memory, "DATA", tmp_path / "radar-data")
    plan = curriculum.choose_next_problem()
    curation = curriculum.curate_evidence([_ranked("unrelated", "generic serving benchmark")], plan)
    proposals = curriculum.propose_tree_growth(plan, curation)
    assert proposals["auto_mutated"] is False
    assert proposals["proposals"]
    assert proposals["proposals"][0]["status"] == "review_required"
