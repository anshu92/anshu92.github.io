from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
os.environ.setdefault("BLOGPIPE_REPO_ROOT", os.path.join(os.path.dirname(__file__), ".."))


def _state() -> dict:
    return {
        "brief": {
            "format_name": "deep_dive",
            "diagram_style": "flowchart",
            "opener_hook": "single_concrete_pain_point_with_falsifier",
            "art_direction": "editorial_illustration",
        },
        "evidence": {
            "primary": {
                "id": "primary",
                "title": "Test Paper",
                "url": "https://arxiv.org/abs/2401.00001",
                "authors": [],
                "abstract": "A paper about a method.",
                "source": "arxiv",
                "tags": ["ml"],
                "pillar": "research",
            },
            "section_evidence": {
                "paper_method": "The method uses a geometry-aware selector.",
                "paper_experiments": "The method reaches 81.67% versus 79.32% for Full LoRA.",
            },
            "contradiction_notes": ["single benchmark only"],
        },
        "body": (
            "Takeaway 81.67%.\n\n"
            "## Why this works\nThe method uses a selector. [cite: primary]\n\n"
            "## Steal This\nUse it for your workflow. [cite: primary]\n"
        ),
    }


def test_planning_brief_fallback_has_required_sections(monkeypatch) -> None:
    from blogpipe.graph.nodes import node_planning_brief

    monkeypatch.setenv("BLOGPIPE_DRY_RUN", "1")
    out = node_planning_brief(_state())
    plan = out["planning_brief"]
    assert "why this works" in plan["mandatory_sections"]
    assert "when to use it" in plan["mandatory_sections"]
    assert plan["reviewer_focus"]
    assert plan["preventive_checks"]
    assert plan["backup_remedies"]


def test_meta_review_aggregates_prior_findings() -> None:
    from blogpipe.graph.nodes import node_meta_review

    state = _state()
    state["review_notes"] = [
        {
            "role": "adversary",
            "pass_review": False,
            "findings": ["missing_falsifier_language"],
            "rewrite_targets": [],
            "summary": "Need a falsifier.",
        },
        {
            "role": "evidence_verifier",
            "pass_review": False,
            "findings": ["citation_count_below_min"],
            "rewrite_targets": [],
            "summary": "Need more cites.",
        },
        {
            "role": "render_reviewer",
            "pass_review": True,
            "findings": [],
            "rewrite_targets": [],
            "summary": "Render looks fine.",
        },
    ]
    out = node_meta_review(state)
    meta = out["meta_review"]
    assert meta["role"] == "meta_reviewer"
    assert not meta["pass_review"]
    assert "reviewer_blocked:evidence_verifier" in meta["findings"]
    assert "citation_count_below_min" in meta["findings"]
    assert "missing_falsifier_language" in meta["metadata"]["all_findings"]


def test_meta_review_requires_render_reviewer_consensus(monkeypatch) -> None:
    from blogpipe.graph.nodes import node_meta_review

    monkeypatch.setenv("BLOGPIPE_REVIEWER_CONSENSUS", "1")
    state = _state()
    state["review_notes"] = [
        {
            "role": "adversary",
            "pass_review": True,
            "findings": [],
            "rewrite_targets": [],
            "summary": "Looks okay.",
        },
        {
            "role": "evidence_verifier",
            "pass_review": True,
            "findings": [],
            "rewrite_targets": [],
            "summary": "Evidence okay.",
        },
    ]
    out = node_meta_review(state)
    assert not out["meta_review"]["pass_review"]
    assert "missing_required_reviewer:render_reviewer" in out["meta_review"]["findings"]


def test_render_reviewer_flags_render_errors(monkeypatch) -> None:
    from blogpipe.graph import nodes
    from blogpipe import package as package_mod

    monkeypatch.setattr(
        package_mod,
        "_render_full_html",
        lambda front, body: (None, ["raw_mermaid_in_render"], [], {
            "mermaid_rendered": False,
            "tables_rendered": False,
            "images_resolved": True,
            "captions_ok": True,
            "density_ok": True,
        }),
    )
    out = nodes.node_render_reviewer(_state())
    note = out["review_notes"][0]
    assert note["role"] == "render_reviewer"
    assert not note["pass_review"]
    assert "raw_mermaid_in_render" in note["findings"]


def test_gap_analyzer_finds_missing_mechanism_and_results() -> None:
    from blogpipe.graph.nodes import node_gap_analyzer

    state = _state()
    state["body"] = "Takeaway 81.67%.\n\n## Why it matters\nA useful result. [cite: primary]\n"
    state["planning_brief"] = {
        "mandatory_sections": ["why this works", "when to use it"],
    }
    out = node_gap_analyzer(state)
    codes = {x["code"] for x in out["gap_analysis"]}
    assert "missing_mechanism_section" in codes
    assert "no_results_table" in codes
    assert out["source_registry"]


def test_section_patcher_adds_mechanism_and_decision_sections() -> None:
    from blogpipe.graph.nodes import node_section_patcher

    state = _state()
    state["body"] = "Takeaway 81.67%.\n\n## Why it matters\nA useful result.\n"
    state["gap_analysis"] = [
        {
            "code": "missing_mechanism_section",
            "message": "missing",
            "section_hint": "Why this works",
            "required_evidence": ["paper_method"],
        },
        {
            "code": "missing_decision_section",
            "message": "missing",
            "section_hint": "What I would test next",
            "required_evidence": ["paper_limitations"],
        },
    ]
    out = node_section_patcher(state)
    assert "## Why this works" in out["body"]
    assert "## What I would test next" in out["body"]
    assert out["citation_audit"]["ok"] is True


def test_section_patcher_removes_unregistered_external_links() -> None:
    from blogpipe.graph.nodes import node_section_patcher

    state = _state()
    state["body"] = (
        "Takeaway 81.67%.\n\n"
        "## Why this works\n"
        "Read [bad source](https://example.com/unknown) and [the paper](https://arxiv.org/abs/2401.00001).\n"
    )
    state["source_registry"] = [
        {
            "kind": "item",
            "source_id": "primary",
            "key": "primary",
            "url": "https://arxiv.org/abs/2401.00001",
            "text": "Test Paper",
            "metadata": {},
        }
    ]
    out = node_section_patcher(state)
    assert "https://example.com/unknown" not in out["body"]
    assert out["citation_audit"]["invalid_links"] == ["https://example.com/unknown"]


def test_gap_analyzer_ignores_verbose_planner_section_sentences() -> None:
    from blogpipe.graph.nodes import node_gap_analyzer

    state = _state()
    state["planning_brief"] = {
        "mandatory_sections": [
            "Introduction: The MoE scaling challenge and the promise of expert upcycling.",
            "why this works",
        ],
    }
    out = node_gap_analyzer(state)
    sections = {x["section_hint"] for x in out["gap_analysis"] if x["code"] == "planning_section_gap"}
    assert "Introduction: The MoE scaling challenge and the promise of expert upcycling." not in sections


def test_failure_memory_record_and_snapshot(monkeypatch, tmp_path) -> None:
    from blogpipe.graph import nodes
    from blogpipe import memory

    monkeypatch.setattr(memory, "CACHE", tmp_path)
    monkeypatch.setenv("BLOGPIPE_FAILURE_MEMORY_LIMIT", "5")
    state = {
        "primary": {"title": "Bad Draft"},
        "quality_report": {
            "overall_status": "blocked",
            "blocking_reasons": [
                {"stage": "draft", "code": "lint:generic_heading_used"},
                {"stage": "editor", "code": "grounding_issue"},
            ],
        },
    }
    nodes._record_failure_memory(state)
    snap = nodes._failure_memory_snapshot(state)
    assert snap
    assert snap[-1]["overall_status"] == "blocked"
    assert "draft:lint:generic_heading_used" in snap[-1]["codes"]


def test_failure_memory_similarity_prefers_related_items(monkeypatch, tmp_path) -> None:
    from blogpipe.graph import nodes
    from blogpipe import memory

    monkeypatch.setattr(memory, "CACHE", tmp_path)
    memory.save_json(
        "failure_memory.json",
        [
            {
                "title": "Vision model draft",
                "codes": ["render:raw_mermaid_in_render"],
                "query_text": "vision model mermaid render benchmark",
            },
            {
                "title": "Database indexing post",
                "codes": ["draft:generic_heading_used"],
                "query_text": "database indexing benchmark table",
            },
        ],
    )
    state = _state()
    state["body"] += "\n```mermaid\nflowchart LR\nA-->B\n```"
    snap = nodes._failure_memory_snapshot(state)
    assert snap
    assert snap[0]["title"] == "Vision model draft"


def test_meta_review_weighted_policy_allows_optional_adversary_failure(monkeypatch) -> None:
    from blogpipe.graph.nodes import node_meta_review

    monkeypatch.setenv("BLOGPIPE_REVIEWER_CONSENSUS", "1")
    monkeypatch.setenv("BLOGPIPE_REVIEWER_MIN_PASS_SCORE", "0.80")
    state = _state()
    state["review_notes"] = [
        {
            "role": "adversary",
            "pass_review": False,
            "findings": ["missing_falsifier_language"],
            "summary": "Need a falsifier.",
        },
        {
            "role": "evidence_verifier",
            "pass_review": True,
            "findings": [],
            "summary": "Evidence is fine.",
        },
        {
            "role": "render_reviewer",
            "pass_review": True,
            "findings": [],
            "summary": "Render is fine.",
        },
    ]
    out = node_meta_review(state)
    meta = out["meta_review"]
    assert meta["pass_review"] is True
    assert meta["metadata"]["weighted_score"] >= 0.8
    assert "missing_falsifier_language" in meta["metadata"]["all_findings"]


def test_build_graph_contains_multi_agent_review_nodes() -> None:
    from blogpipe.graph.build import build_graph

    g = build_graph()
    nodes = set(g.get_graph().nodes.keys())
    for expected in {
        "planner",
        "draft_refine",
        "gap_analyzer",
        "evidence_backfill",
        "section_patcher",
        "adversary_review",
        "evidence_verifier",
        "render_reviewer",
        "meta_reviewer",
        "editor",
    }:
        assert expected in nodes


def test_planning_brief_prompt_uses_explicit_schema_and_delimiters(monkeypatch) -> None:
    from blogpipe.graph.nodes import node_planning_brief

    captured = {}

    def _fake_graph_llm_text(name, system, user, **kwargs):  # noqa: ANN001
        captured["system"] = system
        captured["user"] = user
        return "{}"

    monkeypatch.setenv("BLOGPIPE_DRY_RUN", "0")
    monkeypatch.setattr("blogpipe.config.llm_configured", lambda: True)
    monkeypatch.setattr("blogpipe.graph.nodes.gllm.graph_llm_text", _fake_graph_llm_text)
    node_planning_brief(_state())
    assert "Exact schema" in captured["system"]
    assert '### EDITORIAL_BRIEF' in captured["user"]
    assert '"""' in captured["user"]


def test_grounding_prompt_distinguishes_blocking_vs_advisory(monkeypatch) -> None:
    from blogpipe.graph.critics import grounding_check_node

    captured = {}

    def _fake_graph_llm_text(name, system, user, **kwargs):  # noqa: ANN001
        captured["system"] = system
        return '{"unsupported_claims":[]}'

    monkeypatch.setenv("BLOGPIPE_DRY_RUN", "0")
    monkeypatch.setattr("blogpipe.config.llm_configured", lambda: True)
    monkeypatch.setattr("blogpipe.graph.critics.gllm.graph_llm_text", _fake_graph_llm_text)
    grounding_check_node("Takeaway 81.67%.", '{"primary":{"id":"p"}}')
    assert "Do NOT flag" in captured["system"]
    assert "derived arithmetic restatements" in captured["system"]


def test_node_editor_recomputes_grounding_on_final_polished_body(monkeypatch) -> None:
    from blogpipe.graph import nodes

    seen = {}

    def _fake_global_rubric(body, bundle=None, lint_issues=None):  # noqa: ANN001
        from blogpipe.models import EditorReport
        seen.setdefault("rubric_bodies", []).append(body)
        return EditorReport(rubric_score=10, five_questions_ok=True, pass_gate=True, llm_ok=True)

    def _fake_grounding_check(body, evidence_text):  # noqa: ANN001
        seen["grounding_body"] = body
        return True, [], True

    monkeypatch.setattr(nodes, "global_rubric", _fake_global_rubric)
    monkeypatch.setattr(nodes, "grounding_check_node", _fake_grounding_check)
    monkeypatch.setattr(nodes, "explain_undefined_terms", lambda body, bundle: body.replace("## Conclusion", "## What I would test next"))
    monkeypatch.setattr(nodes, "embed_planned_visuals", lambda body, plan, slug: body)

    state = _state()
    state["body"] = (
        "Takeaway 81.67%.\n\n"
        "## Why this works\nMechanism. [cite: primary]\n\n"
        "## Conclusion\nClose. [cite: primary]\n"
    )
    out = nodes.node_editor(state)
    assert "## What I would test next" in seen["grounding_body"]
    assert out["editor_report"]["grounding_issues"] == []
