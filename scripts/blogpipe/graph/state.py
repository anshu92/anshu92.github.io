"""Shared graph state and small dataclasses."""

from __future__ import annotations

from typing import Any, List, TypedDict

from ..models import EditorialBrief, EvidenceBundle, RankResult


class BlogState(TypedDict, total=False):
    """Merged by LangGraph (partial updates)."""

    brief: dict[str, Any]
    rank_result: dict[str, Any]
    primary: dict[str, Any]
    evidence: dict[str, Any]
    research_trace: dict[str, Any]
    body: str
    outline: List[str]
    section_meta: List[dict[str, Any]]
    budget_exhausted: bool
    warnings: List[str]
    supervisor_decisions: List[str]
    pass_gate: bool
    llm_ok: bool
    editor_report: dict[str, Any]
    draft_lint: dict[str, Any]
    slug: str
    out_path: str
    mermaid_injected: bool
    rewrites_applied: int
    re_research_applied: int
    # Internal flags
    _done_curate: bool
    _done_harvest: bool
    _done_rank: bool
    _done_research: bool
    _done_draft: bool
    _done_editor: bool


def brief_model(d: dict[str, Any]) -> EditorialBrief:
    return EditorialBrief.model_validate(d)


def evidence_model(d: dict[str, Any]) -> EvidenceBundle:
    return EvidenceBundle.model_validate(d)


def rank_model(d: dict[str, Any]) -> RankResult:
    return RankResult.model_validate(d)
