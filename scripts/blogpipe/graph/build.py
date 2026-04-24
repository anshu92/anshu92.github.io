"""Compile the LangGraph StateGraph for the blog pipeline."""

from __future__ import annotations

from typing import Any, Optional

from langgraph.graph import END, START, StateGraph

from .. import config
from .committee_subgraph import committee_node
from .nodes import (
    node_adversary_review,
    node_curate,
    node_draft_refine,
    node_editor,
    node_evidence_verifier,
    node_harvest,
    node_meta_review,
    node_planning_brief,
    node_publish_and_write,
    node_rank,
    node_render_reviewer,
    node_research,
    node_review_gate,
)
from .retry_policies import DEFAULT_RETRY
from .state import BlogState


def build_graph(
    checkpointer: Optional[object] = None,
) -> Any:
    """Return compiled graph: curate → … → write artifacts."""
    g: StateGraph = StateGraph(BlogState)
    g.add_node("curate", node_curate, retry_policy=DEFAULT_RETRY)
    g.add_node("harvest", node_harvest, retry_policy=DEFAULT_RETRY)
    g.add_node("rank", node_rank, retry_policy=DEFAULT_RETRY)
    g.add_node("planner", node_planning_brief, retry_policy=DEFAULT_RETRY)
    g.add_node("draft_refine", node_draft_refine, retry_policy=DEFAULT_RETRY)
    g.add_node("adversary_review", node_adversary_review, retry_policy=DEFAULT_RETRY)
    g.add_node("evidence_verifier", node_evidence_verifier, retry_policy=DEFAULT_RETRY)
    g.add_node("render_reviewer", node_render_reviewer, retry_policy=DEFAULT_RETRY)
    g.add_node("meta_reviewer", node_meta_review, retry_policy=DEFAULT_RETRY)
    g.add_node("editor", node_editor, retry_policy=DEFAULT_RETRY)
    g.add_node("review_gate", node_review_gate, retry_policy=DEFAULT_RETRY)
    g.add_node("publish", node_publish_and_write, retry_policy=DEFAULT_RETRY)
    g.add_edge(START, "curate")
    g.add_edge("curate", "harvest")
    g.add_edge("harvest", "rank")
    if config.committee_enabled():
        g.add_node("committee", committee_node, retry_policy=DEFAULT_RETRY)
        g.add_edge("rank", "committee")
        g.add_edge("committee", "planner")
    else:
        g.add_node("research", node_research, retry_policy=DEFAULT_RETRY)
        g.add_edge("rank", "research")
        g.add_edge("research", "planner")
    g.add_edge("planner", "draft_refine")
    g.add_edge("draft_refine", "adversary_review")
    g.add_edge("adversary_review", "evidence_verifier")
    g.add_edge("evidence_verifier", "render_reviewer")
    g.add_edge("render_reviewer", "meta_reviewer")
    g.add_edge("meta_reviewer", "editor")
    g.add_edge("editor", "review_gate")
    g.add_edge("review_gate", "publish")
    g.add_edge("publish", END)
    return g.compile(checkpointer=checkpointer)
