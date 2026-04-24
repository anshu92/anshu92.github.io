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
from .retry_policies import DEFAULT_RETRY, add_node_with_retry
from .state import BlogState


def build_graph(
    checkpointer: Optional[object] = None,
) -> Any:
    """Return compiled graph: curate → … → write artifacts."""
    g: StateGraph = StateGraph(BlogState)
    add_node_with_retry(g, "curate", node_curate, DEFAULT_RETRY)
    add_node_with_retry(g, "harvest", node_harvest, DEFAULT_RETRY)
    add_node_with_retry(g, "rank", node_rank, DEFAULT_RETRY)
    add_node_with_retry(g, "planner", node_planning_brief, DEFAULT_RETRY)
    add_node_with_retry(g, "draft_refine", node_draft_refine, DEFAULT_RETRY)
    add_node_with_retry(g, "adversary_review", node_adversary_review, DEFAULT_RETRY)
    add_node_with_retry(g, "evidence_verifier", node_evidence_verifier, DEFAULT_RETRY)
    add_node_with_retry(g, "render_reviewer", node_render_reviewer, DEFAULT_RETRY)
    add_node_with_retry(g, "meta_reviewer", node_meta_review, DEFAULT_RETRY)
    add_node_with_retry(g, "editor", node_editor, DEFAULT_RETRY)
    add_node_with_retry(g, "review_gate", node_review_gate, DEFAULT_RETRY)
    add_node_with_retry(g, "publish", node_publish_and_write, DEFAULT_RETRY)
    g.add_edge(START, "curate")
    g.add_edge("curate", "harvest")
    g.add_edge("harvest", "rank")
    if config.committee_enabled():
        add_node_with_retry(g, "committee", committee_node, DEFAULT_RETRY)
        g.add_edge("rank", "committee")
        g.add_edge("committee", "planner")
    else:
        add_node_with_retry(g, "research", node_research, DEFAULT_RETRY)
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
