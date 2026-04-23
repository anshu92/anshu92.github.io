"""Compile the LangGraph StateGraph for the blog pipeline."""

from __future__ import annotations

from typing import Any, Optional

from langgraph.graph import END, START, StateGraph

from .. import config
from .committee_subgraph import get_committee_subgraph
from .nodes import (
    node_curate,
    node_draft_refine,
    node_editor,
    node_harvest,
    node_publish_and_write,
    node_rank,
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
    g.add_node("curate", node_curate, retry=DEFAULT_RETRY)
    g.add_node("harvest", node_harvest, retry=DEFAULT_RETRY)
    g.add_node("rank", node_rank, retry=DEFAULT_RETRY)
    g.add_node("draft_refine", node_draft_refine, retry=DEFAULT_RETRY)
    g.add_node("editor", node_editor, retry=DEFAULT_RETRY)
    g.add_node("review_gate", node_review_gate, retry=DEFAULT_RETRY)
    g.add_node("publish", node_publish_and_write, retry=DEFAULT_RETRY)
    g.add_edge(START, "curate")
    g.add_edge("curate", "harvest")
    g.add_edge("harvest", "rank")
    if config.committee_enabled():
        g.add_node("committee", get_committee_subgraph())
        g.add_edge("rank", "committee")
        g.add_edge("committee", "draft_refine")
    else:
        g.add_node("research", node_research, retry=DEFAULT_RETRY)
        g.add_edge("rank", "research")
        g.add_edge("research", "draft_refine")
    g.add_edge("draft_refine", "editor")
    g.add_edge("editor", "review_gate")
    g.add_edge("review_gate", "publish")
    g.add_edge("publish", END)
    return g.compile(checkpointer=checkpointer)
