"""Compile the LangGraph StateGraph for the blog pipeline."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .. import config
from ..analysts import RUNNERS
from .committee import (
    fan_to_analysts_from_scout,
    make_analyst_node,
    node_scout,
    node_synthesizer,
)
from .nodes import (
    node_curate,
    node_draft_refine,
    node_editor,
    node_harvest,
    node_publish_and_write,
    node_rank,
    node_research,
)
from .state import BlogState


def build_graph() -> object:
    """Return compiled graph: curate → … → write artifacts."""
    g: StateGraph = StateGraph(BlogState)
    g.add_node("curate", node_curate)
    g.add_node("harvest", node_harvest)
    g.add_node("rank", node_rank)
    g.add_node("draft_refine", node_draft_refine)
    g.add_node("editor", node_editor)
    g.add_node("publish", node_publish_and_write)
    g.add_edge(START, "curate")
    g.add_edge("curate", "harvest")
    g.add_edge("harvest", "rank")
    if config.committee_enabled():
        g.add_node("scout", node_scout)
        g.add_node("synthesizer", node_synthesizer)
        for name in RUNNERS:
            g.add_node(f"analyst_{name}", make_analyst_node(name))
        g.add_edge("rank", "scout")
        g.add_conditional_edges("scout", fan_to_analysts_from_scout)
        for name in RUNNERS:
            g.add_edge(f"analyst_{name}", "synthesizer")
        g.add_edge("synthesizer", "draft_refine")
    else:
        g.add_node("research", node_research)
        g.add_edge("rank", "research")
        g.add_edge("research", "draft_refine")
    g.add_edge("draft_refine", "editor")
    g.add_edge("editor", "publish")
    g.add_edge("publish", END)
    return g.compile()
