"""Compiled committee subgraph: scout → supervisor → parallel analysts → synthesizer."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from ..analysts import RUNNERS
from ..llm_chain import budget
from .committee import (
    fan_to_analysts_from_supervisor,
    make_analyst_node,
    node_scout,
    node_synthesizer,
)
from .retry_policies import ANALYST_RETRY, DEFAULT_RETRY
from .state import BlogState
from .supervisor import node_supervisor


def build_committee_graph() -> StateGraph:
    """Uncompiled committee map-reduce."""
    g: StateGraph = StateGraph(BlogState)
    g.add_node("scout", node_scout, retry_policy=DEFAULT_RETRY)
    g.add_node("supervisor", node_supervisor, retry_policy=DEFAULT_RETRY)
    g.add_node("synthesizer", node_synthesizer, retry_policy=DEFAULT_RETRY)
    for name in RUNNERS:
        g.add_node(
            f"analyst_{name}", make_analyst_node(name), retry_policy=ANALYST_RETRY
        )
    g.add_edge(START, "scout")
    g.add_edge("scout", "supervisor")
    g.add_conditional_edges("supervisor", fan_to_analysts_from_supervisor)
    for name in RUNNERS:
        g.add_edge(f"analyst_{name}", "synthesizer")
    g.add_edge("synthesizer", END)
    return g


_COMMITTEE_COMPILED: object | None = None


def get_committee_subgraph() -> object:
    """Single compiled instance (no checkpointer) for `run_partial` and embedding as a node."""
    global _COMMITTEE_COMPILED
    if _COMMITTEE_COMPILED is None:
        _COMMITTEE_COMPILED = build_committee_graph().compile()
    return _COMMITTEE_COMPILED


def committee_node(state: dict[str, Any]) -> dict[str, Any]:
    """Parent-graph node: bill all committee LLM calls to the `committee` stage budget."""
    with budget.stage("committee"):
        g = get_committee_subgraph()
        return g.invoke(state)  # type: ignore[no-untyped-call]


def run_committee_subgraph_state(state: dict[str, Any]) -> dict[str, Any]:
    """Run committee pipeline without parent Send semantics (e.g. run_partial)."""
    with budget.stage("committee"):
        g = get_committee_subgraph()
        return g.invoke(state)  # type: ignore[no-untyped-call]
