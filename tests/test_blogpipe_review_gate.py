"""Review gate interrupt with checkpointer and resume via Command."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


def test_review_gate_stream_interrupts_then_resume(monkeypatch, tmp_path):
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.constants import END, START
    from langgraph.graph import StateGraph
    from langgraph.types import Command

    from blogpipe.graph.nodes import node_review_gate
    from blogpipe.graph.state import BlogState

    monkeypatch.setenv("BLOGPIPE_AUTO_APPROVE", "0")
    # Ensure config re-reads env
    from blogpipe import config

    assert not config.auto_approve_editor_gate()

    g: StateGraph = StateGraph(BlogState)
    g.add_node("review_gate", node_review_gate)
    g.add_edge(START, "review_gate")
    g.add_edge("review_gate", END)
    cg = g.compile(checkpointer=MemorySaver())
    base = {"configurable": {"thread_id": "rev-t1"}}
    chunks = list(
        cg.stream(
            {"pass_gate": False, "editor_report": {"pass_gate": False}},
            config=base,
        )
    )
    assert any(
        isinstance(ch, dict) and "__interrupt__" in ch for ch in chunks
    ), chunks

    list(
        cg.stream(
            Command(resume={"approve": True}),
            config=base,
        )
    )
    st = cg.get_state(base)  # type: ignore[union-attr]
    assert st and st.values
    assert st.values.get("pass_gate") is True
