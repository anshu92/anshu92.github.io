"""Entry: reset usage and LLM cap, compile graph, invoke, return final state."""

from __future__ import annotations

import logging
import os
import sqlite3
import uuid
from typing import Any

from langgraph.types import Command

from .. import config
from ..llm_chain import get_llm_usage, reset_llm_usage, set_llm_call_cap
from .build import build_graph

LOG = logging.getLogger(__name__)


def _thread_id() -> str:
    o = config.graph_thread_id_override()
    if o:
        return o
    rid = os.environ.get("GITHUB_RUN_ID", "")
    if rid:
        return f"gh-{rid}"
    return f"local-{uuid.uuid4().hex[:12]}"


def _make_checkpointer() -> object:
    path = (config.checkpointer_path() or "").strip()
    if path == ":memory:":
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()
    pdir = path.rsplit("/", 1)[0] if "/" in path else "."
    if pdir and pdir not in (".",) and pdir:
        try:
            os.makedirs(pdir, exist_ok=True)
        except OSError:
            pass
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError:  # pragma: no cover
        from langgraph.checkpoint.memory import MemorySaver

        LOG.warning("langgraph-checkpoint-sqlite not installed; using MemorySaver")
        return MemorySaver()
    conn = sqlite3.connect(path, check_same_thread=False)
    return SqliteSaver(conn)


def _langsmith_run_config(tid: str) -> dict[str, Any]:
    return {
        "run_name": f"blogpipe/{tid}",
        "tags": ["blogpipe", "graph"],
    }


def run_graph_pipeline() -> dict[str, Any]:
    set_llm_call_cap(config.llm_call_cap())
    checkpointer = _make_checkpointer()
    g = build_graph(checkpointer=checkpointer)
    tid = _thread_id()
    rlc: dict[str, Any] = {"thread_id": tid, **_langsmith_run_config(tid)}
    cfg: dict[str, Any] = {"configurable": rlc}
    base = {"configurable": {"thread_id": tid}}
    if checkpointer.get_tuple(base) is None:  # type: ignore[union-attr]
        reset_llm_usage()
    if config.graph_stream_enabled():
        for chunk in g.stream({}, config=cfg, stream_mode="updates"):
            if not isinstance(chunk, dict):
                continue
            for node, delta in chunk.items():
                LOG.info(
                    "graph_update node=%s keys=%s",
                    node,
                    sorted((delta or {}).keys()) if isinstance(delta, dict) else None,
                )
    else:
        g.invoke({}, config=cfg)
    snap = g.get_state(cfg)  # type: ignore[union-attr]
    return dict(snap.values) if snap and snap.values is not None else {}


def resume_graph_after_interrupt(thread_id: str) -> dict[str, Any]:
    """Resume a paused run (review_gate) with approval."""
    set_llm_call_cap(config.llm_call_cap())
    checkpointer = _make_checkpointer()
    g = build_graph(checkpointer=checkpointer)
    rlc: dict[str, Any] = {"thread_id": thread_id, **_langsmith_run_config(thread_id)}
    cfg: dict[str, Any] = {"configurable": rlc}
    if config.graph_stream_enabled():
        for chunk in g.stream(
            Command(resume={"approve": True}), config=cfg, stream_mode="updates"
        ):
            if not isinstance(chunk, dict):
                continue
            for node, delta in chunk.items():
                LOG.info(
                    "graph_resume node=%s keys=%s",
                    node,
                    sorted((delta or {}).keys()) if isinstance(delta, dict) else None,
                )
    else:
        g.invoke(Command(resume={"approve": True}), config=cfg)
    snap = g.get_state(cfg)  # type: ignore[union-attr]
    return dict(snap.values) if snap and snap.values is not None else {}


def run_partial(stop_after: str) -> dict[str, Any]:
    """Run graph stages in-process up to stop_after; ``edit`` includes publish+write."""
    from . import nodes  # noqa: PLC0415

    from .committee_subgraph import run_committee_subgraph_state  # noqa: PLC0415

    def _research(s: dict[str, Any]) -> dict[str, Any]:
        if config.committee_enabled():
            return run_committee_subgraph_state(s)
        return nodes.node_research(s)

    order = [
        ("curate", nodes.node_curate),
        ("harvest", nodes.node_harvest),
        ("rank", nodes.node_rank),
        ("research", _research),
        ("draft", nodes.node_draft_refine),
        ("edit", nodes.node_editor),
    ]
    s: dict[str, Any] = {}
    set_llm_call_cap(config.llm_call_cap())
    reset_llm_usage()
    for step_name, fn in order:
        s.update(fn(s))
        if step_name == stop_after:
            if step_name == "edit":
                s.update(nodes.node_review_gate(s))
                s.update(nodes.node_publish_and_write(s))
            return s
    return s
