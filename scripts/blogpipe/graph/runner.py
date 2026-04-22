"""Entry: reset usage and LLM cap, compile graph, invoke, return final state."""

from __future__ import annotations

import logging
from typing import Any

from .. import config
from ..llm_chain import reset_llm_usage, set_llm_call_cap
from .build import build_graph

LOG = logging.getLogger(__name__)


def run_graph_pipeline() -> dict[str, Any]:
    set_llm_call_cap(config.llm_call_cap())
    reset_llm_usage()
    g = build_graph()
    return g.invoke({})


def run_partial(stop_after: str) -> dict[str, Any]:
    """Run graph stages in-process up to stop_after; ``edit`` includes publish+write."""
    from . import nodes  # noqa: PLC0415

    order = [
        ("curate", nodes.node_curate),
        ("harvest", nodes.node_harvest),
        ("rank", nodes.node_rank),
        ("research", nodes.node_research),
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
                s.update(nodes.node_publish_and_write(s))
            return s
    return s
