"""Deterministic routing helpers (LangGraph compiles a linear flow; this logs decisions)."""

from __future__ import annotations

from typing import Any


def log_decision(state: dict[str, Any], action: str) -> list[str]:
    """Return updated supervisor_decisions list for state merge."""
    d = list(state.get("supervisor_decisions") or [])
    d.append(action)
    return d
