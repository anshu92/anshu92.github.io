"""Supervisor: core + optional picks within eligibility."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

os.environ.setdefault("BLOGPIPE_REPO_ROOT", os.path.join(os.path.dirname(__file__), ".."))


def _pack_state():
    return {
        "evidence_pack": {
            "primary": {
                "id": "1",
                "title": "Test title about neural scaling",
                "url": "https://arxiv.org/abs/2401.00001",
                "authors": [],
                "abstract": "We study scaling laws.",
                "source": "arxiv",
                "tags": ["ml"],
                "pillar": "research",
            },
            "calls_used": 0,
            "trace": [],
        }
    }


def test_supervisor_bypass_uses_config_order(monkeypatch):
    from blogpipe.graph import supervisor

    monkeypatch.setenv("BLOGPIPE_SUPERVISOR", "0")
    monkeypatch.setenv("BLOGPIPE_COMMITTEE_ANALYSTS", "methods,web")
    s = _pack_state()
    out = supervisor.node_supervisor(s)
    assert out["selected_analysts"] == ["methods", "web"]


def test_supervisor_dry_run_picks_defaults(monkeypatch):
    from blogpipe.graph import supervisor

    monkeypatch.setenv("BLOGPIPE_SUPERVISOR", "1")
    monkeypatch.setenv("BLOGPIPE_DRY_RUN", "1")
    s = _pack_state()
    out = supervisor.node_supervisor(s)
    # core (methods, empirical, glossary) + 2+ optionals
    sa = out["selected_analysts"]
    for c in ("methods", "empirical", "glossary"):
        if c in sa:
            pass  # if in eligibility, present
    assert "methods" in sa
    assert "empirical" in sa
    assert "glossary" in sa
    assert 6 <= len(sa) <= 9


def test_task_profile_supervisor_route():
    from blogpipe import model_registry

    p = model_registry.get_task_profile("supervisor_route")
    assert p.name == "supervisor_route"
    assert p.max_output == 600
