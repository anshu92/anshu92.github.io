"""Per-stage LLM call budgets (independent of the global call cap)."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
os.environ.setdefault("BLOGPIPE_REPO_ROOT", os.path.join(os.path.dirname(__file__), ".."))


def test_stage_quota_toggles_draft() -> None:
    from blogpipe import llm_chain

    llm_chain.reset_llm_usage()
    llm_chain.set_llm_call_cap(200)
    llm_chain.set_stage_quotas({"draft": 2})
    with llm_chain.stage("draft"):
        assert not llm_chain.is_stage_full("draft")
        assert not llm_chain.is_llm_call_cap_reached()
        llm_chain.bump_llm_ok()
        assert not llm_chain.is_stage_full("draft")
        llm_chain.bump_llm_ok()
        assert llm_chain.is_stage_full("draft")
        assert llm_chain.is_llm_call_cap_reached()
    u = llm_chain.get_llm_usage()
    assert u.get("by_stage", {}).get("draft") == 2


def test_global_cap_takes_precedence() -> None:
    from blogpipe import llm_chain

    llm_chain.reset_llm_usage()
    llm_chain.set_llm_call_cap(2)
    llm_chain.set_stage_quotas({"draft": 100})
    with llm_chain.stage("draft"):
        assert not llm_chain.is_llm_call_cap_reached()
        llm_chain.bump_llm_ok()
        assert not llm_chain.is_llm_call_cap_reached()
        llm_chain.bump_llm_ok()
        assert llm_chain.is_llm_call_cap_reached()


def test_second_stage_independent() -> None:
    from blogpipe import llm_chain

    llm_chain.reset_llm_usage()
    llm_chain.set_llm_call_cap(200)
    llm_chain.set_stage_quotas({"draft": 1, "editor": 1})
    with llm_chain.stage("draft"):
        llm_chain.bump_llm_ok()
        assert llm_chain.is_llm_call_cap_reached()
    with llm_chain.stage("editor"):
        assert not llm_chain.is_llm_call_cap_reached()
        llm_chain.bump_llm_ok()
        assert llm_chain.is_llm_call_cap_reached()


def test_config_stage_quotas_parses_json(monkeypatch: pytest.MonkeyPatch) -> None:
    from blogpipe import config

    monkeypatch.setenv("BLOGPIPE_STAGE_QUOTAS", '{"draft": 3, "editor": 7}')
    q = config.stage_quotas()
    assert q["draft"] == 3
    assert q["editor"] == 7
    assert "committee" in q
