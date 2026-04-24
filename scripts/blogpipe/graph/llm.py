"""LLM helpers for the graph; delegate to openrouter_client + llm_chain cap."""

from __future__ import annotations

import logging
from typing import Any, Optional

from .. import config, openrouter_client
from ..llm_chain import is_llm_call_cap_reached

LOG = logging.getLogger(__name__)


def graph_llm_text(
    tag: str,
    system: str,
    user: str,
    *,
    mode: str = "fast",
    max_tokens: Optional[int] = None,
    task: Optional[str] = None,
    temperature: float = 0.4,
) -> str:
    """Optional tag for dry-run canned paths; delegates to openrouter_client.llm_text."""
    if config.dry_run():
        return _dry_stub(tag, system, user)
    if is_llm_call_cap_reached():
        LOG.warning("graph_llm %s: call cap reached", tag)
        return ""
    if max_tokens is None:
        ovr = (task and config.max_tokens_for_task(task)) or 0
        if ovr > 0:
            max_tokens = ovr
        else:
            max_tokens = (
                config.max_tokens_smart()
                if mode == "smart"
                else config.max_tokens_fast()
            )
    return openrouter_client.llm_text(
        system, user, mode=mode, max_tokens=max_tokens, task=task, temperature=temperature
    )


def _dry_stub(tag: str, system: str, user: str) -> str:
    """Deterministic placeholders for CI smoke without API keys."""
    t = (tag or "").lower()
    if "rubric" in t or "score" in t:
        return (
            '{"rubric_score": 10, "rubric_items": [], '
            '"five_questions": {"problem": "x", "hard": "x", "tried": "x", '
            '"outcomes": "x", "next": "x"}, "five_questions_ok": true}'
        )
    if "ground" in t or "unsupported" in t:
        return '{"unsupported_claims": []}'
    if "critic" in t or "section" in t:
        return '{"verdict": "ok", "query_hint": ""}'
    if "plan" in t:
        return (
            '{"mandatory_claims":["name the main metric with baseline"],'
            '"mandatory_sections":["why this works","when to use it"],'
            '"required_visuals":["one mechanism diagram"],'
            '"likely_failures":["invented metrics","generic headings"],'
            '"reviewer_focus":["evidence grounding","decision usefulness"]}'
        )
    if "adversary" in t:
        return (
            '{"role":"adversary","pass_review":false,'
            '"findings":["Stress-test unsupported advice and vague comparative claims."],'
            '"rewrite_targets":["make tradeoffs explicit"],'
            '"summary":"Adversarial review requests stronger tradeoff language."}'
        )
    if "verify" in t or "verifier" in t:
        return (
            '{"role":"evidence_verifier","pass_review":true,'
            '"findings":[],"rewrite_targets":[],"summary":"Evidence review found no unsupported claims."}'
        )
    if "meta" in t:
        return (
            '{"role":"meta_reviewer","pass_review":true,'
            '"findings":[],"rewrite_targets":[],"summary":"Meta review accepts the draft."}'
        )
    if "rewrite" in t or "revise" in t:
        return "## Section\n\nDry-run rewrite body with [cite: primary].\n"
    if "outline" in t:
        return '{"sections": []}'
    # default: short markdown
    return "Dry-run LLM output for tag=%r." % tag[:40]
