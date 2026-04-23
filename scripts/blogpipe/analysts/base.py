"""Shared LLM + JSON parsing for committee analysts."""

from __future__ import annotations

import json
import re
import logging
from typing import Any

from .. import config, openrouter_client
from ..llm_chain import is_llm_call_cap_reached
from ..models import AnalystNote

LOG = logging.getLogger(__name__)

_SCHEMA_HINT = (
    "Output JSON only: "
    '{"claims": ["…"], "citations": ["…"], "confidence": "low|medium|high", '
    '"contradictions": ["…"], "suggested_section": "H2 or empty"}.'
)


def _parse_json_note(raw: str, role: str) -> AnalystNote:
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return AnalystNote(
            role=role,
            claims=[],
            skipped=True,
        )
    try:
        d: dict[str, Any] = json.loads(m.group(0))
    except (json.JSONDecodeError, TypeError) as e:
        LOG.debug("analyst %s json: %s", role, e)
        return AnalystNote(role=role, claims=[], skipped=True)
    cl = d.get("claims") or d.get("claim") or []
    if isinstance(cl, str):
        cl = [cl]
    ci = d.get("citations") or []
    if isinstance(ci, str):
        ci = [ci]
    co = d.get("contradictions") or []
    if isinstance(co, str):
        co = [co]
    return AnalystNote(
        role=role,
        claims=[str(x)[:2000] for x in cl if str(x).strip()][:12],
        citations=[str(x)[:500] for x in ci if str(x).strip()][:20],
        confidence=str(d.get("confidence", "medium"))[:20],
        contradictions=[str(x)[:2000] for x in co if str(x).strip()][:8],
        suggested_section=str(d.get("suggested_section", "") or "")[:200],
    )


def run_analyst_task(
    role: str,
    task: str,
    system: str,
    user: str,
) -> AnalystNote:
    if is_llm_call_cap_reached() or not config.llm_configured() or config.dry_run():
        return AnalystNote(role=role, claims=[], skipped=True)
    cap = int(config.committee_per_analyst_max_tokens())
    raw = openrouter_client.llm_text(
        system,
        user[:24000],
        max_tokens=cap,
        task=task,
    )
    if not (raw or "").strip():
        return AnalystNote(role=role, claims=[], skipped=True)
    return _parse_json_note(raw, role)
