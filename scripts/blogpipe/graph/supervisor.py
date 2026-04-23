"""Supervisor: picks committee analysts (LLM) or defers to config list. Logs audit strings."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from .. import config, openrouter_client
from ..analysts import RUNNERS
from ..llm_chain import is_llm_call_cap_reached
from ..models import EvidencePack, Item

LOG = logging.getLogger(__name__)

# Always run (when in eligibility set; env may narrow committee list)
_CORE = ("methods", "empirical", "glossary")
_OPTIONAL = (
    "adversarial",
    "related",
    "practitioner",
    "code",
    "web",
    "visual_planner",
)


def log_decision(state: dict[str, Any], action: str) -> list[str]:
    """Return updated supervisor_decisions list for state merge."""
    d = list(state.get("supervisor_decisions") or [])
    d.append(action)
    return d


def _eligibility_set() -> set[str]:
    return {n.strip() for n in config.committee_analysts() if n.strip()}


def _core_optional(elig: set[str]) -> tuple[list[str], list[str]]:
    core = [a for a in _CORE if a in elig]
    opt = [a for a in _OPTIONAL if a in elig]
    return core, opt


def _parse_picks(text: str) -> list[str]:
    if not text or not text.strip():
        return []
    t = text.strip()
    m = re.search(r"\{[\s\S]*\}", t)
    if m:
        t = m.group(0)
    try:
        data = json.loads(t)
    except json.JSONDecodeError:
        return []
    p = data.get("picks")
    if not isinstance(p, list):
        return []
    return [str(x).strip() for x in p if str(x).strip()]


def _llm_picks(
    item: Item, optional_pool: list[str]
) -> list[str] | None:
    if is_llm_call_cap_reached() or not config.llm_configured() or config.dry_run():
        return None
    if not optional_pool:
        return []
    pool = ", ".join(optional_pool)
    sys_ = (
        "Route sub-analysts for a blog pipeline. Pick 2 to 4 slugs from the optional list; pick fewer "
        "if the title clearly needs fewer, or the remaining slugs are a bad fit. "
        'Output JSON only: {"picks": ["slug1", "slug2", ...]}. Example: {"picks": ["related", "code"]}.'
    )
    user = (
        f"Title: {item.title}\n"
        f"Optional slugs: [{pool}]\n"
        "Heuristic: related+visuals for theory; code+web for systems; adversarial+practitioner for "
        "strong applied claims. Slugs from the list only, no new names."
    )
    out = openrouter_client.llm_text(
        sys_,
        user,
        max_tokens=600,
        task="supervisor_route",
    )
    if not out:
        return None
    raw = _parse_picks(out)
    return [p for p in raw if p in set(optional_pool)][:4]


def node_supervisor(state: dict[str, Any]) -> dict[str, Any]:
    """Set selected_analysts: core + 2-4 LLM-picked optionals, filtered by config eligibility."""
    elig = _eligibility_set()
    core, optional_pool = _core_optional(elig)
    if not config.supervisor_enabled():
        # Legacy: all analysts in env order, filtered to RUNNERS
        selected = [n for n in config.committee_analysts() if n in RUNNERS]
        return {
            "selected_analysts": selected,
            "supervisor_decisions": log_decision(state, f"supervisor_bypass={selected}"),
        }
    if not state.get("evidence_pack"):
        return {
            "selected_analysts": list(core) if core else [],
            "supervisor_decisions": log_decision(state, "supervisor_no_pack"),
        }
    try:
        pack = EvidencePack.model_validate(state.get("evidence_pack"))
    except Exception:  # noqa: BLE001
        return {
            "selected_analysts": list(core) if core else [],
            "supervisor_decisions": log_decision(state, "supervisor_invalid_pack"),
        }
    it = pack.primary
    picks = _llm_picks(it, optional_pool)
    if picks is None:
        # dry-run / no LLM / cap: fallback
        k = min(3, max(2, len(optional_pool)))
        picks = optional_pool[:k] if optional_pool else []
    if len(picks) < 2 and len(optional_pool) >= 2:
        for o in optional_pool:
            if o not in picks:
                picks.append(o)
            if len(picks) >= 2:
                break
    if len(picks) > 4:
        picks = picks[:4]
    seen: set[str] = set()
    ordered: list[str] = []
    for a in list(core) + list(picks):
        if a in seen:
            continue
        seen.add(a)
        if a in elig:
            ordered.append(a)
    return {
        "selected_analysts": ordered,
        "supervisor_decisions": log_decision(
            state,
            f"supervisor picks={picks!r} core={core!r}",
        ),
    }
