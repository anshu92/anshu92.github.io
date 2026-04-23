from __future__ import annotations

from ..mcp_enrichment import _web_search
from ..models import EvidencePack
from .base import _SCHEMA_HINT, run_analyst_task


def run(pack: EvidencePack):
    q = f"{pack.primary.title} deployment hardware cost when to use"
    items, _p = _web_search(q[:200])
    ser = "\n".join(
        f"- {it.title} {it.url}\n  {it.abstract[:240]}" for it in (items or [])[:5]
    )
    abs_ = (pack.primary.abstract or "")[:600]
    return run_analyst_task(
        "practitioner",
        "analyst_practitioner",
        f"Adoption: fit, preconditions, gotchas, validation cost. Use search snippets. {_SCHEMA_HINT}",
        f"Title: {pack.primary.title}\nAbstract:\n{abs_}\n\nWeb:\n{ser or '(no results)'}\n",
    )
