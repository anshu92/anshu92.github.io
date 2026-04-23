from __future__ import annotations

from ..mcp_enrichment import _web_search
from ..models import EvidencePack
from .base import _SCHEMA_HINT, run_analyst_task


def run(pack: EvidencePack):
    q = f"{pack.primary.title} blog review discussion replicates"
    items, _p = _web_search(q[:200])
    ser = "\n".join(
        f"- {it.title} {it.url}\n  {it.abstract[:240]}" for it in (items or [])[:6]
    )
    return run_analyst_task(
        "web",
        "analyst_web",
        f"You summarize public reception, blogs, and alternative takes. {_SCHEMA_HINT}",
        f"Paper: {pack.primary.title}\n\nSearch:\n{ser or '(no results)'}\n",
    )
