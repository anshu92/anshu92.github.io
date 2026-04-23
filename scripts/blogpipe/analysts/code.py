from __future__ import annotations

from ..mcp_enrichment import _github_search_repos
from ..models import EvidencePack
from .base import _SCHEMA_HINT, run_analyst_task


def run(pack: EvidencePack):
    q = f"{pack.primary.title} pytorch implementation"
    items = _github_search_repos(q[:180]) or []
    ser = "\n".join(
        f"- {it.title} {it.url}\n  {it.abstract[:240]}" for it in items[:5]
    )
    return run_analyst_task(
        "code",
        "analyst_code",
        f"You link implementations and code ecosystem signals. {_SCHEMA_HINT}",
        f"Paper: {pack.primary.title}\n\nGitHub search:\n{ser or '(no repos found)'}\n",
    )
