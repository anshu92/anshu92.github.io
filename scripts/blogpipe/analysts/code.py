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
    abs_ = (pack.primary.abstract or "")[:600]
    return run_analyst_task(
        "code",
        "analyst_code",
        f"Link repos and code signals to the paper. Put repository URLs you used in citations. "
        f"{_SCHEMA_HINT}",
        f"Title: {pack.primary.title}\nAbstract:\n{abs_}\n\nGitHub search:\n{ser or '(no repos found)'}\n",
    )
