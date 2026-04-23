from __future__ import annotations

from ..models import EvidencePack
from .base import _SCHEMA_HINT, run_analyst_task


def run(pack: EvidencePack):
    se = pack.section_evidence or {}
    keys = (
        "paper_method",
        "paper_approach",
        "method",
        "paper_experiments",
    )
    blob = "\n\n".join(str(se.get(k) or "") for k in keys if (se.get(k) or "").strip())[
        :12000
    ]
    if not blob.strip():
        blob = (pack.primary.abstract or "")[:6000]
    return run_analyst_task(
        "methods",
        "analyst_methods",
        f"You are a methods specialist. Extract algorithmic and architectural claims. {_SCHEMA_HINT}",
        f"Title: {pack.primary.title}\n\nExcerpts:\n{blob}\n",
    )
