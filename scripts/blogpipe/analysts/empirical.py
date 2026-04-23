from __future__ import annotations

from ..models import EvidencePack
from .base import _SCHEMA_HINT, run_analyst_task


def run(pack: EvidencePack):
    se = pack.section_evidence or {}
    keys = (
        "paper_experiments",
        "paper_results",
        "results",
        "paper_baseline",
    )
    blob = "\n\n".join(str(se.get(k) or "") for k in keys if (se.get(k) or "").strip())[
        :12000
    ]
    if not blob.strip() and pack.paper_result_notes:
        blob = "\n".join(pack.paper_result_notes[:20])[:8000]
    if not blob.strip():
        blob = (pack.primary.abstract or "")[:6000]
    return run_analyst_task(
        "empirical",
        "analyst_empirical",
        f"Extract benchmark numbers, ablations, and baselines that appear verbatim in the text. "
        f"Do not infer metrics. {_SCHEMA_HINT}",
        f"Title: {pack.primary.title}\n\nResults / metrics:\n{blob}\n",
    )
