from __future__ import annotations

from ..models import EvidencePack
from .base import _SCHEMA_HINT, run_analyst_task


def run(pack: EvidencePack):
    lims = "\n".join(pack.paper_limitations[:15])
    return run_analyst_task(
        "adversarial",
        "analyst_adversarial",
        f"Critically stress-test claims. Flag overclaim, missing baselines, and threats to validity. {_SCHEMA_HINT}",
        f"Title: {pack.primary.title}\nAbstract:\n{pack.primary.abstract[:4000]}\n\n"
        f"Stated limitations:\n{lims}\n",
    )
