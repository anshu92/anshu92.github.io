from __future__ import annotations

from ..models import EvidencePack
from .base import _SCHEMA_HINT, run_analyst_task


def run(pack: EvidencePack):
    refs = "\n".join(f"- {i.title} — {i.url}" for i in pack.refs[:10])
    cits = "\n".join(f"- {i.title} — {i.url}" for i in pack.cits[:10])
    return run_analyst_task(
        "related",
        "analyst_related",
        f"You place this work in the lineage of prior and follow-up work. {_SCHEMA_HINT}",
        f"Primary: {pack.primary.title}\n\nMain references (Semantic Scholar):\n{refs or '(none)'}\n\nCited by / follow-ups:\n{cits or '(none)'}",
    )
