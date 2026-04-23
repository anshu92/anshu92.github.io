from __future__ import annotations

from ..models import EvidencePack
from .base import _SCHEMA_HINT, run_analyst_task


def run(pack: EvidencePack):
    refs = "\n".join(f"- {i.title} — {i.url}" for i in pack.refs[:10])
    cits = "\n".join(f"- {i.title} — {i.url}" for i in pack.cits[:10])
    abs_ = (pack.primary.abstract or "")[:600]
    return run_analyst_task(
        "related",
        "analyst_related",
        f"Lineage: prior and follow-up work. Put main reference and follow-up URLs in citations. "
        f"{_SCHEMA_HINT}",
        f"Primary: {pack.primary.title}\nAbstract:\n{abs_}\n\n"
        f"Main references (Semantic Scholar):\n{refs or '(none)'}\n\n"
        f"Cited by / follow-ups:\n{cits or '(none)'}",
    )
