"""Glossary analyst: extracts technical terms + 1–2 sentence definitions."""

from __future__ import annotations

from ..models import EvidencePack
from .base import run_analyst_task

_SCHEMA = (
    "Output JSON only: "
    '{"claims": ["TERM — 1–2 sentence definition for an ML practitioner", ...], '
    '"citations": [], "confidence": "low|medium|high", '
    '"contradictions": [], "suggested_section": ""}. '
    "Each claim starts with TERM — . At most 12 terms: acronyms, methods, benchmarks, named "
    "components. Self-contained, factual, prefer paper wording."
)


def run(pack: EvidencePack):
    se = pack.section_evidence or {}
    blob = "\n\n".join(
        str(se.get(k) or "")
        for k in (
            "paper_problem",
            "paper_method",
            "paper_experiments",
            "paper_limitations",
        )
        if (se.get(k) or "").strip()
    )[:10000]
    if not blob.strip():
        blob = (pack.primary.abstract or "")[:6000]
    return run_analyst_task(
        "glossary",
        "analyst_glossary",
        f"Build a precise glossary. {_SCHEMA}",
        f"Title: {pack.primary.title}\n\nExcerpts:\n{blob}\n",
    )
