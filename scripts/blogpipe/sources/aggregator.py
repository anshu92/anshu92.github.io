from __future__ import annotations

import logging

from ..models import Item
from . import (
    aec_scholar,
    arxiv,
    autodesk_research,
    eng_blogs,
    huggingface_papers,
    mlsys_proceedings,
    nvidia_aec,
    paperswithcode,
)

LOG = logging.getLogger(__name__)


def harvest_all() -> list[Item]:
    """Merge all fixed sources; dedupe by URL."""
    batches: list[list[Item]] = [
        huggingface_papers.fetch(),
        arxiv.fetch(),
        paperswithcode.fetch(),
        mlsys_proceedings.fetch(),
        eng_blogs.fetch(),
        autodesk_research.fetch(),
        nvidia_aec.fetch(),
        aec_scholar.fetch(),
    ]
    seen: set[str] = set()
    out: list[Item] = []
    for batch in batches:
        for it in batch:
            key = it.url.split("#", 1)[0].rstrip("/")
            if key in seen:
                continue
            seen.add(key)
            out.append(it)
    LOG.info("harvest_all: %d unique items", len(out))
    return out
