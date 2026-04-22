from __future__ import annotations

import logging

import feedparser

from ..models import Item, Pillar
from ._http import client

LOG = logging.getLogger(__name__)

# MLSys official feed (if available)
MLSYS = "https://proceedings.mlsys.org/rss"


def fetch() -> list[Item]:
    out: list[Item] = []
    for name, url in (("mlsys", MLSYS),):
        try:
            r = client().get(
                url,
                headers={"User-Agent": "blogpipe/0.1 (+https://anshu92.github.io)"},
            )
            r.raise_for_status()
        except Exception as e:
            LOG.debug("mlsys_proceedings %s: %s", name, e)
            continue
        try:
            parsed = feedparser.parse(r.text)
        except Exception:
            continue
        for e in (parsed.entries or [])[:15]:
            title = (e.get("title") or "").strip() or "Paper"
            link = (e.get("link") or "").strip() or url
            summ = (e.get("summary") or "")[:2000]
            out.append(
                Item(
                    id=f"mlsys_{abs(hash((link, title))) & 0xFFFFFFFF}",
                    title=title,
                    url=link,
                    authors=[],
                    abstract=summ,
                    published_at=None,
                    source="mlsys",
                    tags=["mlsys", "systems"],
                    pillar=Pillar.systems,
                )
            )
    return out
