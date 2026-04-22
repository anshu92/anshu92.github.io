from __future__ import annotations

import logging
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Optional

import feedparser

from ..models import Item, Pillar
from ._http import client

LOG = logging.getLogger(__name__)

URLS: list[str] = [
    "https://www.research.autodesk.com/feed",
    "https://www.research.autodesk.com/blog/feed",
]


def fetch() -> list[Item]:
    out: list[Item] = []
    for url in URLS:
        try:
            r = client().get(
                url,
                headers={"User-Agent": "blogpipe/0.1 (+https://anshu92.github.io)"},
            )
            r.raise_for_status()
        except Exception as e:
            LOG.debug("autodesk %s: %s", url, e)
            continue
        try:
            parsed = feedparser.parse(r.text)
        except Exception:
            continue
        for e in (parsed.entries or [])[:20]:
            title = (e.get("title") or "").strip() or "Autodesk"
            link = (e.get("link") or "").strip() or url
            summ = (e.get("summary") or "")[:2000]
            pub = _parse_published(e)
            out.append(
                Item(
                    id=f"adsk_{abs(hash((link, title))) & 0xFFFFFFFF}",
                    title=title,
                    url=link,
                    authors=[],
                    abstract=summ,
                    published_at=pub,
                    source="autodesk_research",
                    tags=["aec", "autodesk"],
                    pillar=Pillar.aec,
                )
            )
    return out


def _parse_published(e: dict) -> Optional[datetime]:
    import time
    from datetime import timezone

    if e.get("published_parsed"):
        try:
            return datetime.fromtimestamp(
                time.mktime(e["published_parsed"]),
                tz=timezone.utc,
            )
        except Exception:
            pass
    p = e.get("published")
    if isinstance(p, str) and p:
        try:
            return parsedate_to_datetime(p)
        except Exception:
            return None
    return None
