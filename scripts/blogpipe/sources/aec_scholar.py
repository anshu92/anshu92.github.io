from __future__ import annotations

import logging
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import feedparser

from .. import config
from ..models import Item, Pillar
from ._http import client

LOG = logging.getLogger(__name__)


def fetch() -> list[Item]:
    url = config.aec_scholar_rss()
    if not url or not url.startswith("http"):
        return []
    out: list[Item] = []
    try:
        r = client().get(
            url,
            headers={"User-Agent": "blogpipe/0.1 (+https://anshu92.github.io)"},
        )
        r.raise_for_status()
    except Exception as e:
        LOG.warning("aec_scholar: %s", e)
        return []
    try:
        parsed = feedparser.parse(r.text)
    except Exception:
        return []
    import time

    for e in (parsed.entries or [])[:25]:
        title = (e.get("title") or "").strip() or "Scholar alert"
        link = (e.get("link") or "").strip() or url
        summ = (e.get("summary") or "")[:2000]
        pub = None
        if e.get("published_parsed"):
            try:
                pub = datetime.fromtimestamp(
                    time.mktime(e["published_parsed"]),
                    tz=timezone.utc,
                )
            except Exception:
                pub = None
        if pub is None and e.get("published"):
            try:
                pub = parsedate_to_datetime(e["published"])
            except Exception:
                pub = None
        out.append(
            Item(
                id=f"aec_scholar_{abs(hash((link, title))) & 0xFFFFFFFF}",
                title=title,
                url=link,
                authors=[],
                abstract=summ,
                published_at=pub,
                source="aec_scholar_rss",
                tags=["aec", "scholar"],
                pillar=Pillar.aec,
            )
        )
    return out
