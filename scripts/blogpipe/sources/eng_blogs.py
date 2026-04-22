from __future__ import annotations

import logging
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Optional

import feedparser

from ..models import Item, Pillar
from ._http import client

LOG = logging.getLogger(__name__)

# Curated, fixed URLs
FEEDS: list[tuple[str, str]] = [
    ("huggingface", "https://huggingface.co/blog/feed.xml"),
    ("databricks", "https://www.databricks.com/feed"),
    ("pytorch", "https://pytorch.org/blog/feed"),
    # Many vendor blogs: best-effort; skip if 404
    (
        "google_ai",
        "https://blog.research.google/feeds/posts/default?alt=rss",
    ),
]


def fetch() -> list[Item]:
    out: list[Item] = []
    for name, url in FEEDS:
        try:
            r = client().get(
                url,
                headers={"User-Agent": "blogpipe/0.1 (+https://anshu92.github.io)"},
            )
            r.raise_for_status()
        except Exception as e:
            LOG.debug("eng_blogs %s: %s", name, e)
            continue
        try:
            parsed = feedparser.parse(r.text)
        except Exception:
            continue
        for e in (parsed.entries or [])[:8]:
            title = (e.get("title") or "").strip() or "Post"
            link = (e.get("link") or "").strip() or url
            summ = (e.get("summary") or e.get("description") or "")[:2000]
            pub = _parse_date(
                e.get("published")
                or e.get("updated")
                or (e.get("published_parsed") and e.get("updated"))
            )
            pillar = Pillar.systems if any(
                w in title.lower()
                for w in (
                    "inference",
                    "gpu",
                    "train",
                    "scale",
                    "throughput",
                    "kernel",
                    "distributed",
                )
            ) else Pillar.applied
            out.append(
                Item(
                    id=f"blog_{name}_{abs(hash((link, title))) & 0xFFFFFFFF}",
                    title=title,
                    url=link,
                    authors=[],
                    abstract=summ,
                    published_at=pub,
                    source=f"eng_blog_{name}",
                    tags=["blog", name],
                    pillar=pillar,
                )
            )
    return out


def _parse_date(v: object) -> Optional[datetime]:
    if v is None:
        return None
    if isinstance(v, str) and v:
        try:
            return parsedate_to_datetime(v)
        except Exception:
            return None
    if hasattr(v, "tm_year"):
        import time

        try:
            return datetime.fromtimestamp(time.mktime(v))  # type: ignore[arg-type]
        except Exception:
            return None
    return None
