from __future__ import annotations

import logging
from datetime import datetime, timezone

import feedparser

from ..models import SourceItem
from ._http import client

LOG = logging.getLogger(__name__)

FEEDS: tuple[tuple[str, str, int], ...] = (
    ("openai_research", "https://openai.com/news/rss.xml", 1),
    ("google_research", "https://blog.research.google/feeds/posts/default?alt=rss", 1),
    ("huggingface", "https://huggingface.co/blog/feed.xml", 1),
    ("pytorch", "https://pytorch.org/feed.xml", 1),
    ("nvidia_developer", "https://developer.nvidia.com/blog/feed/", 1),
    ("autodesk_research", "https://www.research.autodesk.com/feed/", 1),
)


def fetch(window_hours: int = 72) -> list[SourceItem]:
    out: list[SourceItem] = []
    for name, url, tier in FEEDS:
        try:
            resp = client().get(url)
            if resp.status_code >= 400:
                continue
            parsed = feedparser.parse(resp.text)
        except Exception as exc:
            LOG.debug("blog feed %s skipped: %s", name, exc)
            continue
        for entry in (parsed.entries or [])[:10]:
            title = (entry.get("title") or "").strip()
            link = (entry.get("link") or "").strip()
            if not title or not link:
                continue
            out.append(
                SourceItem(
                    canonical_url=link,
                    source_kind="blog",
                    source_name=name,
                    source_tier=tier,
                    title=title,
                    published_at=_entry_dt(entry),
                    updated_at=_entry_dt(entry, key="updated_parsed"),
                    venue_or_blog=name.replace("_", " ").title(),
                    abstract_or_excerpt=(entry.get("summary") or entry.get("description") or "")[:2500],
                    tags=["blog", name],
                ).normalized()
            )
    return out


def _entry_dt(entry: object, key: str = "published_parsed") -> datetime | None:
    parsed = getattr(entry, key, None)
    if not parsed:
        return None
    try:
        return datetime(*parsed[:6], tzinfo=timezone.utc)
    except Exception:
        return None
