from __future__ import annotations

import logging
from datetime import datetime, timezone

import feedparser

from ..models import SourceItem
from ._http import client

LOG = logging.getLogger(__name__)
FEED_URL = "https://aclanthology.org/anthology+abstracts.bib.gz"


def fetch(window_hours: int = 72) -> list[SourceItem]:
    """Best-effort ACL adapter.

    ACL does not expose the same simple daily Atom endpoint as arXiv. For v1 we
    keep this adapter conservative and fixture-testable: if the network path is
    unavailable or too large for a CI run, it quietly returns no items.
    """
    try:
        resp = client().get("https://aclanthology.org/rss.xml")
        if resp.status_code >= 400:
            return []
        parsed = feedparser.parse(resp.text)
    except Exception as exc:
        LOG.debug("acl fetch skipped: %s", exc)
        return []
    out: list[SourceItem] = []
    for entry in (parsed.entries or [])[:25]:
        title = (entry.get("title") or "").strip()
        link = (entry.get("link") or "").strip()
        if not title or not link:
            continue
        out.append(
            SourceItem(
                canonical_url=link,
                source_kind="paper",
                source_name="acl",
                source_tier=1,
                title=title,
                published_at=_entry_dt(entry),
                venue_or_blog="ACL Anthology",
                abstract_or_excerpt=(entry.get("summary") or "")[:2000],
                tags=["paper", "acl", "nlp"],
            ).normalized()
        )
    return out


def _entry_dt(entry: object) -> datetime | None:
    parsed = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
    if not parsed:
        return None
    try:
        return datetime(*parsed[:6], tzinfo=timezone.utc)
    except Exception:
        return None
