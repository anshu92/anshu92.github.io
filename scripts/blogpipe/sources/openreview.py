from __future__ import annotations

import logging
from datetime import datetime, timezone

from ..models import SourceItem
from ._http import client

LOG = logging.getLogger(__name__)


def fetch(window_hours: int = 72) -> list[SourceItem]:
    params = {
        "limit": 50,
        "sort": "tmdate:desc",
        "content.venue": "ICLR.cc/2026/Conference",
    }
    try:
        resp = client().get("https://api2.openreview.net/notes", params=params)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        LOG.debug("openreview fetch skipped: %s", exc)
        return []
    out: list[SourceItem] = []
    for note in payload.get("notes") or []:
        content = note.get("content") or {}
        title = _field(content, "title")
        abstract = _field(content, "abstract")
        note_id = note.get("id") or ""
        if not title or not note_id:
            continue
        out.append(
            SourceItem(
                canonical_url=f"https://openreview.net/forum?id={note_id}",
                source_kind="paper",
                source_name="openreview",
                source_tier=1,
                title=title,
                published_at=_ms_dt(note.get("pdate") or note.get("cdate")),
                updated_at=_ms_dt(note.get("tmdate")),
                venue_or_blog="OpenReview",
                abstract_or_excerpt=abstract,
                tags=["paper", "openreview"],
                extra={"openreview_id": note_id},
            ).normalized()
        )
    return out


def _field(content: dict, key: str) -> str:
    value = content.get(key)
    if isinstance(value, dict):
        value = value.get("value")
    return " ".join(str(value or "").split())


def _ms_dt(value: object) -> datetime | None:
    try:
        ivalue = int(value)
    except (TypeError, ValueError):
        return None
    if ivalue <= 0:
        return None
    return datetime.fromtimestamp(ivalue / 1000.0, tz=timezone.utc)
