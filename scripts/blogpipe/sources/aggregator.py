from __future__ import annotations

import gzip
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from pydantic import TypeAdapter

from .. import memory
from ..models import SourceItem
from . import acl, arxiv, blogs, enrichers, openreview

LOG = logging.getLogger(__name__)
ITEMS = TypeAdapter(list[SourceItem])


def harvest_all(*, window_hours: int = 72, fixtures: str = "") -> list[SourceItem]:
    if fixtures:
        return _load_fixtures(fixtures)
    batches = [
        arxiv.fetch(window_hours),
        acl.fetch(window_hours),
        openreview.fetch(window_hours),
        blogs.fetch(window_hours),
    ]
    merged = _dedupe(_filter_recent([item for batch in batches for item in batch], window_hours=window_hours))
    return enrichers.enrich(merged)


def write_snapshot(items: list[SourceItem]) -> Path:
    memory.ensure_dirs()
    name = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ.jsonl.gz")
    path = memory.DAILY_DATA / name
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for item in items:
            fh.write(item.normalized().model_dump_json() + "\n")
    return path


def _dedupe(items: list[SourceItem]) -> list[SourceItem]:
    out: list[SourceItem] = []
    seen: set[str] = set()
    for item in items:
        norm = item.normalized()
        key = norm.item_id
        if key in seen:
            continue
        seen.add(key)
        out.append(norm)
    return out


def _filter_recent(
    items: list[SourceItem],
    *,
    window_hours: int,
    now: datetime | None = None,
) -> list[SourceItem]:
    """Keep only dated items inside the configured freshness window."""
    current = now or datetime.now(timezone.utc)
    out: list[SourceItem] = []
    for item in items:
        published = item.published_at or item.updated_at
        if not published:
            continue
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        age_hours = (current - published).total_seconds() / 3600.0
        if 0 <= age_hours <= float(window_hours):
            out.append(item)
    LOG.info("recency filter: %d -> %d items within %dh", len(items), len(out), window_hours)
    return out


def _load_fixtures(path: str) -> list[SourceItem]:
    p = Path(path)
    if p.is_dir():
        p = p / "source_items.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    raw = data.get("items", data) if isinstance(data, dict) else data
    return [item.normalized() for item in ITEMS.validate_python(raw)]
