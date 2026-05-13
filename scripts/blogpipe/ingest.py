from __future__ import annotations

import logging

from . import memory, store
from .sources.aggregator import harvest_all, write_snapshot

LOG = logging.getLogger(__name__)


def run(*, window_hours: int = 14 * 24, fixtures: str = "", db: str = "") -> int:
    memory.ensure_dirs()
    items = harvest_all(window_hours=window_hours, fixtures=fixtures)
    with store.connect(db or None) as conn:
        count = store.upsert_items(conn, items)
    snapshot = write_snapshot(items)
    LOG.info("ingest: stored %d items; snapshot=%s", count, snapshot)
    return count
