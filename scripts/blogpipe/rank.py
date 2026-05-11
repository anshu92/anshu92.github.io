from __future__ import annotations

import json
import logging

from . import memory, score, store

LOG = logging.getLogger(__name__)


def run(*, db: str = "", limit: int = 50) -> list[object]:
    memory.ensure_dirs()
    with store.connect(db or None) as conn:
        items = store.load_items(conn, limit=500)
        ranked = score.rank_items(items, limit=limit)
        store.update_scores(conn, ranked)
    payload = {
        "count": len(ranked),
        "items": [r.model_dump(mode="json") for r in ranked],
    }
    (memory.REPORTS / "ranked_items.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    LOG.info("rank: %d ranked items", len(ranked))
    return ranked
