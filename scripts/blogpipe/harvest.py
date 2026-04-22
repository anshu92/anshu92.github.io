"""Run all sources, write reports/harvested.json."""

from __future__ import annotations

import json
import logging

from . import memory
from .models import Item
from .sources.aggregator import harvest_all
from .memory import _ROOT

LOG = logging.getLogger(__name__)


def run() -> list[Item]:
    memory.ensure_dirs()
    items = harvest_all()
    p = _ROOT / "reports" / "harvested.json"
    p.write_text(
        json.dumps(
            {
                "count": len(items),
                "items": [i.model_dump(mode="json") for i in items],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    LOG.info("harvest: %d items", len(items))
    return items
