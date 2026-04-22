from __future__ import annotations

import logging

from ..models import Item, Pillar
from ._http import get_json

LOG = logging.getLogger(__name__)

URL = "https://paperswithcode.com/api/v1/papers/?ordering=-publication_date&page=1&items_per_page=20"


def fetch() -> list[Item]:
    try:
        data = get_json(URL)
    except Exception as e:
        LOG.debug("paperswithcode: %s", e)
        return []
    if not isinstance(data, dict):
        return []
    results = (data or {}).get("results") or []
    out: list[Item] = []
    for p in results:
        if not isinstance(p, dict):
            continue
        pid = str(p.get("id", ""))
        title = (p.get("title") or "Paper").strip()
        url = p.get("url") or f"https://paperswithcode.com/paper/{p.get('slug', '')}"
        ab = (p.get("abstract") or "")[:4000]
        out.append(
            Item(
                id=f"pwc_{pid or title[:40]}",
                title=title,
                url=url,
                authors=[],
                abstract=ab,
                published_at=None,
                source="paperswithcode",
                tags=(p.get("task") and [p["task"]]) or ["ml"],
                pillar=Pillar.research,
            )
        )
    return out
