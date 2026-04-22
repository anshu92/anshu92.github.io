from __future__ import annotations

import logging
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Optional

from ..models import Item, Pillar
from ._http import get_json

LOG = logging.getLogger(__name__)

API = "https://huggingface.co/api/daily_papers"


def fetch() -> list[Item]:
    try:
        data = get_json(API)
    except Exception as e:
        LOG.warning("huggingface_papers: %s", e)
        return []
    if not isinstance(data, list):
        return []
    out: list[Item] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        paper = row.get("paper")
        if not isinstance(paper, dict):
            continue
        title = (paper.get("title") or "").strip() or "Untitled"
        aid = str(paper.get("id") or title)[:200]
        arx = aid.replace("arXiv:", "").strip()
        if arx and arx[0].isdigit():
            url = f"https://arxiv.org/abs/{arx}"
        else:
            url = f"https://huggingface.co/papers/{aid}"
        published = _parse_date(paper.get("publishedAt") or row.get("publishedAt"))
        authors: list[str] = []
        for a in paper.get("authors") or []:
            if isinstance(a, str):
                authors.append(a)
            elif isinstance(a, dict) and a.get("name"):
                authors.append(str(a["name"]))
        ab = (paper.get("summary") or paper.get("abstract") or "")[:2000]
        out.append(
            Item(
                id=f"hf_{aid}",
                title=title,
                url=url,
                authors=authors,
                abstract=ab,
                published_at=published,
                source="huggingface_daily_papers",
                tags=["llm", "paper"],
                pillar=Pillar.research,
                extra={"raw": row},
            )
        )
    return out


def _parse_date(v: Any) -> Optional[datetime]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        try:
            return datetime.fromtimestamp(v, tz=timezone.utc)
        except Exception:
            return None
    s = str(v).strip()
    if not s:
        return None
    try:
        if "T" in s:
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        pass
    try:
        return parsedate_to_datetime(s)
    except Exception:
        return None
