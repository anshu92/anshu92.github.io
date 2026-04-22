from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import feedparser

from ..models import Item, Pillar
from ._http import client

LOG = logging.getLogger(__name__)

# NVIDIA main blog; filter in Python
BLOG = "https://blogs.nvidia.com/feed/"

AEC_PAT = re.compile(
    r"\b(bim|aec|building|construction|infrastructure|cad|omniverse.*aec|"
    r"architect\w*|structural|generative design)\b",
    re.I,
)


def fetch() -> list[Item]:
    out: list[Item] = []
    try:
        r = client().get(
            BLOG, headers={"User-Agent": "blogpipe/0.1 (+https://anshu92.github.io)"}
        )
        r.raise_for_status()
    except Exception as e:
        LOG.warning("nvidia_aec: %s", e)
        return []
    try:
        parsed = feedparser.parse(r.text)
    except Exception:
        return []
    import time

    for e in parsed.entries or []:
        title = (e.get("title") or "").strip()
        link = (e.get("link") or "").strip()
        summ = (e.get("summary") or "")[:2000]
        text = f"{title} {summ}"
        if not AEC_PAT.search(text):
            continue
        pub = None
        if e.get("published_parsed"):
            try:
                pub = datetime.fromtimestamp(
                    time.mktime(e["published_parsed"]),
                    tz=timezone.utc,
                )
            except Exception:
                pub = None
        if pub is None and e.get("published"):
            try:
                pub = parsedate_to_datetime(e["published"])
            except Exception:
                pub = None
        out.append(
            Item(
                id=f"nv_aec_{abs(hash((link, title))) & 0xFFFFFFFF}",
                title=title or "NVIDIA",
                url=link or BLOG,
                authors=[],
                abstract=summ,
                published_at=pub,
                source="nvidia_aec",
                tags=["nvidia", "aec"],
                pillar=Pillar.aec,
            )
        )
        if len(out) >= 10:
            break
    return out
