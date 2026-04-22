from __future__ import annotations

import logging
from typing import Optional
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote

from ..models import Item, Pillar
from ._http import client

LOG = logging.getLogger(__name__)

# Last 2 days, categories LLM + systems
_CATS = "cat:cs.CL+OR+cat:cs.LG+OR+cat:cs.AI+OR+cat:cs.CV+OR+cat:cs.DC+OR+cat:cs.PF"


def _atom_url() -> str:
    base = "http://export.arxiv.org/api/query"
    q = f"search_query={quote(_CATS)}&start=0&maxResults=50&sortBy=submittedDate&sortOrder=descending"
    return f"{base}?{q}"


def fetch() -> list[Item]:
    url = _atom_url()
    try:
        r = client().get(
            url, headers={"User-Agent": "blogpipe/0.1 (+https://anshu92.github.io)"}
        )
        r.raise_for_status()
    except Exception as e:
        LOG.warning("arxiv: %s", e)
        return []
    try:
        root = ET.fromstring(r.text)
    except ET.ParseError as e:
        LOG.warning("arxiv parse: %s", e)
        return []
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out: list[Item] = []
    for ent in root.findall("a:entry", ns):
        title_el = ent.find("a:title", ns)
        id_el = ent.find("a:id", ns)
        summ_el = ent.find("a:summary", ns)
        pub_el = ent.find("a:published", ns)
        title = (title_el.text or "").replace("\n", " ").strip() if title_el is not None else ""
        eid = (id_el.text or "").strip() if id_el is not None else ""
        abstract = (summ_el.text or "").strip() if summ_el is not None else ""
        pub = _parse_ts(pub_el.text if pub_el is not None else None)
        authors: list[str] = []
        for a in ent.findall("a:author", ns):
            n = a.find("a:name", ns)
            if n is not None and n.text:
                authors.append(n.text.strip())
        arxiv_id = eid.rsplit("/abs/", 1)[-1] if eid else title[:64]
        pillar = Pillar.systems if any(
            c in (eid + title) for c in ("/cs.DC", "/cs.PF")
        ) else Pillar.research
        if pillar is Pillar.research and any(
            x in title.lower() for x in ("distributed", "inference", "throughput", "kernel")
        ):
            pillar = Pillar.systems
        out.append(
            Item(
                id=f"arxiv_{arxiv_id}",
                title=title or "arXiv",
                url=eid or f"https://arxiv.org/abs/{arxiv_id}",
                authors=authors,
                abstract=abstract[:4000],
                published_at=pub,
                source="arxiv",
                tags=["arxiv", "ml"],
                pillar=pillar,
            )
        )
    return out


def _parse_ts(s: Optional[str]) -> Optional[datetime]:
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
