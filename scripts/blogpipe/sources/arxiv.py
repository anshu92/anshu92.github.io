from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

from ..models import Author, SourceItem
from ._http import client

LOG = logging.getLogger(__name__)
ARXIV_NS = {"a": "http://www.w3.org/2005/Atom"}
CATEGORIES = "cat:cs.CL OR cat:cs.LG OR cat:cs.AI OR cat:cs.CV OR cat:cs.DC OR cat:cs.PF"


def fetch(window_hours: int = 72) -> list[SourceItem]:
    since = datetime.now(timezone.utc) - timedelta(hours=window_hours)
    date_filter = since.strftime("%Y%m%d%H%M")
    query = f"({CATEGORIES}) AND submittedDate:[{date_filter} TO 999912312359]"
    url = (
        "http://export.arxiv.org/api/query?"
        f"search_query={quote(query)}&sortBy=submittedDate&sortOrder=descending&start=0&max_results=100"
    )
    try:
        resp = client().get(url)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
    except Exception as exc:
        LOG.warning("arxiv fetch failed: %s", exc)
        return []
    out: list[SourceItem] = []
    for entry in root.findall("a:entry", ARXIV_NS):
        item = _entry(entry)
        if item:
            out.append(item)
    return out


def _text(entry: ET.Element, tag: str) -> str:
    node = entry.find(f"a:{tag}", ARXIV_NS)
    return " ".join((node.text or "").split()) if node is not None else ""


def _entry(entry: ET.Element) -> SourceItem | None:
    title = _text(entry, "title")
    url = _text(entry, "id")
    if not title or not url:
        return None
    arxiv_id = url.rsplit("/abs/", 1)[-1]
    authors = []
    for author in entry.findall("a:author", ARXIV_NS):
        name_node = author.find("a:name", ARXIV_NS)
        if name_node is not None and name_node.text:
            authors.append(Author(name=name_node.text.strip()))
    return SourceItem(
        canonical_url=url,
        source_kind="paper",
        source_name="arxiv",
        source_tier=1,
        title=title,
        authors=authors,
        published_at=_parse_dt(_text(entry, "published")),
        updated_at=_parse_dt(_text(entry, "updated")),
        arxiv_id=arxiv_id,
        venue_or_blog="arXiv",
        abstract_or_excerpt=_text(entry, "summary"),
        tags=["paper", "arxiv"],
    ).normalized()


def _parse_dt(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
