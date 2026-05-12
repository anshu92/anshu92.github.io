from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

from .. import config
from ..models import Author, SourceItem
from ._http import client

LOG = logging.getLogger(__name__)
ARXIV_NS = {"a": "http://www.w3.org/2005/Atom"}


@dataclass(frozen=True)
class SearchProfile:
    name: str
    categories: tuple[str, ...]
    terms: tuple[str, ...]


ARXIV_PROFILES: tuple[SearchProfile, ...] = (
    SearchProfile(
        "llm_methods",
        ("cs.CL", "cs.AI", "cs.LG"),
        ("language model", "reasoning", "alignment", "post-training", "long context", "agent", "rag"),
    ),
    SearchProfile(
        "llm_systems",
        ("cs.LG", "cs.DC", "cs.PF"),
        ("inference", "serving", "kv cache", "throughput", "latency", "distributed training", "gpu"),
    ),
    SearchProfile(
        "mle_eval",
        ("cs.LG", "cs.AI"),
        ("evaluation", "benchmark", "reproducibility", "observability", "dataset", "monitoring"),
    ),
    SearchProfile(
        "multimodal_geometry",
        ("cs.CV", "cs.GR", "cs.AI"),
        ("multimodal", "3d", "geometry", "mesh", "point cloud", "cad", "vision-language"),
    ),
    SearchProfile(
        "aec_ai",
        ("cs.CV", "cs.AI", "cs.LG"),
        ("bim", "ifc", "cad", "digital twin", "hvac", "construction", "facility", "building controls", "scan-to-bim"),
    ),
)


def fetch(window_hours: int = 72) -> list[SourceItem]:
    since = datetime.now(timezone.utc) - timedelta(hours=window_hours)
    date_filter = since.strftime("%Y%m%d%H%M")
    max_results = config.profile_results()
    out: list[SourceItem] = []
    for profile in ARXIV_PROFILES:
        out.extend(_fetch_profile(profile, date_filter=date_filter, max_results=max_results))
    return out


def _fetch_profile(profile: SearchProfile, *, date_filter: str, max_results: int) -> list[SourceItem]:
    category_query = " OR ".join(f"cat:{category}" for category in profile.categories)
    term_query = " OR ".join(_term_clause(term) for term in profile.terms)
    query = f"({category_query}) AND ({term_query}) AND submittedDate:[{date_filter} TO 999912312359]"
    url = (
        "http://export.arxiv.org/api/query?"
        f"search_query={quote(query)}&sortBy=submittedDate&sortOrder=descending&start=0&max_results={max_results}"
    )
    try:
        resp = client().get(url)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
    except Exception as exc:
        LOG.warning("arxiv fetch failed for %s: %s", profile.name, exc)
        return []
    out: list[SourceItem] = []
    for entry in root.findall("a:entry", ARXIV_NS):
        item = _entry(entry, search_profile=profile.name)
        if item:
            out.append(item)
    return out


def _term_clause(term: str) -> str:
    return f"all:{term}" if all(ch.isalnum() for ch in term) else f'all:"{term}"'


def _text(entry: ET.Element, tag: str) -> str:
    node = entry.find(f"a:{tag}", ARXIV_NS)
    return " ".join((node.text or "").split()) if node is not None else ""


def _entry(entry: ET.Element, *, search_profile: str = "arxiv_general") -> SourceItem | None:
    title = _text(entry, "title")
    url = _text(entry, "id")
    if not title or not url:
        return None
    arxiv_id = url.rsplit("/abs/", 1)[-1]
    categories = [
        node.attrib.get("term", "")
        for node in entry.findall("a:category", ARXIV_NS)
        if node.attrib.get("term")
    ]
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
        tags=["paper", "arxiv", search_profile, *categories],
        extra={"search_profile": search_profile, "arxiv_categories": categories},
    ).normalized()


def _parse_dt(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
