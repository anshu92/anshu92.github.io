from __future__ import annotations

import logging

from .. import config
from ..models import SourceItem
from ._http import client

LOG = logging.getLogger(__name__)


def enrich(items: list[SourceItem]) -> list[SourceItem]:
    """Attach lightweight identifier/citation metadata where cheap.

    Enrichment is intentionally best-effort. The radar can publish from primary
    metadata alone, so API failures never block ingest.
    """
    enriched: list[SourceItem] = []
    for item in items:
        current = item
        if current.doi:
            current = _openalex(current)
            current = _crossref(current)
            current = _unpaywall(current)
        enriched.append(current.normalized())
    return enriched


def _openalex(item: SourceItem) -> SourceItem:
    params = {"filter": f"doi:{item.doi}", "per-page": 1, "mailto": config.contact_email()}
    if config.openalex_key():
        params["api_key"] = config.openalex_key()
    try:
        resp = client().get("https://api.openalex.org/works", params=params)
        if resp.status_code >= 400:
            return item
        results = (resp.json().get("results") or [])
    except Exception:
        return item
    if not results:
        return item
    work = results[0]
    extra = dict(item.extra)
    extra["openalex_id"] = work.get("id", "")
    extra["citation_count"] = work.get("cited_by_count", 0)
    return item.model_copy(update={"extra": extra})


def _crossref(item: SourceItem) -> SourceItem:
    try:
        resp = client().get(
            f"https://api.crossref.org/works/{item.doi}",
            params={"mailto": config.contact_email()},
        )
        if resp.status_code >= 400:
            return item
        msg = (resp.json().get("message") or {})
    except Exception:
        return item
    extra = dict(item.extra)
    extra["crossref_type"] = msg.get("type", "")
    return item.model_copy(update={"extra": extra})


def _unpaywall(item: SourceItem) -> SourceItem:
    try:
        resp = client().get(
            f"https://api.unpaywall.org/v2/{item.doi}",
            params={"email": config.contact_email()},
        )
        if resp.status_code >= 400:
            return item
        data = resp.json()
    except Exception:
        return item
    extra = dict(item.extra)
    extra["is_oa"] = bool(data.get("is_oa"))
    extra["oa_url"] = ((data.get("best_oa_location") or {}).get("url") or "")
    return item.model_copy(update={"extra": extra})
