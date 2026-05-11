from __future__ import annotations

import logging
import re

from .models import SourceItem

LOG = logging.getLogger(__name__)


def clean_html(html: str) -> str:
    if not html:
        return ""
    try:
        import trafilatura

        extracted = trafilatura.extract(html, include_comments=False, include_tables=True)
        if extracted:
            return normalize_text(extracted)
    except Exception as exc:
        LOG.debug("trafilatura unavailable or failed: %s", exc)
    text = re.sub(r"<(script|style)[\s\S]*?</\1>", " ", html, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return normalize_text(text)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def enrich_body_text(item: SourceItem, html: str = "") -> SourceItem:
    body = clean_html(html) if html else item.body_text
    return item.model_copy(update={"body_text": normalize_text(body)})


def sentence_split(text: str) -> list[str]:
    raw = normalize_text(text)
    if not raw:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", raw)
    return [p.strip() for p in parts if len(p.strip()) >= 40]
