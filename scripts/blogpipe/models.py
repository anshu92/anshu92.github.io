from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, Field


class Author(BaseModel):
    name: str
    affiliation: str = ""


class SourceItem(BaseModel):
    item_id: str = ""
    canonical_url: str
    source_kind: str
    source_name: str
    source_tier: int = 2
    title: str
    authors: list[Author] = Field(default_factory=list)
    published_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    doi: str = ""
    arxiv_id: str = ""
    venue_or_blog: str = ""
    abstract_or_excerpt: str = ""
    body_text: str = ""
    tags: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)

    def stable_id(self) -> str:
        if self.doi:
            return "doi:" + self.doi.lower().strip()
        if self.arxiv_id:
            return "arxiv:" + self.arxiv_id.lower().strip()
        url = canonicalize_url(self.canonical_url)
        if url:
            return "url:" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:20]
        title = " ".join((self.title or "").lower().split())
        return "title:" + hashlib.sha1(title.encode("utf-8")).hexdigest()[:20]

    def normalized(self) -> "SourceItem":
        clone = self.model_copy(deep=True)
        clone.canonical_url = canonicalize_url(clone.canonical_url)
        clone.item_id = clone.item_id or clone.stable_id()
        clone.title = " ".join(clone.title.split())
        clone.abstract_or_excerpt = " ".join((clone.abstract_or_excerpt or "").split())
        clone.body_text = " ".join((clone.body_text or "").split())
        return clone


class TopicScores(BaseModel):
    llm: float = 0.0
    mle: float = 0.0
    aec: float = 0.0
    matched_keywords: list[str] = Field(default_factory=list)

    @property
    def best(self) -> float:
        return max(self.llm, self.mle, self.aec)

    @property
    def tracks(self) -> list[str]:
        out = []
        if self.llm >= 0.25:
            out.append("LLM")
        if self.mle >= 0.25:
            out.append("MLE")
        if self.aec >= 0.25:
            out.append("AEC")
        return out or ["ML"]


class RankedItem(BaseModel):
    item: SourceItem
    topic_scores: TopicScores
    daily_score: float
    deep_dive_score: float
    quality_signals: dict[str, Any] = Field(default_factory=dict)
    citation_signals: dict[str, Any] = Field(default_factory=dict)
    rank_reason: str = ""


class EvidenceChunk(BaseModel):
    evidence_id: str
    item_id: str
    title: str
    url: str
    text: str
    section: str = ""
    evidence_type: str = ""


class EvidencePack(BaseModel):
    kind: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ranked_items: list[RankedItem]
    chunks: list[EvidenceChunk]
    prior_posts: list[dict[str, str]] = Field(default_factory=list)

    def evidence_blob(self) -> str:
        return "\n".join(c.text for c in self.chunks)

    def urls(self) -> list[str]:
        return sorted({r.item.canonical_url for r in self.ranked_items if r.item.canonical_url})

    def as_prompt_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=2, ensure_ascii=False)


class WriteResult(BaseModel):
    ok: bool
    path: str = ""
    title: str = ""
    body: str = ""
    errors: list[str] = Field(default_factory=list)
    repair_attempted: bool = False


def canonicalize_url(url: str) -> str:
    raw = (url or "").strip()
    if not raw:
        return ""
    parts = urlsplit(raw)
    scheme = parts.scheme or "https"
    netloc = parts.netloc.lower()
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((scheme, netloc, path, "", ""))
