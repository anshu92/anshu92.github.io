"""Pydantic models for blogpipe."""

from __future__ import annotations  # noqa: F401 — enables PEP 604 on Python 3.9 + pydantic

from datetime import datetime
from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, Field


class Pillar(str, Enum):
    research = "research"
    systems = "systems"
    applied = "applied"
    foundations = "foundations"
    aec = "aec"


class Item(BaseModel):
    """A normalized source item (paper, blog post, etc.)."""

    id: str
    title: str
    url: str
    authors: list[str] = Field(default_factory=list)
    abstract: str = ""
    published_at: Optional[datetime] = None
    source: str
    tags: list[str] = Field(default_factory=list)
    pillar: Pillar = Pillar.research
    extra: dict[str, Any] = Field(default_factory=dict)


class BenchmarkRow(BaseModel):
    """One row in a results/benchmark table."""

    name: str
    value: str
    unit: str = ""
    baseline: str = ""
    notes: str = ""


class Quote(BaseModel):
    source_id: str
    text: str
    url: str = ""


class EvidenceBundle(BaseModel):
    primary: Item
    ancestors: list[Item] = Field(default_factory=list)
    competitors: list[Item] = Field(default_factory=list)
    followups: list[Item] = Field(default_factory=list)
    benchmarks: list[BenchmarkRow] = Field(default_factory=list)
    aec_links: list[Item] = Field(default_factory=list)
    quotes: list[Quote] = Field(default_factory=list)
    by_id: dict[str, Item] = Field(default_factory=dict)

    def register_ids(self) -> None:
        self.by_id = {self.primary.id: self.primary}
        for it in self.ancestors + self.competitors + self.followups + self.aec_links:
            if it.id not in self.by_id:
                self.by_id[it.id] = it


class PostMeta(BaseModel):
    """Per-post index row for the curator."""

    path: str
    slug: str
    title: str
    date: str
    categories: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    h2_titles: list[str] = Field(default_factory=list)
    tldr: str = ""
    pillar: Pillar = Pillar.research
    word_count: int = 0
    embedding: Optional[List[float]] = None


class EditorialBrief(BaseModel):
    """Steers ranker, drafter, and visuals."""

    pillar_weights: dict[str, float] = Field(default_factory=dict)
    recent_topics: list[dict[str, str]] = Field(
        default_factory=list
    )  # {slug, title, pillar}
    follow_up_candidates: list[str] = Field(default_factory=list)
    avoid_topics: list[str] = Field(default_factory=list)
    voice_guide: str = ""
    suggested_series: Optional[str] = None
    format_name: str = "deep_dive"
    format_rationale: str = ""
    opener_hook: str = "in_medias_res_scene"
    art_direction: str = "editorial_illustration"
    diagram_style: str = "flowchart"
    proactive_topic: Optional[str] = None


class RankResult(BaseModel):
    primary: Item
    score: float = 0.0
    rejected: list[Item] = Field(default_factory=list)
    reasoning: str = ""


class EditorReport(BaseModel):
    """15-point rubric + five-question gate."""

    rubric_score: int = 0
    rubric_items: list[dict[str, Any]] = Field(default_factory=list)
    five_questions: dict[str, str] = Field(default_factory=dict)
    five_questions_ok: bool = True
    critique: dict[str, list[str]] = Field(
        default_factory=lambda: {"missing": [], "weak": [], "cut": []}
    )
    pass_gate: bool = False


# --- Requests for HTTP allow-lists (no user-controlled URLs) ---

ALLOWED_FETCH_HOSTS = frozenset(
    {
        "arxiv.org",
        "export.arxiv.org",
        "api.semanticscholar.org",
        "www.semanticscholar.org",
        "api.openalex.org",
        "openalex.org",
        "paperswithcode.com",
        "huggingface.co",
        "api.github.com",
        "www.research.autodesk.com",
        "research.autodesk.com",
        "blogs.nvidia.com",
    }
)


def is_allowed_url(url: str) -> bool:
    from urllib.parse import urlparse

    try:
        p = urlparse(url)
    except Exception:
        return False
    host = (p.netloc or "").lower().removeprefix("www.")
    for h in ALLOWED_FETCH_HOSTS:
        if host == h or host.endswith("." + h):
            return True
    return False
