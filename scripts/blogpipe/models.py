from __future__ import annotations

import hashlib
import json
import re
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
    ml_engineering: float = 0.0
    applied_research: float = 0.0
    ml_theory: float = 0.0
    aec: float = 0.0
    popular_ml: float = 0.0
    priority_track: str = ""
    matched_keywords: list[str] = Field(default_factory=list)

    def _scores(self) -> dict[str, float]:
        return {
            "ml_engineering": self.ml_engineering,
            "applied_research": self.applied_research,
            "ml_theory": self.ml_theory,
            "aec": self.aec,
            "popular_ml": self.popular_ml,
        }

    @property
    def best(self) -> float:
        from .topics import weighted_best

        return weighted_best(self._scores())

    @property
    def tracks(self) -> list[str]:
        from .topics import active_track_labels

        return active_track_labels(self._scores())


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


class EvidenceCard(BaseModel):
    item_id: str
    title: str
    url: str
    role: str = "primary"
    problem: str = "not found in evidence"
    mechanism: str = "not found in evidence"
    math_or_objective: str = "not found in evidence"
    experiment: str = "not found in evidence"
    limitation: str = "not found in evidence"
    impact: str = "not found in evidence"
    paper_supported_claim: str = "not found in evidence"
    paper_supported_limit: str = "not found in evidence"
    transfer_hypothesis: str = ""
    open_research_question: str = ""
    evidence_ids: dict[str, list[str]] = Field(default_factory=dict)


class EvidencePack(BaseModel):
    kind: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ranked_items: list[RankedItem]
    chunks: list[EvidenceChunk]
    evidence_cards: list[EvidenceCard] = Field(default_factory=list)
    prior_posts: list[dict[str, str]] = Field(default_factory=list)
    curriculum: dict[str, Any] = Field(default_factory=dict)

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


class SelectionItem(BaseModel):
    item_id: str
    role: str = ""
    relevance_label: str = ""
    reason: str = ""
    scores: dict[str, float] = Field(default_factory=dict)
    suggested_tags: list[str] = Field(default_factory=list)


class SelectionResult(BaseModel):
    selected_item_ids: list[str] = Field(default_factory=list)
    items: list[SelectionItem] = Field(default_factory=list)
    suggested_tags: list[str] = Field(default_factory=list)


class CurriculumNode(BaseModel):
    id: str
    problem_statement: str
    why_it_matters: str
    prerequisites: list[str] = Field(default_factory=list)
    concepts_to_teach: list[str] = Field(default_factory=list)
    engineering_questions: list[str] = Field(default_factory=list)
    research_strategy: dict[str, Any] = Field(default_factory=dict)
    evidence_requirements: list[str] = Field(default_factory=list)
    completion_criteria: list[str] = Field(default_factory=list)
    growth_prompts: list[str] = Field(default_factory=list)
    stage: str = ""
    level: str = ""

    @property
    def node_id(self) -> str:
        return self.id

    @property
    def title(self) -> str:
        return self.problem_statement

    @property
    def reader_takeaway(self) -> str:
        return self.completion_criteria[0] if self.completion_criteria else self.why_it_matters

    @property
    def questions(self) -> list[str]:
        return self.engineering_questions


class CurriculumPlan(BaseModel):
    node: CurriculumNode
    selected_by: str
    completed_node_ids: list[str] = Field(default_factory=list)
    tree_version: str = "foundation-models-v2"


class ResearchPlan(BaseModel):
    problem_id: str
    problem_statement: str
    queries: list[str] = Field(default_factory=list)
    source_profiles: list[str] = Field(default_factory=list)
    expected_evidence: list[str] = Field(default_factory=list)
    background_queries: list[str] = Field(default_factory=list)


class EvidenceCurationItem(BaseModel):
    item_id: str
    title: str
    url: str
    role: str = "background"
    fit_score: float = 0.0
    evidence_types: list[str] = Field(default_factory=list)
    reason: str = ""


class EvidenceCurationResult(BaseModel):
    problem_id: str
    problem_statement: str
    selected_item_ids: list[str] = Field(default_factory=list)
    items: list[EvidenceCurationItem] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    sufficient: bool = False
    confidence: float = 0.0


class OutlineSection(BaseModel):
    heading: str
    intent: str
    evidence_ids: list[str] = Field(default_factory=list)
    word_budget: int = 0
    focus_item_ids: list[str] = Field(default_factory=list)
    section_role: str = ""
    split_reason: str = ""


class DailyOutline(BaseModel):
    title: str
    angle: str = ""
    sections: list[OutlineSection] = Field(default_factory=list)
    suggested_tags: list[str] = Field(default_factory=list)


class RoleMarketSignal(BaseModel):
    company: str
    role_title: str
    seniority: str
    source_url: str
    topic_tags: list[str] = Field(default_factory=list)
    skill_phrases: list[str] = Field(default_factory=list)
    systems_phrases: list[str] = Field(default_factory=list)
    evidence_quote: str = ""
    catalogue_relevance: float = 0.0
    freshness_date: str = ""


class RoleMarketReport(BaseModel):
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    companies: list[str] = Field(default_factory=list)
    signals: list[RoleMarketSignal] = Field(default_factory=list)
    recurring_topics: list[str] = Field(default_factory=list)
    topic_gaps: list[str] = Field(default_factory=list)
    attempted_sources: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class CatalogueLesson(BaseModel):
    id: str
    title: str
    series_id: str
    series_title: str
    level: str = "foundations"
    prerequisites: list[str] = Field(default_factory=list)
    concepts: list[str] = Field(default_factory=list)
    engineering_questions: list[str] = Field(default_factory=list)
    how_to: list[str] = Field(default_factory=list)
    topic_tags: list[str] = Field(default_factory=list)
    market_rationale: str = ""


class CatalogueSeries(BaseModel):
    id: str
    title: str
    description: str = ""
    lesson_ids: list[str] = Field(default_factory=list)


class LessonBrief(BaseModel):
    lesson: CatalogueLesson
    title: str
    thesis: str
    concept_arc: list[str] = Field(default_factory=list)
    implementation_steps: list[str] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)
    evaluation_plan: list[str] = Field(default_factory=list)
    principal_decisions: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    source_urls: list[str] = Field(default_factory=list)


class AgentReport(BaseModel):
    agent_name: str
    status: str = "ok"
    summary: str = ""
    output: dict[str, Any] = Field(default_factory=dict)
    findings: list[str] = Field(default_factory=list)


class TableSpec(BaseModel):
    table_id: str
    title: str
    purpose: str
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    placement: str = ""


class ComponentSpec(BaseModel):
    component_id: str
    kind: str
    title: str
    purpose: str
    html: str
    evidence_ids: list[str] = Field(default_factory=list)
    placement: str = ""


class VisualAsset(BaseModel):
    asset_id: str
    artifact_type: str
    title: str
    purpose: str
    evidence_ids: list[str] = Field(default_factory=list)
    placement: str = ""
    content: str = ""
    path: str = ""


class VisualPlan(BaseModel):
    assets: list[VisualAsset] = Field(default_factory=list)
    tables: list[TableSpec] = Field(default_factory=list)
    components: list[ComponentSpec] = Field(default_factory=list)
    mermaid_required: bool = False


class ReviewFinding(BaseModel):
    agent_name: str
    severity: str
    message: str
    evidence_id: str = ""
    path: str = ""


class FinalArticle(BaseModel):
    ok: bool
    title: str = ""
    slug: str = ""
    body: str = ""
    path: str = ""
    errors: list[str] = Field(default_factory=list)
    mermaid: bool = False
    assets: list[str] = Field(default_factory=list)


class SwarmRunReport(BaseModel):
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ingest_count: int = 0
    ranked_count: int = 0
    selected_lesson: CatalogueLesson | None = None
    role_market_report: RoleMarketReport | None = None
    lesson_brief: LessonBrief | None = None
    visual_plan: VisualPlan | None = None
    final_article: FinalArticle
    agent_reports: list[AgentReport] = Field(default_factory=list)
    review_findings: list[ReviewFinding] = Field(default_factory=list)


def canonicalize_url(url: str) -> str:
    raw = (url or "").strip()
    if not raw:
        return ""
    parts = urlsplit(raw)
    scheme = parts.scheme or "https"
    netloc = parts.netloc.lower()
    path = parts.path.rstrip("/") or "/"
    if netloc in {"arxiv.org", "www.arxiv.org"} and path.startswith("/abs/"):
        scheme = "https"
        netloc = "arxiv.org"
        path = re.sub(r"v\d+$", "", path)
    return urlunsplit((scheme, netloc, path, "", ""))
