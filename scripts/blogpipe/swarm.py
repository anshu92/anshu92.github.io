from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from markdown_it import MarkdownIt

from . import config, jsonish, memory, visuals
from .llm import LLMClient
from .models import (
    AgentReport,
    CatalogueLesson,
    CatalogueSeries,
    ComponentSpec,
    EvidencePack,
    FinalArticle,
    LessonBrief,
    RankedItem,
    ReviewFinding,
    RoleMarketReport,
    RoleMarketSignal,
    SourceItem,
    SwarmRunReport,
    TableSpec,
    TopicScores,
    VisualAsset,
    VisualPlan,
)

LOG = logging.getLogger(__name__)

TOP_AI_COMPANIES = [
    "OpenAI",
    "Anthropic",
    "Google DeepMind",
    "Meta AI",
    "NVIDIA",
    "Microsoft AI",
    "Amazon AGI",
    "Apple ML",
    "xAI",
    "Cohere",
    "Mistral",
    "Hugging Face",
    "Databricks/MosaicML",
]

ROLE_SOURCE_URLS = [
    "https://openai.com/careers/search",
    "https://www.anthropic.com/careers",
    "https://deepmind.google/careers",
    "https://www.metacareers.com/jobs",
    "https://www.nvidia.com/en-us/about-nvidia/careers",
    "https://jobs.careers.microsoft.com",
    "https://www.amazon.jobs",
    "https://jobs.apple.com",
    "https://x.ai/careers",
    "https://cohere.com/careers",
    "https://mistral.ai/careers",
    "https://huggingface.co/jobs",
    "https://www.databricks.com/company/careers",
]

SENIORITY_RE = re.compile(r"\b(staff|principal|senior staff|lead|member of technical staff)\b", re.I)
TECH_ROLE_RE = re.compile(
    r"\b(machine learning|ml|ai|research engineer|software engineer|infrastructure|systems|training|inference|data)\b",
    re.I,
)
TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "linear-algebra": ("matrix", "linear algebra", "tensor", "kernel", "cuda", "gpu"),
    "autodiff-optimization": ("autodiff", "gradient", "optimizer", "loss", "training stability"),
    "transformers-attention": ("transformer", "attention", "llm", "sequence model"),
    "distributed-training": ("distributed training", "parallelism", "sharding", "fsdp", "checkpoint", "nccl"),
    "inference-serving": ("inference", "serving", "latency", "throughput", "quantization", "cache"),
    "evaluation-reliability": ("evaluation", "benchmark", "reliability", "monitoring", "failure mode"),
    "retrieval-grounding": ("retrieval", "rag", "grounding", "embedding", "vector"),
    "agents-tools": ("agent", "tool", "planning", "workflow", "orchestration"),
}

FOUNDATION_SOURCE_PACKS: dict[str, list[dict[str, str]]] = {
    "matmul-from-scalar-operations": [
        {
            "url": "https://www.deeplearningbook.org/contents/linear_algebra.html",
            "source_name": "Deep Learning Book",
            "title": "Linear Algebra chapter, Deep Learning Book",
            "body": (
                "The chapter frames linear algebra as the mathematical language for matrices, vectors, dot products, "
                "and transformations used in neural network models. Matrix multiplication is an algorithm that "
                "composes dot products between rows and columns; the shape invariant decides whether the operation "
                "is defined. Practical implementations still need correctness checks because shape mistakes and "
                "numerical error can make a model fail even when code runs."
            ),
        },
        {
            "url": "https://pytorch.org/docs/stable/generated/torch.matmul.html",
            "source_name": "PyTorch",
            "title": "torch.matmul documentation",
            "body": (
                "PyTorch documents torch.matmul as the method for matrix products with dimension-dependent "
                "broadcasting semantics. The API distinguishes vector-vector, matrix-vector, matrix-matrix, and "
                "batched matrix multiplication, so implementation tests should compare expected output shape and "
                "broadcasting behavior. The practical release gate is to compare the reference algorithm against a "
                "trusted library result before profiling throughput."
            ),
        },
        {
            "url": "https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html",
            "source_name": "NVIDIA",
            "title": "Matrix multiplication performance guide",
            "body": (
                "NVIDIA's matrix multiplication performance guide discusses arithmetic intensity, memory movement, "
                "Tensor Core use, and profiling as practical concerns for GEMM workloads. The benchmark question is "
                "whether the bottleneck is arithmetic work, memory access, or layout choices rather than the "
                "algebraic formula. A limitation is that faster kernels preserve the same mathematical contract but "
                "add hardware-sensitive tradeoffs."
            ),
        },
        {
            "url": "https://d2l.ai/chapter_preliminaries/linear-algebra.html",
            "source_name": "Dive into Deep Learning",
            "title": "Linear Algebra preliminaries",
            "body": (
                "Dive into Deep Learning introduces vectors, matrices, tensor products, and linear algebra notation "
                "as the working vocabulary for model implementation. The method is practical: name dimensions, "
                "state the operation, and check the resulting tensor before using the computation inside a model. "
                "This supports reproducibility because a small worked example becomes the oracle for larger kernels."
            ),
        },
    ],
    "reverse-mode-autodiff": [
        {
            "url": "https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html",
            "source_name": "PyTorch",
            "title": "A Gentle Introduction to torch.autograd",
            "body": (
                "PyTorch describes autograd as a mechanism that records operations on tensors and computes gradients "
                "by replaying the graph backward. The practical method is to trace a small computation, identify the "
                "intermediate values needed for gradient calculation, and compare gradients against a numerical "
                "check. A common limitation is memory growth when training stores more intermediates than expected."
            ),
        },
        {
            "url": "https://jax.readthedocs.io/en/latest/automatic-differentiation.html",
            "source_name": "JAX",
            "title": "Automatic differentiation in JAX",
            "body": (
                "JAX presents automatic differentiation as a transformation over Python functions, including "
                "gradient and vector-Jacobian product forms. The method reinforces that the mathematical objective "
                "and the execution graph must agree before optimization work begins. Evaluation should include "
                "gradient checks, failure cases around nondifferentiable operations, and reproducibility gates."
            ),
        },
    ],
    "attention-as-indexed-retrieval": [
        {
            "url": "https://arxiv.org/abs/1706.03762",
            "source_name": "arXiv",
            "title": "Attention Is All You Need",
            "body": (
                "The transformer paper introduces scaled dot-product attention as a method that maps queries, keys, "
                "and values into token-to-token retrieval weights. The equation uses matrix products and a softmax "
                "objective to decide which values each token should read. Practical evaluation includes masking "
                "checks, shape checks, and experiments that compare attention behavior across tasks."
            ),
        },
        {
            "url": "https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html",
            "source_name": "PyTorch",
            "title": "scaled_dot_product_attention documentation",
            "body": (
                "PyTorch documents scaled dot-product attention as an implementation interface for query, key, value, "
                "masking, dropout, and backend behavior. The practical method is to validate tensor shapes, masks, "
                "and numerical behavior before relying on a faster backend. Limitations include backend-specific "
                "tradeoffs and failure cases where masking semantics are misunderstood."
            ),
        },
    ],
}


def run(
    *,
    window_hours: int = 14 * 24,
    fixtures: str = "",
    dry_run: bool = False,
    db: str = "",
) -> dict[str, Any]:
    return SwarmOrchestrator().run(window_hours=window_hours, fixtures=fixtures, dry_run=dry_run, db=db)


class SwarmOrchestrator:
    def __init__(self, *, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()
        self.reports: list[AgentReport] = []
        self.findings: list[ReviewFinding] = []

    def run(
        self,
        *,
        window_hours: int = 14 * 24,
        fixtures: str = "",
        dry_run: bool = False,
        db: str = "",
    ) -> dict[str, Any]:
        from . import evidence, ingest, rank

        memory.ensure_dirs()
        _swarm_reports_dir().mkdir(parents=True, exist_ok=True)
        ingest_count = ingest.run(window_hours=window_hours, fixtures=fixtures, db=db)
        ranked = rank.run(db=db, limit=60, max_age_hours=None if fixtures else window_hours)

        role_report = RoleMarketScout().run()
        self._report("RoleMarketScout", "ok", f"{len(role_report.signals)} role-market signals", role_report.model_dump(mode="json"))
        lesson = CatalogueEditor().choose(role_report)
        self._report("CatalogueEditor", "ok", f"selected {lesson.id}", lesson.model_dump(mode="json"))

        selected = ResearchLead().curate(ranked, lesson)
        self._report(
            "ResearchLead",
            "ok" if len(selected) >= 2 else "blocked",
            f"{len(selected)} evidence items selected",
            {"selected_item_ids": [item.item.item_id for item in selected]},
        )

        if len(selected) < 2:
            final = self._blocked(
                lesson=lesson,
                errors=[f"insufficient_evidence:{len(selected)}/2"],
                title=f"{lesson.title} - blocked",
            )
            return self._finish(
                ingest_count=ingest_count,
                ranked_count=len(ranked),
                lesson=lesson,
                role_report=role_report,
                brief=None,
                visual_plan=None,
                final=final,
            )

        pack = evidence.build_daily_pack(selected, prior_limit=5)
        brief = TechnicalExplainer(self.llm).build(lesson, pack, role_report)
        self._report("TechnicalExplainer", "ok", brief.thesis, brief.model_dump(mode="json"))
        brief = ImplementationEngineer(self.llm).refine(brief, pack)
        self._report("ImplementationEngineer", "ok", "added implementation and evaluation path", brief.model_dump(mode="json"))

        draft_body = ManagingEditor(self.llm).draft_body(brief, pack)
        fact_findings = SkepticalFactChecker().review(draft_body, pack)
        principal_findings = PrincipalReviewer().review(brief, draft_body)
        self._report("SkepticalFactChecker", _status(fact_findings), "draft fact review", {"findings": [f.model_dump() for f in fact_findings]})
        self._report("PrincipalReviewer", _status(principal_findings), "draft principal review", {"findings": [f.model_dump() for f in principal_findings]})

        visual_plan = VisualExplainer().plan(brief, pack, draft_body)
        visual_plan = TableDesigner().design(visual_plan, brief, pack)
        visual_plan = ComponentDesigner().design(visual_plan, brief, pack)
        slug = _slug_for_lesson(brief.title)
        rendered_assets = visuals.render_visual_assets(visual_plan, slug=slug)
        visual_plan = visual_plan.model_copy(update={"assets": rendered_assets})
        visual_errors = visuals.validate_visual_plan(visual_plan, evidence_ids=_evidence_ids(pack))
        self.findings.extend(_findings_from_errors("VisualExplainer", visual_errors))
        self._report("VisualExplainer", "ok" if not visual_errors else "blocked", "visual plan", visual_plan.model_dump(mode="json"))
        self._report("TableDesigner", "ok", f"{len(visual_plan.tables)} table specs", {"tables": [t.model_dump() for t in visual_plan.tables]})
        self._report("ComponentDesigner", "ok", f"{len(visual_plan.components)} component specs", {"components": [c.model_dump() for c in visual_plan.components]})

        final_body = ManagingEditor(self.llm).finalize_body(brief, pack, visual_plan)
        layout_findings = LayoutReviewer().review(final_body, visual_plan)
        final_principal_findings = PrincipalReviewer().review(brief, final_body)
        final_fact_findings = SkepticalFactChecker().review(final_body, pack, visual_plan=visual_plan)
        self.findings.extend(layout_findings)
        self.findings.extend(final_principal_findings)
        self.findings.extend(final_fact_findings)
        self._report("LayoutReviewer", _status(layout_findings), "layout review", {"findings": [f.model_dump() for f in layout_findings]})
        self._report("PrincipalReviewer", _status(final_principal_findings), "final principal review", {"findings": [f.model_dump() for f in final_principal_findings]})

        blocking = [finding.message for finding in self.findings if finding.severity == "error"]
        if blocking:
            final = self._blocked(lesson=lesson, errors=blocking, title=brief.title, slug=slug, body=final_body)
        else:
            final = ManagingEditor(self.llm).publish_or_preview(
                brief=brief,
                body=final_body,
                slug=slug,
                visual_plan=visual_plan,
                dry_run=dry_run,
            )
            if final.ok and not dry_run:
                CatalogueEditor().record_completion(lesson, final)

        return self._finish(
            ingest_count=ingest_count,
            ranked_count=len(ranked),
            lesson=lesson,
            role_report=role_report,
            brief=brief,
            visual_plan=visual_plan,
            final=final,
        )

    def _blocked(
        self,
        *,
        lesson: CatalogueLesson,
        errors: list[str],
        title: str,
        slug: str = "",
        body: str = "",
    ) -> FinalArticle:
        slug = slug or _slug_for_lesson(title or lesson.title)
        payload = {
            "title": title,
            "slug": slug,
            "lesson_id": lesson.id,
            "errors": errors,
        }
        path = _swarm_reports_dir() / f"{slug}.blocked.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return FinalArticle(ok=False, title=title, slug=slug, body=body, path=str(path.relative_to(memory.ROOT)), errors=errors)

    def _finish(
        self,
        *,
        ingest_count: int,
        ranked_count: int,
        lesson: CatalogueLesson,
        role_report: RoleMarketReport,
        brief: LessonBrief | None,
        visual_plan: VisualPlan | None,
        final: FinalArticle,
    ) -> dict[str, Any]:
        report = SwarmRunReport(
            ingest_count=ingest_count,
            ranked_count=ranked_count,
            selected_lesson=lesson,
            role_market_report=role_report,
            lesson_brief=brief,
            visual_plan=visual_plan,
            final_article=final,
            agent_reports=self.reports,
            review_findings=self.findings,
        )
        _write_report("role_market_signal.json", role_report.model_dump(mode="json"))
        _write_report("catalogue_decision.json", lesson.model_dump(mode="json"))
        if brief is not None:
            _write_report("lesson_brief.json", brief.model_dump(mode="json"))
        if visual_plan is not None:
            _write_report("visual_plan.json", visual_plan.model_dump(mode="json"))
        _write_report("agent_reports.json", [item.model_dump(mode="json") for item in self.reports])
        _write_report("review_findings.json", [item.model_dump(mode="json") for item in self.findings])
        _write_report("run_report.json", report.model_dump(mode="json"))
        _write_report("llm_usage.json", self.llm.usage.__dict__)
        return report.model_dump(mode="json")

    def _report(self, agent_name: str, status: str, summary: str, output: dict[str, Any]) -> None:
        self.reports.append(AgentReport(agent_name=agent_name, status=status, summary=summary, output=output))


class RoleMarketScout:
    def run(self) -> RoleMarketReport:
        fixture = os.environ.get("BLOGPIPE_ROLE_MARKET_FIXTURES", "").strip()
        if fixture:
            return self._from_fixture(Path(fixture))
        if os.environ.get("BLOGPIPE_ROLE_MARKET_LIVE", "").strip().lower() in {"1", "true", "yes", "on"}:
            return self._from_live_pages()
        return RoleMarketReport(
            companies=TOP_AI_COMPANIES,
            recurring_topics=["distributed-training", "inference-serving", "evaluation-reliability", "agents-tools"],
            topic_gaps=[],
            attempted_sources=ROLE_SOURCE_URLS,
        )

    def _from_fixture(self, path: Path) -> RoleMarketReport:
        payload = json.loads(path.read_text(encoding="utf-8"))
        postings = payload.get("postings", []) if isinstance(payload, dict) else payload
        if not isinstance(postings, list):
            postings = []
        signals = [signal for posting in postings if (signal := self._signal_from_posting(posting)) is not None]
        return self._report_from_signals(signals, attempted_sources=[str(path)])

    def _from_live_pages(self) -> RoleMarketReport:
        signals: list[RoleMarketSignal] = []
        errors: list[str] = []
        timeout = httpx.Timeout(8.0, connect=5.0)
        for url in ROLE_SOURCE_URLS:
            try:
                response = httpx.get(
                    url,
                    timeout=timeout,
                    headers={"User-Agent": config.user_agent()},
                    follow_redirects=True,
                )
                response.raise_for_status()
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{url}:{type(exc).__name__}")
                continue
            signals.extend(self._signals_from_text(url, response.text))
        report = self._report_from_signals(signals, attempted_sources=ROLE_SOURCE_URLS)
        report.errors.extend(errors[:12])
        return report

    def _signals_from_text(self, url: str, text: str) -> list[RoleMarketSignal]:
        signals: list[RoleMarketSignal] = []
        for posting in _anchor_role_postings(url, text):
            if len(signals) >= 25:
                break
            signal = self._signal_from_posting(posting)
            if signal is not None:
                signals.append(signal)
        seen_titles = {signal.role_title.lower() for signal in signals}
        lines = _html_lines(text)
        for idx, line in enumerate(lines):
            if len(signals) >= 25:
                break
            if not _looks_like_role_title(line):
                continue
            if line.lower() in seen_titles:
                continue
            context = " ".join(lines[max(0, idx - 2) : idx + 8])
            posting = {"company": _company_from_url(url), "title": line, "description": context, "url": url}
            signal = self._signal_from_posting(posting)
            if signal is not None:
                signals.append(signal)
                seen_titles.add(signal.role_title.lower())
        return signals

    def _signal_from_posting(self, posting: object) -> RoleMarketSignal | None:
        if not isinstance(posting, dict):
            return None
        title = str(posting.get("title") or posting.get("role_title") or "").strip()
        description = str(posting.get("description") or posting.get("body") or posting.get("text") or "").strip()
        if not _looks_like_role_title(title):
            return None
        blob = f"{title}\n{description}"
        if not SENIORITY_RE.search(blob) or not TECH_ROLE_RE.search(blob):
            return None
        topics = _topic_tags(blob)
        if not topics:
            return None
        source_url = str(posting.get("url") or posting.get("source_url") or "").strip()
        if not source_url:
            return None
        phrases = _skill_phrases(blob)
        systems = [phrase for phrase in phrases if phrase in _systems_phrase_set()]
        return RoleMarketSignal(
            company=str(posting.get("company") or _company_from_url(source_url) or "unknown").strip(),
            role_title=title[:160],
            seniority=_seniority(blob),
            source_url=source_url,
            topic_tags=topics,
            skill_phrases=phrases[:10],
            systems_phrases=systems[:8],
            evidence_quote=_first_sentence(description or title)[:320],
            catalogue_relevance=round(min(1.0, 0.25 + 0.12 * len(topics) + 0.04 * len(phrases)), 3),
            freshness_date=str(posting.get("freshness_date") or ""),
        )

    def _report_from_signals(self, signals: list[RoleMarketSignal], *, attempted_sources: list[str]) -> RoleMarketReport:
        unique = _dedupe_role_signals(signals)
        topic_counts: dict[str, int] = {}
        for signal in unique:
            for tag in signal.topic_tags:
                topic_counts[tag] = topic_counts.get(tag, 0) + 1
        recurring = [tag for tag, _count in sorted(topic_counts.items(), key=lambda item: (-item[1], item[0]))[:8]]
        return RoleMarketReport(
            companies=TOP_AI_COMPANIES,
            signals=unique,
            recurring_topics=recurring,
            topic_gaps=[],
            attempted_sources=attempted_sources,
        )


class CatalogueEditor:
    def choose(self, role_report: RoleMarketReport) -> CatalogueLesson:
        lessons = _catalogue_lessons()
        forced = os.environ.get("BLOGPIPE_CATALOGUE_LESSON", "").strip()
        by_id = {lesson.id: lesson for lesson in lessons}
        if forced and forced in by_id:
            chosen = by_id[forced]
        else:
            completed = set(_load_catalogue_state().get("completed_lesson_ids", []))
            chosen = next(
                (
                    lesson
                    for lesson in lessons
                    if lesson.id not in completed and set(lesson.prerequisites) <= completed
                ),
                lessons[0],
            )
        market = ", ".join(role_report.recurring_topics[:4])
        if market:
            chosen = chosen.model_copy(update={"market_rationale": f"Role-market weak signals: {market}."})
        return chosen

    def record_completion(self, lesson: CatalogueLesson, final: FinalArticle) -> None:
        state = _load_catalogue_state()
        completed = list(state.get("completed_lesson_ids", []))
        if lesson.id not in completed:
            completed.append(lesson.id)
        history = list(state.get("history", []))
        history.append(
            {
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "lesson_id": lesson.id,
                "title": final.title,
                "path": final.path,
            }
        )
        memory.ensure_dirs()
        _catalogue_state_path().write_text(
            json.dumps({"completed_lesson_ids": completed, "history": history[-100:]}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


class ResearchLead:
    def curate(self, ranked: list[RankedItem], lesson: CatalogueLesson) -> list[RankedItem]:
        selected: list[RankedItem] = []
        seen_urls: set[str] = set()
        for item in _foundation_ranked_items(lesson):
            selected.append(item)
            seen_urls.add(item.item.canonical_url)

        scored = sorted(
            ((_lesson_fit(item, lesson), item) for item in ranked),
            key=lambda pair: pair[0],
            reverse=True,
        )
        for _index, (fit, item) in enumerate(scored):
            if fit <= 0.05 and selected:
                continue
            if len(selected) >= 6:
                break
            url = item.item.canonical_url
            if url and url in seen_urls:
                continue
            role = "primary" if len([entry for entry in selected if entry.item.extra.get("selector_role") == "primary"]) < 3 else "supporting"
            clone = item.model_copy(deep=True)
            clone.item.extra["selector_role"] = role
            clone.item.extra["catalogue_lesson_id"] = lesson.id
            clone.quality_signals["lesson_fit"] = round(fit, 3)
            selected.append(clone)
            if url:
                seen_urls.add(url)
        return selected


class TechnicalExplainer:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def build(self, lesson: CatalogueLesson, pack: EvidencePack, role_report: RoleMarketReport) -> LessonBrief:
        fallback = _default_brief(lesson, pack, role_report)
        data = _agent_json(
            self.llm,
            task="technical_explainer",
            system="You are the TechnicalExplainer agent for a Staff/Principal ML engineering blog. Return JSON only.",
            user=(
                "Create a concise lesson brief with keys: title, thesis, concept_arc, failure_modes, "
                "evaluation_plan, principal_decisions. Use the evidence IDs and source URLs provided.\n\n"
                f"LESSON:\n{lesson.model_dump_json(indent=2)}\n\nEVIDENCE:\n{_pack_prompt(pack)}"
            ),
        )
        if not data:
            return fallback
        return fallback.model_copy(
            update={
                "title": str(data.get("title") or fallback.title)[:180],
                "thesis": str(data.get("thesis") or fallback.thesis)[:500],
                "concept_arc": _string_list(data.get("concept_arc")) or fallback.concept_arc,
                "failure_modes": _string_list(data.get("failure_modes")) or fallback.failure_modes,
                "evaluation_plan": _string_list(data.get("evaluation_plan")) or fallback.evaluation_plan,
                "principal_decisions": _string_list(data.get("principal_decisions")) or fallback.principal_decisions,
            }
        )


class ImplementationEngineer:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def refine(self, brief: LessonBrief, pack: EvidencePack) -> LessonBrief:
        data = _agent_json(
            self.llm,
            task="implementation_engineer",
            system="You are the ImplementationEngineer agent. Return JSON only.",
            user=(
                "Add practical implementation_steps for the lesson. Include pseudocode or benchmark steps where useful. "
                "Return JSON with implementation_steps only.\n\n"
                f"BRIEF:\n{brief.model_dump_json(indent=2)}\n\nEVIDENCE:\n{_pack_prompt(pack)}"
            ),
        )
        steps = _string_list(data.get("implementation_steps")) if data else []
        if not steps:
            steps = brief.implementation_steps or [
                "Write the smallest tensor example and name every shape before optimizing.",
                "Implement the direct algorithm first so later kernels have a correctness oracle.",
                "Add assertions for dimensions, dtype, memory layout, and numerical tolerance.",
                "Benchmark the naive path against the optimized path before changing model code.",
            ]
        return brief.model_copy(update={"implementation_steps": steps})


class SkepticalFactChecker:
    def review(self, body: str, pack: EvidencePack, *, visual_plan: VisualPlan | None = None) -> list[ReviewFinding]:
        findings: list[ReviewFinding] = []
        known = _evidence_ids(pack)
        refs = set(re.findall(r"\[E(\d+)\]", body or ""))
        refs = {f"E{ref}" for ref in refs}
        if not refs:
            findings.append(_finding("SkepticalFactChecker", "error", "no_evidence_ids"))
        for ref in sorted(refs - known):
            findings.append(_finding("SkepticalFactChecker", "error", f"unknown_evidence_id:{ref}", evidence_id=ref))
        for ref in sorted(refs & known):
            chunk = _chunk_by_id(pack).get(ref)
            if chunk and chunk.url and chunk.url not in body:
                findings.append(_finding("SkepticalFactChecker", "error", f"missing_source_link:{ref}", evidence_id=ref))
        cited_urls = {chunk.url for ref in refs & known if (chunk := _chunk_by_id(pack).get(ref)) and chunk.url}
        if len(cited_urls) < 2:
            findings.append(_finding("SkepticalFactChecker", "error", f"too_few_distinct_source_urls:{len(cited_urls)}/2"))
        claim_text = _claim_text(body)
        if re.search(r"\b\d+(?:\.\d+)?%?\b", claim_text):
            evidence_blob = pack.evidence_blob()
            for number in sorted(set(re.findall(r"\b\d+(?:\.\d+)?%?\b", claim_text))):
                if number not in evidence_blob and number not in {"1", "2", "3", "4", "5"}:
                    findings.append(_finding("SkepticalFactChecker", "error", f"unsupported_number:{number}"))
        if visual_plan is not None:
            errors = visuals.validate_visual_plan(visual_plan, evidence_ids=known)
            findings.extend(_findings_from_errors("SkepticalFactChecker", errors))
        return findings


class PrincipalReviewer:
    def review(self, brief: LessonBrief, body: str) -> list[ReviewFinding]:
        findings: list[ReviewFinding] = []
        lower = (body or "").lower()
        required = ("how to implement", "failure modes", "principal engineer")
        for phrase in required:
            if phrase not in lower:
                findings.append(_finding("PrincipalReviewer", "error", f"missing_principal_lesson_section:{phrase}"))
        if not brief.implementation_steps:
            findings.append(_finding("PrincipalReviewer", "error", "missing_implementation_steps"))
        if "career" in lower and "job" in lower:
            findings.append(_finding("PrincipalReviewer", "error", "job_market_roundup_leaked_into_article"))
        words = _word_count(body)
        if words < 650:
            findings.append(_finding("PrincipalReviewer", "error", f"article_too_short:{words}/650"))
        elif words < 900:
            if _is_high_signal_short_article(body):
                findings.append(_finding("PrincipalReviewer", "warning", f"article_short_high_signal:{words}/900"))
            else:
                findings.append(_finding("PrincipalReviewer", "error", f"article_short:{words}/900"))
        return findings


class VisualExplainer:
    def plan(self, brief: LessonBrief, pack: EvidencePack, draft_body: str) -> VisualPlan:
        first_ids = brief.evidence_ids[:3] or sorted(_evidence_ids(pack))[:3]
        if brief.lesson.id == "matmul-from-scalar-operations":
            mermaid = VisualAsset(
                asset_id="matmul-cell-flow",
                artifact_type="mermaid",
                title="One output cell is a row-column dot product",
                purpose="Show the exact data dependency for one matrix-multiplication output cell.",
                evidence_ids=first_ids[:2],
                placement="after_concept",
                content=(
                    "flowchart LR\n"
                    "  A[Row i of A] --> M[Multiply matching entries]\n"
                    "  B[Column j of B] --> M\n"
                    "  M --> S[Accumulate the products]\n"
                    "  S --> C[C[i,j]]"
                ),
            )
            svg = VisualAsset(
                asset_id="matmul-output-cell",
                artifact_type="svg",
                title="How one C cell is produced",
                purpose="Connect the direct loop to the row, column, and accumulator the implementation must preserve.",
                evidence_ids=first_ids[:2],
                placement="after_how",
                content="A row i; B column j; multiply matching entries; accumulate; write C[i,j]",
            )
            return VisualPlan(assets=[mermaid, svg], mermaid_required=True)
        mermaid = VisualAsset(
            asset_id="lesson-flow",
            artifact_type="mermaid",
            title="Lesson flow",
            purpose="Show how the concept moves from invariant to implementation and validation.",
            evidence_ids=first_ids[:2],
            placement="after_concept",
            content=(
                "flowchart LR\n"
                "  A[State the tensor or system invariant] --> B[Run the simplest correct algorithm]\n"
                "  B --> C[Profile the bottleneck]\n"
                "  C --> D[Add the optimized implementation]\n"
                "  D --> E[Gate with tests and benchmarks]"
            ),
        )
        svg = VisualAsset(
            asset_id="implementation-checkpoints",
            artifact_type="svg",
            title="Implementation checkpoints",
            purpose="Keep the lesson grounded in observable engineering checks.",
            evidence_ids=first_ids[:2],
            placement="after_how",
            content="Shape invariant; Correctness oracle; Bottleneck profile; Validation gate",
        )
        return VisualPlan(assets=[mermaid, svg], mermaid_required=True)


class TableDesigner:
    def design(self, plan: VisualPlan, brief: LessonBrief, pack: EvidencePack) -> VisualPlan:
        ids = brief.evidence_ids[:2] or sorted(_evidence_ids(pack))[:2]
        if brief.lesson.id == "matmul-from-scalar-operations":
            table = TableSpec(
                table_id="matmul-implementation-choices",
                title="Which matmul path answers which engineering question?",
                purpose="Compare implementation paths by the question they answer, not by decorative scores.",
                headers=["Question", "Direct loop", "Library matmul", "Tiled or GPU path"],
                rows=[
                    [
                        "What contract is being checked?",
                        "Row-column dot product and output shape",
                        "API semantics, batching, broadcasting, dtype behavior",
                        "Same contract under layout and hardware constraints",
                    ],
                    [
                        "What failure does it expose?",
                        "Index order, accumulator reset, shape mismatch",
                        "Unexpected broadcasting or dtype promotion",
                        "Memory access, layout, or kernel choice bottleneck",
                    ],
                    [
                        "When should it be used?",
                        "As the correctness oracle for small examples",
                        "As the trusted baseline for product code",
                        "After correctness and profiling identify the bottleneck",
                    ],
                ],
                evidence_ids=ids,
                placement="after_failure_modes",
            )
            return plan.model_copy(update={"tables": [*plan.tables, table]})
        rows = [
            ["Shape or state invariant", brief.concept_arc[0] if brief.concept_arc else "Name the invariant before optimizing.", "Prevents silent wrong results"],
            ["Failure mode", brief.failure_modes[0] if brief.failure_modes else "Mismatch between theory and implementation", "Turns review into a concrete test"],
            ["Validation gate", brief.evaluation_plan[0] if brief.evaluation_plan else "Compare against a correctness oracle", "Decides whether to move to optimization"],
        ]
        table = TableSpec(
            table_id="engineering-checks",
            title="Engineering checks for the lesson",
            purpose="Compare the invariant, common failure, and validation gate a Principal engineer should ask for.",
            headers=["Question", "What to inspect", "Why it matters"],
            rows=rows,
            evidence_ids=ids,
            placement="after_failure_modes",
        )
        return plan.model_copy(update={"tables": [*plan.tables, table]})


class ComponentDesigner:
    def design(self, plan: VisualPlan, brief: LessonBrief, pack: EvidencePack) -> VisualPlan:
        decision = brief.principal_decisions[0] if brief.principal_decisions else "Do not optimize until correctness and profiling evidence agree."
        component = ComponentSpec(
            component_id="principal-check",
            kind="callout",
            title="Principal engineer check",
            purpose="Highlight the decision the reader should be able to make after the lesson.",
            html=(
                '<div class="blogpipe-callout" data-kind="principal-check">'
                "<strong>Principal engineer check</strong>"
                f"<p>{_escape_inline(decision)}</p>"
                "</div>"
            ),
            evidence_ids=brief.evidence_ids[:1],
            placement="before_decision",
        )
        return plan.model_copy(update={"components": [*plan.components, component]})


class LayoutReviewer:
    def review(self, body: str, plan: VisualPlan) -> list[ReviewFinding]:
        findings: list[ReviewFinding] = []
        if "visual map" in (body or "").lower():
            findings.append(_finding("LayoutReviewer", "error", "generic_visual_map_present"))
        for table in plan.tables:
            if len(table.headers) > 5:
                findings.append(_finding("LayoutReviewer", "error", f"table_too_wide:{table.table_id}"))
            for row in table.rows:
                if any(len(cell) > 180 for cell in row):
                    findings.append(_finding("LayoutReviewer", "warning", f"table_cell_long:{table.table_id}"))
        if len(plan.components) > 2:
            findings.append(_finding("LayoutReviewer", "warning", "too_many_callouts"))
        try:
            MarkdownIt().parse(body)
        except Exception as exc:  # noqa: BLE001
            findings.append(_finding("LayoutReviewer", "error", f"markdown_parse_error:{exc}"))
        return findings


class ManagingEditor:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def draft_body(self, brief: LessonBrief, pack: EvidencePack) -> str:
        text = _agent_markdown(
            self.llm,
            task="managing_editor",
            system="You are the ManagingEditor agent for a technical ML blog. Return Markdown only.",
            user=(
                "Draft the article body from this brief and evidence. Write for a Staff/Principal ML engineer. "
                "Include concrete how-to content, failure modes, validation gates, and evidence citations with URLs. "
                "Avoid career advice and avoid claims not supported by the evidence.\n\n"
                f"BRIEF:\n{brief.model_dump_json(indent=2)}\n\nEVIDENCE:\n{_pack_prompt(pack)}"
            ),
        )
        if text and "[E" in text and "##" in text:
            return text
        return _article_body(brief, pack, visual_plan=None)

    def finalize_body(self, brief: LessonBrief, pack: EvidencePack, visual_plan: VisualPlan) -> str:
        fallback = _article_body(brief, pack, visual_plan=visual_plan)
        visual_markdown = _visual_markdown_prompt(visual_plan)
        text = _agent_markdown(
            self.llm,
            task="managing_editor",
            system=(
                "You are the ManagingEditor for a technical ML blog. Return final Markdown body only, no frontmatter. "
                "Use the evidence IDs exactly as provided and include the source URL near every cited evidence ID."
            ),
            user=(
                "Write the final article. Target 900-1400 words unless the article is unusually compact and dense. "
                "The post must teach from first principles, include a practical implementation path, name failure "
                "modes, give validation gates, and end with Staff/Principal-level decision criteria. Integrate the "
                "provided visuals/tables/components near the prose that explains them. Do not add new factual claims "
                "without evidence.\n\n"
                f"BRIEF:\n{brief.model_dump_json(indent=2)}\n\n"
                f"EVIDENCE:\n{_pack_prompt(pack)}\n\n"
                f"VISUAL_MARKDOWN:\n{visual_markdown}"
            ),
        )
        candidate = _normalize_article_body(text, pack, visual_plan)
        if candidate and _usable_final_article(candidate, brief, pack, visual_plan):
            return candidate
        return fallback

    def publish_or_preview(
        self,
        *,
        brief: LessonBrief,
        body: str,
        slug: str,
        visual_plan: VisualPlan,
        dry_run: bool,
    ) -> FinalArticle:
        mermaid = "```mermaid" in body
        post = _frontmatter(brief, body=body, mermaid=mermaid) + "\n" + body.strip() + "\n"
        if dry_run:
            path = _swarm_reports_dir() / f"{slug}.preview.md"
        else:
            path = memory.CONTENT_POST / f"{slug}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(post, encoding="utf-8")
        assets = [asset.path for asset in visual_plan.assets if asset.path]
        return FinalArticle(
            ok=True,
            title=brief.title,
            slug=slug,
            body=body,
            path=str(path.relative_to(memory.ROOT)),
            errors=[],
            mermaid=mermaid,
            assets=assets,
        )


def _default_brief(lesson: CatalogueLesson, pack: EvidencePack, role_report: RoleMarketReport) -> LessonBrief:
    chunks = _representative_chunks(pack, limit=8)
    source_urls = pack.urls()
    evidence_ids = [chunk.evidence_id for chunk in chunks]
    return LessonBrief(
        lesson=lesson,
        title=lesson.title,
        thesis=(
            f"{lesson.title} matters because senior ML engineers need to connect the first-principles "
            "mechanism to implementation constraints, failure modes, and release gates."
        ),
        concept_arc=lesson.concepts
        or [
            "Define the invariant before adding scale.",
            "Implement the direct algorithm as an oracle.",
            "Measure where the system stops matching the model.",
        ],
        implementation_steps=lesson.how_to,
        failure_modes=[
            "Shape or state assumptions are implicit and fail silently.",
            "A faster implementation is merged without a correctness oracle.",
            "Benchmarks measure throughput without explaining the bottleneck.",
        ],
        evaluation_plan=[
            "Run a minimal correctness test against a direct implementation.",
            "Profile the bottleneck before changing kernels, data layout, or serving architecture.",
            "Record the release gate a Staff/Principal engineer would use to approve the change.",
        ],
        principal_decisions=[
            "Decide whether the mechanism is understood well enough to optimize, scale, or operationalize.",
            "Name the blocker that would prevent use in a production ML system.",
        ],
        evidence_ids=evidence_ids,
        source_urls=source_urls,
    )


def _article_body(brief: LessonBrief, pack: EvidencePack, *, visual_plan: VisualPlan | None) -> str:
    if brief.lesson.id == "matmul-from-scalar-operations":
        return _matmul_article_body(brief, pack, visual_plan=visual_plan)
    chunks = pack.chunks
    citations = _distinct_citations(pack, limit=4)
    citation = citations[0] if citations else _citation(chunks, 0)
    second = citations[1] if len(citations) > 1 else citation
    third = citations[2] if len(citations) > 2 else second
    fourth = citations[3] if len(citations) > 3 else third
    parts: list[str] = [
        f"# {brief.title}",
        "",
        "## Why this is the next lesson",
        (
            f"{brief.thesis} The practical target is not a paper summary; it is the ability to explain the "
            f"mechanism, implement the simplest useful version, and decide what evidence would justify moving "
            f"to a larger system. A foundations post should leave the reader with a usable engineering test, "
            f"not just a vocabulary list. {citation}"
        ),
        "",
        "## Concept from first principles",
        _bullets(brief.concept_arc, citation=second or citation),
    ]
    if visual_plan:
        parts.extend(_visual_blocks(visual_plan, placement="after_concept"))
    parts.extend(
        [
            "",
            "## Mechanism and engineering intuition",
            (
                "Read the mechanism as a set of invariants. For a tensor lesson that means shapes, axes, memory "
                "layout, and numerical tolerance. For a systems lesson it means ownership of state, data movement, "
                f"latency budget, and rollback behavior. The evidence should be used to sharpen those invariants, "
                f"not to decorate the article with unrelated claims. The useful discipline is to state what must "
                f"remain true when the implementation changes. {second or citation}"
            ),
            "",
            "## How to implement",
            _ordered(brief.implementation_steps, citation=third or citation),
            "",
            "```python",
            "def lesson_oracle(inputs):",
            "    assert inputs, 'name the smallest valid input before optimizing'",
            "    reference = direct_correct_implementation(inputs)",
            "    candidate = optimized_or_scaled_implementation(inputs)",
            "    assert close_enough(reference, candidate)",
            "    return profile(candidate)",
            "```",
            "",
            (
                "Treat that oracle as disposable production code but permanent engineering knowledge. It gives the "
                "team a small input, an expected output, and a place to attach assertions before optimized code "
                f"enters the path. {third or citation}"
            ),
        ]
    )
    if visual_plan:
        parts.extend(_visual_blocks(visual_plan, placement="after_how"))
    parts.extend(
        [
            "",
            "## Failure modes",
            _bullets(brief.failure_modes, citation=third or citation),
        ]
    )
    if visual_plan:
        parts.extend(_visual_blocks(visual_plan, placement="after_failure_modes"))
    parts.extend(
        [
            "",
            "## Evaluation and release gate",
            _ordered(brief.evaluation_plan, citation=fourth or citation),
        ]
    )
    if visual_plan:
        parts.extend(_visual_blocks(visual_plan, placement="before_decision"))
    parts.extend(
        [
            "",
            "## Principal engineer decision",
            _bullets(brief.principal_decisions, citation=citation),
            "",
            (
                "A Principal engineer should leave this lesson with a sharper question: what invariant, benchmark, "
                "or failure case must be made explicit before the next layer of scale is added? That question keeps "
                f"the catalogue technical and prevents frontier-model reading from turning into trend tracking. {citation}"
            ),
        ]
    )
    return "\n".join(part for part in parts if part is not None)


def _matmul_article_body(brief: LessonBrief, pack: EvidencePack, *, visual_plan: VisualPlan | None) -> str:
    citations = _distinct_citations(pack, limit=4)
    c1 = citations[0] if citations else _citation(pack.chunks, 0)
    c2 = citations[1] if len(citations) > 1 else c1
    c3 = citations[2] if len(citations) > 2 else c2
    c4 = citations[3] if len(citations) > 3 else c3
    parts: list[str] = [
        f"# {brief.title}",
        "",
        "## Why this is the next lesson",
        (
            "Matrix multiplication is the first linear-algebra lesson worth making operational because it is small "
            "enough to derive by hand and important enough to sit under dense layers, attention, embeddings, and "
            "many kernel-level performance decisions. The foundation is not the symbol `A @ B`; it is the contract "
            "that every output cell is a dot product between one row of the left input and one column of the right "
            f"input. Once that contract is clear, the implementation questions become concrete: which shapes are "
            f"legal, where does the accumulator reset, what should a reference answer be, and when is it valid to "
            f"replace the direct loop with a library or GPU kernel? {c1}"
        ),
        "",
        "## Concept from first principles",
        (
            "Start with one output cell. If the left matrix has rows indexed by `i` and a shared inner axis `k`, and "
            "the right matrix has columns indexed by `j` over the same inner axis, then `C[i,j]` is the sum of "
            "matching products along `k`. That sentence is the invariant. The outer indices choose the cell to "
            "write; the inner index walks the two input slices that feed that cell. If the shared axis does not "
            "match, there is no well-defined dot product to compute. If the output shape is wrong, the code has "
            f"already lost the algebra before performance enters the discussion. {c1}"
        ),
    ]
    if visual_plan:
        parts.extend(_visual_blocks(visual_plan, placement="after_concept"))
    parts.extend(
        [
            "",
            "## Mechanism and engineering intuition",
            (
                "The direct algorithm is intentionally boring: choose an output row, choose an output column, reset "
                "an accumulator, walk the shared axis, add product terms, and write the accumulator to the output. "
                "That boring loop is valuable because it separates correctness from speed. Library matmul adds API "
                "semantics such as vector, matrix, and batched cases; optimized kernels add layout and hardware "
                f"constraints. None of those should change the row-column contract the direct loop exposes. {c2}"
            ),
            "",
            "## How to implement",
            (
                "Implement the direct path first and keep it as a correctness oracle for small tensors. Name the "
                "dimensions, assert the shared dimension, allocate the output shape, and make the accumulator local "
                "to one output cell. Then compare the result with a trusted library call before measuring anything. "
                f"Only after the oracle and the library agree should profiling decide whether layout, batching, or "
                f"kernel choice is the next problem. {c2}"
            ),
            "",
            "```python",
            "def matmul_reference(a, b):",
            "    m = len(a)",
            "    k = len(a[0])",
            "    assert k == len(b), 'inner dimensions must match'",
            "    n = len(b[0])",
            "    c = [[0.0 for _ in range(n)] for _ in range(m)]",
            "    for i in range(m):",
            "        for j in range(n):",
            "            acc = 0.0",
            "            for kk in range(k):",
            "                acc += a[i][kk] * b[kk][j]",
            "            c[i][j] = acc",
            "    return c",
            "```",
            "",
            (
                "The code review should focus on the contract. The accumulator must reset inside the output-cell "
                "loop. The shared axis must be the same length on both inputs. The output shape must be derived from "
                "the unshared axes, not copied from either input. The comparison with the library should use a "
                "tolerance appropriate to the dtype because floating-point addition order can differ once optimized "
                f"implementations enter the path. {c2}"
            ),
        ]
    )
    if visual_plan:
        parts.extend(_visual_blocks(visual_plan, placement="after_how"))
    parts.extend(
        [
            "",
            "## Failure modes",
            (
                "The common failures are simple enough to miss in a large model. A tensor can have plausible shape "
                "metadata while the intended axes are swapped. Broadcasting can hide a mismatch in higher-level API "
                "calls. An accumulator can live at the wrong loop level and smear one cell's partial sum into the "
                "next. A benchmark can celebrate a faster path while never checking that the result still matches "
                "the reference. These are not academic errors; they are exactly the class of mistakes that become "
                f"hard to diagnose after matmul is buried inside attention, MLP blocks, or batched inference code. {c2}"
            ),
        ]
    )
    if visual_plan:
        parts.extend(_visual_blocks(visual_plan, placement="after_failure_modes"))
    parts.extend(
        [
            "",
            "## Evaluation and release gate",
            (
                "Use a small evaluation ladder. First, run hand-checkable cases where the expected output is obvious. "
                "Then compare random small tensors against the trusted library call. Then add shape-error tests so "
                "illegal inputs fail loudly. Only then should a performance benchmark run. The benchmark should say "
                "what it is diagnosing: arithmetic work, memory access, layout conversion, batching, or kernel "
                "selection. Performance guidance for matrix multiplication is mostly about that diagnosis, because "
                f"faster kernels keep the same mathematical contract while changing how work and data movement are "
                f"organized. {c3}"
            ),
            "",
            "## Principal engineer decision",
            (
                "A Principal engineer should ask whether the team can explain the matmul contract at every boundary "
                "where tensors enter or leave a subsystem. If the team cannot point to the reference oracle, the "
                "trusted-library comparison, the dtype tolerance, and the profiling question, the implementation is "
                "not ready for clever optimization. The next lesson can add memory layout or GPU tiling only after "
                f"this one is stable: output cells are row-column dot products, correctness comes before speed, and "
                f"benchmarks must diagnose a specific bottleneck. {c4}"
            ),
        ]
    )
    if visual_plan:
        parts.extend(_visual_blocks(visual_plan, placement="before_decision"))
    return "\n".join(part for part in parts if part is not None)


def _visual_blocks(plan: VisualPlan, *, placement: str) -> list[str]:
    blocks: list[str] = []
    for asset in plan.assets:
        if asset.placement != placement:
            continue
        if asset.artifact_type == "mermaid":
            blocks.extend(["", visuals.mermaid_block(asset)])
        elif asset.artifact_type == "svg" and asset.path:
            blocks.extend(["", visuals.svg_markdown(asset)])
    for table in plan.tables:
        if table.placement == placement:
            blocks.extend(["", visuals.markdown_table(table)])
    for component in plan.components:
        if component.placement == placement:
            blocks.extend(["", visuals.component_html(component)])
    return blocks


def _visual_markdown_prompt(plan: VisualPlan) -> str:
    parts: list[str] = []
    for placement in ("after_concept", "after_how", "after_failure_modes", "before_decision"):
        blocks = [block for block in _visual_blocks(plan, placement=placement) if block.strip()]
        if blocks:
            parts.append(f"Placement: {placement}\n" + "\n\n".join(blocks))
    return "\n\n".join(parts)


def _normalize_article_body(text: str, pack: EvidencePack, plan: VisualPlan) -> str:
    body = (text or "").strip()
    if not body:
        return ""
    if body.startswith("---"):
        parts = body.split("---", 2)
        if len(parts) == 3:
            body = parts[2].strip()
    body = _ensure_visual_blocks(body, plan)
    body = _ensure_source_links(body, pack)
    return body.strip()


def _ensure_visual_blocks(body: str, plan: VisualPlan) -> str:
    out = body
    for placement in ("after_concept", "after_how", "after_failure_modes", "before_decision"):
        block = "\n\n".join(part for part in _visual_blocks(plan, placement=placement) if part.strip()).strip()
        if not block or block in out:
            continue
        out = _insert_visual_block(out, placement, block)
    return out


def _insert_visual_block(body: str, placement: str, block: str) -> str:
    lines = body.splitlines()
    heading_keywords = {
        "after_concept": ("concept", "first principles"),
        "after_how": ("how to implement", "implementation"),
        "after_failure_modes": ("failure modes",),
        "before_decision": ("principal engineer", "decision"),
    }[placement]
    heading_index = -1
    for idx, line in enumerate(lines):
        lower = line.lower()
        if line.startswith("##") and any(keyword in lower for keyword in heading_keywords):
            heading_index = idx
            break
    if heading_index < 0:
        return body.rstrip() + "\n\n" + block + "\n"
    if placement == "before_decision":
        return "\n".join([*lines[:heading_index], "", block, "", *lines[heading_index:]])
    next_heading = len(lines)
    for idx in range(heading_index + 1, len(lines)):
        if lines[idx].startswith("##"):
            next_heading = idx
            break
    return "\n".join([*lines[:next_heading], "", block, "", *lines[next_heading:]])


def _ensure_source_links(body: str, pack: EvidencePack) -> str:
    refs = sorted({f"E{ref}" for ref in re.findall(r"\[E(\d+)\]", body or "")}, key=lambda value: int(value[1:]))
    if not refs:
        return body
    chunks = _chunk_by_id(pack)
    missing = [ref for ref in refs if (chunk := chunks.get(ref)) and chunk.url and chunk.url not in body]
    if not missing:
        return body
    lines = ["## Source links"]
    for ref in missing:
        chunk = chunks.get(ref)
        if chunk:
            lines.append(f"- [{ref}] {chunk.url}")
    return body.rstrip() + "\n\n" + "\n".join(lines) + "\n"


def _usable_final_article(body: str, brief: LessonBrief, pack: EvidencePack, plan: VisualPlan) -> bool:
    if "[E" not in body or "##" not in body:
        return False
    if _word_count(body) < 650:
        return False
    findings = [
        *SkepticalFactChecker().review(body, pack, visual_plan=plan),
        *PrincipalReviewer().review(brief, body),
    ]
    return not any(finding.severity == "error" for finding in findings)


def _frontmatter(brief: LessonBrief, *, body: str, mermaid: bool) -> str:
    tags = sorted({"ml-systems", "technical-learning", *brief.lesson.topic_tags})
    lines = [
        "---",
        f'title: "{brief.title.replace(chr(34), chr(39))}"',
        f"date: {datetime.now(timezone.utc).date().isoformat()}",
        "draft: true",
        f"mermaid: {'true' if mermaid else 'false'}",
        "tags:",
        *[f'  - "{tag}"' for tag in tags],
        "---",
    ]
    return "\n".join(lines)


def _catalogue_lessons() -> list[CatalogueLesson]:
    series = [
        CatalogueSeries(
            id="linear-algebra-ml-systems",
            title="Linear algebra for ML systems",
            description="First-principles tensor mechanics before kernels and accelerators.",
            lesson_ids=["matmul-from-scalar-operations", "matrix-shapes-and-layout", "gpu-matmul-tiling"],
        ),
        CatalogueSeries(
            id="autodiff-optimization",
            title="Autodiff and optimization",
            lesson_ids=["reverse-mode-autodiff", "optimizer-state-and-stability"],
        ),
        CatalogueSeries(
            id="transformers-attention",
            title="Transformers and attention",
            lesson_ids=["attention-as-indexed-retrieval", "transformer-block-systems-view"],
        ),
        CatalogueSeries(
            id="distributed-training",
            title="Distributed training systems",
            lesson_ids=["distributed-training-parallelism", "checkpointing-and-recovery"],
        ),
        CatalogueSeries(
            id="inference-serving",
            title="Inference and serving systems",
            lesson_ids=["kv-cache-and-batching", "latency-throughput-release-gates"],
        ),
        CatalogueSeries(
            id="evaluation-reliability",
            title="Evaluation and reliability",
            lesson_ids=["benchmark-design-for-ml-systems", "failure-mode-taxonomy"],
        ),
        CatalogueSeries(
            id="retrieval-grounding",
            title="Retrieval, grounding, and multimodal systems",
            lesson_ids=["retrieval-grounding-basics"],
        ),
        CatalogueSeries(
            id="agents-tools",
            title="Agents and tool-using systems",
            lesson_ids=["agent-orchestration-basics"],
        ),
    ]
    by_series = {item.id: item for item in series}
    specs = [
        ("matmul-from-scalar-operations", "Matrix multiplication from scalar multiply-accumulate", "linear-algebra-ml-systems", [], ["scalar multiply-accumulate", "dot products", "shape invariants"], ["How do output cells map to input rows and columns?"], ["Implement triple-loop matmul.", "Compare against a trusted library.", "Profile arithmetic work versus memory access."], ["linear-algebra", "ml-systems"]),
        ("matrix-shapes-and-layout", "Matrix shapes, strides, and memory layout", "linear-algebra-ml-systems", ["matmul-from-scalar-operations"], ["shape contracts", "strides", "contiguity"], ["What breaks when tensors have the right shape but wrong layout?"], ["Print shape and stride at every boundary.", "Add layout conversion tests.", "Benchmark contiguous and non-contiguous paths."], ["linear-algebra", "systems"]),
        ("gpu-matmul-tiling", "GPU matmul tiling and memory hierarchy", "linear-algebra-ml-systems", ["matrix-shapes-and-layout"], ["tiles", "shared memory", "occupancy"], ["Which bottleneck does tiling remove?"], ["Start with a block-level tile.", "Measure occupancy and memory bandwidth.", "Validate numerical drift."], ["gpu", "kernels"]),
        ("reverse-mode-autodiff", "Reverse-mode autodiff as a graph execution problem", "autodiff-optimization", ["matmul-from-scalar-operations"], ["computational graph", "vector-Jacobian product", "gradient accumulation"], ["Where does memory grow during backprop?"], ["Trace a two-operation graph.", "Store intermediates deliberately.", "Check gradients numerically."], ["autodiff", "optimization"]),
        ("optimizer-state-and-stability", "Optimizer state, scale, and training stability", "autodiff-optimization", ["reverse-mode-autodiff"], ["optimizer state", "learning-rate schedule", "stability"], ["Which optimizer states dominate memory?"], ["Log state tensors.", "Ablate schedule changes.", "Define divergence alarms."], ["optimization", "training"]),
        ("attention-as-indexed-retrieval", "Attention as indexed retrieval over tokens", "transformers-attention", ["matmul-from-scalar-operations"], ["queries", "keys", "values", "softmax"], ["What does each token retrieve and why?"], ["Build single-head attention.", "Inspect attention logits.", "Test masking edge cases."], ["attention", "transformers"]),
        ("transformer-block-systems-view", "Transformer blocks as systems components", "transformers-attention", ["attention-as-indexed-retrieval"], ["residual stream", "normalization", "mlp"], ["Where do latency and memory concentrate?"], ["Trace tensors through a block.", "Profile attention and MLP separately.", "Define serving constraints."], ["transformers", "systems"]),
        ("distributed-training-parallelism", "Distributed training parallelism choices", "distributed-training", ["transformer-block-systems-view"], ["data parallel", "tensor parallel", "pipeline parallel", "sharding"], ["Which bottleneck selects the parallelism strategy?"], ["Measure memory headroom.", "Profile communication.", "Run small-scale recovery tests."], ["distributed-training", "systems"]),
        ("checkpointing-and-recovery", "Checkpointing and recovery for long training runs", "distributed-training", ["distributed-training-parallelism"], ["checkpoint cadence", "optimizer state", "fault tolerance"], ["What does it cost to recover?"], ["Measure checkpoint write time.", "Validate restore fidelity.", "Inject restart failures."], ["training", "reliability"]),
        ("kv-cache-and-batching", "KV cache and batching for inference", "inference-serving", ["attention-as-indexed-retrieval"], ["kv cache", "batching", "prefill", "decode"], ["Which latency path dominates?"], ["Separate prefill and decode metrics.", "Test cache eviction.", "Model batch-size tradeoffs."], ["inference", "serving"]),
        ("latency-throughput-release-gates", "Latency, throughput, and serving release gates", "inference-serving", ["kv-cache-and-batching"], ["latency budget", "throughput", "rollback"], ["What blocks production rollout?"], ["Define SLOs.", "Load test representative traffic.", "Add rollback criteria."], ["serving", "reliability"]),
        ("benchmark-design-for-ml-systems", "Benchmark design for ML systems", "evaluation-reliability", ["matmul-from-scalar-operations"], ["benchmark", "dataset", "metric validity"], ["What does the benchmark fail to measure?"], ["Write the decision first.", "Choose metrics tied to failure modes.", "Add counterexamples."], ["evaluation", "reliability"]),
        ("failure-mode-taxonomy", "Failure-mode taxonomy for ML systems", "evaluation-reliability", ["benchmark-design-for-ml-systems"], ["taxonomy", "monitoring", "root cause"], ["Can the team classify failures consistently?"], ["Define failure classes.", "Label examples.", "Attach mitigations."], ["evaluation", "reliability"]),
        ("retrieval-grounding-basics", "Retrieval and grounding basics", "retrieval-grounding", ["benchmark-design-for-ml-systems"], ["retrieval", "grounding", "citation"], ["What makes an answer auditable?"], ["Build a small retrieval index.", "Trace citations.", "Test negative queries."], ["retrieval", "grounding"]),
        ("agent-orchestration-basics", "Agent orchestration basics", "agents-tools", ["benchmark-design-for-ml-systems"], ["orchestrator", "tools", "guardrails"], ["When should a workflow become agentic?"], ["Define state transitions.", "Validate tool outputs.", "Add evaluator loops."], ["agents", "tool-use"]),
    ]
    lessons: list[CatalogueLesson] = []
    for lesson_id, title, series_id, prereqs, concepts, questions, how_to, tags in specs:
        series_item = by_series[series_id]
        lessons.append(
            CatalogueLesson(
                id=lesson_id,
                title=title,
                series_id=series_id,
                series_title=series_item.title,
                prerequisites=list(prereqs),
                concepts=list(concepts),
                engineering_questions=list(questions),
                how_to=list(how_to),
                topic_tags=list(tags),
            )
        )
    return lessons


def _lesson_fit(item: RankedItem, lesson: CatalogueLesson) -> float:
    blob = _ranked_text(item)
    keywords = _keywords([lesson.title, *lesson.concepts, *lesson.engineering_questions, *lesson.how_to, *lesson.topic_tags])
    hits = sum(1 for keyword in keywords if keyword in blob)
    title_hits = sum(1 for keyword in _keywords([lesson.title]) if keyword in blob)
    return min(1.0, 0.12 * hits + 0.12 * title_hits + 0.08 * item.topic_scores.best)


def _foundation_ranked_items(lesson: CatalogueLesson, *, now: datetime | None = None) -> list[RankedItem]:
    specs = FOUNDATION_SOURCE_PACKS.get(lesson.id, [])
    published_at = now or datetime.now(timezone.utc)
    ranked: list[RankedItem] = []
    for index, spec in enumerate(specs, start=1):
        source = SourceItem(
            item_id=f"{lesson.id}-foundation-{index}",
            canonical_url=spec["url"],
            source_kind="reference",
            source_name=spec["source_name"],
            source_tier=1,
            title=spec["title"],
            published_at=published_at,
            abstract_or_excerpt=spec["body"],
            body_text=spec["body"],
            tags=sorted({"foundation", *lesson.topic_tags}),
            extra={
                "selector_role": "primary" if index <= 3 else "supporting",
                "catalogue_lesson_id": lesson.id,
                "foundation_reference": True,
            },
        ).normalized()
        ranked.append(
            RankedItem(
                item=source,
                topic_scores=TopicScores(ml_engineering=0.92, ml_theory=0.72, priority_track=lesson.series_id),
                daily_score=0.96 - index * 0.02,
                deep_dive_score=0.8,
                quality_signals={"lesson_fit": 1.0, "foundation_reference": True},
                citation_signals={"canonical": True},
                rank_reason="Curated canonical reference for foundations-first catalogue coverage.",
            )
        )
    return ranked


def _ranked_text(item: RankedItem) -> str:
    source = item.item
    return " ".join([source.title, source.abstract_or_excerpt, source.body_text, " ".join(source.tags)]).lower()


def _keywords(parts: list[str]) -> list[str]:
    out: list[str] = []
    for part in parts:
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9+-]{2,}", part.lower()):
            if token not in {"and", "the", "for", "with", "from", "into", "what", "when", "where", "how"}:
                out.append(token)
    return sorted(set(out))


def _agent_json(llm: LLMClient, *, task: str, system: str, user: str) -> dict[str, Any]:
    if not llm.configured() and not os.environ.get("BLOGPIPE_FAKE_LLM_RESPONSE") and not os.environ.get("BLOGPIPE_FAKE_LLM_RESPONSES"):
        return {}
    try:
        return jsonish.loads_object(llm.complete(system=system, user=user, task=task))
    except Exception as exc:  # noqa: BLE001
        LOG.warning("%s agent JSON fallback: %s", task, exc)
        return {}


def _agent_markdown(llm: LLMClient, *, task: str, system: str, user: str) -> str:
    if not llm.configured() and not os.environ.get("BLOGPIPE_FAKE_LLM_RESPONSE") and not os.environ.get("BLOGPIPE_FAKE_LLM_RESPONSES"):
        return ""
    try:
        return llm.complete(system=system, user=user, task=task)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("%s agent Markdown fallback: %s", task, exc)
        return ""


def _pack_prompt(pack: EvidencePack) -> str:
    payload = [
        {
            "evidence_id": chunk.evidence_id,
            "title": chunk.title,
            "url": chunk.url,
            "type": chunk.evidence_type,
            "text": chunk.text,
        }
        for chunk in pack.chunks[:12]
    ]
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _representative_chunks(pack: EvidencePack, *, limit: int) -> list[Any]:
    by_source: list[Any] = []
    seen_urls: set[str] = set()
    for chunk in pack.chunks:
        if chunk.url and chunk.url not in seen_urls:
            by_source.append(chunk)
            seen_urls.add(chunk.url)
        if len(by_source) >= limit:
            return by_source
    out = list(by_source)
    seen_ids = {chunk.evidence_id for chunk in out}
    for chunk in pack.chunks:
        if chunk.evidence_id in seen_ids:
            continue
        out.append(chunk)
        if len(out) >= limit:
            break
    return out


def _distinct_citations(pack: EvidencePack, *, limit: int = 4) -> list[str]:
    citations = [f"[{chunk.evidence_id}] {chunk.url}" for chunk in _representative_chunks(pack, limit=limit) if chunk.url]
    while len(citations) < limit:
        fallback = _citation(pack.chunks, len(citations))
        if not fallback:
            break
        citations.append(fallback)
    return citations


def _citation(chunks: list[Any], index: int) -> str:
    if index >= len(chunks):
        return ""
    chunk = chunks[index]
    return f"[{chunk.evidence_id}] {chunk.url}"


def _bullets(items: list[str], *, citation: str) -> str:
    return "\n".join(f"- {item} {citation}".rstrip() for item in items[:6])


def _ordered(items: list[str], *, citation: str) -> str:
    return "\n".join(f"{idx}. {item} {citation}".rstrip() for idx, item in enumerate(items[:6], start=1))


def _slug_for_lesson(title: str) -> str:
    today = datetime.now(timezone.utc).date().isoformat()
    return f"{today}-{memory.slugify(title)}"


def _write_report(name: str, payload: object) -> None:
    path = _swarm_reports_dir() / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _swarm_reports_dir() -> Path:
    return memory.REPORTS / "swarm"


def _catalogue_state_path() -> Path:
    return memory.DATA / "catalogue_state.json"


def _load_catalogue_state() -> dict[str, Any]:
    path = _catalogue_state_path()
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _status(findings: list[ReviewFinding]) -> str:
    return "blocked" if any(f.severity == "error" for f in findings) else "ok"


def _finding(agent: str, severity: str, message: str, *, evidence_id: str = "", path: str = "") -> ReviewFinding:
    return ReviewFinding(agent_name=agent, severity=severity, message=message, evidence_id=evidence_id, path=path)


def _findings_from_errors(agent: str, errors: list[str]) -> list[ReviewFinding]:
    return [_finding(agent, "error", error) for error in errors]


def _evidence_ids(pack: EvidencePack) -> set[str]:
    return {chunk.evidence_id for chunk in pack.chunks}


def _chunk_by_id(pack: EvidencePack) -> dict[str, Any]:
    return {chunk.evidence_id: chunk for chunk in pack.chunks}


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))


def _is_high_signal_short_article(text: str) -> bool:
    lower = (text or "").lower()
    cited_refs = set(re.findall(r"\[E\d+\]", text or ""))
    source_urls = set(re.findall(r"https?://[^\s)]+", text or ""))
    has_visual = "```mermaid" in lower or re.search(r"!\[[^\]]+\]\(/img/posts/[^)]+\.svg\)", text or "") is not None
    has_table = "\n| " in (text or "") and "\n| ---" in (text or "")
    has_code = "```" in (text or "")
    required_sections = all(phrase in lower for phrase in ("how to implement", "failure modes", "principal engineer"))
    return required_sections and has_visual and has_table and has_code and len(cited_refs) >= 4 and len(source_urls) >= 2


def _claim_text(text: str) -> str:
    cleaned = re.sub(r"```[\s\S]*?```", " ", text or "")
    cleaned = re.sub(r"https?://\S+", " ", cleaned)
    cleaned = re.sub(r"/img/posts/\S+", " ", cleaned)
    cleaned = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", " ", cleaned)
    return cleaned


def _topic_tags(text: str) -> list[str]:
    lower = text.lower()
    out: list[str] = []
    for tag, keywords in TOPIC_KEYWORDS.items():
        if any(keyword in lower for keyword in keywords):
            out.append(tag)
    return out


def _skill_phrases(text: str) -> list[str]:
    lower = text.lower()
    phrases = sorted({phrase for keywords in TOPIC_KEYWORDS.values() for phrase in keywords if phrase in lower})
    return phrases


def _systems_phrase_set() -> set[str]:
    return {
        "distributed training",
        "parallelism",
        "sharding",
        "fsdp",
        "checkpoint",
        "nccl",
        "inference",
        "serving",
        "latency",
        "throughput",
        "quantization",
        "cache",
        "evaluation",
        "benchmark",
        "agent",
        "tool",
    }


def _seniority(text: str) -> str:
    match = SENIORITY_RE.search(text)
    return match.group(1).lower() if match else "unknown"


def _first_sentence(text: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", " ".join((text or "").split()))
    return parts[0] if parts else ""


def _anchor_role_postings(url: str, html_text: str) -> list[dict[str, str]]:
    postings: list[dict[str, str]] = []
    for match in re.finditer(r"<a\b(?P<attrs>[^>]*)>(?P<label>[\s\S]{0,900}?)</a>", html_text or "", re.I):
        label = " ".join(_html_lines(match.group("label")))
        if not _looks_like_role_title(label):
            continue
        href_match = re.search(r"\bhref\s*=\s*['\"]([^'\"]+)['\"]", match.group("attrs"), re.I)
        source_url = _absolute_url(url, href_match.group(1)) if href_match else url
        context = _strip_html((html_text or "")[max(0, match.start() - 900) : min(len(html_text or ""), match.end() + 1200)])
        postings.append(
            {
                "company": _company_from_url(url),
                "title": label,
                "description": context,
                "url": source_url,
            }
        )
        if len(postings) >= 50:
            break
    return postings


def _html_lines(text: str) -> list[str]:
    cleaned = re.sub(r"<(script|style)[\s\S]*?</\1>", " ", text or "", flags=re.I)
    cleaned = re.sub(r"<[^>]+>", "\n", cleaned)
    lines: list[str] = []
    for raw in cleaned.splitlines():
        line = re.sub(r"\s+", " ", raw).strip()
        if line:
            lines.append(line)
    return lines


def _looks_like_role_title(title: str) -> bool:
    clean = " ".join((title or "").split())
    if len(clean) < 12 or len(clean) > 150:
        return False
    lower = clean.lower()
    if any(skip in lower for skip in ("careers", "see open roles", "all jobs", "privacy", "cookie", "terms", "login")):
        return False
    if lower.startswith(("our ", "we ", "you ", "the ", "this ", "join ")):
        return False
    if len(clean.split()) > 18:
        return False
    if not SENIORITY_RE.search(clean):
        return False
    if not re.search(r"\b(engineer|scientist|researcher|technical staff|infrastructure|systems|machine learning|ai|ml)\b", clean, re.I):
        return False
    return bool(TECH_ROLE_RE.search(clean))


def _strip_html(text: str) -> str:
    text = re.sub(r"<(script|style)[\s\S]*?</\1>", " ", text or "", flags=re.I)
    text = re.sub(r"<[^>]+>", "\n", text)
    return re.sub(r"\s+", " ", text)


def _absolute_url(base_url: str, href: str) -> str:
    href = (href or "").strip()
    if href.startswith(("http://", "https://")):
        return href
    if href.startswith("//"):
        return f"https:{href}"
    if not href:
        return base_url
    root = base_url.split("//", 1)
    if len(root) != 2:
        return href
    scheme, rest = root
    host = rest.split("/", 1)[0]
    if href.startswith("/"):
        return f"{scheme}//{host}{href}"
    prefix = base_url.rsplit("/", 1)[0]
    return f"{prefix}/{href}"


def _company_from_url(url: str) -> str:
    host = re.sub(r"^www\.", "", (url.split("//")[-1].split("/", 1)[0]).lower())
    mapping = {
        "openai.com": "OpenAI",
        "anthropic.com": "Anthropic",
        "deepmind.google": "Google DeepMind",
        "metacareers.com": "Meta AI",
        "nvidia.com": "NVIDIA",
        "microsoft.com": "Microsoft AI",
        "amazon.jobs": "Amazon AGI",
        "apple.com": "Apple ML",
        "x.ai": "xAI",
        "cohere.com": "Cohere",
        "mistral.ai": "Mistral",
        "huggingface.co": "Hugging Face",
        "databricks.com": "Databricks/MosaicML",
    }
    for key, company in mapping.items():
        if key in host:
            return company
    return host


def _dedupe_role_signals(signals: list[RoleMarketSignal]) -> list[RoleMarketSignal]:
    seen: set[tuple[str, str, str]] = set()
    out: list[RoleMarketSignal] = []
    for signal in signals:
        key = (signal.company.lower(), signal.role_title.lower(), signal.source_url)
        if key in seen:
            continue
        seen.add(key)
        out.append(signal)
    return out


def _escape_inline(text: str) -> str:
    return (
        (text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def seed_sources_for_lesson(lesson: CatalogueLesson, *, now: datetime | None = None) -> list[SourceItem]:
    published_at = now or datetime.now(timezone.utc)
    concept_blob = ", ".join(lesson.concepts + lesson.how_to)
    return [
        SourceItem(
            item_id=f"{lesson.id}-seed-{idx}",
            canonical_url=f"https://example.com/{lesson.id}/seed-{idx}",
            source_kind="paper" if idx < 3 else "blog",
            source_name="fixture",
            source_tier=1,
            title=f"{lesson.title} reference {idx + 1}",
            published_at=published_at,
            abstract_or_excerpt=(
                f"{lesson.title} requires {concept_blob}. The mechanism is an algorithmic path with benchmark "
                "evidence, implementation constraints, failure mode analysis, and production validation gates."
            ),
            tags=list(lesson.topic_tags),
        )
        for idx in range(4)
    ]
