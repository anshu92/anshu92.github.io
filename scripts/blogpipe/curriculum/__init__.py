from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .. import memory
from ..models import (
    CurriculumNode,
    CurriculumPlan,
    EvidenceCurationItem,
    EvidenceCurationResult,
    RankedItem,
    ResearchPlan,
    SelectionItem,
    SelectionResult,
    WriteResult,
)

TREE_VERSION = "foundation-models-v2"
TREE_PATH = Path(__file__).with_name("foundation_models.yaml")


def load_tree(path: str | Path | None = None) -> list[CurriculumNode]:
    raw = yaml.safe_load(Path(path or TREE_PATH).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("curriculum_tree_invalid:root_not_object")
    if str(raw.get("version", "")).strip() != TREE_VERSION:
        raise ValueError(f"curriculum_tree_invalid:version:{raw.get('version')}")
    nodes = [CurriculumNode.model_validate(node) for node in raw.get("nodes", [])]
    validate_tree(nodes)
    return nodes


def validate_tree(nodes: list[CurriculumNode]) -> None:
    if not nodes:
        raise ValueError("curriculum_tree_invalid:no_nodes")
    ids: set[str] = set()
    for node in nodes:
        if node.id in ids:
            raise ValueError(f"curriculum_tree_invalid:duplicate_id:{node.id}")
        ids.add(node.id)
        _validate_problem_node(node)
    for node in nodes:
        missing = sorted(set(node.prerequisites) - ids)
        if missing:
            raise ValueError(f"curriculum_tree_invalid:missing_prerequisite:{node.id}:{','.join(missing)}")


def choose_next_problem(tree: list[CurriculumNode] | None = None) -> CurriculumPlan:
    nodes = tree or load_tree()
    completed = _completed_node_ids()
    forced = os.environ.get("BLOGPIPE_CURRICULUM_NODE", "").strip()
    by_id = {node.id: node for node in nodes}
    if forced:
        if forced not in by_id:
            raise ValueError(f"unknown_curriculum_node:{forced}")
        node = by_id[forced]
        selected_by = "forced_env"
    else:
        completed_set = set(completed)
        node = next(
            (
                candidate
                for candidate in nodes
                if candidate.id not in completed_set and set(candidate.prerequisites) <= completed_set
            ),
            next((candidate for candidate in nodes if candidate.id not in completed_set), nodes[0]),
        )
        selected_by = "next_uncompleted_problem"
    return CurriculumPlan(
        node=node,
        selected_by=selected_by,
        completed_node_ids=completed,
        tree_version=TREE_VERSION,
    )


def plan_curriculum(ranked: list[RankedItem] | None = None) -> CurriculumPlan:
    """Backward-compatible entrypoint; ranked items must not affect problem choice."""
    return choose_next_problem()


def build_research_plan(plan: CurriculumPlan) -> ResearchPlan:
    strategy = plan.node.research_strategy
    queries = _string_list(strategy.get("queries"))
    if not queries:
        queries = _default_queries(plan.node)
    source_profiles = _string_list(strategy.get("source_profiles")) or ["textbooks", "systems", "papers", "engineering_blogs"]
    return ResearchPlan(
        problem_id=plan.node.id,
        problem_statement=plan.node.problem_statement,
        queries=queries,
        source_profiles=source_profiles,
        expected_evidence=list(plan.node.evidence_requirements),
        background_queries=_string_list(strategy.get("background_queries")),
    )


def curate_evidence(ranked: list[RankedItem], plan: CurriculumPlan) -> EvidenceCurationResult:
    scored = sorted(
        (_curation_item(ranked_item, plan.node) for ranked_item in ranked),
        key=lambda item: item.fit_score,
        reverse=True,
    )
    selected = [item for item in scored if item.fit_score >= 0.18][: max(3, min(6, len(scored)))]
    for index, item in enumerate(selected):
        item.role = "primary" if index < 3 and item.fit_score >= 0.28 else "supporting"
    primary_count = sum(1 for item in selected if item.role == "primary")
    gaps = _evidence_gaps(plan.node, selected)
    sufficient = bool(primary_count >= 2 and selected and selected[0].fit_score >= 0.34 and not gaps)
    if not selected:
        gaps.append("no_candidate_evidence_matches_problem")
    elif primary_count < 2:
        gaps.append(f"too_few_primary_problem_evidence_items:{primary_count}/2")
    return EvidenceCurationResult(
        problem_id=plan.node.id,
        problem_statement=plan.node.problem_statement,
        selected_item_ids=[item.item_id for item in selected],
        items=selected,
        gaps=_dedupe_strings(gaps),
        sufficient=sufficient,
        confidence=round(min(1.0, sum(item.fit_score for item in selected[:3]) / 3.0), 3),
    )


def apply_curation(ranked: list[RankedItem], curation: EvidenceCurationResult) -> tuple[list[RankedItem], SelectionResult]:
    by_id = {item.item.item_id: item for item in ranked}
    selected: list[RankedItem] = []
    selection_items: list[SelectionItem] = []
    for curated in curation.items:
        ranked_item = by_id.get(curated.item_id)
        if ranked_item is None:
            continue
        ranked_item.item.extra["selector_role"] = curated.role
        ranked_item.item.extra["curriculum_problem_id"] = curation.problem_id
        ranked_item.item.extra["curriculum_fit"] = curated.fit_score
        ranked_item.quality_signals["curriculum_fit"] = curated.fit_score
        selected.append(ranked_item)
        selection_items.append(
            SelectionItem(
                item_id=curated.item_id,
                role=curated.role,
                relevance_label="problem_evidence",
                reason=curated.reason,
                scores={"problem_fit": curated.fit_score},
                suggested_tags=["research-curriculum"],
            )
        )
    return selected, SelectionResult(
        selected_item_ids=[item.item.item_id for item in selected],
        items=selection_items,
        suggested_tags=["research-curriculum", "ml-systems"],
    )


def research_packet(
    *,
    plan: CurriculumPlan,
    research_plan: ResearchPlan,
    curation: EvidenceCurationResult,
    ranked: list[RankedItem],
) -> dict[str, Any]:
    return {
        "problem_id": plan.node.id,
        "problem_statement": plan.node.problem_statement,
        "research_plan": research_plan.model_dump(mode="json"),
        "evidence_sufficient": curation.sufficient,
        "gaps": list(curation.gaps),
        "candidate_count": len(ranked),
        "top_candidates": [item.model_dump(mode="json") for item in curation.items[:8]],
    }


def propose_tree_growth(plan: CurriculumPlan, curation: EvidenceCurationResult) -> dict[str, Any]:
    proposals: list[dict[str, Any]] = []
    if curation.gaps:
        prompt = plan.node.growth_prompts[0] if plan.node.growth_prompts else "What prerequisite problem should be researched next?"
        proposals.append(
            {
                "proposal_type": "investigate_gap",
                "parent_id": plan.node.id,
                "problem_statement": prompt if prompt.endswith("?") else f"{prompt}?",
                "prerequisites": [plan.node.id],
                "evidence_rationale": "; ".join(curation.gaps[:4]),
                "status": "review_required",
            }
        )
    return {
        "tree_version": plan.tree_version,
        "problem_id": plan.node.id,
        "proposals": proposals,
        "auto_mutated": False,
    }


def record_completion(plan: CurriculumPlan, result: WriteResult) -> None:
    if not result.ok:
        return
    memory.ensure_dirs()
    state = _load_state()
    completions = list(state.get("completions", []))
    completions.append(
        {
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "problem_id": plan.node.id,
            "problem_statement": plan.node.problem_statement,
            "post_title": result.title,
            "post_path": result.path,
            "tree_version": plan.tree_version,
        }
    )
    state["tree_version"] = TREE_VERSION
    state["completions"] = completions[-100:]
    _state_path().write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def _curation_item(ranked_item: RankedItem, node: CurriculumNode) -> EvidenceCurationItem:
    text = _blob(ranked_item)
    concepts = [*node.concepts_to_teach, *node.engineering_questions, *node.evidence_requirements]
    keywords = _keywords([node.problem_statement, *concepts, *_string_list(node.research_strategy.get("queries"))])
    hits = [keyword for keyword in keywords if keyword in text]
    phrase_hits = sum(1 for phrase in concepts if phrase and phrase.lower() in text)
    fit = min(1.0, 0.10 * len(hits) + 0.16 * phrase_hits + 0.08 * ranked_item.topic_scores.best)
    if ranked_item.item.source_kind == "paper":
        fit += 0.04
    if any(word in text for word in ("tutorial", "textbook", "course", "lecture", "implementation", "kernel", "benchmark")):
        fit += 0.08
    fit = round(min(1.0, fit), 3)
    evidence_types = _evidence_types(text)
    return EvidenceCurationItem(
        item_id=ranked_item.item.item_id,
        title=ranked_item.item.title,
        url=ranked_item.item.canonical_url,
        role="background",
        fit_score=fit,
        evidence_types=evidence_types,
        reason=_curation_reason(hits, evidence_types),
    )


def _evidence_gaps(node: CurriculumNode, selected: list[EvidenceCurationItem]) -> list[str]:
    if not selected:
        return list(node.evidence_requirements[:3])
    covered = {evidence_type for item in selected for evidence_type in item.evidence_types}
    gaps: list[str] = []
    for requirement in node.evidence_requirements:
        requirement_l = requirement.lower()
        if "implementation" in requirement_l and "implementation" not in covered:
            gaps.append(requirement)
        elif ("benchmark" in requirement_l or "performance" in requirement_l) and "benchmark" not in covered:
            gaps.append(requirement)
        elif ("concept" in requirement_l or "explanation" in requirement_l) and "conceptual" not in covered:
            gaps.append(requirement)
    return gaps


def _validate_problem_node(node: CurriculumNode) -> None:
    statement = node.problem_statement.strip()
    if not statement.endswith("?"):
        raise ValueError(f"curriculum_tree_invalid:not_problem_question:{node.id}")
    if len(statement.split()) < 8:
        raise ValueError(f"curriculum_tree_invalid:problem_too_short:{node.id}")
    vague = {"transformers", "gpu matmul", "attention", "rag", "alignment", "serving", "evaluation"}
    if statement.lower().strip(" ?") in vague:
        raise ValueError(f"curriculum_tree_invalid:vague_topic:{node.id}")
    if not node.concepts_to_teach or not node.engineering_questions or not node.evidence_requirements:
        raise ValueError(f"curriculum_tree_invalid:incomplete_node:{node.id}")


def _default_queries(node: CurriculumNode) -> list[str]:
    base = node.problem_statement.rstrip("?")
    concepts = " ".join(node.concepts_to_teach[:3])
    return [
        f"{base} {concepts} tutorial implementation",
        f"{base} {concepts} benchmark systems",
        f"{base} {concepts} deep learning foundation models",
    ]


def _completed_node_ids() -> list[str]:
    state = _load_state()
    out: list[str] = []
    for entry in state.get("completions", []):
        problem_id = str(entry.get("problem_id") or entry.get("node_id") or "").strip()
        if problem_id and problem_id not in out:
            out.append(problem_id)
    return out


def _load_state() -> dict[str, object]:
    path = _state_path()
    if not path.is_file():
        return {"tree_version": TREE_VERSION, "completions": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"tree_version": TREE_VERSION, "completions": []}
    if not isinstance(data, dict):
        return {"tree_version": TREE_VERSION, "completions": []}
    data.setdefault("tree_version", TREE_VERSION)
    data.setdefault("completions", [])
    return data


def _state_path() -> Path:
    return memory.DATA / "curriculum_state.json"


def _blob(ranked_item: RankedItem) -> str:
    item = ranked_item.item
    return " ".join(
        [
            item.title,
            item.abstract_or_excerpt,
            item.body_text,
            " ".join(item.tags),
            " ".join(ranked_item.topic_scores.matched_keywords),
            str(item.extra.get("search_profile", "")),
        ]
    ).lower()


def _keywords(values: list[str]) -> list[str]:
    stop = {"how", "does", "what", "why", "when", "where", "which", "the", "and", "or", "to", "a", "an", "of", "on", "in"}
    out: list[str] = []
    for value in values:
        for token in re.findall(r"[a-z0-9][a-z0-9+-]{2,}", value.lower()):
            if token not in stop and token not in out:
                out.append(token)
    return out[:80]


def _evidence_types(text: str) -> list[str]:
    types: list[str] = []
    if any(word in text for word in ("tutorial", "explain", "definition", "concept", "linear map", "multiply-accumulate")):
        types.append("conceptual")
    if any(
        word in text
        for word in (
            "implementation",
            "kernel",
            "cuda",
            "pytorch",
            "numpy",
            "code",
            "algorithm",
            "fsdp",
            "zero",
            "tensor parallel",
            "pipeline parallel",
            "checkpoint",
        )
    ):
        types.append("implementation")
    if any(word in text for word in ("benchmark", "latency", "throughput", "performance", "cache", "memory bandwidth", "flops")):
        types.append("benchmark")
    if any(word in text for word in ("limitation", "failure", "bottleneck", "numerical", "overflow", "precision")):
        types.append("limitation")
    return types or ["background"]


def _curation_reason(hits: list[str], evidence_types: list[str]) -> str:
    if hits:
        return f"Matches problem terms: {', '.join(hits[:6])}; evidence types: {', '.join(evidence_types)}."
    return f"Weak lexical match; evidence types: {', '.join(evidence_types)}."


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _dedupe_strings(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out
