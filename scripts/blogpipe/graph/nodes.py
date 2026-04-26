"""Graph node functions: thin wrappers and draft refine loop."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

from langgraph.types import interrupt

from .. import config, draft, formats, lint, memory, quality, topics
from ..draft import (
    _cleanup_missing_cites,
    _ensure_decision_section,
    _ensure_mechanism_section,
    _polish_body,
    _repair_or_inject_results_table,
    _resolve_cites,
    _sanitize_frontmatter_text,
    _slugify,
    _strip_unresolved,
    _stub_body,
    _unwrap_markdown_fence,
    build_prompt,
    embed_planned_visuals,
    explain_undefined_terms,  # editor-only explainer
    soften_unsupported_numeric_claims,
)
from ..llm_chain import budget, get_llm_usage, is_llm_call_cap_reached
from ..memory import _ROOT
from ..models import CitationAuditReport, EvidenceBundle, GapNote, PlanningBrief, ReviewNote
from ..prompting import render_prompt
from ..source_audit import build_source_registry, strip_unregistered_links
from . import llm as gllm
from . import supervisor as sup
from .critics import (
    filler_detector_flag_section,
    global_rubric,
    grounding_check_node,
    rewrite_section,
    rewrite_section_for_grounding,
    section_critic_llm,
)
from ..evidence_coverage import (
    coverage_report,
    render_unused_facts,
    unused_facts_for_section,
)
from .state import evidence_model, brief_model

LOG = logging.getLogger(__name__)

_MAX_RESEARCH_WINGS = 2
_MAX_REWRITES = 2


def _source_registry(bundle: EvidenceBundle) -> list[dict[str, Any]]:
    return [x.model_dump() for x in build_source_registry(bundle)]


def _gap_notes(body: str, bundle: EvidenceBundle, planning: PlanningBrief) -> list[GapNote]:
    issues = lint.structural_issues(body, bundle)
    out: list[GapNote] = []
    if "no_results_table" in issues:
        out.append(
            GapNote(
                code="no_results_table",
                message="Results table is missing or was removed.",
                section_hint="Numbers the paper actually gives us",
                required_evidence=["paper_experiments", "benchmarks"],
                query_hint=f"{bundle.primary.title} benchmark results baseline metrics",
            )
        )
    if "missing_mechanism_section" in issues:
        out.append(
            GapNote(
                code="missing_mechanism_section",
                message="Draft does not explain how the method works.",
                section_hint="Why this works",
                required_evidence=["paper_method"],
                query_hint=f"{bundle.primary.title} method mechanism",
            )
        )
    if "missing_decision_section" in issues:
        out.append(
            GapNote(
                code="missing_decision_section",
                message="Draft does not tell the reader what to test next.",
                section_hint="What I would test next",
                required_evidence=["paper_limitations", "paper_reproducibility"],
                query_hint=f"{bundle.primary.title} limitations reproducibility",
            )
        )
    if "comparative_claim_missing_metric" in issues:
        out.append(
            GapNote(
                code="comparative_claim_missing_metric",
                message="Comparative claim is not anchored to a metric or baseline.",
                section_hint="Numbers the paper actually gives us",
                required_evidence=["paper_experiments", "benchmarks"],
                query_hint=f"{bundle.primary.title} compare baseline metric",
            )
        )
    if "citation_count_below_min" in issues or "evaluative_paragraph_without_citation" in issues:
        out.append(
            GapNote(
                code="citation_support_gap",
                message="Draft needs stronger inline support for evaluative statements.",
                section_hint="body",
                required_evidence=["paper_method", "paper_experiments"],
                query_hint=f"{bundle.primary.title} citation support",
            )
        )
    for wanted in planning.mandatory_sections:
        wanted_clean = (wanted or "").strip()
        if not wanted_clean:
            continue
        normalized = wanted_clean.lower()
        # Only treat concise role-like section requirements as enforceable.
        # Long sentence-level planner outputs become noise rather than useful gaps.
        if len(wanted_clean) > 48 or ":" in wanted_clean:
            continue
        if normalized not in (body or "").lower():
            out.append(
                GapNote(
                    code="planning_section_gap",
                    message=f"Planned section missing: {wanted_clean}",
                    section_hint=wanted_clean,
                    required_evidence=[],
                    query_hint=f"{bundle.primary.title} {wanted_clean}",
                    blocking=False,
                )
            )
    # deterministic dedupe
    seen: set[tuple[str, str]] = set()
    deduped: list[GapNote] = []
    for note in out:
        key = (note.code, note.section_hint)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(note)
    return deduped


def _backfill_snippet(bundle: EvidenceBundle, note: GapNote) -> str:
    chunks: list[str] = []
    for key in note.required_evidence:
        text = (bundle.section_evidence.get(key) or "").strip()
        if text:
            chunks.append(text[:800])
    if note.code == "no_results_table" and bundle.benchmarks:
        rows = [
            f"- {b.name}: {b.value}{(' ' + b.unit) if b.unit else ''} vs {b.baseline or 'n/a'}"
            for b in bundle.benchmarks[:4]
        ]
        chunks.append("\n".join(rows))
    if not chunks and bundle.quotes:
        chunks.append("\n".join(f'- "{q.text[:220]}"' for q in bundle.quotes[:2]))
    return "\n\n".join(x for x in chunks if x).strip()[:2000]


def _json_obj(text: str) -> dict[str, Any]:
    m = re.search(r"\{[\s\S]*\}", text or "")
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def _fallback_planning_brief(bundle: EvidenceBundle) -> PlanningBrief:
    method = (bundle.section_evidence.get("paper_method") or "").strip()
    experiments = (bundle.section_evidence.get("paper_experiments") or "").strip()
    failures = list(bundle.contradiction_notes[:2])
    if not failures:
        failures = ["invented metrics", "generic structure"]
    claims: list[str] = []
    if experiments:
        claims.append(experiments[:220])
    if method:
        claims.append(method[:220])
    return PlanningBrief(
        mandatory_claims=claims[:3],
        mandatory_sections=["why this works", "what would falsify this", "when to use it"],
        required_visuals=["one mechanism diagram"],
        likely_failures=failures[:3],
        preventive_checks=["verify numeric claims against evidence", "confirm mermaid and table render cleanly"],
        backup_remedies=["drop unsupported metrics", "replace broken visuals with summary-only output"],
        reviewer_focus=["evidence grounding", "tradeoffs", "decision usefulness"],
    )


def _parse_planning_brief(raw: str, bundle: EvidenceBundle) -> PlanningBrief:
    d = _json_obj(raw)
    if not d:
        return _fallback_planning_brief(bundle)
    try:
        return PlanningBrief.model_validate(d)
    except Exception:
        return _fallback_planning_brief(bundle)


def _failure_memory_query(state: dict[str, Any]) -> str:
    primary = state.get("primary") or {}
    evidence = state.get("evidence") or {}
    draft_lint = state.get("draft_lint") or {}
    brief = state.get("brief") or {}
    chunks: list[str] = []
    if isinstance(primary, dict):
        chunks.extend(
            [
                str(primary.get("title") or ""),
                str(primary.get("abstract") or ""),
                " ".join(str(x) for x in (primary.get("tags") or [])[:8]),
            ]
        )
    if isinstance(evidence, dict):
        chunks.extend(str(x) for x in (evidence.get("contradiction_notes") or [])[:6])
    if isinstance(draft_lint, dict):
        chunks.extend(str(x) for x in (draft_lint.get("structural") or [])[:8])
        chunks.extend(str(x) for x in (draft_lint.get("unsupported_numeric_claims") or [])[:6])
    if isinstance(brief, dict):
        chunks.extend(
            [
                str(brief.get("format_name") or ""),
                str(brief.get("diagram_style") or ""),
                str(brief.get("opener_hook") or ""),
            ]
        )
    body = str(state.get("body") or "")
    if body:
        headings = re.findall(r"^##\s+(.+?)\s*$", body, re.M)
        chunks.extend(headings[:6])
    return "\n".join(x for x in chunks if x).strip()


def _failure_memory_snapshot(state: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    if state:
        return memory.retrieve_similar_json_items(
            "failure_memory.json",
            _failure_memory_query(state),
            limit=config.failure_memory_limit(),
        )
    data = memory.load_json("failure_memory.json", [])
    if not isinstance(data, list):
        return []
    return [x for x in data if isinstance(x, dict)][-config.failure_memory_limit():]


def _record_failure_memory(state: dict[str, Any]) -> None:
    qd = state.get("quality_report") or {}
    reasons = qd.get("blocking_reasons") or []
    if not isinstance(reasons, list) or not reasons:
        return
    codes: list[str] = []
    for reason in reasons:
        if not isinstance(reason, dict):
            continue
        code = str(reason.get("code") or "").strip()
        stage = str(reason.get("stage") or "").strip()
        if code:
            codes.append(f"{stage}:{code}" if stage else code)
    item = {
        "title": ((state.get("primary") or {}).get("title") if isinstance(state.get("primary"), dict) else "") or "",
        "overall_status": str(qd.get("overall_status") or ""),
        "codes": codes[:10],
        "query_text": _failure_memory_query(state),
        "signals": {
            "format_name": ((state.get("brief") or {}).get("format_name") if isinstance(state.get("brief"), dict) else "") or "",
            "diagram_style": ((state.get("brief") or {}).get("diagram_style") if isinstance(state.get("brief"), dict) else "") or "",
            "blocking_stages": sorted(
                {
                    str(reason.get("stage") or "").strip()
                    for reason in reasons
                    if isinstance(reason, dict) and str(reason.get("stage") or "").strip()
                }
            ),
        },
    }
    memory.append_json_list(
        "failure_memory.json",
        item,
        limit=config.failure_memory_limit(),
    )


def _parse_review_note(raw: str, *, role: str, fallback_findings: list[str]) -> ReviewNote:
    d = _json_obj(raw)
    if d:
        d.setdefault("role", role)
        try:
            return ReviewNote.model_validate(d)
        except Exception:
            pass
    return ReviewNote(
        role=role,
        pass_review=not bool(fallback_findings),
        findings=list(fallback_findings),
        rewrite_targets=[],
        summary=(fallback_findings[0] if fallback_findings else f"{role} found no blocking issues."),
    )


def _rewrite_full_body(body: str, bundle: EvidenceBundle, planning: PlanningBrief, note: ReviewNote) -> str:
    if config.dry_run() or not config.llm_configured() or is_llm_call_cap_reached():
        return body
    system = (
        "Revise the full markdown draft for a technical blog. Keep the same paper and headings where possible. "
        "Address the review findings directly. Preserve citations and mermaid/table structure. "
        "Add concrete tradeoffs, evidence-backed claims, and practitioner guidance where missing. "
        "Output markdown only."
    )
    user = (
        f"PLANNING_BRIEF:\n{planning.model_dump_json(indent=2)}\n\n"
        f"REVIEW_NOTE:\n{note.model_dump_json(indent=2)}\n\n"
        f"EVIDENCE_JSON:\n{bundle.model_dump_json()[:18000]}\n\n"
        f"DRAFT_MD:\n{body[:20000]}"
    )
    out = gllm.graph_llm_text(
        f"rewrite_full_{note.role}",
        system,
        user,
        mode="smart",
        max_tokens=config.max_tokens_smart(),
        task="draft_full",
    )
    if out.strip():
        return _unwrap_markdown_fence(out)
    return body


def _slugify(title: str) -> str:
    s = re.sub(r"[^\w\s-]", "", title.lower())
    s = re.sub(r"[-\s]+", "-", s).strip("-")
    return s[:80] or "post"


def _section_body_for_title(body: str, title: str) -> str:
    pat = re.compile(
        r"^##\s+" + re.escape(title) + r"\s*\n([\s\S]*?)(?=^##\s|\Z)",
        re.M,
    )
    m = pat.search(body)
    return (m.group(1) or "").strip() if m else ""


def _replace_section_body(full: str, title: str, new_inner: str) -> str:
    new_block = f"## {title}\n{new_inner.strip()}\n"
    pat = re.compile(
        r"^##\s+" + re.escape(title) + r"\s*\n[\s\S]*?(?=^##\s|\Z)",
        re.M,
    )
    if not pat.search(full):
        return full.rstrip() + "\n\n" + new_block
    return pat.sub(new_block.rstrip() + "\n", full, count=1)


def _discover_headings(body: str) -> list[str]:
    """Extract the H2 headings the writer actually chose, in order."""
    seen: list[str] = []
    for m in re.finditer(r"^##\s+(.+?)\s*$", body or "", re.M):
        title = m.group(1).strip()
        if title and title not in seen:
            seen.append(title)
    return seen


def _draft_looks_truncated_for_rescue(body: str) -> bool:
    """Heuristic: long prose that ends mid-sentence (no . ! ? on last line)."""
    t = (body or "").rstrip()
    if not t or len(t.split()) < 200:
        return False
    if t.count("```") % 2 == 1:
        return False
    last = t[-1]
    if last in (".", "!", "?", ":", '"', "'"):
        if last in ".!?\"'":
            return False
    lines = [ln for ln in t.splitlines() if ln.strip()]
    if not lines:
        return True
    last_line = lines[-1].rstrip()
    if re.search(r"[.!?][\"'»)]*\s*$", last_line):
        return False
    if last in "`" and t.endswith("```"):
        return False
    if last in ("#", ">"):
        return False
    return bool(last.isalnum() or last in (",", ";", "-"))


def node_curate(_state: dict[str, Any]) -> dict[str, Any]:
    from .. import curator  # noqa: PLC0415

    memory.ensure_dirs()
    curator.run()
    brief = json.loads((_ROOT / "reports" / "editorial_brief.json").read_text())
    return {
        "brief": brief,
        "_done_curate": True,
        "supervisor_decisions": sup.log_decision(_state, "curate"),
    }


def node_harvest(_state: dict[str, Any]) -> dict[str, Any]:
    from .. import harvest as hv  # noqa: PLC0415

    hv.run()
    return {
        "_done_harvest": True,
        "supervisor_decisions": sup.log_decision(_state, "harvest"),
    }


def node_rank(state: dict[str, Any]) -> dict[str, Any]:
    from .. import rank  # noqa: PLC0415

    rr = rank.run()
    d = json.loads(rr.model_dump_json())
    p = d.get("primary") or {}
    return {
        "rank_result": d,
        "primary": p,
        "_done_rank": True,
        "supervisor_decisions": sup.log_decision(state, "rank"),
    }


def node_research(state: dict[str, Any]) -> dict[str, Any]:
    """Legacy single-node research (``BLOGPIPE_COMMITTEE_DISABLED=1`` or CLI ``research``)."""
    from .. import research  # noqa: PLC0415

    with budget.stage("paper_reader"):
        research.run()
    print("stage=research (graph) done", flush=True)
    ev = json.loads((_ROOT / "reports" / "evidence_bundle.json").read_text())
    tr = (
        json.loads((_ROOT / "reports" / "research_trace.json").read_text())
        if (_ROOT / "reports" / "research_trace.json").is_file()
        else {}
    )
    return {
        "evidence": ev,
        "source_registry": _source_registry(EvidenceBundle.model_validate(ev)),
        "research_trace": tr,
        "_done_research": True,
        "supervisor_decisions": sup.log_decision(state, "research"),
    }


def _re_search_snippet(
    title: str, query_hint: str, bundle: EvidenceBundle
) -> str:
    from ..mcp_enrichment import _web_search  # noqa: PLC0415

    q = (query_hint or f"{title} research paper {bundle.primary.title}")[:200]
    items, _prov = _web_search(q)
    if not items:
        return ""
    return "\n".join(
        f"- {it.title} — {it.url}\n  {it.abstract[:220]}" for it in items[:3]
    )


def node_planning_brief(state: dict[str, Any]) -> dict[str, Any]:
    """Create an explicit pre-draft plan for required claims, sections, and likely failure modes."""
    bundle = evidence_model(state["evidence"])
    brief = brief_model(state["brief"])
    failure_memory = _failure_memory_snapshot(state)
    if config.dry_run() or not config.llm_configured() or is_llm_call_cap_reached():
        plan = _fallback_planning_brief(bundle)
    else:
        system = render_prompt("graph_planner")
        user = (
            "### EDITORIAL_BRIEF\n"
            f'"""\n{brief.model_dump_json(indent=2)}\n"""\n\n'
            "### EVIDENCE_JSON\n"
            f'"""\n{bundle.model_dump_json()[:18000]}\n"""\n\n'
            "### RECENT_FAILURE_MEMORY\n"
            f'"""\n{json.dumps(failure_memory, indent=2)[:4000]}\n"""'
        )
        raw = gllm.graph_llm_text(
            "planning_brief",
            system,
            user,
            mode="smart",
            max_tokens=1600,
            task="planning_brief",
            temperature=config.verifier_temperature(),
        )
        plan = _parse_planning_brief(raw, bundle)
    return {
        "planning_brief": plan.model_dump(),
        "_done_planning": True,
        "supervisor_decisions": sup.log_decision(state, "planning_brief"),
    }


def node_draft_refine(state: dict[str, Any]) -> dict[str, Any]:
    """One-shot draft via build_prompt, then per-section critic + optional re-search/rewrite."""
    with budget.stage("draft"):
        return _node_draft_refine_inner(state)


def _node_draft_refine_inner(state: dict[str, Any]) -> dict[str, Any]:
    memory.ensure_dirs()
    bundle = evidence_model(state["evidence"])
    brief = brief_model(state["brief"])
    planning = PlanningBrief.model_validate(state.get("planning_brief") or {})
    fmt = formats.FORMATS.get(brief.format_name) or formats.FORMATS["deep_dive"]
    rewrites = 0
    re_re = 0
    warnings: list[str] = []

    if is_llm_call_cap_reached():
        warnings.append("call_cap_at_draft_start")
        body = _stub_body(bundle, brief, fmt)
        return {
            "body": body,
            "warnings": list(state.get("warnings") or []) + warnings,
            "rewrites_applied": 0,
            "re_research_applied": 0,
            "budget_exhausted": True,
            "_done_draft": True,
        }

    if config.dry_run() or not config.llm_configured():
        body = _stub_body(bundle, brief, fmt)
        return {
            "body": body,
            "rewrites_applied": 0,
            "re_research_applied": 0,
            "_done_draft": True,
        }

    system, user = build_prompt(bundle, brief, fmt)
    if planning.model_fields_set:
        user += (
            "\n\nPLANNING_BRIEF (must satisfy this before stopping):\n"
            + planning.model_dump_json(indent=2)
        )
    body = gllm.graph_llm_text(
        "full_draft",
        system,
        user,
        max_tokens=config.max_tokens_smart(),
        task="draft_full",
    )
    if (body or "").strip():
        body = _unwrap_markdown_fence(body)
        if _draft_looks_truncated_for_rescue(body) and not is_llm_call_cap_reached():
            cont = gllm.graph_llm_text(
                "draft_continuation",
                "Continue from the end only; do not repeat prior lines; complete unfinished sections. "
                "Output markdown only, no frontmatter, no retake of the first-line takeaway.",
                f"So far (continue from the end, do not repeat):\n\n{body[-12000:]}",
                mode="fast",
                max_tokens=min(config.max_tokens_fast(), 4096),
                task="draft_full",
            )
            if (cont or "").strip():
                body = (body.rstrip() + "\n\n" + cont.strip()).strip()
    if not (body or "").strip():
        body = _stub_body(bundle, brief, fmt)
        warnings.append("full_draft_empty_used_stub")
    body = _resolve_cites(body, bundle)
    if _strip_unresolved(body):
        rep_txt = gllm.graph_llm_text(
            "cite_repair",
            "Replace [missing cite: ...] with paraphrase or remove the line. Return the full body only, "
            "no fences, no wrapper.",
            body[:12000],
            max_tokens=2048,
            task="draft_cite_repair",
        )
        if rep_txt.strip():
            body = _unwrap_markdown_fence(rep_txt)
        else:
            body = _cleanup_missing_cites(body)
    else:
        body = _cleanup_missing_cites(body)

    ev_excerpt = user
    if "EVIDENCE:" in user:
        ev_excerpt = user.split("EVIDENCE:", 1)[-1][:8000]

    headings = _discover_headings(body)
    critic_used = 0
    grounding_rewrites_used = 0
    _MAX_GROUNDING_REWRITES = 3
    _GROUNDING_VERDICTS = {
        "vague",
        "missing_number",
        "no_tradeoffs",
        "no_named_alternatives",
        "only_marketing_nouns",
        "opinion_unsupported",
        "no_limits",
    }
    for idx, title in enumerate(headings):
        if re_re >= _MAX_RESEARCH_WINGS and rewrites >= _MAX_REWRITES:
            break
        if is_llm_call_cap_reached():
            warnings.append("call_cap_during_refinement")
            break
        block = _section_body_for_title(body, title)
        if not block.strip():
            continue
        need, _reason = filler_detector_flag_section(title, block)
        want_critic = need or (idx < 3)
        if not want_critic:
            cr = {"verdict": "ok", "query_hint": ""}
        else:
            allow = critic_used < 4
            cr, used_llm = section_critic_llm(title, block, ev_excerpt, allow_llm=allow)
            if used_llm:
                critic_used += 1
        verdict = str(cr.get("verdict", "ok"))
        qhint = str(cr.get("query_hint", "") or "")
        if not need and verdict in ("ok", "vague"):
            if verdict == "vague" and rewrites < _MAX_REWRITES:
                need = True
        if not need and verdict == "ok":
            continue
        if verdict in ("ok",) and not need:
            continue
        if verdict == "empty_placeholder" and re_re < _MAX_RESEARCH_WINGS:
            k = f"refine_{title[:32]}"
            snip = _re_search_snippet(title, qhint, bundle)
            re_re += 1
            if snip:
                bundle.section_evidence[k] = (
                    (bundle.section_evidence.get(k) or "") + "\n" + snip
                )[-4000:]
            ex = bundle.model_dump_json(indent=0)
            extra = bundle.section_evidence.get(k, "") if snip else ""
            nblock = rewrite_section(
                title, block, (ex + "\n" + extra)[:20000], verdict
            )
            rewrites += 1
            if nblock.strip():
                body = _replace_section_body(body, title, nblock)
        elif (
            verdict in _GROUNDING_VERDICTS
            and grounding_rewrites_used < _MAX_GROUNDING_REWRITES
            and rewrites < _MAX_REWRITES
        ):
            with budget.stage("polish"):
                unused = unused_facts_for_section(title, body, bundle)
                facts_text = render_unused_facts(unused)
                nblock = rewrite_section_for_grounding(
                    title,
                    block,
                    bundle.model_dump_json()[:18000],
                    facts_text,
                    verdict,
                )
            grounding_rewrites_used += 1
            rewrites += 1
            if nblock.strip():
                body = _replace_section_body(body, title, nblock)
        elif rewrites < _MAX_REWRITES:
            nblock = rewrite_section(
                title, block, bundle.model_dump_json()[:20000], verdict
            )
            rewrites += 1
            if nblock.strip():
                body = _replace_section_body(body, title, nblock)

    body = _polish_body(body, bundle, brief)
    body = embed_planned_visuals(
        body, bundle.visual_plan, _slugify(bundle.primary.title)
    )
    struct = lint.structural_issues(body, bundle)
    unsupported = lint.unsupported_numeric_claims(body, bundle.model_dump_json())
    print(
        f"stage=draft_graph rewrites={rewrites} re_research={re_re} structural={len(struct)} unsupported={len(unsupported)}",
        flush=True,
    )
    return {
        "body": body,
        "evidence": json.loads(bundle.model_dump_json()),
        "draft_lint": {
            "structural": struct,
            "unsupported_numeric_claims": unsupported,
            "missing_planned_visuals": lint.missing_planned_visuals(
                body, bundle.visual_plan
            ),
            "stage": "draft_graph",
        },
        "rewrites_applied": rewrites,
        "re_research_applied": re_re,
        "warnings": list(state.get("warnings") or []) + warnings,
        "_done_draft": True,
    }


def node_gap_analyzer(state: dict[str, Any]) -> dict[str, Any]:
    """Deterministic draft-gap analysis before reviewer loops."""
    bundle = evidence_model(state["evidence"])
    body = state.get("body") or ""
    planning = PlanningBrief.model_validate(state.get("planning_brief") or {})
    gaps = _gap_notes(body, bundle, planning)
    registry = _source_registry(bundle)
    return {
        "gap_analysis": [x.model_dump() for x in gaps],
        "source_registry": registry,
        "_done_gap_analysis": True,
        "supervisor_decisions": sup.log_decision(state, "gap_analyzer"),
    }


def node_evidence_backfill(state: dict[str, Any]) -> dict[str, Any]:
    """Attach targeted evidence snippets for the gaps the analyzer found."""
    bundle = evidence_model(state["evidence"])
    gaps = [GapNote.model_validate(x) for x in (state.get("gap_analysis") or [])]
    changed = False
    for note in gaps[:6]:
        snippet = _backfill_snippet(bundle, note)
        if not snippet:
            continue
        key = f"gap_backfill:{note.code}"
        existing = (bundle.section_evidence.get(key) or "").strip()
        if snippet != existing:
            bundle.section_evidence[key] = snippet
            changed = True
    payload: dict[str, Any] = {
        "source_registry": state.get("source_registry") or _source_registry(bundle),
        "_done_backfill": True,
        "supervisor_decisions": sup.log_decision(state, "evidence_backfill"),
    }
    if changed:
        payload["evidence"] = json.loads(bundle.model_dump_json())
    return payload


def node_section_patcher(state: dict[str, Any]) -> dict[str, Any]:
    """Patch the draft body from gap analysis and audit outgoing citation links."""
    bundle = evidence_model(state["evidence"])
    body = state.get("body") or ""
    brief = brief_model(state["brief"])
    gaps = [GapNote.model_validate(x) for x in (state.get("gap_analysis") or [])]
    if gaps:
        body = _repair_or_inject_results_table(body, bundle)
        body = _ensure_mechanism_section(body, bundle)
        body = _ensure_decision_section(body, bundle)
        body = _polish_body(body, bundle, brief)
    registry = [
        x if isinstance(x, dict) else x.model_dump()
        for x in (state.get("source_registry") or _source_registry(bundle))
    ]
    from ..models import SourceRegistryEntry  # noqa: PLC0415

    model_registry = [SourceRegistryEntry.model_validate(x) for x in registry]
    body, audit = strip_unregistered_links(body, model_registry)
    return {
        "body": body,
        "citation_audit": audit.model_dump(),
        "source_registry": registry,
        "_done_section_patcher": True,
        "supervisor_decisions": sup.log_decision(state, "section_patcher"),
    }


def node_adversary_review(state: dict[str, Any]) -> dict[str, Any]:
    """Adversarial reviewer: stress-test tradeoffs, falsifiers, and advice sections."""
    body = state.get("body") or ""
    bundle = evidence_model(state["evidence"])
    planning = PlanningBrief.model_validate(state.get("planning_brief") or {})
    failure_memory = _failure_memory_snapshot(state)
    findings: list[str] = []
    lint_issues = lint.structural_issues(body, bundle)
    for key in ("no_tradeoffs_paragraph", "missing_decision_section", "advice_without_traceability"):
        if key in lint_issues:
            findings.append(key)
    if "What Would Falsify This" not in body and "falsify" not in body.lower():
        findings.append("missing_falsifier_language")
    if config.dry_run() or not config.llm_configured() or is_llm_call_cap_reached():
        note = _parse_review_note("", role="adversary", fallback_findings=findings)
    else:
        raw = gllm.graph_llm_text(
            "adversary_review",
            render_prompt("graph_adversary"),
            (
                "### PLANNING_BRIEF\n"
                f'"""\n{planning.model_dump_json(indent=2)}\n"""\n\n'
                "### RECENT_FAILURE_MEMORY\n"
                f'"""\n{json.dumps(failure_memory, indent=2)[:3000]}\n"""\n\n'
                "### DRAFT_MD\n"
                f'"""\n{body[:12000]}\n"""\n\n'
                "### EVIDENCE_JSON\n"
                f'"""\n{bundle.model_dump_json()[:12000]}\n"""'
            ),
            mode="smart",
            max_tokens=1200,
            task="draft_section_critic",
            temperature=config.verifier_temperature(),
        )
        note = _parse_review_note(raw, role="adversary", fallback_findings=findings)
    new_body = body
    if note.findings:
        new_body = _rewrite_full_body(body, bundle, planning, note)
    return {
        "body": new_body,
        "review_notes": [note.model_dump()],
        "_done_adversary": True,
        "supervisor_decisions": sup.log_decision(state, "adversary_review"),
    }


def node_evidence_verifier(state: dict[str, Any]) -> dict[str, Any]:
    """Verifier reviewer: catch unsupported claims and unresolved citations before editor gate."""
    body = state.get("body") or ""
    bundle = evidence_model(state["evidence"])
    planning = PlanningBrief.model_validate(state.get("planning_brief") or {})
    failure_memory = _failure_memory_snapshot(state)
    def _verifier_findings(text: str) -> list[str]:
        out = lint.unsupported_numeric_claims(text, bundle.model_dump_json())
        lint_issues_local = lint.structural_issues(text, bundle)
        if "unresolved_source_alias" in lint_issues_local:
            out.append("unresolved_source_alias")
        if "citation_count_below_min" in lint_issues_local:
            out.append("citation_count_below_min")
        return out

    findings = _verifier_findings(body)
    if findings:
        softened = soften_unsupported_numeric_claims(body, bundle.model_dump_json())
        if softened != body:
            body = softened
            findings = _verifier_findings(body)
    if config.dry_run() or not config.llm_configured() or is_llm_call_cap_reached():
        note = _parse_review_note("", role="evidence_verifier", fallback_findings=findings)
    else:
        raw = gllm.graph_llm_text(
            "evidence_verifier",
            render_prompt("graph_evidence_verifier"),
            "### RECENT_FAILURE_MEMORY\n"
            f'"""\n{json.dumps(failure_memory, indent=2)[:3000]}\n"""\n\n'
            "### DRAFT_MD\n"
            f'"""\n{body[:14000]}\n"""\n\n'
            "### EVIDENCE_JSON\n"
            f'"""\n{bundle.model_dump_json()[:16000]}\n"""',
            mode="smart",
            max_tokens=1200,
            task="editor_grounding",
            temperature=config.verifier_temperature(),
        )
        note = _parse_review_note(raw, role="evidence_verifier", fallback_findings=findings)
    new_body = body
    if note.findings:
        new_body = _rewrite_full_body(body, bundle, planning, note)
    return {
        "body": new_body,
        "review_notes": [note.model_dump()],
        "_done_verifier": True,
        "supervisor_decisions": sup.log_decision(state, "evidence_verifier"),
    }


def node_render_reviewer(state: dict[str, Any]) -> dict[str, Any]:
    """Render reviewer: validate HTML rendering semantics before package step."""
    from .. import package as package_mod  # noqa: PLC0415

    body = state.get("body") or ""
    bundle = evidence_model(state["evidence"])
    front = {"title": bundle.primary.title}
    findings: list[str] = []
    render_report = {}
    full_html, errors, warnings, flags = package_mod._render_full_html(front, body)
    if errors:
        findings.extend(errors)
    if not full_html:
        findings.append("render_html_generation_failed")
    render_report = {
        "html_valid": not bool(errors),
        "errors": list(errors),
        "warnings": list(warnings),
        **flags,
    }
    note = ReviewNote(
        role="render_reviewer",
        pass_review=not bool(findings),
        findings=list(dict.fromkeys(findings)),
        rewrite_targets=[],
        summary=(
            "Render review found no blocking issues."
            if not findings
            else "Render review found issues with HTML/diagram/table rendering."
        ),
    )
    return {
        "render_report": render_report,
        "review_notes": [note.model_dump()],
        "_done_render_review": True,
        "supervisor_decisions": sup.log_decision(state, "render_reviewer"),
    }


def node_meta_review(state: dict[str, Any]) -> dict[str, Any]:
    """Meta reviewer: synthesize reviewer notes into a final pre-editor summary."""
    notes = [ReviewNote.model_validate(x) for x in (state.get("review_notes") or [])]
    all_findings: list[str] = []
    blocking_findings: list[str] = []
    summaries: list[str] = []
    by_role = {note.role: note for note in notes}
    weights = config.reviewer_weights()
    for note in notes:
        all_findings.extend(note.findings)
        if note.summary:
            summaries.append(note.summary)
    all_findings = list(dict.fromkeys(all_findings))
    weighted_score = 0.0
    if config.reviewer_consensus_required():
        required_roles = {"evidence_verifier", "render_reviewer", "adversary"}
        missing_roles = sorted(required_roles - set(by_role))
        for role in missing_roles:
            blocking_findings.append(f"missing_required_reviewer:{role}")
        for role in ("evidence_verifier", "render_reviewer"):
            note = by_role.get(role)
            if note is not None and not note.pass_review:
                blocking_findings.append(f"reviewer_blocked:{role}")
                blocking_findings.extend(note.findings)
        total_weight = sum(weights.get(role, 1.0) for role in required_roles)
        passed_weight = sum(
            weights.get(role, 1.0)
            for role in required_roles
            if (note := by_role.get(role)) is not None and note.pass_review
        )
        weighted_score = (passed_weight / total_weight) if total_weight > 0 else 0.0
        threshold = config.reviewer_min_pass_score()
        if weighted_score < threshold:
            blocking_findings.append(
                f"reviewer_weight_below_threshold:{weighted_score:.2f}<{threshold:.2f}"
            )
    blocking_findings = list(dict.fromkeys(blocking_findings))
    meta = ReviewNote(
        role="meta_reviewer",
        pass_review=not bool(blocking_findings),
        reviewer_weight=1.0,
        findings=blocking_findings,
        rewrite_targets=[],
        summary=" | ".join(summaries[:3]) if summaries else "Meta review found no additional issues.",
        metadata={
            "all_findings": all_findings,
            "weighted_score": weighted_score,
            "reviewer_weights": weights,
        },
    )
    warnings = list(state.get("warnings") or [])
    if blocking_findings:
        warnings.append("meta_review_requires_editor_attention")
    return {
        "meta_review": meta.model_dump(),
        "review_notes": [meta.model_dump()],
        "warnings": warnings,
        "_done_meta_review": True,
        "supervisor_decisions": sup.log_decision(state, "meta_review"),
    }


def node_editor(state: dict[str, Any]) -> dict[str, Any]:
    """Rubric, grounding, lint gating, EditorReport fields."""
    body = state.get("body") or ""
    evidence_json = state.get("evidence") or {}
    ev_text = json.dumps(evidence_json)[:22000] if evidence_json else "{}"

    bundle_for_explainer: EvidenceBundle | None = None
    if isinstance(evidence_json, dict) and evidence_json:
        try:
            bundle_for_explainer = EvidenceBundle.model_validate(evidence_json)
        except Exception:
            bundle_for_explainer = None

    with budget.stage("polish"):
        if bundle_for_explainer is not None:
            body = explain_undefined_terms(body, bundle_for_explainer)
            body = embed_planned_visuals(
                body,
                bundle_for_explainer.visual_plan,
                _slugify(bundle_for_explainer.primary.title),
            )
            state["body"] = body
    final_lint_issues = list(
        dict.fromkeys(lint.structural_issues(body, bundle_for_explainer))
    )
    with budget.stage("editor"):
        rep = global_rubric(
            body, bundle=bundle_for_explainer, lint_issues=final_lint_issues
        )
        g_ok, g_iss, g_llm = grounding_check_node(body, ev_text)
    if bundle_for_explainer is not None:
        undefined_after = lint.undefined_acronyms(
            body, draft._glossary_terms_from_bundle(bundle_for_explainer)
        )
    else:
        undefined_after = lint.undefined_acronyms(body, [])
    citation_audit = CitationAuditReport.model_validate(state.get("citation_audit") or {})
    lint_issues = final_lint_issues
    det_ground = lint.unsupported_numeric_claims(body, ev_text)
    if citation_audit.invalid_links:
        det_ground.extend([f"unregistered_citation_link:{u}" for u in citation_audit.invalid_links[:6]])
    usage = get_llm_usage()
    need_llm = bool(config.llm_configured() and not config.dry_run())
    llm_ok = True
    ed_warn: list[str] = list(state.get("warnings") or [])
    review_notes = [ReviewNote.model_validate(x) for x in (state.get("review_notes") or [])]
    meta_findings: list[str] = []
    for note in review_notes:
        if note.role == "meta_reviewer":
            meta_findings.extend(note.findings)
    if need_llm:
        if int(usage.get("ok", 0) or 0) < 3:
            llm_ok = False
            ed_warn.append("llm_successes_below_threshold")
        if not g_llm:
            llm_ok = False
            ed_warn.append("grounding_llm_no_response")
    if state.get("budget_exhausted"):
        ed_warn.append("budget_exhausted_during_draft")
        llm_ok = False
    if llm_ok and "empty_placeholder_section" in lint_issues:
        rep = rep.model_copy(update={"pass_gate": False})
    if llm_ok and "pov_phrase_without_opinion" in lint_issues:
        rep = rep.model_copy(update={"pass_gate": False})
    if llm_ok and "generic_heading_used" in lint_issues:
        rep = rep.model_copy(update={"pass_gate": False})
    if llm_ok and "collective_research_voice" in lint_issues:
        rep = rep.model_copy(update={"pass_gate": False})
    if llm_ok and "templated_heading_used" in lint_issues:
        rep = rep.model_copy(update={"pass_gate": False})
    if llm_ok and "duplicate_heading" in lint_issues:
        rep = rep.model_copy(update={"pass_gate": False})
    if llm_ok and "takeaway_repeated_as_heading" in lint_issues:
        rep = rep.model_copy(update={"pass_gate": False})
    if llm_ok and "no_results_table" in lint_issues:
        rep = rep.model_copy(update={"pass_gate": False})
    if llm_ok and "fewer_than_required_h2_sections" in lint_issues:
        rep = rep.model_copy(update={"pass_gate": False})
    if llm_ok and "unfenced_mermaid_block" in lint_issues:
        rep = rep.model_copy(update={"pass_gate": False})
    if llm_ok and "citation_count_below_min" in lint_issues:
        rep = rep.model_copy(update={"pass_gate": False})
    if llm_ok and "redundant_adjacent_sections" in lint_issues:
        rep = rep.model_copy(update={"pass_gate": False})
    if llm_ok and "fake_results_table" in lint_issues:
        rep = rep.model_copy(update={"pass_gate": False})
    if llm_ok and "mermaid_is_taxonomy" in lint_issues:
        rep = rep.model_copy(update={"pass_gate": False})
    if llm_ok and "no_tradeoffs_paragraph" in lint_issues:
        rep = rep.model_copy(update={"pass_gate": False})
    if llm_ok and "unpaired_improvement_claims" in lint_issues:
        rep = rep.model_copy(update={"pass_gate": False})
    if meta_findings:
        rep = rep.model_copy(update={"pass_gate": False})
    if det_ground:
        rep = rep.model_copy(update={"pass_gate": False})
    if not llm_ok:
        rep = rep.model_copy(update={"pass_gate": False})
    rep = rep.model_copy(
        update={
            "lint_issues": lint_issues,
            "grounding_ok": bool(g_ok and not det_ground),
            "grounding_issues": list(dict.fromkeys(g_iss + det_ground)),
            "llm_ok": llm_ok,
            "editor_warnings": ed_warn,
        }
    )
    qrep = quality.from_editor(rep)
    d = rep.model_dump()
    d["undefined_acronyms"] = undefined_after
    d["missing_planned_visuals"] = (
        lint.missing_planned_visuals(
            body, bundle_for_explainer.visual_plan
        )
        if bundle_for_explainer is not None
        else {"missing_figures": [], "missing_equations": []}
    )
    d["pass_gate"] = qrep.pass_gate
    d["review_notes"] = [n.model_dump() for n in review_notes]
    d["citation_audit"] = citation_audit.model_dump()
    return {
        "editor_report": d,
        "llm_ok": llm_ok,
        "quality_report": qrep.model_dump(),
        "blocking_reasons": [x.model_dump() for x in qrep.blocking_reasons],
        "overall_status": qrep.overall_status,
        "pass_gate": bool(qrep.pass_gate),
        "warnings": ed_warn,
        "_done_editor": True,
    }


def node_review_gate(state: dict[str, Any]) -> dict[str, Any]:
    """HITL: pause for approval when the editor gate failed (requires checkpointer on graph)."""
    if bool(state.get("pass_gate")) or config.auto_approve_editor_gate():
        return {}
    decision = interrupt(
        {
            "reason": "pass_gate_false",
            "editor_report": state.get("editor_report"),
        }
    )
    if isinstance(decision, dict) and decision.get("approve"):
        return {"pass_gate": True}
    return {
        "warnings": list(state.get("warnings") or []) + ["human_rejected_publish"],
    }


def node_publish_and_write(state: dict[str, Any]) -> dict[str, Any]:
    """Write markdown, draft_path, reports, llm_usage.json, merge research trace."""
    memory.ensure_dirs()
    bundle = evidence_model(state["evidence"])
    body = (state.get("body") or "").strip() + "\n"
    ed = state.get("editor_report") or {}
    score = int(ed.get("rubric_score", 0))
    tr = state.get("research_trace") or {}
    u = get_llm_usage()
    u["llm_cap"] = config.llm_call_cap()
    u["llm_calls"] = int(u.get("ok", 0) or 0) + int(u.get("fail", 0) or 0)
    u["budget_exhausted"] = bool(
        is_llm_call_cap_reached() or state.get("budget_exhausted")
    )
    u["supervisor_decisions"] = state.get("supervisor_decisions", [])
    u["tokens_in"] = u.get("tokens_in", 0)
    u["tokens_out"] = u.get("tokens_out", 0)
    u["usd_spent"] = u.get("usd_spent", 0.0)
    u["by_task"] = u.get("by_task", {})
    u["by_model"] = u.get("by_model", {})
    if state.get("committee_synthesis"):
        u["committee_synthesis"] = state.get("committee_synthesis")
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    slug = _slugify(bundle.primary.title)
    title = _sanitize_frontmatter_text(bundle.primary.title, 200)
    takeaway = (
        _sanitize_frontmatter_text(body.split("\n", 1)[0].strip(), 300)
        if body.strip()
        else "Draft"
    )
    description = _sanitize_frontmatter_text(takeaway, 160)
    fm = f"""---
date: "{day}"
draft: true
title: "{title}"
description: "{description}"
categories: ["Machine Learning"]
tags: {json.dumps(bundle.primary.tags[:8] + ["blogpipe"])}
math: true
mermaid: true
one_sentence_takeaway: "{takeaway}"
image: /img/posts/{slug}/hero.png
rubric_score: {score}
---

"""
    full = fm + body.lstrip()
    if config.dry_run():
        out = _ROOT / "reports" / f"draft_preview_{day}-{slug}.md"
    else:
        out = _ROOT / "content" / "post" / f"{day}-{slug}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(full, encoding="utf-8")
    (_ROOT / "reports" / "draft_path.txt").write_text(
        str(out.relative_to(_ROOT)), encoding="utf-8"
    )
    if state.get("draft_lint"):
        (_ROOT / "reports" / "draft_lint.json").write_text(
            json.dumps(state["draft_lint"], indent=2),
            encoding="utf-8",
        )
    if state.get("planning_brief"):
        (_ROOT / "reports" / "planning_brief.json").write_text(
            json.dumps(state["planning_brief"], indent=2),
            encoding="utf-8",
        )
    if state.get("source_registry"):
        (_ROOT / "reports" / "source_registry.json").write_text(
            json.dumps(state["source_registry"], indent=2),
            encoding="utf-8",
        )
    if state.get("gap_analysis"):
        (_ROOT / "reports" / "gap_analysis.json").write_text(
            json.dumps(state["gap_analysis"], indent=2),
            encoding="utf-8",
        )
    if state.get("citation_audit"):
        (_ROOT / "reports" / "citation_audit.json").write_text(
            json.dumps(state["citation_audit"], indent=2),
            encoding="utf-8",
        )
    if state.get("meta_review"):
        (_ROOT / "reports" / "meta_review.json").write_text(
            json.dumps(state["meta_review"], indent=2),
            encoding="utf-8",
        )
    if state.get("review_notes"):
        (_ROOT / "reports" / "review_notes.json").write_text(
            json.dumps(state["review_notes"], indent=2),
            encoding="utf-8",
        )
    ed_path = _ROOT / "reports" / "editor_report.json"
    ed_path.write_text(json.dumps(ed, indent=2), encoding="utf-8")
    qd = state.get("quality_report") or {}
    if qd:
        (_ROOT / "reports" / "quality_report.json").write_text(
            json.dumps(qd, indent=2),
            encoding="utf-8",
        )
    (_ROOT / "reports" / "llm_usage.json").write_text(
        json.dumps(u, indent=2), encoding="utf-8"
    )
    tr2 = dict(tr) if tr else {}
    tr2["rewrites_applied"] = state.get("rewrites_applied", 0)
    tr2["re_research_applied"] = state.get("re_research_applied", 0)
    (_ROOT / "reports" / "research_trace.json").write_text(
        json.dumps(tr2, indent=2), encoding="utf-8"
    )
    u_total = u["llm_calls"]
    print(
        f"stage=edit calls_ok={u.get('ok', 0)}/{max(1, u_total)} pass_gate={ed.get('pass_gate')} "
        f"llm_ok={ed.get('llm_ok', True)}",
        flush=True,
    )
    learned: dict[str, list[str]] = {}
    try:
        with budget.stage("extras"):
            learned = topics.update_themes_from_draft(body, bundle.primary)
    except Exception as e:  # never block publish on the keyword learner
        LOG.warning("topics: keyword learning skipped: %s", e)
    if learned:
        (_ROOT / "reports" / "learned_keywords.json").write_text(
            json.dumps(learned, indent=2, sort_keys=True), encoding="utf-8"
        )
    try:
        _record_failure_memory(state)
    except Exception as e:  # noqa: BLE001
        LOG.warning("failure memory update skipped: %s", e)
    return {
        "slug": slug,
        "out_path": str(out),
        "learned_keywords": learned,
        "supervisor_decisions": sup.log_decision(state, "write_artifacts"),
    }
