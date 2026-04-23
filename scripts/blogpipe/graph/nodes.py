"""Graph node functions: thin wrappers and draft refine loop."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

from langgraph.types import interrupt

from .. import config, draft, formats, lint, memory
from ..draft import (
    _cleanup_missing_cites,
    _polish_body,
    _resolve_cites,
    _sanitize_frontmatter_text,
    _slugify,
    _strip_unresolved,
    _stub_body,
    _unwrap_markdown_fence,
    build_prompt,
    embed_planned_visuals,
    explain_undefined_terms,
)
from ..llm_chain import get_llm_usage, is_llm_call_cap_reached
from ..memory import _ROOT
from ..models import EvidenceBundle
from . import llm as gllm
from . import supervisor as sup
from .critics import (
    filler_detector_flag_section,
    global_rubric,
    grounding_check_node,
    rewrite_section,
    section_critic_llm,
)
from .state import evidence_model, brief_model

LOG = logging.getLogger(__name__)

_MAX_RESEARCH_WINGS = 2
_MAX_REWRITES = 2


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


def node_draft_refine(state: dict[str, Any]) -> dict[str, Any]:
    """One-shot draft via build_prompt, then per-section critic + optional re-search/rewrite."""
    memory.ensure_dirs()
    bundle = evidence_model(state["evidence"])
    brief = brief_model(state["brief"])
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
    body = gllm.graph_llm_text(
        "full_draft",
        system,
        user,
        max_tokens=config.max_tokens_smart(),
        task="draft_full",
    )
    if not (body or "").strip():
        body = _stub_body(bundle, brief, fmt)
        warnings.append("full_draft_empty_used_stub")
    body = _unwrap_markdown_fence(body)
    body = _resolve_cites(body, bundle)
    if _strip_unresolved(body):
        rep_txt = gllm.graph_llm_text(
            "cite_repair",
            "Fix [missing cite: ...] with paraphrase or remove. Output markdown body only, no fences.",
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
    for title in headings:
        if re_re >= _MAX_RESEARCH_WINGS and rewrites >= _MAX_REWRITES:
            break
        if is_llm_call_cap_reached():
            warnings.append("call_cap_during_refinement")
            break
        block = _section_body_for_title(body, title)
        if not block.strip():
            continue
        need, _reason = filler_detector_flag_section(title, block)
        cr = section_critic_llm(title, block, ev_excerpt)
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
        elif rewrites < _MAX_REWRITES:
            nblock = rewrite_section(
                title, block, bundle.model_dump_json()[:20000], verdict
            )
            rewrites += 1
            if nblock.strip():
                body = _replace_section_body(body, title, nblock)

    body = _polish_body(body, bundle, brief)
    body = explain_undefined_terms(body, bundle)
    body = embed_planned_visuals(
        body, bundle.visual_plan, _slugify(bundle.primary.title)
    )
    struct = lint.structural_issues(body)
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


def node_editor(state: dict[str, Any]) -> dict[str, Any]:
    """Rubric, grounding, lint gating, EditorReport fields."""
    body = state.get("body") or ""
    evidence_json = state.get("evidence") or {}
    ev_text = json.dumps(evidence_json)[:22000] if evidence_json else "{}"

    rep = global_rubric(body)
    g_ok, g_iss, g_llm = grounding_check_node(body, ev_text)
    bundle_for_explainer: EvidenceBundle | None = None
    if isinstance(evidence_json, dict) and evidence_json:
        try:
            bundle_for_explainer = EvidenceBundle.model_validate(evidence_json)
        except Exception:
            bundle_for_explainer = None
    if bundle_for_explainer is not None:
        body = explain_undefined_terms(body, bundle_for_explainer)
        body = embed_planned_visuals(
            body,
            bundle_for_explainer.visual_plan,
            _slugify(bundle_for_explainer.primary.title),
        )
        state["body"] = body
        undefined_after = lint.undefined_acronyms(
            body, draft._glossary_terms_from_bundle(bundle_for_explainer)
        )
    else:
        undefined_after = lint.undefined_acronyms(body, [])
    lint_issues = list(dict.fromkeys(lint.structural_issues(body)))
    det_ground = lint.unsupported_numeric_claims(body, ev_text)
    usage = get_llm_usage()
    need_llm = bool(config.llm_configured() and not config.dry_run())
    llm_ok = True
    ed_warn: list[str] = list(state.get("warnings") or [])
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
    if det_ground:
        rep = rep.model_copy(update={"pass_gate": False})
    if not llm_ok:
        rep = rep.model_copy(update={"pass_gate": False})
    d = rep.model_dump()
    d["lint_issues"] = lint_issues
    d["grounding_ok"] = bool(g_ok and not det_ground)
    d["grounding_issues"] = list(dict.fromkeys(g_iss + det_ground))
    d["undefined_acronyms"] = undefined_after
    d["missing_planned_visuals"] = (
        lint.missing_planned_visuals(
            body, bundle_for_explainer.visual_plan
        )
        if bundle_for_explainer is not None
        else {"missing_figures": [], "missing_equations": []}
    )
    d["llm_ok"] = llm_ok
    d["editor_warnings"] = ed_warn
    return {
        "editor_report": d,
        "llm_ok": llm_ok,
        "pass_gate": bool(d.get("pass_gate", False)),
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
        _sanitize_frontmatter_text(body.split("\n", 1)[0].strip(), 500)
        if body.strip()
        else "Draft"
    )
    fm = f"""---
date: "{day}"
draft: true
title: "{title}"
description: "{takeaway[:160]}"
categories: ["Machine Learning"]
tags: {json.dumps(bundle.primary.tags[:8] + ["blogpipe"])}
math: true
mermaid: true
one_sentence_takeaway: "{takeaway[:300]}"
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
    ed_path = _ROOT / "reports" / "editor_report.json"
    ed_path.write_text(json.dumps(ed, indent=2), encoding="utf-8")
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
    return {
        "slug": slug,
        "out_path": str(out),
        "supervisor_decisions": sup.log_decision(state, "write_artifacts"),
    }
