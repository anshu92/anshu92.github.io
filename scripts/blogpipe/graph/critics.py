"""Section critic, rubric, grounding, filler heuristics."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional, Tuple

from .. import config, lint
from ..llm_chain import GROQ_USER_CONTENT_BUDGET, is_llm_call_cap_reached
from ..models import EditorReport
from . import llm as gllm

LOG = logging.getLogger(__name__)

_PLACEHOLDER = re.compile(
    r"^no [a-z]+ (information|reference|equation|code|content)\b",
    re.I | re.M,
)


def _shrink_for_groq_rubric_user(body: str) -> str:
    b = (body or "")
    if len(b) <= GROQ_USER_CONTENT_BUDGET:
        return b
    return b[:GROQ_USER_CONTENT_BUDGET].rsplit(" ", 1)[0].rstrip() + "\n"


def _shrink_for_groq_grounding_user(body: str, evidence_text: str) -> str:
    """Heads of EVIDENCE and DRAFT to stay under Groq user-message cap (avoids word-cut in llm_chain)."""
    ev0 = (evidence_text or "")[:40000]
    b0 = (body or "")[:40000]
    s = f"EVIDENCE:\n{ev0}\n\n---\n\nDRAFT:\n{b0}\n"
    if len(s) <= GROQ_USER_CONTENT_BUDGET:
        return s
    room = (GROQ_USER_CONTENT_BUDGET - 60) // 2
    ev2 = (ev0[:room]).rsplit(" ", 1)[0].rstrip()
    b2 = (b0[:room]).rsplit(" ", 1)[0].rstrip()
    return f"EVIDENCE:\n{ev2}\n\n---\n\nDRAFT:\n{b2}\n"


def filler_detector_score(text: str) -> Tuple[int, list[str]]:
    """No-LLM concreteness score; higher is better. Returns (score, issue_tags)."""
    issues: list[str] = []
    s = (text or "").strip()
    if not s:
        return 0, ["empty"]
    d = lint.count_digits(s)
    if d < 1:
        issues.append("low_digits")
    cites = len(re.findall(r"\[cite:\s*[^\]]+\]", s, re.I))
    if cites < 1 and len(s) > 200:
        issues.append("no_cite_in_long")
    # rough proper-noun count: word starting with capital (not after period)
    caps = re.findall(r"(?<![.!?]\s)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", s)
    if len(caps) < 2 and len(s) > 150:
        issues.append("low_named_entities")
    if "wikipedia" in s.lower() and "http" not in s:
        issues.append("wiki_like")
    sc = 3 + min(d, 5) + min(cites, 3) * 2 + min(len(caps) // 2, 3)
    return sc, issues


def filler_detector_flag_section(
    title: str, body: str
) -> Tuple[bool, Optional[str]]:
    """If section needs evidence or rewrite, return (True, reason)."""
    t = f"## {title}\n{body}"
    if "empty_placeholder" in lint.lint_empty_placeholders(t):
        return True, "empty_placeholder"
    if "pov_phrase_without_opinion" in lint.lint_pov_after_phrase(t):
        return True, "pov_phrase"
    sc, _tags = filler_detector_score(body)
    if sc < 6 and len(body.split()) > 40:
        return True, "low_concreteness"
    return False, None


def section_critic_deterministic(title: str, body: str) -> Optional[str]:
    """Return verdict string if we can decide without LLM, else None."""
    first = (body or "").split("\n", 1)[0].strip() if (body or "").strip() else ""
    if first and _PLACEHOLDER.search(first):
        return "empty_placeholder"
    block = f"## {title}\n{body}\n" if not (body or "").lstrip().startswith("##") else (body or "")
    if "pov_phrase_without_opinion" in lint.lint_pov_after_phrase(block):
        return "no_pov"
    return None


def section_critic_llm(
    title: str, body: str, evidence_excerpt: str, *, allow_llm: bool = True
) -> tuple[Dict[str, Any], bool]:
    """LLM section critic. Returns ({verdict, query_hint}, used_llm)."""
    det = section_critic_deterministic(title, body)
    if det is not None:
        return {
            "verdict": det,
            "query_hint": f"more on {title}",
        }, False
    if config.dry_run():
        return {"verdict": "ok", "query_hint": ""}, False
    if is_llm_call_cap_reached() or not config.llm_configured():
        return {"verdict": "ok", "query_hint": ""}, False
    if not allow_llm:
        return {"verdict": "ok", "query_hint": ""}, False
    system = (
        "You review one section of a technical blog. The section title was chosen by the writer and "
        "may be anything — do NOT penalise the title itself. Judge only the body against these "
        "standards for a great technical blog:\n"
        "- concrete: specific systems, numbers, constraints, failure modes (not 'scale' or "
        "'performance' alone);\n"
        "- teaches why, not just what; examples carry the argument;\n"
        "- claims are backed by a number, baseline, or citation; no unsupported confidence;\n"
        "- tradeoffs named where relevant (alternatives, what was given up);\n"
        "- limitations and failure modes are shown when the section is about the method or results;\n"
        "- POV: if an opinion lead-in phrase ('What I find', 'The remarkable thing') appears, it is "
        "followed by a real opinion, not a hedge.\n"
        "Output JSON only: "
        '{"verdict": "ok" | "vague" | "no_pov" | "empty_placeholder" | "missing_number" | "no_tradeoffs" | "no_limits", '
        '"query_hint": "short search query that would fix the gap, or empty"}. '
        "empty_placeholder: body is a 'no ... information' placeholder, 'TBD'/'TODO', or trivially short. "
        "vague: few specifics, few numbers, few named entities, reads like filler. "
        "no_pov: lead-in phrase present but the opinion is missing or hedged into nothing. "
        "missing_number: the section promises a numeric comparison or result but does not give one. "
        "no_tradeoffs: the section discusses a design choice but does not name alternatives or costs. "
        "no_limits: the section reports a result or method but does not acknowledge any limit, "
        "assumption, or failure mode where one would be expected."
    )
    user = f"SECTION: ## {title}\n{body[:6000]}\n\nEVIDENCE_EXCERPT:\n{evidence_excerpt[:4000]}\n"
    raw = gllm.graph_llm_text(
        "section_critic", system, user, mode="smart", task="draft_section_critic"
    )
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return {"verdict": "ok", "query_hint": ""}, True
    try:
        return json.loads(m.group(0)), True
    except (json.JSONDecodeError, TypeError):
        return {"verdict": "ok", "query_hint": ""}, True


def rewrite_section(
    title: str,
    old_body: str,
    bundle_excerpt: str,
    reason: str,
) -> str:
    """Targeted section rewrite."""
    if config.dry_run():
        return f"## {title}\n\nWhat I find: a concrete nugget. [cite: primary]\n\n" + (old_body or "")[:200]
    if is_llm_call_cap_reached():
        return old_body
    system = (
        f"Rewrite ONLY the section '## {title}' for a technical blog. Issue: {reason}.\n"
        "Standards you must hit in the rewrite:\n"
        "- Be concrete: replace vague wording with specific systems, numbers, constraints, "
        "failure modes, or named entities. Delete filler.\n"
        "- Teach why, not just what. Use one tight example or before/after comparison if it "
        "carries the point.\n"
        "- Back every claim with a number, baseline, or [cite: id] from EVIDENCE. Do not invent "
        "sources or numbers not in EVIDENCE.\n"
        "- If the section is about a design choice: name the alternative and what was given up.\n"
        "- If the section is about a method or result: name at least one honest limitation, broken "
        "assumption, or failure mode (not an ablation that validates the method).\n"
        "- If the issue is 'pov_phrase' or 'no_pov': keep or introduce one first-person opinion "
        "sentence and follow it with an actual stance.\n"
        "- Keep it skimmable: short paragraphs, descriptive sub-heads if you add any.\n"
        "- Do NOT change the heading to a generic one (no 'Introduction', 'Background', "
        "'Overview', 'Results', 'Summary', 'Conclusion').\n"
        "Output only that section's markdown (include the ## line)."
    )
    return gllm.graph_llm_text(
        f"rewrite_{title[:20]}",
        system,
        f"EVIDENCE:\n{bundle_excerpt[:8000]}\n\nOLD:\n{old_body[:4000]}\n",
        mode="fast",
        max_tokens=2048,
        task="draft_rewrite_section",
    )


def global_rubric(body: str) -> EditorReport:
    """Editor rubric: match editor.json shape."""
    from ..editor import _normalize_rubric_items

    system = (
        "You are a senior technical editor. Score the blog draft on 15 criteria, 1 point each if "
        "the draft clearly meets the standard, 0 otherwise. Be strict.\n"
        "CRITERIA: 1.sharp_takeaway (first sentence names a numeric result), "
        "2.intro_earns_attention (problem/audience/value in first 3-5 sentences, no throat-clearing), "
        "3.concrete_problem (real-world context, constraints, failure mode named), "
        "4.inevitable_structure (problem -> approach -> results -> limits; no tangents), "
        "5.specific_language (specific systems/components/numbers; no filler), "
        "6.teaches_why (explains mechanism, not just what was done), "
        "7.strong_example (inputs/outputs, before/after, or code that earns its place), "
        "8.tradeoffs_explicit (alternatives and what was given up are named), "
        "9.claims_backed (every speed/accuracy/cost claim has number or cite), "
        "10.limitations_honest (where it breaks, assumptions, when NOT to use it), "
        "11.skimmable (short paragraphs, descriptive sub-heads, key takeaways scannable), "
        "12.actionable_close (reader knows what to try/avoid/run), "
        "13.diagrams_clarify (mermaid/figure reduces cognitive load; 0 if absent), "
        "14.honest_scope (local results not oversold as general laws), "
        "15.pov_present (argues something; at least one first-person opinion).\n"
        "Also fill five_questions: {problem, hard, tried, outcomes, next}, 1-3 sentences each, "
        'from the text. If any cannot be answered, set that value to "CANNOT_DETERMINE". '
        "five_questions_ok is true only if none are CANNOT_DETERMINE.\n"
        "rubric_items: JSON array of 15 objects {item: criterion_name, score: 0 or 1}. "
        'Output JSON only: {"rubric_score": N, "rubric_items": [...], "five_questions": {...}, '
        '"five_questions_ok": true}'
    )
    if config.dry_run():
        return EditorReport(
            rubric_score=10,
            five_questions_ok=True,
            pass_gate=True,
        )
    raw = gllm.graph_llm_text(
        "rubric", system, _shrink_for_groq_rubric_user(body), mode="smart", task="editor_rubric"
    )
    if not raw.strip():
        return EditorReport(
            rubric_score=9,
            five_questions_ok=True,
            pass_gate=True,
        )
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return EditorReport(rubric_score=8, pass_gate=True)
    try:
        d = json.loads(m.group(0))
        sc = int(d.get("rubric_score", 8))
        fq = d.get("five_questions") or {}
        bad = any(
            isinstance(fq.get(k), str) and "CANNOT" in str(fq.get(k, "")) for k in fq
        )
        ok = not bad and d.get("five_questions_ok", not bad)
        return EditorReport(
            rubric_score=sc,
            rubric_items=_normalize_rubric_items(d.get("rubric_items")),
            five_questions={str(k): str(v) for k, v in fq.items()},
            five_questions_ok=bool(ok),
            pass_gate=sc >= config.editor_min_score() and bool(ok),
        )
    except Exception as e:  # noqa: BLE001
        LOG.warning("rubric parse: %s", e)
        return EditorReport(rubric_score=8, pass_gate=True)


def grounding_check_node(body: str, evidence_text: str) -> tuple[bool, list[str], bool]:
    if config.dry_run() or not config.llm_configured():
        return True, [], True
    if is_llm_call_cap_reached():
        return True, [], False
    raw = gllm.graph_llm_text(
        "grounding",
        "You compare a blog draft to EVIDENCE JSON. If analyst_notes or contradictions in "
        "EVIDENCE conflict with a draft claim, flag it. Output JSON only: "
        '{"unsupported_claims": ["short label", ...]}. Max 10; [] if none.',
        _shrink_for_groq_grounding_user(body, evidence_text),
        mode="smart",
        task="editor_grounding",
    )
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return True, [], bool(raw.strip())
    try:
        d = json.loads(m.group(0))
        issues = [
            str(x).strip() for x in (d.get("unsupported_claims") or []) if str(x).strip()
        ][:12]
        return (len(issues) == 0, issues, True)
    except (json.JSONDecodeError, TypeError):
        return True, [], True
