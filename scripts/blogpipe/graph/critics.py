"""Section critic, rubric, grounding, filler heuristics."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional, Tuple

from .. import config, lint
from ..llm_chain import GROQ_USER_CONTENT_BUDGET, is_llm_call_cap_reached
from ..models import EditorReport
from ..rubric_prompt import RUBRIC_SYSTEM, shrink_rubric_user
from . import llm as gllm

LOG = logging.getLogger(__name__)

_PLACEHOLDER = re.compile(
    r"^no [a-z]+ (information|reference|equation|code|content)\b",
    re.I | re.M,
)


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
        "You review one section of a technical blog. Ignore the specific section title; judge only the "
        "body. Good sections are concrete, teach why with examples, back claims, name tradeoffs and "
        "limits when relevant, and follow POV lead-ins with a real stance (not a hedge).\n"
        "Verdicts: empty_placeholder = placeholder line, TBD, or trivially short. vague = few "
        "numbers/entities, filler. no_pov = lead-in with no real opinion. missing_number = promised "
        "numeric result missing. no_tradeoffs = design choice with no named alternative/cost. "
        "no_limits = method/result with no expected limit. only_marketing_nouns = only abstract nouns, "
        "no measurable mechanism. opinion_unsupported = I/we opinion not tied to a fact in the section. "
        "no_named_alternatives = tradeoff language but no EVIDENCE-named alternative, only 'traditional "
        "approaches' style vague refs.\n"
        "Use a fair prompting standard: if the evidence is mixed, prefer the narrowest critique over a broad one.\n"
        'Output JSON only. Example: {"verdict": "ok", "query_hint": ""}\n'
        'Schema: {"verdict": "ok" | "vague" | "no_pov" | "empty_placeholder" | "missing_number" | '
        '"no_tradeoffs" | "no_limits" | "only_marketing_nouns" | "opinion_unsupported" | '
        '"no_named_alternatives", "query_hint": "short search query, or empty"}'
    )
    user = (
        "### SECTION\n"
        f'"""\n## {title}\n{body[:6000]}\n"""\n\n'
        "### EVIDENCE_EXCERPT\n"
        f'"""\n{evidence_excerpt[:4000]}\n"""'
    )
    raw = gllm.graph_llm_text(
        "section_critic", system, user, mode="smart", task="draft_section_critic"
        , temperature=config.verifier_temperature()
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
        "Concrete, skimmable paragraphs; [cite: id] and numbers from EVIDENCE only. Design section: "
        "name the alternative and cost. Method/result section: one real limitation (not a validating "
        "ablation). For no_pov: one first-person sentence with a stance. Keep a specific ## title, "
        "not Introduction/Results/Conclusion. Output that section’s markdown with the ## line only."
    )
    return gllm.graph_llm_text(
        f"rewrite_{title[:20]}",
        system,
        f"EVIDENCE:\n{bundle_excerpt[:8000]}\n\nOLD:\n{old_body[:4000]}\n",
        mode="fast",
        max_tokens=2048,
        task="draft_rewrite_section",
    )


def rewrite_section_for_grounding(
    title: str,
    old_body: str,
    bundle_excerpt: str,
    unused_facts_text: str,
    reason: str,
) -> str:
    """Targeted rewrite that injects unused facts (numbers, named entities, claims)."""
    if config.dry_run():
        return old_body
    if is_llm_call_cap_reached():
        return old_body
    system = (
        f"Rewrite ONLY the section '## {title}' for a senior-engineer blog. Critic: {reason}. "
        "The draft was too abstract. Meet all of: (1) a number from UNUSED NUMBERS with a NAMED "
        "baseline or alternative (no 'prior work' vagueness), (2) one UNUSED NAMED ENTITIES line "
        "compared to the primary method on one axis (cost, accuracy, etc.), (3) any first-person line "
        "tied to UNUSED ANALYST CLAIMS or quotes with [cite: id], (4) a named 'road not taken' with a "
        "contrastive marker, (5) a falsification-style limitation (cheapest test that would break the "
        "main claim). Keep existing concrete text. EVIDENCE and UNUSED only—no new facts. "
        "Output the section including '## ' line only."
    )
    user = (
        f"EVIDENCE:\n{(bundle_excerpt or '')[:8000]}\n\n"
        f"UNUSED FACTS (pull from these; do not invent):\n{unused_facts_text or '(none)'}\n\n"
        f"OLD SECTION:\n{(old_body or '')[:4000]}\n"
    )
    return gllm.graph_llm_text(
        f"rewrite_grounded_{title[:20]}",
        system,
        user,
        mode="fast",
        max_tokens=2048,
        task="draft_rewrite_section",
    )


_FLOOR_LINT_KEYS = {
    "fake_results_table",
    "mermaid_is_taxonomy",
    "redundant_adjacent_sections",
    "no_tradeoffs_paragraph",
    "unpaired_improvement_claims",
}


def _apply_rubric_floor(
    score: int,
    body: str,
    bundle,
    lint_issues: list[str] | None,
) -> tuple[int, list[str], dict | None]:
    """Cap `score` based on deterministic measurements. Returns (new_score, reasons, util_report)."""
    reasons: list[str] = []
    floored = score
    text = body or ""

    def _apply(name: str, cap: int, extra: str = "") -> None:
        nonlocal floored
        # Always record the reason when a condition fires, even if a previous
        # cap already dropped the score below this cap (so the editor report
        # surfaces every contributing factor).
        suffix = f":{extra}" if extra else ""
        reasons.append(f"{name}{suffix}:cap={cap}")
        if floored > cap:
            floored = cap

    has_cite_marker = bool(re.search(r"\[cite:\s*[^\]]+\]", text, re.I))
    has_md_link = bool(re.search(r"\]\(https?://[^)]+\)", text))
    if not has_cite_marker and not has_md_link:
        _apply("no_cites", config.rubric_floor_no_cites_max())
    if not lint.has_numeric_claim(text):
        _apply("no_numbers", config.rubric_floor_no_numbers_max())
    fired = sorted(set(lint_issues or []) & _FLOOR_LINT_KEYS)
    if fired:
        _apply("lints", config.rubric_floor_lint_max(), ",".join(fired))
    util_report = None
    if bundle is not None:
        try:
            from ..evidence_coverage import coverage_report

            util_report = coverage_report(text, bundle)
        except Exception:  # noqa: BLE001
            util_report = None
        if util_report is not None:
            thr = config.rubric_floor_low_util_threshold()
            if util_report.get("score", 1.0) < thr:
                _apply(
                    "low_util",
                    config.rubric_floor_low_util_max(),
                    f"score={util_report.get('score')}:thr={thr}",
                )
    return floored, reasons, util_report


def global_rubric(
    body: str,
    *,
    bundle=None,
    lint_issues: list[str] | None = None,
) -> EditorReport:
    """Editor rubric: match editor.json shape, with deterministic floor."""
    from ..editor import _normalize_rubric_items

    def _finalize(
        raw_score: int,
        rubric_items: list,
        fq_dict: dict,
        ok: bool,
    ) -> EditorReport:
        floored, reasons, util = _apply_rubric_floor(
            raw_score, body, bundle, lint_issues
        )
        floor_applied = floored != raw_score
        return EditorReport(
            rubric_score=floored,
            rubric_score_raw=raw_score,
            rubric_items=rubric_items,
            five_questions={str(k): str(v) for k, v in (fq_dict or {}).items()},
            five_questions_ok=bool(ok),
            pass_gate=floored >= config.editor_min_score() and bool(ok),
            rubric_floor_applied=floor_applied,
            rubric_floor_reasons=reasons,
            evidence_utilization=util or {},
        )

    if config.dry_run():
        return _finalize(10, [], {}, True)
    raw = gllm.graph_llm_text(
        "rubric", RUBRIC_SYSTEM, shrink_rubric_user(body), mode="smart", task="editor_rubric"
    )
    if not raw.strip():
        return _finalize(9, [], {}, True)
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return _finalize(8, [], {}, True)
    try:
        d = json.loads(m.group(0))
        sc = int(d.get("rubric_score", 8))
        fq = d.get("five_questions") or {}
        bad = any(
            isinstance(fq.get(k), str) and "CANNOT" in str(fq.get(k, "")) for k in fq
        )
        ok = not bad and d.get("five_questions_ok", not bad)
        return _finalize(
            sc, _normalize_rubric_items(d.get("rubric_items")), fq, bool(ok)
        )
    except Exception as e:  # noqa: BLE001
        LOG.warning("rubric parse: %s", e)
        return _finalize(8, [], {}, True)


def grounding_check_node(body: str, evidence_text: str) -> tuple[bool, list[str], bool]:
    if config.dry_run() or not config.llm_configured():
        return True, [], True
    if is_llm_call_cap_reached():
        return True, [], False
    raw = gllm.graph_llm_text(
        "grounding",
        "You compare a blog draft to EVIDENCE JSON.\n"
        "Flag only unsupported factual claims that materially conflict with, overstate, or invent evidence.\n"
        "Do NOT flag:\n"
        "- author synthesis that is clearly labeled as judgment,\n"
        "- reproducibility suggestions or implementation sketches that are not presented as paper-reported facts,\n"
        "- derived arithmetic restatements when the underlying benchmark numbers are cited and the wording does not exaggerate the claim.\n"
        "Do flag:\n"
        "- invented metrics or counts,\n"
        "- unsupported comparative claims,\n"
        "- contradictions with analyst_notes or contradiction notes,\n"
        "- setup details stated as factual but absent from evidence.\n"
        'Output JSON only with this schema: {"unsupported_claims": ["short factual label", ...]}. Max 10.',
        _shrink_for_groq_grounding_user(body, evidence_text),
        mode="smart",
        task="editor_grounding",
        temperature=config.verifier_temperature(),
    )
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return True, [], bool(raw.strip())
    try:
        d = json.loads(m.group(0))
        issues = [
            str(x).strip() for x in (d.get("unsupported_claims") or []) if str(x).strip()
        ][:10]
        return (len(issues) == 0, issues, True)
    except (json.JSONDecodeError, TypeError):
        return True, [], True
