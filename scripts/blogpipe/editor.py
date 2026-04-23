"""Editor pass: rubric score, optional rewrite.

When using ``python -m blogpipe graph``, rubric/grounding run inside the graph after drafting.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from . import config, lint, memory, openrouter_client
from .llm_chain import get_llm_usage
from .models import EditorReport
from .memory import _ROOT

LOG = logging.getLogger(__name__)


def _normalize_rubric_items(raw: object) -> list[dict[str, Any]]:
    """Coerce LLM output: strings or loose dicts into {item, score} dicts."""
    if not raw or not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for x in raw:
        if isinstance(x, str):
            t = x.strip()
            if t:
                out.append({"item": t, "score": 1})
            continue
        if isinstance(x, dict):
            d: dict[str, Any] = dict(x)
            if "item" not in d and "name" in d:
                d["item"] = d.pop("name")
            if "item" not in d and "criterion" in d:
                d["item"] = d.pop("criterion")
            it = d.get("item")
            if it is not None and str(it).strip():
                d.setdefault("score", 1)
                out.append(d)
    return out


def _unwrap_markdown_fence(text: str) -> str:
    t = text.strip()
    m = re.match(r"^```(?:markdown|md)?\n([\s\S]*?)\n```$", t, re.I)
    return m.group(1).strip() if m else t


def _looks_like_markdown_body(text: str) -> bool:
    t = _unwrap_markdown_fence(text)
    if "## " not in t:
        return False
    bad_starts = (
        "i've revised",
        "here is the revised",
        "here's the revised",
        "i updated",
        "i improved",
    )
    return not t.lower().startswith(bad_starts)


def _load_draft_path() -> Path:
    return _ROOT / (_ROOT / "reports" / "draft_path.txt").read_text().strip()


def _split_frontmatter(text: str) -> tuple[str, str]:
    m = re.match(r"^((?:---\s*\n.*?\n---\s*\n))", text, re.DOTALL)
    if not m:
        return "", text
    return m.group(1), text[m.end() :]


def _rubric_llm(md: str) -> EditorReport:
    system = (
        "You are a senior technical editor. Score the blog draft on 15 criteria, 1 point each if "
        "the draft clearly meets the standard, 0 otherwise. Be strict: the bar is a post a senior "
        "engineer would recommend in a Slack channel.\n"
        "\nCRITERIA (score each 0 or 1):\n"
        "1. sharp_takeaway: First sentence names a concrete result with a number; one-sentence summary "
        "of the whole post is possible.\n"
        "2. intro_earns_attention: In the first 3-5 sentences, the reader knows the problem, who it "
        "matters to, and what they will learn. No throat-clearing, no 'In recent years'.\n"
        "3. concrete_problem: Real-world context, constraints, and failure mode are named. Avoids "
        "vague words ('scale', 'performance', 'robustness') without specifics.\n"
        "4. inevitable_structure: Flow reads problem -> context -> approach -> implementation -> "
        "results -> limits. Each section answers one clear question. No tangents.\n"
        "5. specific_language: Specific systems, components, datasets, APIs, models, numbers. No "
        "filler, no repeated caveats.\n"
        "6. teaches_why: Explains why something works, not just what was done. Provides enough "
        "background for an informed outsider to follow.\n"
        "7. strong_example: At least one concrete example with inputs/outputs or before/after "
        "behaviour that carries the argument, not decorates it.\n"
        "8. tradeoffs_explicit: Names alternatives considered and what was given up (latency, cost, "
        "simplicity, accuracy, maintainability).\n"
        "9. claims_backed: Every improvement/speed/accuracy claim is supported by a number, "
        "benchmark, baseline, or citation. No unsupported confidence.\n"
        "10. limitations_honest: Describes where the method breaks, assumptions required, and when "
        "NOT to use it. The success path is not the only path shown.\n"
        "11. skimmable: Short paragraphs, descriptive sub-headings, key takeaways easy to spot on a "
        "scroll, code blocks that matter.\n"
        "12. actionable_close: Reader ends knowing what to try, avoid, run, or think differently "
        "about. Not a restatement of the intro.\n"
        "13. diagrams_clarify: Mermaid/figure reduces cognitive load; does not need a paragraph to "
        "decode. If absent, score 0.\n"
        "14. honest_scope: Local results are not sold as general laws. Sample size, dataset, scale, "
        "or team constraints are acknowledged where relevant.\n"
        "15. pov_present: The post argues something — why a design worked, why an assumption is "
        "wrong, why a tradeoff is underrated. At least one first-person opinion sentence.\n"
        "\nAlso fill five_questions: {problem, hard, tried, outcomes, next}, each 1-3 sentences "
        "quoted or paraphrased from the text. If any cannot be answered from the post, set that "
        "value to \"CANNOT_DETERMINE\". five_questions_ok is true only if none are CANNOT_DETERMINE.\n"
        "\nrubric_items MUST be a JSON array of 15 objects (one per criterion above), each "
        '{"item": "sharp_takeaway", "score": 0 or 1}. Not an array of strings.\n'
        'Output JSON only: {"rubric_score": N, "rubric_items": ['
        '{"item":"sharp_takeaway","score":1}, ...], "five_questions": {...}, '
        '"five_questions_ok": true}'
    )
    raw = openrouter_client.llm_text(
        system,
        md[:24000],
        mode="smart",
        max_tokens=config.max_tokens_smart(),
        task="editor_rubric",
    )
    if not raw.strip():
        return EditorReport(
            rubric_score=9,
            rubric_items=[],
            five_questions={},
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
    except Exception as e:
        LOG.warning("editor parse: %s", e)
        return EditorReport(rubric_score=8, pass_gate=True)


def _grounding_check(body: str) -> tuple[bool, list[str], bool]:
    """Flags claims vs evidence. Third value: LLM returned parseable/usable content."""
    ev_p = _ROOT / "reports" / "evidence_bundle.json"
    if not ev_p.is_file() or config.dry_run() or not config.llm_configured():
        return True, [], True
    try:
        ev_text = ev_p.read_text(encoding="utf-8")[:22000]
    except OSError:
        return True, [], True
    raw = openrouter_client.llm_text(
        "You compare a blog draft to EVIDENCE JSON. "
        'Output JSON only: {"unsupported_claims": ["short label", ...]}. '
        "Flag only specific numbers, dates, competitive claims, or paper results that are not present in EVIDENCE. "
        "Max 10 items; use [] if none.",
        f"EVIDENCE_JSON:\n{ev_text}\n\n---\n\nDRAFT_MD:\n{body[:20000]}\n",
        mode="smart",
        max_tokens=config.max_tokens_smart(),
        task="editor_grounding",
    )
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return True, [], bool(raw.strip())
    try:
        d = json.loads(m.group(0))
        issues = [str(x).strip() for x in (d.get("unsupported_claims") or []) if str(x).strip()][:12]
        return (len(issues) == 0, issues, True)
    except (json.JSONDecodeError, TypeError):
        return True, [], True


def _inject_score(path: Path, score: int) -> None:
    t = path.read_text(encoding="utf-8")
    if re.search(r"^rubric_score:\s", t, re.M):
        t = re.sub(
            r"^rubric_score:\s*.*$", f"rubric_score: {score}", t, count=1, flags=re.M
        )
    else:
        t = t.replace("---\n", f"---\nrubric_score: {score}\n", 1)
    path.write_text(t, encoding="utf-8")


def run() -> EditorReport:
    memory.ensure_dirs()
    path = _load_draft_path()
    text = path.read_text(encoding="utf-8")
    front, body = _split_frontmatter(text)
    rep = _rubric_llm(body)
    loops = 0
    while (
        not rep.pass_gate
        and loops < config.editor_max_loops()
        and not config.dry_run()
        and config.llm_configured()
    ):
        loops += 1
        fix = openrouter_client.llm_text(
            f"Revise the markdown body to pass rubric >= {config.editor_min_score()} and make all "
            "five questions (problem, hard, tried, outcomes, next) answerable in one skim. Current "
            f"score {rep.rubric_score}.\n"
            "Editing standards to apply:\n"
            "- Open with a sharp takeaway containing a concrete number; remove throat-clearing.\n"
            "- Replace vague wording ('performance', 'scale', 'robust') with specifics: components, "
            "constraints, numbers, failure modes.\n"
            "- Every speed/accuracy/cost claim needs a number, baseline, or [cite:] backing.\n"
            "- Name tradeoffs where a design choice is described; name at least one limitation where "
            "a method or result is described.\n"
            "- Keep the narrative tight: each section answers one question that leads to the next.\n"
            "- Close with a concrete next action for the reader, not a restatement of the intro.\n"
            "- Keep at least one first-person opinion sentence with a real stance.\n"
            "- Short paragraphs. Descriptive, story-specific ## headings (rename, add, remove, or "
            "reorder as needed — NO 'Introduction', 'Background', 'Overview', 'Results', 'Summary', "
            "'Conclusion').\n"
            "Output full markdown body only, no frontmatter, no commentary, no code fences.",
            body[:20000],
            mode="smart",
            max_tokens=config.max_tokens_smart(),
            task="draft_full",
        )
        if fix.strip() and _looks_like_markdown_body(fix):
            body = _unwrap_markdown_fence(fix)
            path.write_text(front + body, encoding="utf-8")
        rep = _rubric_llm(body)
    g_ok, g_issues, g_llm = _grounding_check(body)
    ev_p = _ROOT / "reports" / "evidence_bundle.json"
    ev_text = ev_p.read_text(encoding="utf-8")[:22000] if ev_p.is_file() else ""
    lint_issues = list(dict.fromkeys(lint.structural_issues(body)))
    det_ground = lint.unsupported_numeric_claims(body, ev_text) if ev_text else []
    if (
        (g_issues or det_ground)
        and not config.dry_run()
        and config.llm_configured()
    ):
        fix = openrouter_client.llm_text(
            "Revise the markdown body to remove unsupported claims flagged against EVIDENCE. "
            "Delete or soften any claim that is not directly supported. Do not add new facts. "
            "Keep the topic fixed to the same paper, preserve the table and mermaid block, and "
            "return the full markdown body only.",
            "UNSUPPORTED CLAIMS:\n- "
            + "\n- ".join(list(dict.fromkeys(g_issues + det_ground))[:10])
            + f"\n\nBODY:\n{body[:20000]}\n\nEVIDENCE_JSON:\n{ev_text}",
            mode="smart",
            max_tokens=config.max_tokens_smart(),
            task="editor_grounding",
        )
        if fix.strip() and _looks_like_markdown_body(fix):
            body = _unwrap_markdown_fence(fix)
            path.write_text(front + body, encoding="utf-8")
            rep = _rubric_llm(body)
            g_ok, g_issues, g_llm = _grounding_check(body)
            lint_issues = list(dict.fromkeys(lint.structural_issues(body)))
            det_ground = lint.unsupported_numeric_claims(body, ev_text) if ev_text else []
    _inject_score(path, rep.rubric_score)
    usage = get_llm_usage()
    need_llm = bool(config.llm_configured() and not config.dry_run())
    llm_ok = True
    editor_warnings: list[str] = []
    if need_llm:
        if int(usage.get("ok", 0) or 0) < 2:
            llm_ok = False
            editor_warnings.append("llm_successes_below_threshold")
        if not g_llm:
            llm_ok = False
            editor_warnings.append("grounding_llm_no_response")
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
    if g_issues:
        rep = rep.model_copy(update={"pass_gate": False})
    if det_ground:
        rep = rep.model_copy(update={"pass_gate": False})
    if not llm_ok:
        rep = rep.model_copy(update={"pass_gate": False})
    u_total = int(usage.get("ok", 0) or 0) + int(usage.get("fail", 0) or 0)
    print(
        f"stage=edit calls_ok={usage.get('ok', 0)}/{max(1, u_total)} pass_gate={rep.pass_gate} "
        f"llm_ok={llm_ok}",
        flush=True,
    )
    (_ROOT / "reports" / "llm_usage.json").write_text(
        json.dumps(usage, indent=2),
        encoding="utf-8",
    )
    rep = rep.model_copy(
        update={
            "lint_issues": lint_issues,
            "grounding_ok": bool(g_ok and not det_ground),
            "grounding_issues": list(dict.fromkeys(g_issues + det_ground)),
            "llm_ok": llm_ok,
            "editor_warnings": editor_warnings,
        }
    )
    (_ROOT / "reports" / "editor_report.json").write_text(
        rep.model_dump_json(indent=2), encoding="utf-8"
    )
    return rep
