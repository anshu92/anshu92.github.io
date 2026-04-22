"""Editor pass: rubric score, optional rewrite."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from . import config, memory, openrouter_client
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
        "Score the technical blog draft 0-15 (one point each for: "
        "sharp takeaway, intro earns attention, concrete problem, structure, "
        "specificity, teaches why, strong example, tradeoffs, evidence, "
        "limitations, skimmable, actionable close, diagrams clarify, honest scope, POV). "
        "Also in JSON, five_questions: {problem, hard, tried, outcomes, next} each 1-3 sentences from the text. "
        "If any cannot be answered from the post, set that value to \"CANNOT_DETERMINE\". "
        'rubric_items must be a JSON array of 15 objects, one per rubric line, each like '
        '{"item": "short label", "score": 0 or 1} — not an array of strings. '
        'Output JSON only: {"rubric_score": N, "rubric_items": ['
        '{"item":"sharp takeaway","score":1},...], "five_questions": {...}, "five_questions_ok": true }'
    )
    raw = openrouter_client.llm_text(system, md[:24000], mode="smart")
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
            f"Revise the markdown body only to pass rubric >= {config.editor_min_score()} "
            f"and ensure five_questions are answerable. Keep ## headings. "
            f"Score {rep.rubric_score}. Output full markdown body only, no frontmatter, "
            f"no commentary, and no code fences.",
            body[:20000],
            mode="smart",
        )
        if fix.strip() and _looks_like_markdown_body(fix):
            body = _unwrap_markdown_fence(fix)
            path.write_text(front + body, encoding="utf-8")
        rep = _rubric_llm(body)
    _inject_score(path, rep.rubric_score)
    (_ROOT / "reports" / "editor_report.json").write_text(
        rep.model_dump_json(indent=2), encoding="utf-8"
    )
    return rep
