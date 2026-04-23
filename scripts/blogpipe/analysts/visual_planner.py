"""Visual planner: 0-3 concept figures and 0-3 key equations; never decorative clutter."""

from __future__ import annotations

import json
import re
import logging
from typing import Any

from .. import config, openrouter_client
from ..llm_chain import is_llm_call_cap_reached
from ..models import AnalystNote, EvidencePack, VisualPlan, FigureSpec, EquationSpec

LOG = logging.getLogger(__name__)

_BASE_SCHEMA = (
    "Output JSON only, no other text. "
    "Shape: figures[] of {id, kind in concept|architecture|comparison|plot, prompt (no text in image), "
    "alt, caption, placement_hint}; equations[] of {id, latex (KaTeX-escaped), caption, "
    "placement_hint}; confidence low|medium|high. Up to 3 each; [] allowed. "
    "Figure only if prose cannot carry the idea; equation only for a core definition or objective. "
    "id: lowercase a-z0-9_, max 32, unique. "
    'Example: {"figures": [], "equations": [], "confidence": "low"}.'
)


_ALT_PLACEHOLDERS = {
    "", "accessibility", "alt", "alt text", "image", "figure",
    "<short alt text describing what the figure shows for screen readers>",
}


def _normalize_fig(d: dict[str, Any]) -> dict[str, Any] | None:
    raw = str(d.get("id") or "fig")
    safe = re.sub(r"[^a-z0-9_]", "", raw.lower()[:32]) or "fig"
    kind = str(d.get("kind") or "concept")[:20]
    if kind not in ("concept", "architecture", "comparison", "plot"):
        kind = "concept"
    alt = str(d.get("alt") or "").strip()
    caption = str(d.get("caption") or "").strip()
    # Guard: if the LLM echoed the schema placeholder, fall back to caption or id.
    if alt.lower() in _ALT_PLACEHOLDERS:
        alt = caption[:200] or safe.replace("_", " ")
    return {
        "id": safe,
        "kind": kind,
        "prompt": str(d.get("prompt") or "")[:2000],
        "alt": alt[:200] or safe.replace("_", " "),
        "caption": caption[:300],
        "placement_hint": str(d.get("placement_hint") or "")[:500],
    }


def _normalize_eq(d: dict[str, Any]) -> dict[str, Any] | None:
    raw = str(d.get("id") or "eq")
    safe = re.sub(r"[^a-z0-9_]", "", raw.lower()[:32]) or "eq"
    latex = str(d.get("latex") or "").strip()
    if not latex:
        return None
    return {
        "id": safe,
        "latex": latex[:4000],
        "caption": str(d.get("caption") or "")[:300],
        "placement_hint": str(d.get("placement_hint") or "")[:500],
    }


def _parse_raw_to_plan(raw: str) -> VisualPlan | None:
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except (json.JSONDecodeError, TypeError):
        return None
    figs: list[FigureSpec] = []
    for x in (d.get("figures") or [])[:3]:
        if not isinstance(x, dict):
            continue
        n = _normalize_fig(x)
        if n and n.get("prompt"):
            try:
                figs.append(FigureSpec.model_validate(n))
            except Exception:
                continue
    eqs: list[EquationSpec] = []
    for x in (d.get("equations") or [])[:3]:
        if not isinstance(x, dict):
            continue
        n = _normalize_eq(x)
        if n:
            try:
                eqs.append(EquationSpec.model_validate(n))
            except Exception:
                continue
    return VisualPlan(figures=figs, equations=eqs)


def run(pack: EvidencePack) -> AnalystNote:
    if is_llm_call_cap_reached() or not config.llm_configured() or config.dry_run():
        return AnalystNote(role="visual_planner", claims=[], skipped=True)
    se = pack.section_evidence or {}
    blob = "\n\n".join(
        str(se.get(k) or "")
        for k in (
            "paper_problem",
            "paper_method",
            "paper_experiments",
            "paper_limitations",
        )
        if (se.get(k) or "").strip()
    )[:10000]
    if not blob.strip():
        blob = (pack.primary.abstract or "")[:6000]
    system = f"Visual plan for a technical ML post. {_BASE_SCHEMA}"
    user = f"Title: {pack.primary.title}\n\nExcerpts:\n{blob}\n"
    raw = openrouter_client.llm_text(
        system,
        user,
        max_tokens=1500,
        task="visual_planner",
    )
    if not (raw or "").strip():
        return AnalystNote(role="visual_planner", claims=[], skipped=True)
    plan = _parse_raw_to_plan(raw)
    if plan is None:
        return AnalystNote(role="visual_planner", claims=[], skipped=True)
    if not plan.figures and not plan.equations:
        return AnalystNote(
            role="visual_planner",
            claims=[json.dumps({"figures": [], "equations": []})],
            confidence="low",
        )
    payload = plan.model_dump()
    return AnalystNote(
        role="visual_planner",
        claims=[json.dumps(payload)],
        confidence="high" if (plan.figures or plan.equations) else "low",
    )


def parse_visual_plan_from_analysts(notes: list[AnalystNote]) -> VisualPlan | None:
    """Merge visual_planner output into a VisualPlan."""
    for n in notes:
        if n.role != "visual_planner" or n.skipped or not n.claims:
            continue
        for c in n.claims:
            c = c.strip()
            if not c.startswith("{"):
                continue
            try:
                d = json.loads(c)
            except json.JSONDecodeError:
                continue
            try:
                return VisualPlan.model_validate(d)
            except Exception as e:  # noqa: BLE001
                LOG.debug("visual_plan validate: %s", e)
                continue
    return None
