"""Committee of analysts: scout → parallel analysts → synthesizer + bundle."""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List

import httpx
from langgraph.types import Send

from .. import config, memory, openrouter_client, research
from ..analysts import RUNNERS
from ..llm_chain import get_llm_usage, is_llm_call_cap_reached
from ..memory import _ROOT
from ..models import AnalystNote, EvidencePack, Item
from . import supervisor as sup

LOG = logging.getLogger(__name__)


def _synthesize_narrative(notes: list[AnalystNote], pack: EvidencePack) -> str:
    if is_llm_call_cap_reached() or not config.llm_configured() or config.dry_run():
        return "Committee: aggregate analyst points into one editorial through-line in draft."
    blob = "\n\n".join(
        f"### {n.role}\n"
        + "\n".join(f"- {c}" for c in (n.claims or [])[:8])
        + ("\nCaveats: " + "; ".join(n.contradictions[:4]) if n.contradictions else "")
        for n in notes
    )[:20000]
    return (
        openrouter_client.llm_text(
            "Write 2-4 short paragraphs: unified angle for a technical post, key tensions, "
            "and what a reader should take away. Plain prose, no JSON.",
            f"Title: {pack.primary.title}\n\nAnalyst memos:\n{blob}\n",
            max_tokens=1800,
            task="committee_synthesis",
        )
        or ""
    )


def node_scout(state: Dict[str, Any]) -> Dict[str, Any]:
    p_raw = state.get("primary")
    if not p_raw and state.get("rank_result"):
        p_raw = (state.get("rank_result") or {}).get("primary")
    if not p_raw:
        p_raw = json.loads(
            (_ROOT / "reports" / "rank_result.json").read_text("utf-8")
        ).get("primary", {})
    primary = Item.model_validate(p_raw)
    pack, c, tr = research.gather_evidence_pack(primary)
    return {
        "evidence_pack": json.loads(
            pack.model_dump_json()
        ),
        "primary": json.loads(pack.primary.model_dump_json()),
        "research_trace": {
            "scout": True,
            "calls": c,
            "trace": tr,
        },
        "supervisor_decisions": sup.log_decision(
            state,  # type: ignore[arg-type]
            "scout",
        ),
    }


def make_analyst_node(name: str) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    def _node(state: Dict[str, Any]) -> Dict[str, Any]:
        ep = state.get("evidence_pack")
        if not ep:
            return {"committee_notes": [AnalystNote(role=name, skipped=True).model_dump()]}
        pack = EvidencePack.model_validate(ep)
        if is_llm_call_cap_reached():
            return {
                "committee_notes": [
                    AnalystNote(role=name, claims=[], skipped=True).model_dump()
                ],
            }
        run = RUNNERS.get(name)
        if not run:
            return {
                "committee_notes": [AnalystNote(role=name, skipped=True).model_dump()],
            }
        try:
            note = run(pack)
        except (httpx.RequestError, json.JSONDecodeError) as e:
            raise e
        except Exception as e:  # noqa: BLE001
            LOG.warning("analyst %s: %s", name, e)
            note = AnalystNote(role=name, claims=[], skipped=True)
        return {"committee_notes": [note.model_dump()]}

    return _node


def node_synthesizer(state: Dict[str, Any]) -> Dict[str, Any]:
    memory.ensure_dirs()
    ep = state.get("evidence_pack")
    if not ep:
        return {
            "evidence": {},
            "research_trace": state.get("research_trace") or {},
            "_done_research": True,
        }
    pack = EvidencePack.model_validate(ep)
    raw_notes = list(state.get("committee_notes") or [])
    nmodels: list[AnalystNote] = []
    for x in raw_notes:
        if isinstance(x, dict):
            try:
                nmodels.append(AnalystNote.model_validate(x))
            except Exception:  # noqa: BLE001
                continue
    syn = _synthesize_narrative(nmodels, pack)
    tr = state.get("research_trace") or {}
    t0: list[dict[str, Any]] = list(
        (tr.get("trace") or []) if isinstance(tr, dict) else []
    )
    c0 = int((tr.get("calls") or 0) if isinstance(tr, dict) else 0) or 0
    b, c, tr2 = research.finalize_evidence_from_pack(
        pack, c0, t0, analyst_notes=nmodels, committee_synthesis=syn
    )
    (_ROOT / "reports" / "evidence_bundle.json").write_text(
        b.model_dump_json(indent=2), encoding="utf-8"
    )
    u0 = get_llm_usage()
    sink: dict = getattr(research, "_RESEARCH_SINK", {})  # type: ignore[misc,assignment]
    trace_out: dict[str, Any] = {
        "calls": c,
        "trace": tr2,
        "committee": True,
        "ss_cache_hits": int(sink.get("ss_cache_hits", 0) or 0),
        "ss_429": int(sink.get("ss_429", 0) or 0),
    }
    trace_out["llm_ok"] = u0.get("ok", 0)
    report = {
        "narrative": syn,
        "n_analysts": len(nmodels),
        "n_skipped": sum(1 for n in nmodels if n.skipped),
        "analysts": [n.model_dump() for n in nmodels],
    }
    (_ROOT / "reports" / "committee_report.json").write_text(
        json.dumps({"committee_synthesis": report, "narrative": syn}, indent=2),
        encoding="utf-8",
    )
    return {
        "evidence": json.loads(b.model_dump_json()),
        "research_trace": trace_out,
        "committee_synthesis": {"narrative": syn, "report": report},
        "_done_research": True,
        "supervisor_decisions": sup.log_decision(
            state,  # type: ignore[arg-type]
            "synthesizer",
        ),
    }


def fan_to_analysts_from_supervisor(
    state: Dict[str, Any],
) -> List[Send]:
    """Map-reduce: one Send per selected analyst, or to synthesizer if nothing to run."""
    out: list[Send] = []
    pick = list(state.get("selected_analysts") or config.committee_analysts())
    for n in pick:
        if n in RUNNERS:
            out.append(Send(f"analyst_{n}", state))
    if not out:
        out = [Send("synthesizer", state)]
    return out
