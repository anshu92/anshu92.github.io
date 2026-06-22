from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

from pydantic import TypeAdapter

from . import assets, config, evidence, ingest, memory, outline as outline_mod, rank, score, selector, store, writer
from .llm import LLMClient
from .models import RankedItem, WriteResult

LOG = logging.getLogger(__name__)
RANKED = TypeAdapter(list[RankedItem])


@dataclass(frozen=True)
class AgentRunPlan:
    daily_required: bool
    requested_deep_dives: int
    allowed_deep_dives: int
    remaining_runtime_seconds: float
    deep_dive_min_budget_seconds: float
    rationale: str


def run_all(
    *,
    window_hours: int = 14 * 24,
    fixtures: str = "",
    dry_run: bool = False,
    db: str = "",
    max_deep_dives: int = 1,
) -> dict[str, object]:
    ingest_count = ingest.run(window_hours=window_hours, fixtures=fixtures, db=db)
    ranked = rank.run(db=db, max_age_hours=None if fixtures else window_hours)
    client = LLMClient()
    initial_plan = _plan_agent_run(client, requested_deep_dives=max_deep_dives)
    daily = write_daily(
        ranked=ranked,
        dry_run=dry_run,
        llm=client,
        db=db,
        fallback_max_age_hours=None if fixtures else window_hours,
    )
    if daily.ok:
        final_plan = _plan_agent_run(client, requested_deep_dives=max_deep_dives)
        deep = write_deep_dives(ranked=ranked, max_new=final_plan.allowed_deep_dives, dry_run=dry_run, llm=client)
        if "/img/posts/" in daily.body:
            render_assets()
    else:
        LOG.warning("daily writer blocked publication: %s", daily.errors)
        final_plan = _plan_agent_run(client, requested_deep_dives=0)
        deep = []
    result = {
        "ingest_count": ingest_count,
        "ranked_count": len(ranked),
        "daily": daily.model_dump(),
        "deep_dives": [d.model_dump() for d in deep],
        "agent_plan": {"initial": asdict(initial_plan), "final": asdict(final_plan)},
    }
    memory.ensure_dirs()
    (memory.REPORTS / "run_report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (memory.REPORTS / "llm_usage.json").write_text(
        json.dumps(client.usage.__dict__, indent=2),
        encoding="utf-8",
    )
    return result


def _plan_agent_run(client: LLMClient, *, requested_deep_dives: int) -> AgentRunPlan:
    remaining = client._remaining_runtime_seconds() if isinstance(client, LLMClient) else 0.0
    min_budget = config.agent_deep_dive_min_budget_seconds()
    requested = max(0, requested_deep_dives)
    if requested <= 0:
        allowed = 0
        rationale = "deep_dives_not_requested"
    elif remaining < min_budget:
        allowed = 0
        rationale = "skip_optional_deep_dives_to_preserve_github_actions_budget"
    else:
        allowed = requested
        rationale = "runtime_budget_allows_optional_deep_dives"
    return AgentRunPlan(
        daily_required=True,
        requested_deep_dives=requested,
        allowed_deep_dives=allowed,
        remaining_runtime_seconds=round(remaining, 3),
        deep_dive_min_budget_seconds=round(min_budget, 3),
        rationale=rationale,
    )


def write_daily(
    *,
    ranked: list[RankedItem] | None = None,
    dry_run: bool = False,
    llm: LLMClient | None = None,
    db: str = "",
    fallback_max_age_hours: int | None = 14 * 24,
) -> WriteResult:
    ranked = load_ranked() if ranked is None else ranked
    client = llm or LLMClient()
    required_items = min(3, config.daily_primary_papers())
    ranked = _augment_ranked_with_store_papers(
        ranked,
        required_papers=min(3, config.daily_primary_papers()),
        db=db,
        max_age_hours=fallback_max_age_hours,
    )
    if len(ranked) < required_items:
        result = _blocked_daily([f"insufficient_ranked_items:{len(ranked)}/{required_items}"])
        _write_daily_reports(result, client)
        return result
    try:
        shortlist, selection = selector.select_daily_items(ranked, llm=client)
    except selector.SelectionError as exc:
        result = _blocked_daily([str(exc)])
        _write_daily_reports(result, client)
        return result
    pack = evidence.build_daily_pack(shortlist)
    try:
        daily_outline = outline_mod.generate_daily_outline(pack, selection=selection, llm=client)
    except outline_mod.OutlineError as exc:
        result = _blocked_daily([str(exc)])
        _write_daily_reports(result, client)
        return result
    try:
        result = writer.write_daily(pack, outline=daily_outline, selection=selection, llm=client, dry_run=dry_run)
    except RuntimeError as exc:
        result = _blocked_daily([f"daily_writer_failed:{exc}"])
        _write_daily_reports(result, client)
        return result
    memory.ensure_dirs()
    (memory.REPORTS / "daily_selection.json").write_text(
        selection.model_dump_json(indent=2),
        encoding="utf-8",
    )
    (memory.REPORTS / "daily_outline.json").write_text(
        daily_outline.model_dump_json(indent=2),
        encoding="utf-8",
    )
    (memory.REPORTS / "daily_selected_items.json").write_text(
        json.dumps({"items": [r.model_dump(mode="json") for r in shortlist]}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_daily_reports(result, client)
    return result


def _augment_ranked_with_store_papers(
    ranked: list[RankedItem],
    *,
    required_papers: int,
    db: str = "",
    max_age_hours: int | None = 72,
) -> list[RankedItem]:
    paper_count = sum(1 for item in ranked if item.item.source_kind == "paper")
    if paper_count >= required_papers:
        return ranked

    existing_ids = {item.item.item_id for item in ranked}
    with store.connect(db or None) as conn:
        stored_items = store.load_items(conn, limit=500)

    fallback_papers = [
        item
        for item in stored_items
        if item.source_kind == "paper" and item.normalized().item_id not in existing_ids
    ]
    if not fallback_papers:
        return ranked

    fallback_ranked = score.rank_items(
        fallback_papers,
        limit=50,
        max_age_hours=max_age_hours,
    )
    if not fallback_ranked:
        return ranked

    combined = [*ranked, *fallback_ranked]
    combined.sort(key=lambda item: item.daily_score, reverse=True)
    combined = score._diversify(combined)
    recovered = sum(1 for item in combined if item.item.source_kind == "paper")
    LOG.warning(
        "daily writer recovered ranked paper pool from store fallback: %d -> %d papers",
        paper_count,
        recovered,
    )
    return combined


def _write_daily_reports(result: WriteResult, client: LLMClient) -> None:
    memory.ensure_dirs()
    (memory.REPORTS / "daily_write_result.json").write_text(
        result.model_dump_json(indent=2),
        encoding="utf-8",
    )
    (memory.REPORTS / "llm_usage.json").write_text(
        json.dumps(client.usage.__dict__, indent=2),
        encoding="utf-8",
    )


def _blocked_daily(errors: list[str]) -> WriteResult:
    from datetime import datetime, timezone

    today = datetime.now(timezone.utc).date().isoformat()
    title = f"Research Radar blocked - {today}"
    slug = f"{today}-research-radar"
    result = WriteResult(ok=False, title=title, errors=errors)
    memory.ensure_dirs()
    (memory.REPORTS / f"{slug}.blocked.json").write_text(
        json.dumps({"title": title, "slug": slug, "errors": errors}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result


def write_deep_dives(
    *,
    ranked: list[RankedItem] | None = None,
    max_new: int = 1,
    dry_run: bool = False,
    llm: LLMClient | None = None,
) -> list[WriteResult]:
    ranked = load_ranked() if ranked is None else ranked
    client = llm or LLMClient()
    results: list[WriteResult] = []
    for item in score.deep_dive_shortlist(ranked, maximum=max_new):
        pack = evidence.build_deep_dive_pack(item)
        results.append(writer.write_deep_dive(pack, llm=client, dry_run=dry_run))
    (memory.REPORTS / "deep_dive_write_result.json").write_text(
        json.dumps([r.model_dump() for r in results], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return results


def render_assets(*, ranked: list[RankedItem] | None = None) -> list[str]:
    selected_path = memory.REPORTS / "daily_selected_items.json"
    if ranked is None and selected_path.is_file():
        payload = json.loads(selected_path.read_text(encoding="utf-8"))
        ranked = RANKED.validate_python(payload.get("items", []))
    ranked = ranked or load_ranked()
    shortlist = score.daily_shortlist(ranked)
    from datetime import datetime, timezone

    slug = f"{datetime.now(timezone.utc).date().isoformat()}-research-radar"
    paths = assets.render_daily_assets(shortlist, slug)
    return [str(path.relative_to(memory.ROOT)) for path in paths]


def load_ranked() -> list[RankedItem]:
    path = memory.REPORTS / "ranked_items.json"
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return RANKED.validate_python(payload.get("items", []))
    with store.connect() as conn:
        return rank.run(db=str(store.db_path()))
