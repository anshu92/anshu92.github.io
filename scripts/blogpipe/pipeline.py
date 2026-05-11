from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import TypeAdapter

from . import assets, evidence, ingest, memory, rank, score, store, writer
from .llm import LLMClient
from .models import RankedItem, WriteResult

LOG = logging.getLogger(__name__)
RANKED = TypeAdapter(list[RankedItem])


def run_all(
    *,
    window_hours: int = 72,
    fixtures: str = "",
    dry_run: bool = False,
    db: str = "",
    max_deep_dives: int = 1,
) -> dict[str, object]:
    ingest_count = ingest.run(window_hours=window_hours, fixtures=fixtures, db=db)
    ranked = rank.run(db=db)
    daily = write_daily(ranked=ranked, dry_run=dry_run)
    deep = write_deep_dives(ranked=ranked, max_new=max_deep_dives, dry_run=dry_run)
    render_assets(ranked=ranked)
    result = {
        "ingest_count": ingest_count,
        "ranked_count": len(ranked),
        "daily": daily.model_dump(),
        "deep_dives": [d.model_dump() for d in deep],
    }
    memory.ensure_dirs()
    (memory.REPORTS / "run_report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result


def write_daily(*, ranked: list[RankedItem] | None = None, dry_run: bool = False) -> WriteResult:
    ranked = ranked or load_ranked()
    shortlist = score.daily_shortlist(ranked)
    pack = evidence.build_daily_pack(shortlist)
    client = LLMClient()
    result = writer.write_daily(pack, llm=client, dry_run=dry_run)
    (memory.REPORTS / "daily_write_result.json").write_text(
        result.model_dump_json(indent=2),
        encoding="utf-8",
    )
    (memory.REPORTS / "llm_usage.json").write_text(
        json.dumps(client.usage.__dict__, indent=2),
        encoding="utf-8",
    )
    return result


def write_deep_dives(
    *,
    ranked: list[RankedItem] | None = None,
    max_new: int = 1,
    dry_run: bool = False,
) -> list[WriteResult]:
    ranked = ranked or load_ranked()
    client = LLMClient()
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
