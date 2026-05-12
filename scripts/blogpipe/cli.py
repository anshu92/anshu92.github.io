from __future__ import annotations

import argparse
import logging
import sys

from . import config, logging_utils

logging_utils.setup_logging()
LOG = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="blogpipe", description="Evidence-grounded research radar")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest_p = sub.add_parser("ingest")
    ingest_p.add_argument("--window", default="72h")
    ingest_p.add_argument("--fixtures", default="")
    ingest_p.add_argument("--db", default="")

    rank_p = sub.add_parser("rank")
    rank_p.add_argument("--db", default="")
    rank_p.add_argument("--limit", type=int, default=50)
    rank_p.add_argument("--max-age", default="72h")

    daily_p = sub.add_parser("write-daily")
    daily_p.add_argument("--dry-run", action="store_true")

    deep_p = sub.add_parser("write-deep-dives")
    deep_p.add_argument("--max-new", type=int, default=1)
    deep_p.add_argument("--dry-run", action="store_true")

    assets_p = sub.add_parser("render-assets")
    assets_p.set_defaults(render_assets=True)

    run_p = sub.add_parser("run")
    run_p.add_argument("--window", default="72h")
    run_p.add_argument("--fixtures", default="")
    run_p.add_argument("--dry-run", action="store_true")
    run_p.add_argument("--db", default="")
    run_p.add_argument("--max-deep-dives", type=int, default=1)

    args = parser.parse_args(argv)
    try:
        if args.command == "ingest":
            from . import ingest

            ingest.run(window_hours=_window_hours(args.window), fixtures=args.fixtures, db=args.db)
        elif args.command == "rank":
            from . import rank

            rank.run(db=args.db, limit=args.limit, max_age_hours=_window_hours(args.max_age))
        elif args.command == "write-daily":
            from . import pipeline

            pipeline.write_daily(dry_run=args.dry_run or config.dry_run_env())
        elif args.command == "write-deep-dives":
            from . import pipeline

            pipeline.write_deep_dives(max_new=args.max_new, dry_run=args.dry_run or config.dry_run_env())
        elif args.command == "render-assets":
            from . import pipeline

            pipeline.render_assets()
        elif args.command == "run":
            from . import pipeline

            pipeline.run_all(
                window_hours=_window_hours(args.window),
                fixtures=args.fixtures,
                dry_run=args.dry_run or config.dry_run_env(),
                db=args.db,
                max_deep_dives=args.max_deep_dives,
            )
    except Exception as exc:
        LOG.exception("command failed: %s", exc)
        return 1
    return 0


def _window_hours(value: str) -> int:
    raw = (value or "72h").strip().lower()
    if raw.endswith("h"):
        return max(1, int(raw[:-1]))
    if raw.endswith("d"):
        return max(1, int(raw[:-1]) * 24)
    return max(1, int(raw))


if __name__ == "__main__":
    sys.exit(main())
