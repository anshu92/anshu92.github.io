from __future__ import annotations

import argparse
import logging
import sys

from . import config, logging_utils

logging_utils.setup_logging()
LOG = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="blogpipe", description="Agent-swarm technical blog pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    swarm_p = sub.add_parser("swarm", help="Run the technical blog agent swarm")
    swarm_sub = swarm_p.add_subparsers(dest="swarm_command", required=True)
    run_p = swarm_sub.add_parser("run", help="Generate one review-gated technical lesson draft")
    run_p.add_argument("--window", default="14d")
    run_p.add_argument("--fixtures", default="")
    run_p.add_argument("--dry-run", action="store_true")
    run_p.add_argument("--db", default="")

    curriculum_p = sub.add_parser("curriculum", help="Prepare and validate evidence-first research reports")
    curriculum_sub = curriculum_p.add_subparsers(dest="curriculum_command", required=True)
    curriculum_sub.add_parser("status", help="Print canonical curriculum progress as JSON")

    prepare_p = curriculum_sub.add_parser("prepare", help="Create an empty report skeleton without overwriting files")
    prepare_p.add_argument("module_id")
    prepare_p.add_argument("--report", default="")
    prepare_p.add_argument("--dry-run", action="store_true")

    validate_p = curriculum_sub.add_parser("validate", help="Validate the curriculum and one module's report files")
    validate_p.add_argument("module_id", nargs="?", default="")
    validate_p.add_argument("--report", default="")

    args = parser.parse_args(argv)
    try:
        if args.command == "swarm" and args.swarm_command == "run":
            from . import swarm

            swarm.run(
                window_hours=_window_hours(args.window),
                fixtures=args.fixtures,
                dry_run=args.dry_run or config.dry_run_env(),
                db=args.db,
            )
        elif args.command == "curriculum":
            from . import research_curriculum

            if args.curriculum_command == "status":
                research_curriculum.print_status()
            elif args.curriculum_command == "prepare":
                output = research_curriculum.prepare_report(
                    args.module_id,
                    report_id=args.report,
                    dry_run=args.dry_run,
                )
                print(output)
            elif args.curriculum_command == "validate":
                curriculum = research_curriculum.load_curriculum()
                errors = (
                    research_curriculum.validate_report(args.module_id, report_id=args.report)
                    if args.module_id
                    else research_curriculum.validate_curriculum(curriculum)
                )
                if errors:
                    raise ValueError("research_report_invalid:" + ";".join(errors))
                print("research curriculum validation passed")
    except Exception as exc:  # noqa: BLE001
        LOG.exception("command failed: %s", exc)
        return 1
    return 0


def _window_hours(value: str) -> int:
    raw = (value or "14d").strip().lower()
    if raw.endswith("h"):
        return max(1, int(raw[:-1]))
    if raw.endswith("d"):
        return max(1, int(raw[:-1]) * 24)
    return max(1, int(raw))


if __name__ == "__main__":
    sys.exit(main())
