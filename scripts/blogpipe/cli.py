"""Blogpipe CLI."""

from __future__ import annotations

import argparse
import logging
import sys

from . import logging_utils

logging_utils.setup_logging()
LOG = logging.getLogger(__name__)


def main() -> int:
    p = argparse.ArgumentParser(prog="blogpipe", description="ML blog pipeline")
    p.add_argument(
        "command",
        choices=[
            "curate",
            "harvest",
            "rank",
            "research",
            "draft",
            "edit",
            "visuals",
            "package",
            "run",
        ],
    )
    a = p.parse_args()
    try:
        if a.command == "curate":
            from . import curator

            curator.run()
        elif a.command == "harvest":
            from . import harvest as hv

            hv.run()
        elif a.command == "rank":
            from . import rank

            rank.run()
        elif a.command == "research":
            from . import research

            research.run()
        elif a.command == "draft":
            from . import draft

            draft.run()
        elif a.command == "edit":
            from . import editor

            editor.run()
        elif a.command == "visuals":
            from . import visuals

            visuals.run()
        elif a.command == "package":
            from . import package

            package.run()
        elif a.command == "run":
            from . import curator, draft, editor, harvest, package, rank, research, visuals

            curator.run()
            harvest.run()
            rank.run()
            research.run()
            draft.run()
            editor.run()
            visuals.run()
            package.run()
    except Exception as e:
        LOG.exception("command failed: %s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
