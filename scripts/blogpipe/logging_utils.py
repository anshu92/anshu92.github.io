from __future__ import annotations

import logging
import os


def setup_logging() -> None:
    level = os.environ.get("BLOGPIPE_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )
