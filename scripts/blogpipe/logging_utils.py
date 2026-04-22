"""Log filtering to avoid echoing secret env values."""

from __future__ import annotations

import logging
import re

_SECRET_ENV_NAMES = frozenset(
    {
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "SEMANTIC_SCHOLAR_API_KEY",
        "FAL_API_KEY",
        "EMAIL_PASSWORD",
    }
)


def setup_logging() -> None:
    class RedactFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            msg = str(record.getMessage())
            for name in _SECRET_ENV_NAMES:
                v = __import__("os").environ.get(name, "")
                if v and len(v) > 6:
                    msg = msg.replace(v, "[REDACTED]")
            record.msg = re.sub(
                r"(api[_-]?key|bearer|token)=[\w-]+", r"\1=[REDACTED]", msg, flags=re.I
            )
            record.args = ()
            return True

    root = logging.getLogger()
    if not any(isinstance(f, RedactFilter) for f in root.filters):
        root.addFilter(RedactFilter())
    root.setLevel(logging.INFO)
