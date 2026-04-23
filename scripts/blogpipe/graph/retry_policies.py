"""LangGraph per-node `retry=` policies (transient I/O, not logic errors)."""

from __future__ import annotations

import json

import httpx
from langgraph.types import RetryPolicy

# httpx/transport failures and JSON shape drops from model output during fan-out
_TRANSIENT = (
    httpx.TransportError,
    httpx.TimeoutException,
    httpx.HTTPStatusError,
    json.JSONDecodeError,
    TimeoutError,
)

# Default: draft/editor/scout and most nodes
DEFAULT_RETRY = RetryPolicy(
    max_attempts=3,
    initial_interval=1.0,
    backoff_factor=2.0,
    max_interval=20.0,
    jitter=True,
    retry_on=_TRANSIENT,
)

# Committee analysts: one extra retry on the tight budget
ANALYST_RETRY = RetryPolicy(
    max_attempts=2,
    initial_interval=2.0,
    backoff_factor=2.0,
    max_interval=20.0,
    jitter=True,
    retry_on=_TRANSIENT,
)
