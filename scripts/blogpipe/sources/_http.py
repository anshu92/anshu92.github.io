from __future__ import annotations

import httpx

from .. import config

TIMEOUT = httpx.Timeout(30.0, connect=10.0)
_CLIENT: httpx.Client | None = None


def client() -> httpx.Client:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = httpx.Client(
            follow_redirects=True,
            timeout=TIMEOUT,
            headers={"User-Agent": config.user_agent()},
        )
    return _CLIENT
