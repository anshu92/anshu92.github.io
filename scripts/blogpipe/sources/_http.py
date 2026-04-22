from __future__ import annotations

import httpx
from typing import Optional

DEFAULT_TIMEOUT = httpx.Timeout(30.0, connect=10.0)

_client: Optional[httpx.Client] = None


def client() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(verify=True, follow_redirects=True, timeout=DEFAULT_TIMEOUT)
    return _client


def get_json(url: str) -> object:
    r = client().get(url, headers={"User-Agent": "blogpipe/0.1 (+https://anshu92.github.io)"})
    r.raise_for_status()
    return r.json()
