from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

import httpx

from . import config

LOG = logging.getLogger(__name__)


@dataclass
class LLMUsage:
    calls: int = 0
    failures: int = 0
    prompt_tokens_est: int = 0
    completion_tokens_est: int = 0
    model: str = ""


@dataclass
class LLMClient:
    cfg: config.LLMConfig = field(default_factory=config.llm_config)
    usage: LLMUsage = field(default_factory=LLMUsage)

    def configured(self) -> bool:
        return bool(self.cfg.api_key and self.cfg.model and self.cfg.max_calls > 0)

    def complete(self, *, system: str, user: str, max_tokens: int | None = None) -> str:
        fake = _fake_response(system, self.usage.calls)
        if fake:
            self.usage.calls += 1
            self.usage.model = "fake"
            return fake
        if not self.configured():
            raise RuntimeError("BLOGPIPE_LLM_API_KEY is required for non-dry-run writing")
        if self.usage.calls >= self.cfg.max_calls:
            raise RuntimeError("BLOGPIPE_LLM_MAX_CALLS reached")
        self.usage.calls += 1
        self.usage.model = self.cfg.model
        self.usage.prompt_tokens_est += (len(system) + len(user)) // 4
        payload = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.cfg.temperature,
            "max_tokens": max_tokens or self.cfg.max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                resp = httpx.post(
                    f"{self.cfg.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=httpx.Timeout(120.0, connect=10.0),
                )
                if resp.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                resp.raise_for_status()
                data = resp.json()
                text = data["choices"][0]["message"]["content"]
                self.usage.completion_tokens_est += len(text) // 4
                return text
            except Exception as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(2 ** attempt)
        self.usage.failures += 1
        raise RuntimeError(f"LLM completion failed: {last_error}") from last_error


def write_usage(path: str, client: LLMClient) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(client.usage.__dict__, fh, indent=2)


def _fake_response(system: str, call_index: int) -> str:
    sequence = config._get("BLOGPIPE_FAKE_LLM_RESPONSES")
    if sequence:
        try:
            values = json.loads(sequence)
        except json.JSONDecodeError:
            values = []
        if isinstance(values, list) and call_index < len(values):
            return str(values[call_index])
    lower = (system or "").lower()
    if "research radar selector" in lower:
        return config._get("BLOGPIPE_FAKE_SELECTOR_RESPONSE")
    if "research radar outline" in lower:
        return config._get("BLOGPIPE_FAKE_OUTLINE_RESPONSE")
    if "repair markdown" in lower:
        return config._get("BLOGPIPE_FAKE_REPAIR_RESPONSE", config._get("BLOGPIPE_FAKE_LLM_RESPONSE"))
    return config._get("BLOGPIPE_FAKE_LLM_RESPONSE", "")
