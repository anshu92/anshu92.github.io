from __future__ import annotations

import json
import logging
import signal
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import httpx

from . import config

LOG = logging.getLogger(__name__)
RETRY_STATUS_CODES = {429, 500, 502, 503, 504}
FAST_TASKS = {"selector", "outline", "outline_repair", "quality_review", "repair"}
SMART_TASKS = {"draft", "draft_section", "editor"}


@dataclass
class LLMUsage:
    calls: int = 0
    failures: int = 0
    prompt_tokens_est: int = 0
    completion_tokens_est: int = 0
    model: str = ""
    calls_by_task: dict[str, int] = field(default_factory=dict)
    model_by_task: dict[str, str] = field(default_factory=dict)


@dataclass
class LLMClient:
    cfg: config.LLMConfig = field(default_factory=config.llm_config)
    usage: LLMUsage = field(default_factory=LLMUsage)
    started_at: float = field(default_factory=time.monotonic)

    def configured(self) -> bool:
        return bool(self.cfg.api_key and self.cfg.model and self.cfg.max_calls > 0)

    def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int | None = None,
        task: str | None = None,
    ) -> str:
        fake = _fake_response(system, self.usage.calls)
        task_name = (task or "default").strip().lower() or "default"
        model_chain = self._model_chain(task_name)
        model = model_chain[0]
        if fake:
            self.usage.calls += 1
            self.usage.model = model
            self.usage.calls_by_task[task_name] = self.usage.calls_by_task.get(task_name, 0) + 1
            self.usage.model_by_task[task_name] = model
            return fake
        if not self.configured():
            raise RuntimeError("BLOGPIPE_LLM_API_KEY is required for non-dry-run writing")
        if self.usage.calls >= self.cfg.max_calls:
            raise RuntimeError("BLOGPIPE_LLM_MAX_CALLS reached")
        if self._runtime_exhausted():
            raise RuntimeError("BLOGPIPE_LLM_MAX_RUNTIME_SECONDS reached")
        self.usage.calls += 1
        self.usage.model = model
        self.usage.calls_by_task[task_name] = self.usage.calls_by_task.get(task_name, 0) + 1
        self.usage.model_by_task[task_name] = model
        self.usage.prompt_tokens_est += (len(system) + len(user)) // 4
        last_error: Exception | None = None
        for chain_model in model_chain:
            base_url, api_key = self._endpoint_for_model(chain_model)
            if not api_key:
                last_error = RuntimeError(f"missing_api_key:{chain_model}")
                LOG.warning("llm task=%s model=%s missing api key; trying next model", task_name, chain_model)
                continue
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": chain_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": self.cfg.temperature,
                "max_tokens": max_tokens or self.cfg.max_tokens,
            }
            for attempt in range(3):
                remaining = self._remaining_runtime_seconds()
                if remaining <= 1.0:
                    last_error = RuntimeError("llm_runtime_budget_exhausted")
                    break
                try:
                    task_timeout = self._task_timeout_seconds(task_name)
                    read_timeout = max(1.0, min(task_timeout, remaining))
                    with _wall_clock_deadline(read_timeout):
                        resp = httpx.post(
                            f"{base_url}/chat/completions",
                            headers=headers,
                            json=payload,
                            timeout=httpx.Timeout(read_timeout, connect=min(10.0, read_timeout)),
                        )
                    if resp.status_code in RETRY_STATUS_CODES and attempt < 2:
                        time.sleep(2 ** attempt)
                        continue
                    if resp.status_code in RETRY_STATUS_CODES:
                        last_error = RuntimeError(f"retriable_status:{resp.status_code}")
                        LOG.warning(
                            "llm task=%s model=%s exhausted retries status=%s; trying next model",
                            task_name,
                            chain_model,
                            resp.status_code,
                        )
                        break
                    if resp.is_error:
                        last_error = RuntimeError(f"http_status:{resp.status_code}")
                        if _can_fallback_status(resp.status_code):
                            LOG.warning(
                                "llm task=%s model=%s failed status=%s; trying next model",
                                task_name,
                                chain_model,
                                resp.status_code,
                            )
                            break
                        resp.raise_for_status()
                    data = resp.json()
                    text = data["choices"][0]["message"]["content"]
                    self.usage.model = chain_model
                    self.usage.model_by_task[task_name] = chain_model
                    self.usage.completion_tokens_est += len(text) // 4
                    return text
                except Exception as exc:
                    last_error = exc
                    if isinstance(exc, TimeoutError):
                        LOG.warning(
                            "llm task=%s model=%s exceeded wall-clock timeout (%s); trying next model",
                            task_name,
                            chain_model,
                            exc,
                        )
                        break
                    if attempt < 2:
                        time.sleep(2 ** attempt)
                        continue
                    LOG.warning(
                        "llm task=%s model=%s failed after retries (%s); trying next model",
                        task_name,
                        chain_model,
                        exc,
                    )
                    break
        self.usage.failures += 1
        raise RuntimeError(
            f"LLM completion failed for task={task_name}; chain={model_chain}; last_error={last_error}"
        ) from last_error

    def _remaining_runtime_seconds(self) -> float:
        return max(0.0, self.cfg.max_runtime_seconds - (time.monotonic() - self.started_at))

    def _runtime_exhausted(self) -> bool:
        return self._remaining_runtime_seconds() <= 1.0

    def _task_timeout_seconds(self, task: str) -> float:
        return self.cfg.smart_timeout_seconds if task in SMART_TASKS else self.cfg.fast_timeout_seconds

    def _model_chain(self, task: str) -> list[str]:
        fallback = self._fallback_chain(task)
        explicit = self.cfg.model_chain_by_task.get(task, [])
        if explicit:
            return _unique_models([*explicit, *fallback])
        if self.cfg.model_chain_default:
            preferred = [self.cfg.model_by_task.get(task, ""), *self.cfg.model_chain_default]
            return _unique_models([*preferred, *fallback])
        return _unique_models(fallback)

    def _fallback_chain(self, task: str) -> list[str]:
        task_override = self.cfg.model_by_task.get(task, "")
        bias = ""
        if task in SMART_TASKS:
            bias = self.cfg.model_smart
        elif task in FAST_TASKS:
            bias = self.cfg.model_fast
        return [
            task_override,
            bias,
            self.cfg.model_primary,
            self.cfg.model_legacy,
            self._base_model(),
            *self._openrouter_fallback_models(),
        ]

    def _base_model(self) -> str:
        return "" if self.cfg.model == "openrouter/free" and self.cfg.openrouter_api_key else self.cfg.model

    def _openrouter_fallback_models(self) -> list[str]:
        if self.cfg.openrouter_api_key or "openrouter" in self.cfg.base_url.lower():
            return list(self.cfg.openrouter_free_models)
        return []

    def _endpoint_for_model(self, model: str) -> tuple[str, str]:
        if _is_openrouter_model(model):
            key = self.cfg.openrouter_api_key
            if not key and "openrouter" in self.cfg.base_url.lower():
                key = self.cfg.api_key
            return self.cfg.openrouter_base_url, key
        return self.cfg.base_url, self.cfg.api_key


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
    if "quality review" in lower:
        return config._get("BLOGPIPE_FAKE_QUALITY_RESPONSE")
    return config._get("BLOGPIPE_FAKE_LLM_RESPONSE", "")


def _unique_models(models: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in models:
        model = (raw or "").strip()
        if not model or model in seen:
            continue
        seen.add(model)
        out.append(model)
    return out


def _is_openrouter_model(model: str) -> bool:
    normalized = (model or "").strip().lower()
    return normalized.startswith("openrouter/") or "/" in normalized


def _can_fallback_status(status_code: int) -> bool:
    if status_code in RETRY_STATUS_CODES:
        return True
    if status_code in {401, 403}:
        return False
    return status_code >= 400


@contextmanager
def _wall_clock_deadline(seconds: float):
    # httpx read timeouts are inactivity timers, not whole-request deadlines.
    # CI runs on Unix in the main thread, so SIGALRM gives us a hard ceiling for
    # providers that slowly stream or hold the response open.
    if (
        seconds <= 0
        or threading.current_thread() is not threading.main_thread()
        or not hasattr(signal, "SIGALRM")
        or not hasattr(signal, "setitimer")
    ):
        yield
        return

    def _raise_timeout(_signum, _frame):  # noqa: ANN001
        raise TimeoutError(f"llm_wall_clock_timeout:{seconds:.1f}s")

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)
