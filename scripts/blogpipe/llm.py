from __future__ import annotations

import json
import logging
import signal
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable

import httpx

from . import config

LOG = logging.getLogger(__name__)
RETRY_STATUS_CODES = {429, 500, 502, 503, 504}
FAST_TASKS = {"selector", "outline", "outline_repair", "quality_review", "repair"}
SMART_TASKS = {"draft", "draft_section", "editor"}
CompletionRejector = Callable[[str], str | None]


class _RejectedCompletionError(RuntimeError):
    def __init__(self, reason: str, text: str) -> None:
        self.reason = reason
        self.text = text
        super().__init__(f"rejected_completion:{reason}")


@dataclass
class RejectedCompletionTracker:
    best_text: str = ""
    best_score: int = 1_000_000

    def observe(self, text: str, reason: str) -> None:
        score = _rejection_error_count(reason)
        if score < self.best_score:
            self.best_score = score
            self.best_text = text

    def prefer(self, fallback_text: str, fallback_reason: str) -> str:
        fallback_score = _rejection_error_count(fallback_reason)
        if self.best_text and self.best_score < fallback_score:
            LOG.warning(
                "llm using best rejected completion (score=%s vs %s)",
                self.best_score,
                fallback_score,
            )
            return self.best_text
        return fallback_text


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
    _rate_limit_hits: int = field(default=0, init=False, repr=False)

    def configured(self) -> bool:
        return bool(self.cfg.api_key and self.cfg.model and self.cfg.max_calls > 0)

    def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int | None = None,
        task: str | None = None,
        reject_completion: CompletionRejector | None = None,
        rejected_tracker: RejectedCompletionTracker | None = None,
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
        last_rejected_text = ""
        tried_models: list[str] = []
        for chain_model in model_chain:
            if self._should_skip_slow_openrouter_model(chain_model, task_name):
                continue
            tried_models.append(chain_model)
            text, last_error = self._complete_with_model(
                chain_model=chain_model,
                task_name=task_name,
                system=system,
                user=user,
                max_tokens=max_tokens,
                reject_completion=reject_completion,
            )
            if text is not None:
                return text
            if isinstance(last_error, _RejectedCompletionError):
                last_rejected_text = last_error.text
                if rejected_tracker is not None:
                    rejected_tracker.observe(last_error.text, last_error.reason)
            mirror_models = self._openrouter_free_models_after_native_failure(
                chain_model,
                last_error,
                tried=tried_models,
            )
            for mirror in mirror_models:
                if self._should_skip_slow_openrouter_model(mirror, task_name):
                    continue
                tried_models.append(mirror)
                LOG.warning(
                    "llm task=%s native model=%s failed (%s); trying openrouter free model=%s",
                    task_name,
                    chain_model,
                    last_error,
                    mirror,
                )
                text, last_error = self._complete_with_model(
                    chain_model=mirror,
                    task_name=task_name,
                    system=system,
                    user=user,
                    max_tokens=max_tokens,
                    reject_completion=reject_completion,
                )
                if text is not None:
                    return text
                if isinstance(last_error, _RejectedCompletionError):
                    last_rejected_text = last_error.text
                    if rejected_tracker is not None:
                        rejected_tracker.observe(last_error.text, last_error.reason)
            self._pause_after_rate_limit(last_error)
        emergency_chain = self._emergency_openrouter_models(task_name, tried=tried_models, last_error=last_error)
        for chain_model in emergency_chain:
            if self._should_skip_slow_openrouter_model(chain_model, task_name):
                continue
            tried_models.append(chain_model)
            text, last_error = self._complete_with_model(
                chain_model=chain_model,
                task_name=task_name,
                system=system,
                user=user,
                max_tokens=max_tokens,
                reject_completion=reject_completion,
            )
            if text is not None:
                return text
            if isinstance(last_error, _RejectedCompletionError):
                last_rejected_text = last_error.text
                if rejected_tracker is not None:
                    rejected_tracker.observe(last_error.text, last_error.reason)
            self._pause_after_rate_limit(last_error)
        if last_rejected_text:
            fallback_reason = last_error.reason if isinstance(last_error, _RejectedCompletionError) else ""
            if rejected_tracker is not None:
                last_rejected_text = rejected_tracker.prefer(last_rejected_text, fallback_reason)
            LOG.warning(
                "llm task=%s all models returned rejected completions; returning last completion for validation",
                task_name,
            )
            return last_rejected_text
        self.usage.failures += 1
        raise RuntimeError(
            f"LLM completion failed for task={task_name}; chain={model_chain}; last_error={last_error}"
        ) from last_error

    def _complete_with_model(
        self,
        *,
        chain_model: str,
        task_name: str,
        system: str,
        user: str,
        max_tokens: int | None,
        reject_completion: CompletionRejector | None,
    ) -> tuple[str | None, Exception | None]:
        if self._should_skip_slow_openrouter_model(chain_model, task_name):
            return None, RuntimeError("llm_runtime_budget_low_for_slow_model")
        base_url, api_key = self._endpoint_for_model(chain_model)
        if not api_key:
            return None, RuntimeError(f"missing_api_key:{chain_model}")
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
        last_error: Exception | None = None
        for attempt in range(3):
            remaining = self._remaining_runtime_seconds()
            if remaining <= 1.0:
                return None, RuntimeError("llm_runtime_budget_exhausted")
            try:
                task_timeout = self._task_timeout_seconds(task_name, chain_model)
                read_timeout = max(1.0, min(task_timeout, remaining))
                with _wall_clock_deadline(read_timeout):
                    resp = httpx.post(
                        f"{base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=httpx.Timeout(read_timeout, connect=min(10.0, read_timeout)),
                    )
                if resp.status_code in RETRY_STATUS_CODES and attempt < 2:
                    delay = 2 ** attempt
                    if resp.status_code in {429, 503}:
                        self._rate_limit_hits += 1
                        delay = min(45.0, 8.0 * (attempt + 1))
                    time.sleep(delay)
                    continue
                if resp.status_code in RETRY_STATUS_CODES:
                    last_error = RuntimeError(f"retriable_status:{resp.status_code}")
                    if resp.status_code in {429, 503}:
                        self._rate_limit_hits += 1
                    LOG.warning(
                        "llm task=%s model=%s exhausted retries status=%s; trying next model",
                        task_name,
                        chain_model,
                        resp.status_code,
                    )
                    return None, last_error
                if resp.is_error:
                    last_error = RuntimeError(f"http_status:{resp.status_code}")
                    if _can_fallback_status(resp.status_code):
                        LOG.warning(
                            "llm task=%s model=%s failed status=%s; trying next model",
                            task_name,
                            chain_model,
                            resp.status_code,
                        )
                        return None, last_error
                    resp.raise_for_status()
                data = resp.json()
                text = data["choices"][0]["message"]["content"]
                if text is None:
                    last_error = RuntimeError("empty_completion_content")
                    LOG.warning(
                        "llm task=%s model=%s returned empty content; trying next model",
                        task_name,
                        chain_model,
                    )
                    return None, last_error
                text = str(text)
                self.usage.model = chain_model
                self.usage.model_by_task[task_name] = chain_model
                self.usage.completion_tokens_est += len(text) // 4
                if reject_completion is not None:
                    rejection_reason = reject_completion(text)
                    if rejection_reason:
                        last_error = _RejectedCompletionError(rejection_reason, text)
                        LOG.warning(
                            "llm task=%s model=%s returned rejected completion (%s); trying next model",
                            task_name,
                            chain_model,
                            rejection_reason,
                        )
                        return None, last_error
                return text, None
            except Exception as exc:
                last_error = exc
                if isinstance(exc, TimeoutError):
                    LOG.warning(
                        "llm task=%s model=%s exceeded wall-clock timeout (%s); trying next model",
                        task_name,
                        chain_model,
                        exc,
                    )
                    return None, last_error
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                LOG.warning(
                    "llm task=%s model=%s failed after retries (%s); trying next model",
                    task_name,
                    chain_model,
                    exc,
                )
                return None, last_error
        return None, last_error

    def _pause_after_rate_limit(self, last_error: Exception | None) -> None:
        if not _is_rate_limit_error(last_error):
            return
        cooldown = config.llm_rate_limit_cooldown_seconds()
        if cooldown <= 0:
            return
        remaining = self._remaining_runtime_seconds()
        if remaining <= cooldown + 5.0:
            return
        LOG.warning("llm pausing %.0fs after rate limit before next model", cooldown)
        time.sleep(cooldown)

    def _remaining_runtime_seconds(self) -> float:
        return max(0.0, self.cfg.max_runtime_seconds - (time.monotonic() - self.started_at))

    def _runtime_exhausted(self) -> bool:
        return self._remaining_runtime_seconds() <= 1.0

    def _task_timeout_seconds(self, task: str, model: str = "") -> float:
        base = self.cfg.smart_timeout_seconds if task in SMART_TASKS else self.cfg.fast_timeout_seconds
        if _is_openrouter_model(model):
            return max(base, self.cfg.openrouter_timeout_seconds)
        return base

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
            *self._openrouter_fallback_models(task),
        ]

    def _base_model(self) -> str:
        return "" if self.cfg.model == "openrouter/free" and self.cfg.openrouter_api_key else self.cfg.model

    def _openrouter_fallback_models(self, task: str) -> list[str]:
        if not (self.cfg.openrouter_api_key or "openrouter" in self.cfg.base_url.lower()):
            return []
        if _gemini_native_endpoint(self.cfg.base_url) and config.openrouter_smart_fallback_enabled():
            return config.openrouter_chain_models(custom=self.cfg.openrouter_free_models)
        return list(self.cfg.openrouter_free_models)

    def _emergency_openrouter_models(
        self,
        task: str,
        *,
        tried: list[str],
        last_error: Exception | None,
    ) -> list[str]:
        if not self.cfg.openrouter_api_key or not _gemini_native_endpoint(self.cfg.base_url):
            return []
        if not config.openrouter_smart_fallback_enabled():
            return []
        if not _openrouter_fallback_eligible_error(last_error):
            return []
        seen = {model for model in tried if model}
        out: list[str] = []
        for model in config.DEFAULT_OPENROUTER_SMART_EMERGENCY:
            if model not in seen:
                out.append(model)
        if not out:
            for model in self.cfg.openrouter_free_models[:3]:
                if model not in seen:
                    out.append(model)
        if out:
            LOG.warning(
                "llm task=%s native chain exhausted (%s); trying emergency openrouter models=%s",
                task,
                last_error,
                out,
            )
        return out

    def _openrouter_free_models_after_native_failure(
        self,
        model: str,
        error: Exception | None,
        *,
        tried: list[str],
    ) -> list[str]:
        if not _openrouter_fallback_eligible_error(error):
            return []
        if not self.cfg.openrouter_api_key or not _gemini_native_endpoint(self.cfg.base_url):
            return []
        if not config.openrouter_smart_fallback_enabled():
            return []
        if _is_openrouter_model(model):
            return []
        return config.openrouter_free_models_after_rate_limit(
            config.DEFAULT_OPENROUTER_RATE_LIMIT_FALLBACK,
            tried=tried,
            limit=config.openrouter_rate_limit_fallback_limit(),
        )

    def _should_skip_slow_openrouter_model(self, model: str, task_name: str) -> bool:
        normalized = (model or "").strip().lower()
        if normalized not in config.SLOW_OPENROUTER_MODELS:
            return False
        remaining = self._remaining_runtime_seconds()
        min_budget = config.llm_slow_openrouter_min_budget_seconds()
        if remaining >= min_budget:
            return False
        LOG.warning(
            "llm task=%s skipping slow openrouter model=%s; %.0fs runtime remaining (<%.0fs)",
            task_name,
            model,
            remaining,
            min_budget,
        )
        return True

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


def _gemini_native_endpoint(base_url: str) -> bool:
    return "generativelanguage.googleapis.com" in (base_url or "").lower()


def _is_openrouter_model(model: str) -> bool:
    normalized = (model or "").strip().lower()
    return normalized.startswith("openrouter/") or "/" in normalized


def _is_rate_limit_error(error: Exception | None) -> bool:
    if error is None:
        return False
    message = str(error).lower()
    return any(token in message for token in ("429", "503", "retriable_status:429", "retriable_status:503"))


def _openrouter_fallback_eligible_error(error: Exception | None) -> bool:
    if error is None:
        return False
    if isinstance(error, _RejectedCompletionError):
        return True
    if isinstance(error, TimeoutError):
        return True
    return _is_rate_limit_error(error)


def _rejection_error_count(reason: str) -> int:
    if reason.startswith("outline_invalid:"):
        payload = reason.split("outline_invalid:", 1)[1]
        return len([part for part in payload.split(",") if part.strip()])
    return 1_000_000


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
