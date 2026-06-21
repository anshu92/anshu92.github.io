from __future__ import annotations

import httpx
import pytest

from blogpipe import config
from blogpipe.llm import LLMClient, RejectedCompletionTracker, _RejectedCompletionError


class _StubResponse:
    def __init__(self, status_code: int, payload: dict | None = None):
        self.status_code = status_code
        self._payload = payload or {}
        self.is_error = status_code >= 400

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.is_error:
            req = httpx.Request("POST", "https://example.test/chat/completions")
            res = httpx.Response(self.status_code, request=req)
            raise httpx.HTTPStatusError("status error", request=req, response=res)


def test_task_chain_fallback_tries_next_model(monkeypatch):
    monkeypatch.setenv("BLOGPIPE_LLM_BASE_URL", "https://example.test")
    monkeypatch.setenv("BLOGPIPE_LLM_API_KEY", "test-key")
    monkeypatch.setenv("BLOGPIPE_LLM_MODEL", "base-model")
    monkeypatch.setenv("BLOGPIPE_LLM_CHAIN_DRAFT", "chain-a,chain-b")
    monkeypatch.delenv("BLOGPIPE_FAKE_LLM_RESPONSE", raising=False)
    monkeypatch.setattr("blogpipe.llm.time.sleep", lambda _: None)

    attempted: list[str] = []

    def _post(url, headers, json, timeout):  # noqa: ANN001
        attempted.append(str(json["model"]))
        if json["model"] == "chain-a":
            return _StubResponse(500)
        return _StubResponse(200, {"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr("blogpipe.llm.httpx.post", _post)
    llm = LLMClient()
    out = llm.complete(system="sys", user="usr", task="draft")
    assert out == "ok"
    assert attempted[:3] == ["chain-a", "chain-a", "chain-a"]
    assert attempted[-1] == "chain-b"
    assert llm.usage.calls == 1
    assert llm.usage.model_by_task["draft"] == "chain-b"


def test_task_without_chain_uses_bias_order(monkeypatch):
    monkeypatch.setenv("BLOGPIPE_LLM_BASE_URL", "https://example.test")
    monkeypatch.setenv("BLOGPIPE_LLM_API_KEY", "test-key")
    monkeypatch.setenv("BLOGPIPE_LLM_MODEL", "base-model")
    monkeypatch.setenv("BLOGPIPE_MODEL", "legacy-model")
    monkeypatch.setenv("BLOGPIPE_LLM_MODEL_FAST", "fast-model")
    monkeypatch.setenv("BLOGPIPE_LLM_MODEL_SELECTOR", "selector-model")
    monkeypatch.delenv("BLOGPIPE_FAKE_LLM_RESPONSE", raising=False)
    monkeypatch.setattr("blogpipe.llm.time.sleep", lambda _: None)

    attempted: list[str] = []

    def _post(url, headers, json, timeout):  # noqa: ANN001
        attempted.append(str(json["model"]))
        return _StubResponse(200, {"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr("blogpipe.llm.httpx.post", _post)
    llm = LLMClient()
    llm.complete(system="sys", user="usr", task="selector")
    assert attempted[0] == "selector-model"


def test_default_gemini_roster_documents_current_models():
    assert "gemini-3.5-flash" in config.DEFAULT_GEMINI_CHAIN_FAST
    assert "gemini-3.1-flash-lite" in config.DEFAULT_GEMINI_CHAIN_FAST
    assert "gemini-2.0-flash" not in config.DEFAULT_GEMINI_CHAIN_FAST
    assert "gemini-3.1-flash-lite-preview" not in config.DEFAULT_GEMINI_CHAIN_FAST
    assert config.DEFAULT_GEMINI_MODEL_FAST == "gemini-3.5-flash"
    assert config.DEFAULT_GEMINI_MODEL_SMART == "gemini-3.1-pro-preview"
    assert "gemini-2.5-pro" in config.DEFAULT_GEMINI_CHAIN_SMART


def test_default_openrouter_free_roster_documents_current_models():
    assert config.DEFAULT_OPENROUTER_FREE_MODELS[-1] == "openrouter/free"
    assert config.DEFAULT_OPENROUTER_RATE_LIMIT_FALLBACK[0] == "openrouter/free"
    assert config.DEFAULT_OPENROUTER_RATE_LIMIT_FALLBACK[1] == "meta-llama/llama-3.3-70b-instruct:free"
    assert "moonshotai/kimi-k2.6:free" not in config.DEFAULT_OPENROUTER_FREE_MODELS
    assert "cognitivecomputations/dolphin-mistral-24b-venice-edition:free" in config.DEFAULT_OPENROUTER_FREE_MODELS
    assert "poolside/laguna-xs.2:free" in config.DEFAULT_OPENROUTER_FREE_MODELS
    assert "nvidia/nemotron-3-nano-30b-a3b:free" in config.DEFAULT_OPENROUTER_FREE_MODELS


def test_openrouter_free_roster_appended_when_key_exists(monkeypatch):
    monkeypatch.setenv("BLOGPIPE_LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")
    monkeypatch.setenv("BLOGPIPE_LLM_API_KEY", "gemini-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    monkeypatch.setenv("BLOGPIPE_LLM_MODEL_FAST", "gemini-2.5-flash")
    monkeypatch.delenv("BLOGPIPE_LLM_CHAIN_OUTLINE", raising=False)
    monkeypatch.delenv("BLOGPIPE_FAKE_LLM_RESPONSE", raising=False)

    llm = LLMClient()
    chain = llm._model_chain("outline")
    assert chain[0] == "gemini-2.5-flash"
    assert "openrouter/free" in chain
    assert chain.index("openrouter/free") < chain.index("nvidia/nemotron-3-ultra-550b-a55b:free")
    assert "qwen/qwen3-next-80b-a3b-instruct:free" in chain
    assert "nvidia/nemotron-3-ultra-550b-a55b:free" in chain
    assert "cognitivecomputations/dolphin-mistral-24b-venice-edition:free" in chain
    assert "poolside/laguna-xs.2:free" in chain
    assert "moonshotai/kimi-k2.6:free" not in chain
    assert "minimax/minimax-m2.5:free" not in chain
    assert "arcee-ai/trinity-large-thinking:free" not in chain
    assert "inclusionai/ring-2.6-1t:free" not in chain


def test_openrouter_free_roster_can_be_overridden(monkeypatch):
    monkeypatch.setenv("BLOGPIPE_LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")
    monkeypatch.setenv("BLOGPIPE_LLM_API_KEY", "gemini-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    monkeypatch.setenv("BLOGPIPE_OPENROUTER_FREE_MODELS", "openai/gpt-oss-120b:free,openrouter/free")
    monkeypatch.delenv("BLOGPIPE_FAKE_LLM_RESPONSE", raising=False)

    llm = LLMClient()
    chain = llm._model_chain("outline")
    assert "openai/gpt-oss-120b:free" in chain
    assert "inclusionai/ring-2.6-1t:free" not in chain


def test_llm_uses_task_specific_timeout(monkeypatch):
    monkeypatch.setenv("BLOGPIPE_LLM_BASE_URL", "https://example.test")
    monkeypatch.setenv("BLOGPIPE_LLM_API_KEY", "test-key")
    monkeypatch.setenv("BLOGPIPE_LLM_MODEL", "base-model")
    monkeypatch.setenv("BLOGPIPE_LLM_SMART_TIMEOUT_SECONDS", "33")
    monkeypatch.delenv("BLOGPIPE_FAKE_LLM_RESPONSE", raising=False)

    seen: list[float] = []

    def _post(url, headers, json, timeout):  # noqa: ANN001
        seen.append(float(timeout.read))
        return _StubResponse(200, {"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr("blogpipe.llm.httpx.post", _post)
    llm = LLMClient()
    assert llm.complete(system="sys", user="usr", task="draft") == "ok"
    assert seen == [33.0]


def test_llm_runtime_budget_blocks_new_calls(monkeypatch):
    monkeypatch.setenv("BLOGPIPE_LLM_BASE_URL", "https://example.test")
    monkeypatch.setenv("BLOGPIPE_LLM_API_KEY", "test-key")
    monkeypatch.setenv("BLOGPIPE_LLM_MODEL", "base-model")
    monkeypatch.setenv("BLOGPIPE_LLM_MAX_RUNTIME_SECONDS", "60")
    monkeypatch.delenv("BLOGPIPE_FAKE_LLM_RESPONSE", raising=False)
    llm = LLMClient()
    monkeypatch.setattr("blogpipe.llm.time.monotonic", lambda: llm.started_at + 60.0)
    with pytest.raises(RuntimeError, match="BLOGPIPE_LLM_MAX_RUNTIME_SECONDS reached"):
        llm.complete(system="sys", user="usr", task="outline")


def test_gemini_endpoint_includes_openrouter_in_all_task_chains(monkeypatch):
    monkeypatch.setenv(
        "BLOGPIPE_LLM_BASE_URL",
        "https://generativelanguage.googleapis.com/v1beta/openai",
    )
    monkeypatch.setenv("BLOGPIPE_LLM_API_KEY", "gemini-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.setenv("BLOGPIPE_OPENROUTER_FREE_MODELS", "openrouter/free,meta-llama/llama-3.3-70b-instruct:free")
    llm = LLMClient()
    assert "openrouter/free" in llm._model_chain("draft")
    assert "openrouter/free" in llm._model_chain("editor")
    assert "openrouter/free" in llm._model_chain("outline")
    assert "openrouter/free" in llm._model_chain("repair")


def test_emergency_openrouter_models_only_after_rate_limit(monkeypatch):
    monkeypatch.setenv(
        "BLOGPIPE_LLM_BASE_URL",
        "https://generativelanguage.googleapis.com/v1beta/openai",
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    llm = LLMClient()
    models = llm._emergency_openrouter_models(
        "draft",
        tried=["gemini-3.1-pro-preview"],
        last_error=RuntimeError("retriable_status:429"),
    )
    assert models[0] == "openrouter/free"
    assert "meta-llama/llama-3.3-70b-instruct:free" in models
    assert "google/gemini-3.1-pro-preview" not in models
    assert llm._emergency_openrouter_models(
        "draft",
        tried=["gemini-3.1-pro-preview"],
        last_error=RuntimeError("http_status:400"),
    ) == []
    emergency_after_reject = llm._emergency_openrouter_models(
        "draft",
        tried=["gemini-3.1-pro-preview", "openrouter/free"],
        last_error=_RejectedCompletionError("outline_invalid", "{}"),
    )
    assert emergency_after_reject[0] == "meta-llama/llama-3.3-70b-instruct:free"
    outline_models = llm._emergency_openrouter_models(
        "outline_repair",
        tried=["gemini-3.5-flash"],
        last_error=RuntimeError("retriable_status:429"),
    )
    assert outline_models[0] == "openrouter/free"


def test_openrouter_smart_fallback_defaults_on_when_key_present(monkeypatch):
    monkeypatch.delenv("BLOGPIPE_OPENROUTER_SMART_FALLBACK", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    assert config.openrouter_smart_fallback_enabled() is True


def test_openrouter_smart_fallback_can_be_disabled(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.setenv("BLOGPIPE_OPENROUTER_SMART_FALLBACK", "0")
    assert config.openrouter_smart_fallback_enabled() is False


def test_rate_limit_on_native_gemini_tries_openrouter_free_models(monkeypatch):
    monkeypatch.setenv(
        "BLOGPIPE_LLM_BASE_URL",
        "https://generativelanguage.googleapis.com/v1beta/openai",
    )
    monkeypatch.setenv("BLOGPIPE_LLM_API_KEY", "gemini-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.setenv("BLOGPIPE_LLM_MODEL", "gemini-3.1-pro-preview")
    monkeypatch.setenv("BLOGPIPE_LLM_CHAIN_DRAFT", "gemini-3.1-pro-preview")
    monkeypatch.delenv("BLOGPIPE_FAKE_LLM_RESPONSE", raising=False)
    monkeypatch.setattr("blogpipe.llm.time.sleep", lambda _: None)

    attempted: list[str] = []

    def _post(url, headers, json, timeout):  # noqa: ANN001
        attempted.append(str(json["model"]))
        if json["model"] == "gemini-3.1-pro-preview":
            return _StubResponse(429)
        if json["model"] == "openrouter/free":
            return _StubResponse(200, {"choices": [{"message": {"content": "free ok"}}]})
        return _StubResponse(500)

    monkeypatch.setattr("blogpipe.llm.httpx.post", _post)
    llm = LLMClient()
    out = llm.complete(system="sys", user="usr", task="draft")
    assert out == "free ok"
    assert attempted[:3] == ["gemini-3.1-pro-preview"] * 3
    assert attempted[3] == "openrouter/free"


def test_rejected_completion_tracker_prefers_fewer_outline_errors():
    tracker = RejectedCompletionTracker()
    tracker.observe("worse", "outline_invalid:a,b,c,d,e")
    tracker.observe("better", "outline_invalid:a,b,c,d")
    chosen = tracker.prefer("worse", "outline_invalid:a,b,c,d,e")
    assert chosen == "better"


def test_rejected_native_completion_tries_openrouter_free_models(monkeypatch):
    monkeypatch.setenv(
        "BLOGPIPE_LLM_BASE_URL",
        "https://generativelanguage.googleapis.com/v1beta/openai",
    )
    monkeypatch.setenv("BLOGPIPE_LLM_API_KEY", "gemini-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.setenv("BLOGPIPE_LLM_CHAIN_OUTLINE", "gemini-3.5-flash")
    monkeypatch.delenv("BLOGPIPE_FAKE_LLM_RESPONSE", raising=False)
    monkeypatch.setattr("blogpipe.llm.time.sleep", lambda _: None)

    attempted: list[str] = []

    def _post(url, headers, json, timeout):  # noqa: ANN001
        attempted.append(str(json["model"]))
        if json["model"] == "gemini-3.5-flash":
            return _StubResponse(200, {"choices": [{"message": {"content": "bad outline"}}]})
        if json["model"] == "openrouter/free":
            return _StubResponse(200, {"choices": [{"message": {"content": "good outline"}}]})
        return _StubResponse(500)

    monkeypatch.setattr("blogpipe.llm.httpx.post", _post)
    llm = LLMClient()
    out = llm.complete(
        system="sys",
        user="usr",
        task="outline",
        reject_completion=lambda text: "invalid" if text == "bad outline" else None,
    )
    assert out == "good outline"
    assert attempted[0] == "gemini-3.5-flash"
    assert "openrouter/free" in attempted


def test_openrouter_models_use_longer_timeout(monkeypatch):
    monkeypatch.setenv(
        "BLOGPIPE_LLM_BASE_URL",
        "https://generativelanguage.googleapis.com/v1beta/openai",
    )
    monkeypatch.setenv("BLOGPIPE_LLM_API_KEY", "gemini-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.setenv("BLOGPIPE_LLM_CHAIN_OUTLINE", "openrouter/free")
    monkeypatch.setenv("BLOGPIPE_LLM_FAST_TIMEOUT_SECONDS", "45")
    monkeypatch.setenv("BLOGPIPE_LLM_OPENROUTER_TIMEOUT_SECONDS", "120")
    monkeypatch.delenv("BLOGPIPE_FAKE_LLM_RESPONSE", raising=False)

    seen: list[float] = []

    def _post(url, headers, json, timeout):  # noqa: ANN001
        seen.append(float(timeout.read))
        return _StubResponse(200, {"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr("blogpipe.llm.httpx.post", _post)
    llm = LLMClient()
    assert llm.complete(system="sys", user="usr", task="outline") == "ok"
    assert seen == [120.0]


def test_slow_openrouter_free_models_are_skipped_when_runtime_budget_is_low(monkeypatch):
    monkeypatch.setenv("BLOGPIPE_LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")
    monkeypatch.setenv("BLOGPIPE_LLM_API_KEY", "gemini-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.setenv(
        "BLOGPIPE_LLM_CHAIN_OUTLINE",
        "nvidia/nemotron-3-ultra-550b-a55b:free,openrouter/free",
    )
    monkeypatch.setenv("BLOGPIPE_LLM_MAX_RUNTIME_SECONDS", "1200")
    monkeypatch.setenv("BLOGPIPE_LLM_SLOW_OPENROUTER_MIN_BUDGET_SECONDS", "180")
    monkeypatch.delenv("BLOGPIPE_FAKE_LLM_RESPONSE", raising=False)

    attempted: list[str] = []

    def _post(url, headers, json, timeout):  # noqa: ANN001
        attempted.append(str(json["model"]))
        return _StubResponse(200, {"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr("blogpipe.llm.httpx.post", _post)
    llm = LLMClient()
    monkeypatch.setattr("blogpipe.llm.time.monotonic", lambda: llm.started_at + 1100.0)

    assert llm.complete(system="sys", user="usr", task="outline") == "ok"
    assert attempted == ["openrouter/free"]


def test_llm_wall_clock_timeout_skips_to_next_model(monkeypatch):
    monkeypatch.setenv("BLOGPIPE_LLM_BASE_URL", "https://example.test")
    monkeypatch.setenv("BLOGPIPE_LLM_API_KEY", "test-key")
    monkeypatch.setenv("BLOGPIPE_LLM_MODEL", "base-model")
    monkeypatch.setenv("BLOGPIPE_LLM_CHAIN_SELECTOR", "slow-model,backup-model")
    monkeypatch.delenv("BLOGPIPE_FAKE_LLM_RESPONSE", raising=False)
    monkeypatch.setattr("blogpipe.llm.time.sleep", lambda _: None)

    attempted: list[str] = []

    def _post(url, headers, json, timeout):  # noqa: ANN001
        attempted.append(str(json["model"]))
        if json["model"] == "slow-model":
            raise TimeoutError("llm_wall_clock_timeout:45.0s")
        return _StubResponse(200, {"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr("blogpipe.llm.httpx.post", _post)
    llm = LLMClient()
    assert llm.complete(system="sys", user="usr", task="selector") == "ok"
    assert attempted == ["slow-model", "backup-model"]


def test_rejected_completion_tries_next_model(monkeypatch):
    monkeypatch.setenv("BLOGPIPE_LLM_BASE_URL", "https://example.test")
    monkeypatch.setenv("BLOGPIPE_LLM_API_KEY", "test-key")
    monkeypatch.setenv("BLOGPIPE_LLM_MODEL", "base-model")
    monkeypatch.setenv("BLOGPIPE_LLM_CHAIN_DRAFT", "short-model,usable-model")
    monkeypatch.delenv("BLOGPIPE_FAKE_LLM_RESPONSE", raising=False)
    monkeypatch.setattr("blogpipe.llm.time.sleep", lambda _: None)

    attempted: list[str] = []

    def _post(url, headers, json, timeout):  # noqa: ANN001
        attempted.append(str(json["model"]))
        if json["model"] == "short-model":
            return _StubResponse(200, {"choices": [{"message": {"content": "tiny"}}]})
        return _StubResponse(200, {"choices": [{"message": {"content": "usable draft"}}]})

    monkeypatch.setattr("blogpipe.llm.httpx.post", _post)
    llm = LLMClient()
    out = llm.complete(
        system="sys",
        user="usr",
        task="draft",
        reject_completion=lambda text: "too_short" if text == "tiny" else None,
    )
    assert out == "usable draft"
    assert attempted == ["short-model", "usable-model"]
