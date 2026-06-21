from __future__ import annotations

import httpx
import pytest

from blogpipe import config
from blogpipe.llm import LLMClient


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
    assert config.DEFAULT_GEMINI_MODEL_SMART == "gemini-3.1-pro-preview"
    assert "gemini-2.5-pro" in config.DEFAULT_GEMINI_CHAIN_SMART


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
    assert chain[-1] == "openrouter/free"
    assert "qwen/qwen3-next-80b-a3b-instruct:free" in chain
    assert "nvidia/nemotron-3-ultra-550b-a55b:free" in chain
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


def test_gemini_endpoint_skips_openrouter_for_smart_tasks(monkeypatch):
    monkeypatch.setenv(
        "BLOGPIPE_LLM_BASE_URL",
        "https://generativelanguage.googleapis.com/v1beta/openai",
    )
    monkeypatch.setenv("BLOGPIPE_LLM_API_KEY", "gemini-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.setenv("BLOGPIPE_OPENROUTER_FREE_MODELS", "openrouter/free,meta-llama/llama-3.3-70b-instruct:free")
    llm = LLMClient()
    assert "openrouter/free" not in llm._model_chain("draft")
    assert "openrouter/free" in llm._model_chain("outline")
    assert "openrouter/free" in llm._model_chain("repair")


def test_emergency_openrouter_models_only_after_rate_limit(monkeypatch):
    monkeypatch.setenv(
        "BLOGPIPE_LLM_BASE_URL",
        "https://generativelanguage.googleapis.com/v1beta/openai",
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.setenv("BLOGPIPE_OPENROUTER_SMART_FALLBACK", "1")
    llm = LLMClient()
    models = llm._emergency_openrouter_models(
        "draft",
        tried=["gemini-3.1-pro-preview"],
        last_error=RuntimeError("retriable_status:429"),
    )
    assert models[0] == "qwen/qwen3-next-80b-a3b-instruct:free"
    assert len(models) == 3
    assert llm._emergency_openrouter_models(
        "draft",
        tried=["gemini-3.1-pro-preview"],
        last_error=RuntimeError("http_status:400"),
    ) == []


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
