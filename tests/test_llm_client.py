from __future__ import annotations

import httpx

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
    assert chain[-11:] == [
        "minimax/minimax-m2.5:free",
        "nvidia/nemotron-3-super-120b-a12b:free",
        "arcee-ai/trinity-large-thinking:free",
        "openai/gpt-oss-120b:free",
        "qwen/qwen3-coder:free",
        "inclusionai/ring-2.6-1t:free",
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
        "google/gemma-4-31b-it:free",
        "google/gemma-4-26b-a4b-it:free",
        "meta-llama/llama-3.3-70b-instruct:free",
        "openrouter/free",
    ]
    assert "inclusionai/ring-2.6-1t:free" in chain
    assert "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free" in chain


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
