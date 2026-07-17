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
    assert attempted[:2] == ["chain-a", "chain-a"]
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


def test_openai_api_key_defaults_to_openai_models(monkeypatch):
    monkeypatch.delenv("BLOGPIPE_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("BLOGPIPE_LLM_API_KEY", raising=False)
    monkeypatch.delenv("BLOGPIPE_LLM_MODEL", raising=False)
    monkeypatch.delenv("BLOGPIPE_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")

    llm_cfg = config.llm_config()
    llm = LLMClient(cfg=llm_cfg)

    assert llm_cfg.base_url == config.DEFAULT_OPENAI_BASE_URL
    assert llm_cfg.api_key == "openai-key"
    assert llm_cfg.model == config.DEFAULT_OPENAI_MODEL_FAST
    assert llm_cfg.model_fast == config.DEFAULT_OPENAI_MODEL_FAST
    assert llm_cfg.model_smart == config.DEFAULT_OPENAI_MODEL_SMART
    assert llm._model_chain("selector")[0] == config.DEFAULT_OPENAI_MODEL_FAST
    assert llm._model_chain("draft")[:2] == [
        config.DEFAULT_OPENAI_MODEL_SMART,
        config.DEFAULT_OPENAI_MODEL_FAST,
    ]


def test_base_url_override_uses_matching_provider_key(monkeypatch):
    monkeypatch.delenv("BLOGPIPE_LLM_API_KEY", raising=False)
    monkeypatch.delenv("BLOGPIPE_LLM_MODEL", raising=False)
    monkeypatch.delenv("BLOGPIPE_MODEL", raising=False)
    monkeypatch.setenv("BLOGPIPE_LLM_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")

    llm_cfg = config.llm_config()

    assert llm_cfg.api_key == "openrouter-key"
    assert llm_cfg.model == "openrouter/free"


def test_default_openrouter_free_roster_documents_current_models():
    assert config.DEFAULT_OPENROUTER_FREE_MODELS[-1] == "openrouter/free"
    assert config.DEFAULT_OPENROUTER_RATE_LIMIT_FALLBACK[0] == "openrouter/free"
    assert config.DEFAULT_OPENROUTER_RATE_LIMIT_FALLBACK[1] == "meta-llama/llama-3.3-70b-instruct:free"
    assert "moonshotai/kimi-k2.6:free" not in config.DEFAULT_OPENROUTER_FREE_MODELS
    assert "cognitivecomputations/dolphin-mistral-24b-venice-edition:free" in config.DEFAULT_OPENROUTER_FREE_MODELS
    assert "poolside/laguna-xs.2:free" in config.DEFAULT_OPENROUTER_FREE_MODELS
    assert "nvidia/nemotron-3-nano-30b-a3b:free" in config.DEFAULT_OPENROUTER_FREE_MODELS


def test_openrouter_free_roster_not_appended_when_key_exists(monkeypatch):
    monkeypatch.setenv("BLOGPIPE_LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")
    monkeypatch.setenv("BLOGPIPE_LLM_API_KEY", "gemini-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    monkeypatch.setenv("BLOGPIPE_LLM_MODEL_FAST", "gemini-2.5-flash")
    monkeypatch.delenv("BLOGPIPE_LLM_CHAIN_OUTLINE", raising=False)
    monkeypatch.delenv("BLOGPIPE_FAKE_LLM_RESPONSE", raising=False)

    llm = LLMClient()
    chain = llm._model_chain("outline")
    assert chain[0] == "gemini-2.5-flash"
    assert "openrouter/free" not in chain
    assert "qwen/qwen3-next-80b-a3b-instruct:free" not in chain


def test_gemini_api_key_defaults_to_gemini_primary_without_openrouter_fallback(monkeypatch):
    monkeypatch.delenv("BLOGPIPE_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("BLOGPIPE_LLM_API_KEY", raising=False)
    monkeypatch.delenv("BLOGPIPE_LLM_MODEL", raising=False)
    monkeypatch.delenv("BLOGPIPE_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")

    llm_cfg = config.llm_config()
    assert "generativelanguage.googleapis.com" in llm_cfg.base_url
    assert llm_cfg.api_key == "gemini-key"
    assert llm_cfg.model == config.DEFAULT_GEMINI_MODEL_FAST
    assert llm_cfg.model_fast == config.DEFAULT_GEMINI_MODEL_FAST
    assert llm_cfg.model_smart == config.DEFAULT_GEMINI_MODEL_SMART
    assert llm_cfg.openrouter_api_key == "openrouter-key"
    assert LLMClient(cfg=llm_cfg)._model_chain("outline") == [config.DEFAULT_GEMINI_MODEL_FAST]


def test_openrouter_free_roster_override_is_not_automatic_fallback(monkeypatch):
    monkeypatch.setenv("BLOGPIPE_LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")
    monkeypatch.setenv("BLOGPIPE_LLM_API_KEY", "gemini-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    monkeypatch.setenv("BLOGPIPE_OPENROUTER_FREE_MODELS", "openai/gpt-oss-120b:free,openrouter/free")
    monkeypatch.delenv("BLOGPIPE_FAKE_LLM_RESPONSE", raising=False)

    llm = LLMClient()
    chain = llm._model_chain("outline")
    assert "openai/gpt-oss-120b:free" not in chain
    assert "openrouter/free" not in chain
    assert "inclusionai/ring-2.6-1t:free" not in chain


def test_dynamic_openrouter_free_roster_ranks_free_text_models(monkeypatch):
    monkeypatch.setattr(config, "_OPENROUTER_DYNAMIC_FREE_MODELS_CACHE", None)
    monkeypatch.setenv("BLOGPIPE_OPENROUTER_DYNAMIC_FREE_MODELS", "1")
    monkeypatch.setenv("BLOGPIPE_OPENROUTER_DYNAMIC_FREE_MODEL_LIMIT", "3")
    monkeypatch.setenv("BLOGPIPE_OPENROUTER_MODELS_TIMEOUT_SECONDS", "2")

    payload = {
        "data": [
            {
                "id": "nvidia/nemotron-3.5-content-safety:free",
                "name": "NVIDIA Content Safety (free)",
                "description": "guardrail model",
                "created": 1780581864,
                "context_length": 128000,
                "architecture": {"input_modalities": ["text"], "output_modalities": ["text"], "modality": "text->text"},
                "pricing": {"prompt": "0", "completion": "0"},
                "benchmarks": {"artificial_analysis": {"agentic_index": 90, "coding_index": 90, "intelligence_index": 90}},
                "supported_parameters": ["max_tokens"],
            },
            {
                "id": "cohere/north-mini-code:free",
                "name": "Cohere North Mini Code (free)",
                "description": "agentic coding model",
                "created": 1781723748,
                "context_length": 256000,
                "architecture": {"input_modalities": ["text"], "output_modalities": ["text"], "modality": "text->text"},
                "pricing": {"prompt": "0", "completion": "0"},
                "benchmarks": {"artificial_analysis": {"agentic_index": 45, "coding_index": 70, "intelligence_index": 55}},
                "supported_parameters": ["max_tokens", "structured_outputs", "tools"],
            },
            {
                "id": "older/good-free:free",
                "name": "Older Good Free",
                "description": "general model",
                "created": 1770000000,
                "context_length": 64000,
                "architecture": {"input_modalities": ["text"], "output_modalities": ["text"], "modality": "text->text"},
                "pricing": {"prompt": "0", "completion": "0"},
                "benchmarks": {"artificial_analysis": {"agentic_index": 30, "coding_index": 50, "intelligence_index": 40}},
                "supported_parameters": ["max_tokens"],
            },
            {
                "id": "paid/high-performer",
                "name": "Paid High Performer",
                "created": 1781723748,
                "context_length": 1000000,
                "architecture": {"input_modalities": ["text"], "output_modalities": ["text"], "modality": "text->text"},
                "pricing": {"prompt": "0.1", "completion": "0.1"},
                "benchmarks": {"artificial_analysis": {"agentic_index": 100, "coding_index": 100, "intelligence_index": 100}},
                "supported_parameters": ["max_tokens"],
            },
        ]
    }

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return payload

    monkeypatch.setattr("blogpipe.config.httpx.get", lambda *args, **kwargs: Response())

    models = config.discover_openrouter_free_models(base_url="https://openrouter.ai/api/v1", api_key="key")

    assert models[:2] == ["cohere/north-mini-code:free", "older/good-free:free"]
    assert "openrouter/free" in models
    assert "paid/high-performer" not in models
    assert "nvidia/nemotron-3.5-content-safety:free" not in models


def test_dynamic_openrouter_discovery_falls_back_to_static_roster(monkeypatch):
    monkeypatch.setattr(config, "_OPENROUTER_DYNAMIC_FREE_MODELS_CACHE", None)
    monkeypatch.setenv("BLOGPIPE_OPENROUTER_DYNAMIC_FREE_MODELS", "1")

    def _raise(*args, **kwargs):
        raise RuntimeError("network unavailable")

    monkeypatch.setattr("blogpipe.config.httpx.get", _raise)

    models = config.discover_openrouter_free_models(base_url="https://openrouter.ai/api/v1", api_key="")
    llm_cfg = config.llm_config()

    assert models == []
    assert llm_cfg.openrouter_free_models == []


def test_dynamic_openrouter_discovery_can_feed_llm_config(monkeypatch):
    monkeypatch.setattr(config, "_OPENROUTER_DYNAMIC_FREE_MODELS_CACHE", None)
    monkeypatch.setenv("BLOGPIPE_LLM_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("BLOGPIPE_LLM_API_KEY", "or-key")
    monkeypatch.setenv("BLOGPIPE_OPENROUTER_DYNAMIC_FREE_MODELS", "1")
    monkeypatch.delenv("BLOGPIPE_OPENROUTER_FREE_MODELS", raising=False)

    payload = {
        "data": [
            {
                "id": "cohere/north-mini-code:free",
                "name": "Cohere North Mini Code (free)",
                "created": 1781723748,
                "context_length": 256000,
                "architecture": {"input_modalities": ["text"], "output_modalities": ["text"], "modality": "text->text"},
                "pricing": {"prompt": "0", "completion": "0"},
                "benchmarks": {"artificial_analysis": {"agentic_index": 45, "coding_index": 70, "intelligence_index": 55}},
                "supported_parameters": ["max_tokens", "structured_outputs", "tools"],
            }
        ]
    }

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return payload

    monkeypatch.setattr("blogpipe.config.httpx.get", lambda *args, **kwargs: Response())

    llm_cfg = config.llm_config()

    assert llm_cfg.openrouter_free_models[0] == "cohere/north-mini-code:free"
    assert "openrouter/free" in llm_cfg.openrouter_free_models


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


def test_gemini_endpoint_does_not_include_openrouter_in_task_chains(monkeypatch):
    monkeypatch.setenv(
        "BLOGPIPE_LLM_BASE_URL",
        "https://generativelanguage.googleapis.com/v1beta/openai",
    )
    monkeypatch.setenv("BLOGPIPE_LLM_API_KEY", "gemini-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.setenv("BLOGPIPE_OPENROUTER_FREE_MODELS", "openrouter/free,meta-llama/llama-3.3-70b-instruct:free")
    llm = LLMClient()
    assert "openrouter/free" not in llm._model_chain("draft")
    assert "openrouter/free" not in llm._model_chain("editor")
    assert "openrouter/free" not in llm._model_chain("outline")
    assert "openrouter/free" not in llm._model_chain("repair")


def test_emergency_openrouter_models_are_disabled(monkeypatch):
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
    assert models == []
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
    assert emergency_after_reject == []
    outline_models = llm._emergency_openrouter_models(
        "outline_repair",
        tried=["gemini-3.5-flash"],
        last_error=RuntimeError("retriable_status:429"),
    )
    assert outline_models == []


def test_openrouter_smart_fallback_defaults_off_when_key_present(monkeypatch):
    monkeypatch.delenv("BLOGPIPE_OPENROUTER_SMART_FALLBACK", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    assert config.openrouter_smart_fallback_enabled() is False


def test_openrouter_smart_fallback_can_be_disabled(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.setenv("BLOGPIPE_OPENROUTER_SMART_FALLBACK", "0")
    assert config.openrouter_smart_fallback_enabled() is False


def test_rate_limit_on_native_gemini_does_not_try_openrouter_free_models(monkeypatch):
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
    with pytest.raises(RuntimeError, match="LLM completion failed"):
        llm.complete(system="sys", user="usr", task="draft")
    assert attempted[:2] == ["gemini-3.1-pro-preview"] * 2
    assert "openrouter/free" not in attempted


def test_native_rate_limit_does_not_enter_openrouter_circuit(monkeypatch):
    monkeypatch.setenv(
        "BLOGPIPE_LLM_BASE_URL",
        "https://generativelanguage.googleapis.com/v1beta/openai",
    )
    monkeypatch.setenv("BLOGPIPE_LLM_API_KEY", "gemini-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.setenv("BLOGPIPE_LLM_MODEL", "gemini-a")
    monkeypatch.setenv("BLOGPIPE_LLM_CHAIN_DRAFT", "gemini-a,gemini-b")
    monkeypatch.setenv("BLOGPIPE_OPENROUTER_RATE_LIMIT_FALLBACK_LIMIT", "4")
    monkeypatch.setenv("BLOGPIPE_OPENROUTER_RATE_LIMIT_CIRCUIT_BREAKER_HITS", "2")
    monkeypatch.delenv("BLOGPIPE_FAKE_LLM_RESPONSE", raising=False)
    monkeypatch.setattr("blogpipe.llm.time.sleep", lambda _: None)

    attempted: list[str] = []

    def _post(url, headers, json, timeout):  # noqa: ANN001
        model = str(json["model"])
        attempted.append(model)
        if model.startswith("gemini-"):
            return _StubResponse(429)
        return _StubResponse(429)

    monkeypatch.setattr("blogpipe.llm.httpx.post", _post)
    llm = LLMClient()
    with pytest.raises(RuntimeError, match="LLM completion failed"):
        llm.complete(system="sys", user="usr", task="draft")

    openrouter_attempts = [model for model in attempted if model == "openrouter/free" or model.endswith(":free")]
    assert openrouter_attempts == []
    assert llm._openrouter_rate_limit_hits == 0
    assert attempted.count("gemini-a") == 2
    assert attempted.count("gemini-b") == 2


def test_task_chains_remain_task_scoped_without_openrouter_fallback(monkeypatch):
    monkeypatch.setenv(
        "BLOGPIPE_LLM_BASE_URL",
        "https://generativelanguage.googleapis.com/v1beta/openai",
    )
    monkeypatch.setenv("BLOGPIPE_LLM_API_KEY", "gemini-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.setenv("BLOGPIPE_LLM_MODEL", "gemini-a")
    monkeypatch.setenv("BLOGPIPE_LLM_CHAIN_OUTLINE_REPAIR", "gemini-a")
    monkeypatch.setenv("BLOGPIPE_LLM_CHAIN_DRAFT", "draft-ok")
    monkeypatch.setenv("BLOGPIPE_OPENROUTER_RATE_LIMIT_CIRCUIT_BREAKER_HITS", "2")
    monkeypatch.delenv("BLOGPIPE_FAKE_LLM_RESPONSE", raising=False)
    monkeypatch.setattr("blogpipe.llm.time.sleep", lambda _: None)

    attempted: list[tuple[str, str]] = []

    def _post(url, headers, json, timeout):  # noqa: ANN001
        model = str(json["model"])
        task = "outline_repair" if model == "gemini-a" else "draft"
        attempted.append((task, model))
        if model.startswith("gemini-"):
            return _StubResponse(429)
        return _StubResponse(200, {"choices": [{"message": {"content": "draft ok"}}]})

    monkeypatch.setattr("blogpipe.llm.httpx.post", _post)
    llm = LLMClient()
    with pytest.raises(RuntimeError, match="LLM completion failed"):
        llm.complete(system="sys", user="usr", task="outline_repair")

    out = llm.complete(system="sys", user="usr", task="draft")
    assert out == "draft ok"
    assert llm._openrouter_rate_limit_hits_by_task.get("outline_repair", 0) == 0
    assert llm._openrouter_rate_limit_hits_by_task.get("draft", 0) == 0
    assert ("draft", "draft-ok") in attempted


def test_rejected_completion_tracker_prefers_fewer_outline_errors():
    tracker = RejectedCompletionTracker()
    tracker.observe("worse", "outline_invalid:a,b,c,d,e")
    tracker.observe("better", "outline_invalid:a,b,c,d")
    chosen = tracker.prefer("worse", "outline_invalid:a,b,c,d,e")
    assert chosen == "better"


def test_rejected_native_completion_does_not_try_openrouter_free_models(monkeypatch):
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
    assert out == "bad outline"
    assert attempted[0] == "gemini-3.5-flash"
    assert "openrouter/free" not in attempted


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
