from __future__ import annotations

import unittest

from unittest.mock import MagicMock

from blogpipe.llm_chain import (
    _BLACKLIST_TTL_DAILY,
    _BLACKLIST_TTL_MED,
    _BLACKLIST_TTL_SHORT,
    _blacklist_ttl_for_body,
    _retry_delay_seconds_429,
)


class BlacklistTtlTests(unittest.TestCase):
    def test_groq_model_not_found_is_daily(self) -> None:
        body = (
            '{"error":{"message":"The model `moonshotai/kimi-k2-instruct` does not exist '
            'or you do not have access to it.","type":"invalid_request_error",'
            '"code":"model_not_found"}}'
        )
        self.assertEqual(_blacklist_ttl_for_body(404, body), _BLACKLIST_TTL_DAILY)

    def test_openrouter_no_endpoints_is_daily(self) -> None:
        body = '{"error":{"message":"No endpoints found for arcee-ai/trinity-large-preview:free.","code":404}}'
        self.assertEqual(_blacklist_ttl_for_body(404, body), _BLACKLIST_TTL_DAILY)

    def test_openrouter_temporarily_rate_limited_is_medium(self) -> None:
        body = (
            'qwen/qwen3-coder:free is temporarily rate-limited upstream. '
            "Please retry shortly"
        )
        self.assertEqual(_blacklist_ttl_for_body(429, body), _BLACKLIST_TTL_MED)

    def test_groq_per_minute_tpm_is_short(self) -> None:
        body = (
            "Rate limit reached for model llama-4-scout in organization X service tier "
            "on_demand on tokens per minute (TPM): Limit 30000, Used 25518."
        )
        self.assertEqual(_blacklist_ttl_for_body(429, body), _BLACKLIST_TTL_SHORT)

    def test_daily_quota_is_daily(self) -> None:
        body = "You have exhausted free-models-per-day quota."
        self.assertEqual(_blacklist_ttl_for_body(429, body), _BLACKLIST_TTL_DAILY)

    def test_503_is_short(self) -> None:
        body = "upstream connect error"
        self.assertEqual(_blacklist_ttl_for_body(503, body), _BLACKLIST_TTL_SHORT)

    def test_unknown_status_is_none(self) -> None:
        self.assertIsNone(_blacklist_ttl_for_body(200, "ok"))

    def test_retry_delay_parses_gemini_message(self) -> None:
        r = MagicMock()
        r.headers = {}
        r.text = 'Please retry in 12.5s. Quota exceeded.'
        d = _retry_delay_seconds_429(r)
        self.assertGreaterEqual(d, 12.5)
        self.assertLessEqual(d, 90.0)


if __name__ == "__main__":
    unittest.main()
