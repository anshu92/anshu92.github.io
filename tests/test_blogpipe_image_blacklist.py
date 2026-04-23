from __future__ import annotations

from blogpipe.llm_chain import is_blacklist_key_active, register_blacklist_key


def test_register_and_query_blacklist() -> None:
    k = f"__test_key_{id(object())}"
    register_blacklist_key(k, 3600.0)
    assert is_blacklist_key_active(k) is True


def test_empty_key_inactive() -> None:
    assert is_blacklist_key_active("") is False
