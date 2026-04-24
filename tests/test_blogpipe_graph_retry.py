"""Retry policy constants."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


def test_retry_policies_tuned():
    from blogpipe.graph import retry_policies

    assert retry_policies.DEFAULT_RETRY.max_attempts == 3
    assert retry_policies.ANALYST_RETRY.max_attempts == 2


def test_add_node_with_retry_supports_new_retry_policy_keyword() -> None:
    from blogpipe.graph.retry_policies import DEFAULT_RETRY, add_node_with_retry

    class NewGraph:
        def __init__(self) -> None:
            self.calls = []

        def add_node(self, name, node, retry_policy=None):  # noqa: ANN001
            self.calls.append((name, node, retry_policy))

    g = NewGraph()
    fn = lambda: None
    add_node_with_retry(g, "x", fn, DEFAULT_RETRY)
    assert g.calls == [("x", fn, DEFAULT_RETRY)]


def test_add_node_with_retry_falls_back_to_legacy_retry_keyword() -> None:
    from blogpipe.graph.retry_policies import DEFAULT_RETRY, add_node_with_retry

    class OldGraph:
        def __init__(self) -> None:
            self.calls = []

        def add_node(self, name, node, retry=None):  # noqa: ANN001
            self.calls.append((name, node, retry))

    g = OldGraph()
    fn = lambda: None
    add_node_with_retry(g, "x", fn, DEFAULT_RETRY)
    assert g.calls == [("x", fn, DEFAULT_RETRY)]
