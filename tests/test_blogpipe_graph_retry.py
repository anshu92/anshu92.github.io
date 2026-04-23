"""Retry policy constants."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


def test_retry_policies_tuned():
    from blogpipe.graph import retry_policies

    assert retry_policies.DEFAULT_RETRY.max_attempts == 3
    assert retry_policies.ANALYST_RETRY.max_attempts == 2
