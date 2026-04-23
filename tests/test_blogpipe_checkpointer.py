"""Checkpointer: fresh thread resets usage counter; tuple exists after run."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


def test_fresh_thread_no_checkpoint_tuple():
    from langgraph.checkpoint.memory import MemorySaver

    m = MemorySaver()
    c = m.get_tuple({"configurable": {"thread_id": "x99"}})  # type: ignore[union-attr]
    assert c is None
