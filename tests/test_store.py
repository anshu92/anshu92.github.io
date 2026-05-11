from __future__ import annotations

from blogpipe.models import SourceItem
from blogpipe import store


def test_store_upsert_and_fts(tmp_path):
    db = tmp_path / "items.sqlite"
    item = SourceItem(
        canonical_url="https://arxiv.org/abs/2605.00001",
        source_kind="paper",
        source_name="arxiv",
        title="Cache Aware Inference",
        abstract_or_excerpt="KV cache movement reduces latency for long context agents.",
        arxiv_id="2605.00001",
    )
    with store.connect(db) as conn:
        assert store.upsert_items(conn, [item]) == 1
        assert store.upsert_items(conn, [item]) == 1
        rows = store.load_items(conn)
        assert len(rows) == 1
        hits = store.search(conn, "latency")
        assert hits[0].arxiv_id == "2605.00001"
