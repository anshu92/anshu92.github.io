"""Committee subgraph invoke."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

os.environ["BLOGPIPE_DRY_RUN"] = "1"
os.environ["BLOGPIPE_COMMITTEE_DISABLED"] = "0"
os.environ["BLOGPIPE_REPO_ROOT"] = os.path.join(os.path.dirname(__file__), "..")


def test_committee_subgraph_runs_synthesizer(monkeypatch, tmp_path):
    from blogpipe import research
    from blogpipe.graph.committee_subgraph import run_committee_subgraph_state
    from blogpipe.models import EvidencePack, Item, Pillar

    root = str(tmp_path)
    os.environ["BLOGPIPE_REPO_ROOT"] = root
    (tmp_path / "reports").mkdir(parents=True)
    (tmp_path / "content" / "post").mkdir(parents=True)

    def _fake_pack(_primary: Item) -> tuple:
        p = Item(
            id="t1",
            title="Test paper",
            url="https://arxiv.org/abs/2401.00001",
            authors=[],
            abstract="We propose X.",
            source="test",
            tags=["t"],
            pillar=Pillar.research,
        )
        return EvidencePack(primary=p, calls_used=0, trace=[{"tool": "fake"}]), 0, []

    monkeypatch.setattr(research, "gather_evidence_pack", _fake_pack)
    monkeypatch.setattr(research, "openalex_search", lambda q: [])

    s: dict = {
        "primary": {
            "id": "t1",
            "title": "Test paper",
            "url": "https://arxiv.org/abs/2401.00001",
            "authors": [],
            "abstract": "We propose X.",
            "source": "test",
            "tags": ["t"],
            "pillar": "research",
        }
    }
    out = run_committee_subgraph_state(s)
    assert out.get("_done_research")
    assert "evidence" in out
