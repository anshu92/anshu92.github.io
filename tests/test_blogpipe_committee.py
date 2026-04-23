"""Committee pipeline and model registry."""

from __future__ import annotations

import os
import json

# Repo root: tests/ is sibling to scripts/
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

os.environ["BLOGPIPE_DRY_RUN"] = "1"
os.environ["BLOGPIPE_COMMITTEE_DISABLED"] = "0"
os.environ["BLOGPIPE_REPO_ROOT"] = os.path.join(os.path.dirname(__file__), "..")


def test_select_chain_returns_models():
    from blogpipe import model_registry

    ch = model_registry.select_chain(
        "analyst_methods",
        2000,
        prefer_free=True,
        usd_budget_remaining=0.0,
    )
    assert len(ch) >= 1
    assert all(len(p) == 2 for p in ch)


def test_parse_analyst_json():
    from blogpipe.analysts.base import _parse_json_note

    n = _parse_json_note(
        '{"claims": ["a"], "citations": ["u"], "confidence": "high", '
        '"contradictions": ["c"], "suggested_section": "Results"}',
        "methods",
    )
    assert n.role == "methods"
    assert "a" in n.claims
    assert n.contradictions == ["c"]


def test_run_committee_stages_dry_run(monkeypatch, tmp_path):
    from blogpipe.graph import committee
    from blogpipe import research
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
    out = committee.run_committee_research_stages(s)
    assert out.get("_done_research")
    assert "evidence" in out


def test_evidence_bundle_has_analyst_fields():
    from blogpipe.models import AnalystNote, EvidenceBundle, Item, Pillar

    b = EvidenceBundle(
        primary=Item(
            id="1",
            title="P",
            url="u",
            source="s",
        ),
        analyst_notes=[AnalystNote(role="web", claims=["c"])],
        committee_synthesis="synth",
    )
    d = json.loads(b.model_dump_json())
    assert d["analyst_notes"][0]["role"] == "web"
    assert d["committee_synthesis"] == "synth"
