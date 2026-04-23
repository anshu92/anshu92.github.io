"""Topic theme classifier and rank-time gate."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

os.environ.setdefault("BLOGPIPE_REPO_ROOT", os.path.join(os.path.dirname(__file__), ".."))


def _item(title: str, abstract: str = "", source: str = "arxiv"):
    from blogpipe.models import Item, Pillar

    return Item(
        id="t-" + title[:16].lower().replace(" ", "_"),
        title=title,
        url="https://example.test/" + title[:16],
        abstract=abstract,
        source=source,
        tags=["ml"],
        pillar=Pillar.research,
    )


def test_themes_cover_six_categories():
    from blogpipe import topics

    ids = {t.id for t in topics.THEMES}
    assert ids == {
        "llm_scaling",
        "training_recipes",
        "aec_ml",
        "new_models",
        "problem_oriented_ml",
        "ml_engineering_deep_dive",
    }


def test_in_scope_matches_relevant_paper():
    from blogpipe import topics

    it = _item(
        "FlashAttention-3: faster attention kernels for H100",
        "We present a new GPU kernel for transformer inference throughput.",
    )
    matches = topics.matched_themes(it)
    assert "llm_scaling" in matches
    assert topics.is_in_scope(it)
    assert topics.relevance_score(it) > 0


def test_out_of_scope_paper_is_dropped():
    from blogpipe import topics

    it = _item(
        "A taxonomy of medieval ceramics",
        "Catalog of pottery glazes by region from 1100 to 1400 CE.",
    )
    assert topics.matched_themes(it) == []
    assert not topics.is_in_scope(it)


def test_aec_paper_matches_aec_theme():
    from blogpipe import topics

    it = _item(
        "Scan-to-BIM with point cloud transformers",
        "Spatial reasoning over LiDAR scans for IFC reconstruction in construction.",
    )
    assert "aec_ml" in topics.matched_themes(it)


def test_rank_filter_drops_off_topic(monkeypatch, tmp_path):
    from blogpipe import rank

    monkeypatch.setenv("BLOGPIPE_REQUIRE_TOPIC_MATCH", "1")
    items = [
        _item("FlashAttention-3 on H100", "transformer inference kernel"),
        _item("Medieval ceramics survey", "pottery"),
    ]
    out = rank._filter_off_topic(items)
    assert len(out) == 1
    assert "FlashAttention" in out[0].title


def test_rank_filter_keeps_when_all_off_topic(monkeypatch):
    from blogpipe import rank

    monkeypatch.setenv("BLOGPIPE_REQUIRE_TOPIC_MATCH", "1")
    items = [
        _item("Medieval ceramics survey", "pottery"),
        _item("Renaissance frescoes", "art history"),
    ]
    out = rank._filter_off_topic(items)
    assert len(out) == 2  # fallback: do not strand the pipeline


def test_heuristic_score_rewards_theme_matches(monkeypatch):
    from blogpipe import rank
    from blogpipe.models import EditorialBrief

    monkeypatch.setenv("BLOGPIPE_TOPIC_RELEVANCE_WEIGHT", "0.6")
    brief = EditorialBrief(pillar_weights={"research": 0.2, "systems": 0.2})
    on = _item("FlashAttention-3 on H100", "transformer inference kernel throughput")
    off = _item("Medieval ceramics survey", "pottery")
    assert rank._heuristic_score(on, brief) > rank._heuristic_score(off, brief)


def test_add_learned_keywords_persists_and_dedupes(tmp_path, monkeypatch):
    from blogpipe import memory, topics

    monkeypatch.setattr(memory, "CACHE", tmp_path)
    added = topics.add_learned_keywords(
        "llm_scaling",
        ["rope scaling", "kv eviction", "fp8", "the", "kv eviction", "model"],
    )
    # 'the' and 'model' are stopwords; second 'kv eviction' is a dupe; 'fp8' is built-in.
    assert "rope scaling" in added
    assert "kv eviction" in added
    assert "fp8" not in added
    assert "model" not in added
    assert added.count("kv eviction") == 1
    raw = (tmp_path / "topic_keywords.json").read_text()
    assert "rope scaling" in raw

    again = topics.add_learned_keywords("llm_scaling", ["rope scaling"])
    assert again == []


def test_add_learned_keywords_unknown_theme_is_noop(tmp_path, monkeypatch):
    from blogpipe import memory, topics

    monkeypatch.setattr(memory, "CACHE", tmp_path)
    assert topics.add_learned_keywords("not_a_theme", ["fp8 v2"]) == []
    assert not (tmp_path / "topic_keywords.json").exists()


def test_learned_keywords_are_used_in_classification(tmp_path, monkeypatch):
    from blogpipe import memory, topics

    monkeypatch.setattr(memory, "CACHE", tmp_path)
    topics.add_learned_keywords("aec_ml", ["bridge load testing"])
    it = _item(
        "ML for bridge load testing of cable-stayed structures",
        "We use a transformer over sensor streams.",
    )
    assert "aec_ml" in topics.matched_themes(it)


def test_update_themes_from_draft_dry_run_returns_empty(monkeypatch, tmp_path):
    from blogpipe import memory, topics

    monkeypatch.setattr(memory, "CACHE", tmp_path)
    monkeypatch.setenv("BLOGPIPE_DRY_RUN", "1")
    primary = _item("FlashAttention-3 on H100", "transformer inference kernel")
    out = topics.update_themes_from_draft(
        "## intro\nWe study paged attention and rope scaling.\n", primary
    )
    assert out == {}  # dry run => no LLM call => no learning


def test_update_themes_from_draft_persists_when_llm_returns(monkeypatch, tmp_path):
    from blogpipe import memory, openrouter_client, topics

    monkeypatch.setattr(memory, "CACHE", tmp_path)
    monkeypatch.setenv("BLOGPIPE_DRY_RUN", "0")
    monkeypatch.setenv("GROQ_API_KEY", "fake-test-key")

    def fake_llm_text(system, user, **_kw):
        return '{"keywords": ["paged attention", "rope scaling", "kv eviction"]}'

    monkeypatch.setattr(openrouter_client, "llm_text", fake_llm_text)
    primary = _item("FlashAttention-3 on H100", "transformer inference kernel")
    out = topics.update_themes_from_draft(
        "## intro\nA discussion of inference kernels and throughput.\n", primary
    )
    assert "llm_scaling" in out
    assert "kv eviction" in out["llm_scaling"]
    # second draft with the same keywords should not double-add
    again = topics.update_themes_from_draft(
        "## intro\nMore inference kernel notes.\n", primary
    )
    assert again.get("llm_scaling") in (None, [])
