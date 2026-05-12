from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime, timezone

from blogpipe.sources import arxiv
from blogpipe.sources.aggregator import _dedupe, _filter_recent, harvest_all


def test_fixture_harvest_loads_normalized_items():
    items = harvest_all(fixtures="tests/fixtures")
    assert len(items) == 5
    assert all(item.item_id for item in items)


def test_arxiv_entry_parser():
    xml = """
    <entry xmlns="http://www.w3.org/2005/Atom">
      <id>https://arxiv.org/abs/2605.00001</id>
      <title> Test Paper </title>
      <summary> We propose a benchmark. </summary>
      <published>2026-05-10T00:00:00Z</published>
      <updated>2026-05-10T00:00:00Z</updated>
      <author><name>Ada Lovelace</name></author>
    </entry>
    """
    item = arxiv._entry(ET.fromstring(xml), search_profile="llm_methods")
    assert item is not None
    assert item.arxiv_id == "2605.00001"
    assert item.authors[0].name == "Ada Lovelace"
    assert item.extra["search_profile"] == "llm_methods"


def test_arxiv_entry_normalizes_versioned_abs_url():
    xml = """
    <entry xmlns="http://www.w3.org/2005/Atom">
      <id>http://arxiv.org/abs/2605.10187v1</id>
      <title> Versioned Paper </title>
      <summary> We propose a benchmark. </summary>
      <published>2026-05-10T00:00:00Z</published>
      <updated>2026-05-10T00:00:00Z</updated>
    </entry>
    """
    item = arxiv._entry(ET.fromstring(xml), search_profile="llm_methods")
    assert item is not None
    assert item.arxiv_id == "2605.10187"
    assert item.canonical_url == "https://arxiv.org/abs/2605.10187"


def test_arxiv_fetch_fans_out_and_tags_profiles(monkeypatch):
    xml = """
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>https://arxiv.org/abs/2605.00001</id>
        <title> Test Paper </title>
        <summary> We propose a language model benchmark. </summary>
        <published>2026-05-10T00:00:00Z</published>
        <updated>2026-05-10T00:00:00Z</updated>
        <category term="cs.CL" />
        <author><name>Ada Lovelace</name></author>
      </entry>
    </feed>
    """
    calls = []

    class Response:
        text = xml

        def raise_for_status(self):
            return None

    class FakeClient:
        def get(self, url):
            calls.append(url)
            return Response()

    monkeypatch.setattr(arxiv, "client", lambda: FakeClient())
    monkeypatch.setattr(arxiv.config, "profile_results", lambda: 5)
    items = arxiv.fetch(window_hours=72)
    assert len(calls) == len(arxiv.ARXIV_PROFILES)
    assert all(url.startswith("https://export.arxiv.org/api/query?") for url in calls)
    assert len(items) == len(arxiv.ARXIV_PROFILES)
    assert {item.extra["search_profile"] for item in items} == {profile.name for profile in arxiv.ARXIV_PROFILES}


def test_aggregator_dedupes_and_keeps_recent_only():
    now = datetime(2026, 5, 11, tzinfo=timezone.utc)
    fresh = arxiv._entry(
        ET.fromstring(
            """
            <entry xmlns="http://www.w3.org/2005/Atom">
              <id>https://arxiv.org/abs/2605.00001</id>
              <title> Fresh Paper </title>
              <summary> We propose a benchmark. </summary>
              <published>2026-05-10T00:00:00Z</published>
              <updated>2026-05-10T00:00:00Z</updated>
            </entry>
            """
        ),
        search_profile="llm_methods",
    )
    stale = fresh.model_copy(update={"canonical_url": "https://arxiv.org/abs/2604.00001", "arxiv_id": "2604.00001", "item_id": "", "published_at": datetime(2026, 4, 1, tzinfo=timezone.utc)})
    assert fresh is not None
    recent = _filter_recent([fresh, fresh, stale.normalized()], window_hours=72, now=now)
    deduped = _dedupe(recent)
    assert [item.arxiv_id for item in deduped] == ["2605.00001"]
