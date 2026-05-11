from __future__ import annotations

import xml.etree.ElementTree as ET

from blogpipe.sources import arxiv
from blogpipe.sources.aggregator import harvest_all


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
    item = arxiv._entry(ET.fromstring(xml))
    assert item is not None
    assert item.arxiv_id == "2605.00001"
    assert item.authors[0].name == "Ada Lovelace"
