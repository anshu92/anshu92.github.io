from __future__ import annotations

import json

from blogpipe.analysts.visual_planner import _parse_raw_to_plan, parse_visual_plan_from_analysts
from blogpipe.models import AnalystNote, VisualPlan


def test_parse_raw_minimal() -> None:
    raw = '{"figures": [{"id": "a1", "kind": "concept", "prompt": "x", "alt": "y", "caption": "c", "placement_hint": "h"}], "equations": []}'
    p = _parse_raw_to_plan(raw)
    assert p and len(p.figures) == 1
    assert p.figures[0].id == "a1"
    assert p.figures[0].prompt == "x"


def test_parse_from_analyst_note() -> None:
    v = VisualPlan(
        figures=[],
    )
    n = AnalystNote(
        role="visual_planner",
        claims=[json.dumps(v.model_dump())],
    )
    p = parse_visual_plan_from_analysts([n])
    assert p is not None
    assert p.figures == []


def test_parse_skips_glossary() -> None:
    n = AnalystNote(role="glossary", claims=["X — y"])
    assert parse_visual_plan_from_analysts([n]) is None
