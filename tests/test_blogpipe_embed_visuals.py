from __future__ import annotations

from blogpipe.draft import embed_planned_visuals
from blogpipe.models import EquationSpec, FigureSpec, VisualPlan


def test_embed_figure_idempotent() -> None:
    plan = VisualPlan(
        figures=[FigureSpec(id="f1", prompt="p", alt="A", placement_hint="One")],
    )
    body = "## One\nx\n"
    s1 = embed_planned_visuals(body, plan, "slug")
    s2 = embed_planned_visuals(s1, plan, "slug")
    assert s1 == s2
    assert "/figures/f1.png" in s1


def test_embed_equation_inserts() -> None:
    plan = VisualPlan(
        equations=[
            EquationSpec(id="e1", latex="E=mc^2", placement_hint="Two"),
        ],
    )
    body = "## Two\nnada\n"
    out = embed_planned_visuals(body, plan, "slug")
    assert "E=mc^2" in out
    assert "$$" in out


def test_empty_plan_is_noop() -> None:
    assert embed_planned_visuals("hi", None, "s") == "hi"
    assert embed_planned_visuals("hi", VisualPlan(), "s") == "hi"
