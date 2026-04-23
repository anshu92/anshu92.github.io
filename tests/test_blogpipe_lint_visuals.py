from __future__ import annotations

from blogpipe.lint import missing_planned_visuals
from blogpipe.models import EquationSpec, FigureSpec, VisualPlan


def test_missing_figure_detected() -> None:
    p = VisualPlan(figures=[FigureSpec(id="x", prompt="a", alt="a", placement_hint="h")])
    d = missing_planned_visuals("no image here", p)
    assert d["missing_figures"] == ["x"]


def test_figure_ok_when_present() -> None:
    p = VisualPlan(figures=[FigureSpec(id="x", prompt="a", alt="a", placement_hint="h")])
    d = missing_planned_visuals("![a](/p/figures/x.png)", p)
    assert d["missing_figures"] == []


def test_missing_equation() -> None:
    p = VisualPlan(
        equations=[EquationSpec(id="e1", latex="\\sum_i x_i", placement_hint="h")],
    )
    d = missing_planned_visuals("## Hi", p)
    assert "e1" in d["missing_equations"]


def test_equation_ok_when_latex_in_body() -> None:
    p = VisualPlan(
        equations=[EquationSpec(id="e1", latex="a+b", placement_hint="h")],
    )
    d = missing_planned_visuals("We have a+b in prose", p)
    assert d["missing_equations"] == []
    d2 = missing_planned_visuals("x", p)
    assert "e1" in d2["missing_equations"]
