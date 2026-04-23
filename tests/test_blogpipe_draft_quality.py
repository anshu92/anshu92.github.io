"""Regression tests for the six draft-quality bugs surfaced after a 30-min run."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
os.environ.setdefault("BLOGPIPE_REPO_ROOT", os.path.join(os.path.dirname(__file__), ".."))


# --- Equation double-wrapping ---------------------------------------------


def test_normalize_latex_strips_double_dollar():
    from blogpipe.draft import _normalize_latex

    assert _normalize_latex("$$ x = y $$") == "x = y"
    assert _normalize_latex("$ x $") == "x"


def test_normalize_latex_strips_paren_and_bracket_wrappers():
    from blogpipe.draft import _normalize_latex

    assert _normalize_latex(r"\( a + b \)") == "a + b"
    assert _normalize_latex(r"\[ a + b \]") == "a + b"
    assert _normalize_latex(r"$$\(\Delta\text{Acc}\)$$") == r"\Delta\text{Acc}"


def test_equation_md_emits_single_pair_of_dollars():
    from blogpipe.draft import _equation_md
    from blogpipe.models import EquationSpec

    spec = EquationSpec(id="eq1", latex=r"$$\(\Delta\text{Acc}\)$$", caption="cap")
    md = _equation_md(spec)
    assert md.count("$$") == 2  # exactly one open + one close
    assert r"\(" not in md
    assert r"\)" not in md


# --- Visual planner alt placeholder ---------------------------------------


def test_visual_planner_alt_placeholder_is_replaced():
    from blogpipe.analysts.visual_planner import _normalize_fig

    out = _normalize_fig(
        {
            "id": "fig1",
            "kind": "concept",
            "prompt": "p",
            "alt": "accessibility",
            "caption": "BEHEMOTH benchmark architecture",
        }
    )
    assert out is not None
    assert out["alt"].lower() != "accessibility"
    assert "BEHEMOTH" in out["alt"]


def test_visual_planner_alt_placeholder_falls_back_to_id():
    from blogpipe.analysts.visual_planner import _normalize_fig

    out = _normalize_fig(
        {"id": "scaling_curves", "kind": "plot", "prompt": "p", "alt": "image", "caption": ""}
    )
    assert out is not None
    assert out["alt"] == "scaling curves"


# --- Explainer must preserve markdown structure ---------------------------


def test_explainer_preserves_structure_rejects_link_corruption():
    from blogpipe.draft import _explainer_preserves_structure

    before = "Take. [Self-Evolving LLM Memory](https://arxiv.org/abs/2604.11610) is good."
    after = (
        "Take. [Self-Evolving LLM (Large Language Model) — a model] Memory]"
        "(https://arxiv.org/abs/2604.11610) is good."
    )
    assert not _explainer_preserves_structure(before, after)


def test_explainer_preserves_structure_accepts_safe_rewrite():
    from blogpipe.draft import _explainer_preserves_structure

    before = (
        "Takeaway sentence.\n## Heading\nWe use [paper](https://x.test). RDP is unclear here."
    )
    after = (
        "Takeaway sentence.\n## Heading\nWe use [paper](https://x.test). "
        "RDP (Ramer-Douglas-Peucker) is unclear here."
    )
    assert _explainer_preserves_structure(before, after)


def test_explainer_preserves_structure_rejects_heading_change():
    from blogpipe.draft import _explainer_preserves_structure

    before = "T.\n## Method\nbody"
    after = "T.\n## Method (LLM-based)\nbody"
    assert not _explainer_preserves_structure(before, after)


def test_explainer_preserves_structure_rejects_takeaway_change():
    from blogpipe.draft import _explainer_preserves_structure

    before = "Original takeaway.\n## H\nbody"
    after = "Original takeaway with extra LLM (Large Language Model) gloss.\n## H\nbody"
    assert not _explainer_preserves_structure(before, after)


# --- Frontmatter sentence boundary + cruft ---------------------------------


def test_sanitize_frontmatter_text_cuts_at_sentence():
    from blogpipe.draft import _sanitize_frontmatter_text

    src = (
        "A lightweight memory module lifts macro accuracy from 37.84% to 46.00%, "
        "and a cluster-based evolution strategy adds another 9.04% relative gain. "
        "Second sentence here."
    )
    out = _sanitize_frontmatter_text(src, 200)
    assert out.endswith(".")
    assert "Second sentence" not in out


def test_sanitize_frontmatter_text_strips_explainer_cruft():
    from blogpipe.draft import _sanitize_frontmatter_text

    src = (
        "A lightweight memory module wins. Self-Evolving LLM (Large Language Model) — "
        "a type of artificial intelligence model. Tail."
    )
    out = _sanitize_frontmatter_text(src, 240)
    assert "Large Language Model" not in out
    assert "type of artificial intelligence" not in out


def test_sanitize_frontmatter_text_caps_without_mid_word_cut():
    from blogpipe.draft import _sanitize_frontmatter_text

    src = "alpha bravo charlie delta echo foxtrot golf hotel " * 5
    out = _sanitize_frontmatter_text(src, 50)
    assert len(out) <= 51  # 50 + ellipsis
    # The last token before the ellipsis must be a whole word from the input.
    body = out.rstrip("…").rstrip()
    assert body.split()[-1] in {
        "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
    }


# --- Groq truncation no longer leaks marker -------------------------------


def test_groq_truncation_has_no_textual_marker():
    from blogpipe import llm_chain

    big = "A " * 5000
    out = llm_chain._messages_for_provider([{"role": "user", "content": big}], "groq")
    payload = out[0]["content"]
    assert "[truncated" not in payload
    assert "Groq request size" not in payload
    assert len(payload) <= llm_chain._GROQ_CONTENT_CAP + 1


# --- Continuation rescue (truncated full_draft) -----------------------------


def test_draft_truncation_heuristic_long_prose_no_terminal():
    from blogpipe.graph import nodes

    long_body = ("word " * 210 + "incomplete end").rstrip()
    assert nodes._draft_looks_truncated_for_rescue(long_body)
    assert not nodes._draft_looks_truncated_for_rescue(long_body + ".")
    assert not nodes._draft_looks_truncated_for_rescue("short " * 3)


def test_continuation_rescue_merges_continuation_onto_draft() -> None:
    """Mimic node_draft_refine append when draft_continuation returns prose."""
    base = ("word " * 200 + " stops mid without period").rstrip()
    cont = "A complete ending sentence."
    merged = (base.rstrip() + "\n\n" + cont.strip()).strip()
    assert "stops mid" in merged
    assert cont in merged
    assert merged.endswith(".")


# --- Citation anchors -----------------------------------------------------


def test_resolve_cites_uses_arxiv_id_anchor():
    from blogpipe.draft import _resolve_cites
    from blogpipe.models import EvidenceBundle, Item, Pillar

    b = EvidenceBundle(
        primary=Item(
            id="hf_2604.11610",
            title="Self-Evolving LLM Memory Extraction Across Heterogeneous Tasks",
            url="https://arxiv.org/abs/2604.11610",
            source="huggingface_daily_papers",
            tags=[],
            pillar=Pillar.research,
        )
    )
    md = "Foo [cite: hf_2604.11610]. Later, again [cite: hf_2604.11610]."
    out = _resolve_cites(md, b)
    # First mention uses the short arXiv id, not the long title.
    assert "[arXiv:2604.11610](https://arxiv.org/abs/2604.11610)" in out
    # Long paper title should not appear repeatedly as the anchor.
    assert out.count("Self-Evolving LLM Memory Extraction") == 0
    # Second mention to the *primary* shortens further to "the paper".
    assert "[the paper](https://arxiv.org/abs/2604.11610)" in out
