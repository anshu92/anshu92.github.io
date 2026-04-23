from __future__ import annotations

import unittest

from blogpipe.graph.critics import _apply_rubric_floor
from blogpipe.models import (
    AnalystNote,
    BenchmarkRow,
    EvidenceBundle,
    Item,
)


def _bundle() -> EvidenceBundle:
    return EvidenceBundle(
        primary=Item(
            id="primary",
            title="Sample paper",
            url="https://arxiv.org/abs/0000.0001",
            abstract="abstract",
            source="arxiv",
        ),
        competitors=[
            Item(
                id="c1",
                title="Full Fine-Tuning Baseline",
                url="https://arxiv.org/abs/0000.0002",
                abstract="x",
                source="arxiv",
            ),
            Item(
                id="c2",
                title="Naive Rank Pruning",
                url="https://arxiv.org/abs/0000.0003",
                abstract="x",
                source="arxiv",
            ),
        ],
        benchmarks=[
            BenchmarkRow(name="MMLU-Math", value="81.7%", baseline="79.3%"),
            BenchmarkRow(name="Inference Cost", value="30%", baseline="100%"),
        ],
        analyst_notes=[
            AnalystNote(
                role="reviewer",
                claims=[
                    "Selecting 13 layers beats full LoRA.",
                    "Polyline simplification anchors the geometric framing.",
                ],
            )
        ],
    )


class RubricFloorTests(unittest.TestCase):
    def test_floor_caps_when_no_cites_and_no_numbers(self) -> None:
        body = "## A\nAbstract prose with no numbers and no citations whatsoever.\n"
        floored, reasons, util = _apply_rubric_floor(13, body, None, [])
        self.assertLessEqual(floored, 5)
        self.assertTrue(any("no_cites" in r for r in reasons))
        self.assertTrue(any("no_numbers" in r for r in reasons))

    def test_floor_caps_when_lints_fire(self) -> None:
        body = (
            "Take.\n\n"
            "## A\nClaim 81.7% [cite: primary]. Compared to Full Fine-Tuning Baseline.\n"
        )
        floored, reasons, _ = _apply_rubric_floor(
            13, body, None, ["fake_results_table"]
        )
        self.assertLessEqual(floored, 6)
        self.assertTrue(any("lints:" in r and "fake_results_table" in r for r in reasons))

    def test_floor_caps_when_utilization_low(self) -> None:
        body = (
            "## A\nClaim 1.0% [cite: primary]. Some prose with one citation and a number.\n"
        )
        floored, reasons, util = _apply_rubric_floor(13, body, _bundle(), [])
        self.assertIsNotNone(util)
        # high score (13) gets floored down because util is well below the 0.20 threshold
        self.assertLessEqual(floored, 6)
        self.assertTrue(any("low_util" in r for r in reasons))

    def test_floor_does_not_drop_already_low_scores(self) -> None:
        body = "Empty.\n"
        floored, reasons, _ = _apply_rubric_floor(3, body, None, [])
        # Score stays at 3 because it was already below every cap; reasons may still
        # be recorded for visibility in editor_report.json.
        self.assertEqual(floored, 3)

    def test_floor_passes_when_all_signals_strong(self) -> None:
        body = (
            "RDP-LoRA reaches 81.7% on MMLU-Math vs Full Fine-Tuning Baseline at 79.3% "
            "[cite: primary]. Selecting 13 layers beats full LoRA, anchoring the polyline "
            "simplification framing in the cartography literature. Inference Cost drops 30% "
            "compared to Naive Rank Pruning [cite: c2].\n"
        )
        floored, reasons, util = _apply_rubric_floor(13, body, _bundle(), [])
        # No floor reasons should fire and the score should remain at the LLM-given value.
        self.assertEqual(floored, 13)
        self.assertEqual(reasons, [])


if __name__ == "__main__":
    unittest.main()
