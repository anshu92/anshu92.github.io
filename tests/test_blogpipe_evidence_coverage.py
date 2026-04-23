from __future__ import annotations

import unittest

from blogpipe.evidence_coverage import (
    coverage_report,
    harvested_facts,
    render_unused_facts,
    unused_facts_for_section,
)
from blogpipe.models import (
    AnalystNote,
    BenchmarkRow,
    EvidenceBundle,
    Item,
    Quote,
)


def _bundle() -> EvidenceBundle:
    return EvidenceBundle(
        primary=Item(
            id="primary",
            title="RDP-LoRA: Geometric Layer Selection for PEFT",
            url="https://arxiv.org/abs/0000.0001",
            abstract="abstract",
            source="arxiv",
        ),
        competitors=[
            Item(
                id="c1",
                title="Full Fine-Tuning Baseline",
                url="https://arxiv.org/abs/0000.0002",
                abstract="full ft",
                source="arxiv",
            ),
            Item(
                id="c2",
                title="Naive LoRA Rank Pruning",
                url="https://arxiv.org/abs/0000.0003",
                abstract="rank pruning",
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
                    "Selecting 13 of 36 layers via Ramer-Douglas-Peucker beats full LoRA.",
                    "The geometric framing borrows from cartography polyline simplification.",
                ],
            )
        ],
        quotes=[
            Quote(
                source_id="primary",
                text="Adapting only 13 of 36 transformer blocks reaches 81.67% on MMLU-Math.",
            )
        ],
    )


class HarvestedFactsTests(unittest.TestCase):
    def test_buckets_populated(self) -> None:
        facts = harvested_facts(_bundle())
        self.assertTrue(len(facts["numbers"]) >= 2)
        self.assertTrue(any("full fine" in e.lower() for e in facts["entities"]))
        self.assertTrue(facts["analyst_claims"])
        self.assertTrue(facts["quotes"])


class CoverageReportTests(unittest.TestCase):
    def test_zero_coverage_body_scores_low(self) -> None:
        body = "## A\n\nGeneric prose with no numbers and no named entities.\n"
        rep = coverage_report(body, _bundle())
        self.assertLess(rep["score"], 0.3)
        self.assertGreater(len(rep["unused"]["numbers"]), 0)
        self.assertGreater(len(rep["unused"]["entities"]), 0)

    def test_full_coverage_body_scores_high(self) -> None:
        body = (
            "## Result\n"
            "RDP-LoRA reaches 81.7% on MMLU-Math, beating Full Fine-Tuning Baseline at 79.3% "
            "and Naive LoRA Rank Pruning at a 30% Inference Cost. Selecting 13 of 36 layers "
            "via Ramer-Douglas-Peucker beats full LoRA, and the geometric framing borrows "
            "from cartography polyline simplification. Adapting only 13 of 36 transformer "
            "blocks reaches 81.67% on MMLU-Math.\n"
        )
        rep = coverage_report(body, _bundle())
        self.assertGreaterEqual(rep["score"], 0.7)


class UnusedFactsTests(unittest.TestCase):
    def test_unused_facts_for_section_returns_uncovered(self) -> None:
        body = "## A\n\nNothing covered here.\n"
        unused = unused_facts_for_section("A", body, _bundle())
        self.assertGreater(len(unused["numbers"]), 0)
        self.assertGreater(len(unused["entities"]), 0)

    def test_render_unused_facts_renders_named_sections(self) -> None:
        unused = {
            "numbers": ["81.7%", "30%"],
            "entities": ["Full Fine-Tuning Baseline"],
            "analyst_claims": ["Selecting 13 of 36 layers beats full LoRA."],
            "quotes": ["Adapting only 13 of 36 blocks reaches 81.67%"],
        }
        text = render_unused_facts(unused)
        self.assertIn("UNUSED NUMBERS", text)
        self.assertIn("UNUSED NAMED ENTITIES", text)
        self.assertIn("UNUSED ANALYST CLAIMS", text)
        self.assertIn("UNUSED PAPER QUOTES", text)


if __name__ == "__main__":
    unittest.main()
