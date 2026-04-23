from __future__ import annotations

import unittest
from unittest.mock import patch

from blogpipe.paper_reader import build_paper_evidence_from_text
from blogpipe.models import Item, Pillar


SAMPLE_PAPER = """
Abstract

We propose a geometry-driven method for selecting LoRA layers. The paper evaluates the method on MMLU-Math.

1. Introduction

Fine-tuning large language models is expensive and layer selection is often heuristic.

3. Methodology

The authors model hidden states as a trajectory and apply the Ramer-Douglas-Peucker algorithm to identify structural pivots.

4.2. Core Experiments on Qwen3-8B-Base

We analyze MMLU-Math performance. The unadapted model achieves 74.25%, while Full LoRA reaches 79.32%. Geometry-Selected Sparse LoRA yields 81.67%, outperforming random sparse LoRA at 75.56%.

Training Configuration

Model: Qwen3-8B-Base. Dataset: OrcaMath. Benchmark: MMLU-Math.

Limitations

The design space is not exhaustively explored and the analysis is conducted on a single benchmark. Variance analysis across multiple seeds is left for future work.

6. Conclusion

Geometry-selected sparse LoRA improves over random layer selection while adapting fewer layers.
""".strip()


class PaperReaderTests(unittest.TestCase):
    def test_build_paper_evidence_extracts_sections_and_limits(self) -> None:
        primary = Item(
            id="arxiv_2604.19321",
            title="RDP LoRA",
            url="https://arxiv.org/abs/2604.19321",
            abstract="short abstract",
            source="arxiv",
            pillar=Pillar.research,
        )
        evidence = build_paper_evidence_from_text(primary, SAMPLE_PAPER, pdf_url="https://arxiv.org/pdf/2604.19321.pdf")

        self.assertIn("Introduction", evidence.outline)
        self.assertIn("paper_method", evidence.section_evidence)
        self.assertIn("paper_experiments", evidence.section_evidence)
        self.assertIn("paper_limitations", evidence.section_evidence)
        self.assertTrue(any("81.67%" in x for x in evidence.result_notes))
        self.assertTrue(any("single benchmark" in x.lower() for x in evidence.limitation_notes))
        self.assertTrue(any("Qwen3-8B-Base" in x for x in evidence.reproducibility_notes))
        self.assertGreaterEqual(len(evidence.quotes), 1)

    @patch("blogpipe.paper_reader.openrouter_client.llm_text")
    @patch("blogpipe.paper_reader.config.llm_configured", return_value=True)
    @patch("blogpipe.paper_reader.config.dry_run", return_value=False)
    def test_build_paper_evidence_prefers_llm_section_summaries(
        self,
        _dry_run: object,
        _llm_configured: object,
        mock_llm_text: object,
    ) -> None:
        mock_llm_text.return_value = """
        {
          "problem": "The paper asks how to choose LoRA layers without relying on heuristics alone.",
          "method": "The authors use hidden-state trajectories and the Ramer-Douglas-Peucker algorithm to identify structural pivots for sparse adaptation.",
          "setup": "The main setup evaluates Qwen3-8B-Base on OrcaMath with MMLU-Math as the benchmark.",
          "results": [
            {
              "summary": "Geometry-Selected Sparse LoRA reaches 81.67% on MMLU-Math, ahead of Full LoRA at 79.32% and random sparse LoRA at 75.56%.",
              "support": "Geometry-Selected Sparse LoRA yields 81.67%, outperforming random sparse LoRA at 75.56%."
            }
          ],
          "limitations": [
            {
              "summary": "The evidence comes from a single benchmark and leaves seed variance for future work.",
              "support": "The design space is not exhaustively explored and the analysis is conducted on a single benchmark. Variance analysis across multiple seeds is left for future work."
            }
          ],
          "reproducibility": [
            {
              "summary": "The main setup uses Qwen3-8B-Base with OrcaMath and MMLU-Math.",
              "support": "Model: Qwen3-8B-Base. Dataset: OrcaMath. Benchmark: MMLU-Math."
            }
          ],
          "quotes": [
            "Model: Qwen3-8B-Base. Dataset: OrcaMath. Benchmark: MMLU-Math."
          ]
        }
        """
        primary = Item(
            id="arxiv_2604.19321",
            title="RDP LoRA",
            url="https://arxiv.org/abs/2604.19321",
            abstract="short abstract",
            source="arxiv",
            pillar=Pillar.research,
        )

        evidence = build_paper_evidence_from_text(
            primary,
            SAMPLE_PAPER,
            pdf_url="https://arxiv.org/pdf/2604.19321.pdf",
        )

        self.assertEqual(1, evidence.llm_chunks_used)
        self.assertIn(
            "choose LoRA layers without relying on heuristics",
            evidence.section_evidence["paper_problem"],
        )
        self.assertIn(
            "hidden-state trajectories",
            evidence.section_evidence["paper_method"],
        )
        self.assertIn(
            "Qwen3-8B-Base on OrcaMath with MMLU-Math",
            evidence.section_evidence["paper_setup"],
        )
        self.assertIn("81.67%", evidence.result_notes[0])
        self.assertIn("single benchmark", evidence.limitation_notes[0].lower())
        self.assertIn("OrcaMath", evidence.reproducibility_notes[0])
        self.assertTrue(
            any("Model: Qwen3-8B-Base. Dataset: OrcaMath. Benchmark: MMLU-Math." in q.text for q in evidence.quotes)
        )


if __name__ == "__main__":
    unittest.main()
