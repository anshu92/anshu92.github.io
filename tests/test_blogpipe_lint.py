from __future__ import annotations

import textwrap
import unittest

from blogpipe import lint


MICRO_ABSTRACT = (
    "Edge devices such as smartwatches and smart glasses cannot continuously run even the "
    "smallest 100M-1B parameter language models due to power and compute constraints, yet "
    "cloud inference introduces multi-second latencies that break the illusion of a responsive "
    "assistant. We introduce micro language models (μLMs): ultra-compact models (8M-30M "
    "parameters) that instantly generate the first 4-8 words of a contextually grounded "
    "response on-device, while a cloud model completes it; thus, masking the cloud latency. "
    "We show that useful language generation survives at this extreme scale with our models "
    "matching several 70M-256M-class existing models. We design a collaborative generation "
    "framework that reframes the cloud model as a continuator rather than a respondent, "
    "achieving seamless mid-sentence handoffs and structured graceful recovery via three error "
    "correction methods when the local opener goes wrong."
)


BAD_BODY = textwrap.dedent(
    """
    ## Instant On-Device Language Generation with Micro Language Models
    ## Instant On-Device Language Generation with Micro Language Models

    Micro language models (μLMs) offer a solution to the challenge of enabling instant language generation on edge devices with limited power and compute resources. These ultra-compact models, ranging from 8M to 30M parameters, can instantly generate the first 4-8 words of a contextually grounded response on-device, while a cloud model completes it.

    ## Performance Comparison

    Our μLMs match the performance of larger models (70M-256M parameters) while being significantly more compact. Our models achieve comparable results to prior state-of-the-art models, with μLM (8M) achieving 85% response quality and μLM (16M) achieving 88% response quality, compared to 90% for prior state-of-the-art models.

    | Method | Metric | Baseline |
    | --- | --- | --- |
    | μLM (8M) | Response Quality | 85% (prior SOTA: 90%) |
    | μLM (16M) | Response Quality | 88% (prior SOTA: 90%) |
    | Prior SOTA | Response Quality | 90% |

    ## Next Steps
    More experiments are needed.

    ## Next Steps
    Even more experiments are needed.
    """
).strip()


class LintRegressionTests(unittest.TestCase):
    def test_structural_issues_flag_repeated_templated_output(self) -> None:
        issues = lint.structural_issues(BAD_BODY)
        self.assertIn("takeaway_is_heading", issues)
        self.assertIn("takeaway_repeated_as_heading", issues)
        self.assertIn("duplicate_heading", issues)
        self.assertIn("templated_heading_used", issues)

    def test_unsupported_numeric_claims_catch_invented_scores(self) -> None:
        unsupported = lint.unsupported_numeric_claims(BAD_BODY, MICRO_ABSTRACT)
        self.assertIn("85%", unsupported)
        self.assertIn("88%", unsupported)
        self.assertIn("90%", unsupported)

    def test_collective_research_voice_is_flagged(self) -> None:
        body = (
            "We introduce a geometry-driven adapter policy that beats the baseline.\n\n"
            "Our models improve over random selection on the reported benchmark."
        )
        self.assertIn("collective_research_voice", lint.structural_issues(body))

    def test_undefined_acronyms_flag_first_use(self) -> None:
        body = (
            "RDP LoRA reaches 81.67% MMLU-Math accuracy.\n\n"
            "The authors apply RDP to identify breakpoints in MMLU evaluation runs."
        )
        flagged = lint.undefined_acronyms(body, [])
        self.assertIn("RDP", flagged)
        self.assertIn("MMLU", flagged)

    def test_undefined_acronyms_respect_inline_expansion(self) -> None:
        body = (
            "Ramer-Douglas-Peucker (RDP) yields 81.67% on the math split.\n\n"
            "MMLU (Massive Multitask Language Understanding) is the reported benchmark."
        )
        flagged = lint.undefined_acronyms(body, [])
        self.assertNotIn("RDP", flagged)
        self.assertNotIn("MMLU", flagged)

    def test_undefined_acronyms_respect_glossary(self) -> None:
        body = "RDP LoRA reaches 81.67% MMLU-Math accuracy.\n"
        flagged = lint.undefined_acronyms(
            body, ["RDP — Ramer-Douglas-Peucker polygon simplification."]
        )
        self.assertNotIn("RDP", flagged)


class StructuralLintAdditionsTests(unittest.TestCase):
    def test_h2_minimum_flags_h3_only_body(self) -> None:
        body = (
            "Takeaway 81% accuracy.\n\n"
            "### One\nFirst.\n\n"
            "### Two\nSecond.\n"
        )
        self.assertIn("fewer_than_required_h2_sections", lint.lint_h2_minimum(body))

    def test_h2_minimum_passes_with_two_h2s(self) -> None:
        body = (
            "Takeaway 81% accuracy.\n\n"
            "## A\nFirst.\n\n"
            "## B\nSecond.\n"
        )
        self.assertEqual([], lint.lint_h2_minimum(body))

    def test_unfenced_mermaid_block_is_flagged(self) -> None:
        body = (
            "Takeaway 1.0%.\n\n"
            "## Heading\n"
            "graph TD\n    A --> B\n    B --> C\n\n"
            "More prose.\n"
        )
        self.assertIn("unfenced_mermaid_block", lint.lint_unfenced_mermaid(body))

    def test_fenced_mermaid_block_is_not_flagged(self) -> None:
        body = (
            "Takeaway 1.0%.\n\n"
            "## Heading\n"
            "```mermaid\nflowchart LR\n  A --> B\n```\n"
        )
        self.assertEqual([], lint.lint_unfenced_mermaid(body))

    def test_citation_minimum_flags_missing_cites(self) -> None:
        body = "Takeaway 1.0%.\n\n## A\nProse without citations.\n\n## B\nMore prose.\n"
        self.assertIn("citation_count_below_min", lint.lint_citation_minimum(body))

    def test_citation_minimum_accepts_cite_tokens(self) -> None:
        body = (
            "Takeaway 1.0%.\n\n## A\nClaim. [cite: primary]\n\n"
            "## B\nMore. [cite: alpha]\n"
        )
        self.assertEqual([], lint.lint_citation_minimum(body))

    def test_citation_minimum_accepts_resolved_links(self) -> None:
        body = (
            "Takeaway 1.0%.\n\n"
            "## A\nClaim [arXiv:1234](https://arxiv.org/abs/1234).\n\n"
            "## B\n[the paper](https://arxiv.org/abs/1234).\n"
        )
        self.assertEqual([], lint.lint_citation_minimum(body))


class PolishRescueTests(unittest.TestCase):
    def test_promote_h3_when_no_h2_promotes_all(self) -> None:
        from blogpipe.draft import _promote_h3_when_no_h2

        body = "Take 1.0%.\n\n### One\na\n\n### Two\nb\n"
        out = _promote_h3_when_no_h2(body)
        self.assertIn("## One", out)
        self.assertIn("## Two", out)
        self.assertNotIn("### One", out)

    def test_promote_h3_when_no_h2_skips_when_h2_present(self) -> None:
        from blogpipe.draft import _promote_h3_when_no_h2

        body = "Take 1.0%.\n\n## Real\nx\n\n### Sub\ny\n"
        out = _promote_h3_when_no_h2(body)
        self.assertIn("### Sub", out)

    def test_wrap_unfenced_mermaid_blocks_wraps_graph(self) -> None:
        from blogpipe.draft import _wrap_unfenced_mermaid_blocks

        body = (
            "Take 1.0%.\n\n"
            "## A\n"
            "graph TD\n    X --> Y\n    Y --> Z\n\n"
            "Prose after.\n"
        )
        out = _wrap_unfenced_mermaid_blocks(body)
        self.assertIn("```mermaid\ngraph TD", out)
        self.assertIn("Y --> Z\n```", out)
        self.assertIn("Prose after.", out)
        self.assertEqual([], lint.lint_unfenced_mermaid(out))

    def test_wrap_unfenced_mermaid_leaves_existing_fenced_block(self) -> None:
        from blogpipe.draft import _wrap_unfenced_mermaid_blocks

        body = (
            "Take 1.0%.\n\n"
            "## A\n"
            "```mermaid\nflowchart LR\n  A --> B\n```\n"
        )
        self.assertEqual(body, _wrap_unfenced_mermaid_blocks(body))


class ExpertVoiceLintTests(unittest.TestCase):
    """The five new lints introduced for the expert-voice upgrade."""

    def test_section_redundancy_flags_near_duplicate_adjacent_sections(self) -> None:
        body = (
            "Take 5%.\n\n"
            "## The Need for a Unified Framework\n"
            "The conventional approach hinders generalizability and adaptability of 3D vision "
            "systems. A model trained for shape classification might not perform well on 3D "
            "generation tasks, and vice versa. This limitation hinders comprehensive 3D vision "
            "systems that can understand and generate 3D content seamlessly across tasks.\n\n"
            "## The Challenge of Unified 3D Intelligence\n"
            "The conventional approach hinders generalizability and adaptability of 3D vision "
            "systems. A model trained for shape classification cannot perform well on 3D "
            "generation tasks, and vice versa. This limitation hinders comprehensive 3D vision "
            "systems that understand and generate 3D content seamlessly across tasks.\n"
        )
        self.assertIn(
            "redundant_adjacent_sections", lint.lint_section_redundancy(body)
        )

    def test_section_redundancy_passes_distinct_sections(self) -> None:
        body = (
            "Take 5%.\n\n"
            "## Mesh Head\nA cross-model interface bridging diffusion and decoders.\n\n"
            "## Chain of Mesh\nIterative latent prompting for editing 3D chairs.\n"
        )
        self.assertEqual([], lint.lint_section_redundancy(body))

    def test_fake_results_table_flags_method_metric_baseline_prose_only(self) -> None:
        body = (
            "Take 30%.\n\n"
            "## Results\n"
            "Method Metric Baseline UniMesh "
            "3D Mesh Generation Time Prior SOTA UniMesh "
            "30% reduction in generation time Prior SOTA\n"
        )
        self.assertIn("fake_results_table", lint.lint_fake_results_table(body))

    def test_fake_results_table_passes_when_real_table_exists(self) -> None:
        body = (
            "Take 30%.\n\n"
            "## Results\n"
            "| Method | Metric | Baseline |\n"
            "| --- | --- | --- |\n"
            "| UniMesh | 30% time | Prior SOTA |\n"
        )
        self.assertEqual([], lint.lint_fake_results_table(body))

    def test_mermaid_taxonomy_flags_single_sink_no_verbs(self) -> None:
        body = (
            "Take 1%.\n\n"
            "## Diagram\n"
            "```mermaid\n"
            "graph LR\n"
            "A --> B\n"
            "A --> C\n"
            "B --> D\n"
            "B --> E\n"
            "B --> F\n"
            "C --> G\n"
            "C --> H\n"
            "C --> I\n"
            "D --> J\n"
            "E --> J\n"
            "F --> J\n"
            "G --> J\n"
            "H --> J\n"
            "I --> J\n"
            "J --> K\n"
            "```\n"
        )
        self.assertIn("mermaid_is_taxonomy", lint.lint_mermaid_taxonomy(body))

    def test_mermaid_taxonomy_passes_with_measurable_edges(self) -> None:
        body = (
            "Take 1%.\n\n"
            "## Diagram\n"
            "```mermaid\n"
            "graph LR\n"
            "A -->|encode 13 layers| B\n"
            "B -->|reduce 30%| C\n"
            "C -->|select rank-8 LoRA| D\n"
            "D -->|train| E\n"
            "E -->|measure 81.7%| F\n"
            "```\n"
        )
        self.assertEqual([], lint.lint_mermaid_taxonomy(body))

    def test_tradeoffs_present_passes_with_named_alt_and_marker(self) -> None:
        body = (
            "Take.\n\n"
            "## Tradeoffs\n"
            "Compared to Full Fine-Tuning, RDP-LoRA cuts trainable params at the cost of "
            "raw flexibility, vs. naive rank pruning which simply trims weights.\n"
        )
        self.assertEqual([], lint.lint_tradeoffs_present(body))

    def test_tradeoffs_present_flags_marketing_alternative(self) -> None:
        body = (
            "Take.\n\n"
            "## Tradeoffs\n"
            "An alternative approach could involve maintaining separate models for generation "
            "and understanding, but the unified framework wins.\n"
        )
        self.assertIn(
            "no_tradeoffs_paragraph", lint.lint_tradeoffs_present(body)
        )

    def test_baseline_pairing_passes_when_baseline_named(self) -> None:
        body = (
            "Take.\n\n"
            "## Numbers\n"
            "RDP-LoRA reaches 81.7% MMLU-Math vs. 79.3% for Full Fine-Tuning baseline.\n"
            "Inference cost drops 30% compared to LoRA-rank pruning baseline.\n"
        )
        self.assertEqual([], lint.lint_baseline_pairing(body))

    def test_baseline_pairing_flags_unpaired_improvement(self) -> None:
        body = (
            "Take.\n\n"
            "## Numbers\n"
            "UniMesh achieves a 30% reduction in generation time, demonstrating strong "
            "performance and significantly improving over many possible alternatives.\n"
            "It also shows a 25% increase in throughput, opening new possibilities for "
            "downstream work.\n"
        )
        self.assertIn(
            "unpaired_improvement_claims", lint.lint_baseline_pairing(body)
        )


if __name__ == "__main__":
    unittest.main()
