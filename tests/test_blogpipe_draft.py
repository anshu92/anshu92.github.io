from __future__ import annotations

import textwrap
import unittest
from unittest.mock import patch

from blogpipe import formats, lint
from blogpipe.draft import (
    _glossary_terms_from_bundle,
    _polish_body,
    _repair_or_inject_results_table,
    _stub_body,
    build_prompt,
    explain_undefined_terms,
    soften_unsupported_numeric_claims,
)
from blogpipe.models import AnalystNote, EditorialBrief, EvidenceBundle, Item, Pillar


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


class DraftPolishTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bundle = EvidenceBundle(
            primary=Item(
                id="primary",
                title="Micro Language Models Enable Instant Responses",
                url="https://huggingface.co/papers/2604.19642",
                abstract=MICRO_ABSTRACT,
                source="huggingface_daily_papers",
                tags=["llm", "paper"],
                pillar=Pillar.research,
            )
        )
        self.bundle.section_evidence = {
            "paper_problem": (
                "Fine-tuning large language models is expensive and layer selection is often heuristic."
            ),
            "paper_method": (
                "We model hidden states as a trajectory and apply the Ramer-Douglas-Peucker "
                "algorithm to identify structural pivots."
            ),
            "paper_setup": "Model: Qwen3-8B-Base. Dataset: OrcaMath. Benchmark: MMLU-Math.",
            "paper_experiments": (
                "The unadapted model achieves 74.25%, while Full LoRA reaches 79.32%. "
                "Geometry-Selected Sparse LoRA yields 81.67%, outperforming random sparse "
                "LoRA at 75.56%."
            ),
            "paper_limitations": (
                "The design space is not exhaustively explored and the analysis is conducted "
                "on a single benchmark."
            ),
            "paper_reproducibility": (
                "Model: Qwen3-8B-Base. Dataset: OrcaMath. Benchmark: MMLU-Math."
            ),
        }
        self.bundle.contradiction_notes = [
            "The design space is not exhaustively explored and the analysis is conducted on a single benchmark."
        ]
        self.bundle.register_ids()
        self.brief = EditorialBrief(diagram_style="flowchart")

    def test_polish_body_normalizes_takeaway_and_repairs_table(self) -> None:
        body = textwrap.dedent(
            """
            ## Instant On-Device Language Generation with Micro Language Models
            ## Instant On-Device Language Generation with Micro Language Models

            Micro language models (μLMs) offer a solution to the challenge of enabling instant language generation on edge devices with limited power and compute resources.

            ## Performance Comparison

            Our μLMs match the performance of larger models (70M-256M parameters) while being significantly more compact. Our models achieve comparable results to prior state-of-the-art models, with μLM (8M) achieving 85% response quality and μLM (16M) achieving 88% response quality, compared to 90% for prior state-of-the-art models.

            | Method | Metric | Baseline |
            | --- | --- | --- |
            | μLM (8M) | Response Quality | 85% (prior SOTA: 90%) |
            | μLM (16M) | Response Quality | 88% (prior SOTA: 90%) |
            | Prior SOTA | Response Quality | 90% |
            """
        ).strip()

        polished = _polish_body(body, self.bundle, self.brief)

        self.assertFalse(polished.splitlines()[0].startswith("#"))
        self.assertNotIn("85%", polished)
        self.assertNotIn("88%", polished)
        self.assertNotIn("90%", polished)
        self.assertIn("74.25%", polished)
        self.assertIn("81.67%", polished)
        issues = lint.structural_issues(polished)
        self.assertNotIn("takeaway_is_heading", issues)
        self.assertNotIn("duplicate_heading", issues)
        self.assertNotIn("takeaway_repeated_as_heading", issues)
        self.assertNotIn("no_results_table", issues)

    def test_polish_body_rewrites_collective_research_voice(self) -> None:
        body = textwrap.dedent(
            """
            We model hidden states as a trajectory and evaluate the policy on MMLU-Math.

            Our models improve over random selection.

            We evaluate our geometric layer selection method on Qwen3-8B-Base.
            """
        ).strip()

        polished = _polish_body(body, self.bundle, self.brief)

        self.assertNotIn("We model", polished)
        self.assertNotIn("Our models", polished)
        self.assertIn("The authors' models", polished)
        self.assertIn("The authors evaluate their geometric layer selection method", polished)
        self.assertNotIn("collective_research_voice", lint.structural_issues(polished))

    def test_glossary_terms_collected_from_bundle(self) -> None:
        self.bundle.analyst_notes = [
            AnalystNote(
                role="glossary",
                claims=[
                    "RDP — Ramer-Douglas-Peucker polygon simplification.",
                    "MMLU — Massive Multitask Language Understanding benchmark.",
                ],
            ),
            AnalystNote(role="methods", claims=["unrelated"]),
        ]
        terms = _glossary_terms_from_bundle(self.bundle)
        self.assertEqual(len(terms), 2)
        self.assertTrue(any(t.startswith("RDP") for t in terms))

    def test_explain_undefined_terms_is_noop_in_dry_run(self) -> None:
        body = "RDP LoRA reaches 81.67% MMLU-Math.\n"
        self.bundle.analyst_notes = []
        out = explain_undefined_terms(body, self.bundle)
        self.assertEqual(out, body)

    def test_stub_body_is_grounded_and_structurally_valid(self) -> None:
        stub = _stub_body(self.bundle, self.brief, formats.FORMATS["deep_dive"])
        self.assertIn("Qwen3-8B-Base", stub)
        self.assertIn("OrcaMath", stub)
        self.assertIn("MMLU-Math", stub)
        self.assertIn("single benchmark", stub.lower())
        self.assertIn("In my view", stub)
        self.assertNotIn("We model", stub)
        self.assertEqual([], lint.structural_issues(stub))
        self.assertEqual([], lint.unsupported_numeric_claims(stub, self.bundle.model_dump_json()))

    def test_build_prompt_forbids_derived_arithmetic_summaries(self) -> None:
        system, _user = build_prompt(self.bundle, self.brief, formats.FORMATS["deep_dive"])
        self.assertIn("Do not invent derived arithmetic summaries", system)
        self.assertIn("do not turn them into a new unsupported claim", system)
        self.assertIn("How to apply in real-world scenarios", system)
        self.assertIn("game-changer", system)
        self.assertIn("The authors propose", system)
        self.assertIn("plain text only", system)

    def test_soften_unsupported_numeric_claims_rewrites_derived_delta(self) -> None:
        body = (
            "Takeaway 81.67%.\n\n"
            "## Why this works\n"
            "Picking the wrong subset tanks math reasoning scores by up to 6 points.[cite: primary]\n\n"
            "## Numbers on Qwen3-8B-Base\n"
            "| Method | Metric | Baseline |\n"
            "| --- | --- | --- |\n"
            "| RDP-selected layers | 81.67 MMLU-Math | n/a |\n"
            "| random 13-layer selection | 75.56 MMLU-Math | n/a |\n"
        )
        rewritten = (
            "Takeaway 81.67%.\n\n"
            "## Why this works\n"
            "The reported setup shows 81.67% for RDP-selected layers versus 75.56% for random 13-layer selection.[cite: primary]\n\n"
            "## Numbers on Qwen3-8B-Base\n"
            "| Method | Metric | Baseline |\n"
            "| --- | --- | --- |\n"
            "| RDP-selected layers | 81.67 MMLU-Math | n/a |\n"
            "| random 13-layer selection | 75.56 MMLU-Math | n/a |\n"
        )
        with (
            patch("blogpipe.config.llm_configured", return_value=True),
            patch("blogpipe.config.dry_run", return_value=False),
            patch("blogpipe.openrouter_client.llm_text", return_value=rewritten),
        ):
            out = soften_unsupported_numeric_claims(body, self.bundle.model_dump_json())
        self.assertNotIn("up to 6 points", out)
        self.assertIn("81.67%", out)
        self.assertIn("75.56%", out)
        self.assertEqual([], lint.unsupported_numeric_claims(out, self.bundle.model_dump_json()))

    def test_polish_body_normalizes_generic_advice_headings(self) -> None:
        body = (
            "Takeaway 81.67%.\n\n"
            "## Why OpenMobile is a game-changer in mobile agent research\nBody. [cite: primary]\n\n"
            "## How to apply OpenMobile in real-world scenarios\nAdvice. [cite: primary]\n\n"
            "## Conclusion\nClose. [cite: primary]\n"
        )
        polished = _polish_body(body, self.bundle, self.brief)
        self.assertIn("## Why this matters in practice", polished)
        self.assertIn("## What a practitioner should test next", polished)
        self.assertIn("## What I would test next", polished)
        self.assertNotIn("game-changer", polished.lower())
        self.assertNotIn("real-world scenarios", polished.lower())

    def test_polish_body_removes_illustrative_scores_and_injects_mechanism_section(self) -> None:
        bundle = self.bundle.model_copy(update={"benchmarks": []})
        bundle.section_evidence["paper_experiments"] = ""
        body = (
            "Takeaway 4 metrics.\n\n"
            "## Fine-tuning Boosts Spatial Capabilities\n"
            "The paper suggests stronger performance on both synthetic and real tasks.\n\n"
            "*Note: Scores are illustrative based on the paper's claims of significant improvement over prior SOTA models.*\n"
        )
        polished = _polish_body(body, bundle, self.brief)
        self.assertNotIn("Scores are illustrative", polished)
        self.assertIn("## Why this works", polished)
        self.assertIn("Ramer-Douglas-Peucker", polished)
        self.assertNotIn("significant improvement over prior SOTA", polished)

    def test_polish_body_repairs_corrupted_comparative_phrases(self) -> None:
        body = (
            "Takeaway 32%.\n\n"
            "## The MoE Scaling Bottleneck\n"
            "The scaling laws suggest that model quality suggests stronger performance predictably with total parameters.\n\n"
            "This creates a bottleneck: to suggests stronger performance quality via more experts, you need more compute.\n"
        )
        polished = _polish_body(body, self.bundle, self.brief)
        self.assertNotIn("suggests stronger performance predictably", polished)
        self.assertNotIn("to suggests stronger performance quality", polished)
        self.assertIn("model quality scales predictably", polished)
        self.assertIn("improve model quality via more experts", polished)

    def test_ensure_decision_section_adds_actionable_first_person_next_step(self) -> None:
        bundle = self.bundle.model_copy()
        bundle.section_evidence["paper_limitations"] = "The analysis is conducted on a single benchmark."
        body = (
            "Takeaway 81.67%.\n\n"
            "## Why this works\nMechanism. [cite: primary]\n"
        )
        polished = _polish_body(body, bundle, self.brief)
        self.assertIn("## What I would test next", polished)
        self.assertIn("I would start by reproducing", polished)

    def test_repair_or_inject_results_table_requires_verified_benchmark_rows(self) -> None:
        body = (
            "Takeaway 1.0%.\n\n"
            "## Why this works\nMechanism. [cite: primary]\n\n"
            "```mermaid\nflowchart LR\nA --> B\n```\n"
        )
        bundle = self.bundle.model_copy(update={"benchmarks": []})
        bundle.section_evidence["paper_experiments"] = ""
        repaired = _repair_or_inject_results_table(body, bundle)
        self.assertNotIn("| Method | Metric | Baseline |", repaired)

    def test_repair_or_inject_results_table_removes_unsupported_fake_table_when_no_verified_rows(self) -> None:
        body = (
            "Takeaway 1.0%.\n\n"
            "## Why this works\nMechanism. [cite: primary]\n\n"
            "| Method | Metric | Baseline |\n"
            "| --- | --- | --- |\n"
            "| UniT | 85% | baseline 60% |\n"
        )
        bundle = self.bundle.model_copy(update={"benchmarks": []})
        bundle.section_evidence["paper_experiments"] = ""
        repaired = _repair_or_inject_results_table(body, bundle)
        self.assertNotIn("| Method | Metric | Baseline |", repaired)

    def test_polish_body_removes_unknown_cite_markers(self) -> None:
        """LLM rewrites sometimes use evidence section keys as [cite: …]; those are not item ids."""
        body = (
            "Takeaway 81.67%.\n\n"
            "## Why this works\nMechanism. [cite: paper_method]\n\n"
            "```mermaid\nflowchart LR\nA --> B\n```\n\n"
            "| Method | Metric | Baseline |\n"
            "| --- | --- | --- |\n"
            "| x | 81.67% | n/a |\n"
        )
        polished = _polish_body(body, self.bundle, self.brief)
        self.assertNotIn("[cite: paper_method]", polished)
        self.assertNotIn("[missing cite:", polished)


if __name__ == "__main__":
    unittest.main()
