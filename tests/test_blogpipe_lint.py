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


if __name__ == "__main__":
    unittest.main()
