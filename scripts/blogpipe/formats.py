"""PostFormat catalog: length bands, voice hints, and topic GOALS.

Posts do NOT follow rigid titles or pre-assigned headings. ``required_sections``
holds high-level *content goals* (what the post must accomplish) that the writer
turns into its own, story-specific H2 headings. Never emit these strings as
literal ``## ...`` lines in the final post.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .models import Pillar


@dataclass
class PostFormat:
    name: str
    length_min: int
    length_max: int
    required_sections: list[str]  # content GOALS, not literal headings
    optional_sections: list[str]
    voice_override: str
    hero_art_hint: str
    diagram_style: str
    pillar_preference: Optional[Pillar] = None
    max_per_10: int = 3  # cap rare formats

    @property
    def length_band(self) -> tuple[int, int]:
        return (self.length_min, self.length_max)

    def goals_markdown(self) -> str:
        return "\n".join(f"- {s}" for s in self.required_sections if s)


FORMATS: dict[str, PostFormat] = {}


def _add(fmt: PostFormat) -> None:
    FORMATS[fmt.name] = fmt


# Default goals shared by most formats. These are content objectives, NOT headings.
_CORE_GOALS = [
    "Lead with a one-sentence takeaway that names a concrete number or result",
    "Explain why the problem is hard in plain English before any equations",
    "Describe the core method's intuition and how it differs from what came before",
    "Show at least one numeric result compared to a named baseline in both prose and a markdown table",
    "Discuss at least one honest limitation, failure mode, or reproduction barrier (not authors' ablations that validate the method)",
    "Include one first-person judgment and one concrete engineering habit the reader can steal",
    "Use [cite: id] for every external claim, matching one of the evidence ids provided",
    "Include exactly one mermaid diagram that clarifies the method",
]


_add(
    PostFormat(
        name="deep_dive",
        length_min=2000,
        length_max=3500,
        required_sections=_CORE_GOALS
        + [
            "Name the assumption, baseline, or engineering habit this work should update",
        ],
        optional_sections=[
            "Appendix for implementation details if warranted",
        ],
        voice_override="Analytical, peer-to-peer, prefer concrete systems language.",
        hero_art_hint="Editorial, layered illustration with a single visual metaphor.",
        diagram_style="flowchart",
        pillar_preference=None,
    )
)

_add(
    PostFormat(
        name="case_study",
        length_min=1200,
        length_max=2000,
        required_sections=_CORE_GOALS
        + [
            "Anchor the story in a specific constraint or deadline",
            "Be honest about what you would do differently next time",
        ],
        optional_sections=[],
        voice_override="First-person where helpful; narrative, but still number-backed.",
        hero_art_hint="Situational / building-site adjacent without stock cliches.",
        diagram_style="block",
        pillar_preference=Pillar.aec,
    )
)

_add(
    PostFormat(
        name="benchmark_shootout",
        length_min=900,
        length_max=1800,
        required_sections=_CORE_GOALS
        + [
            "Define the evaluation rig and the metrics that matter before reporting numbers",
            "End with a when-to-pick-which recommendation grounded in the numbers",
        ],
        optional_sections=["A visible results chart or table"],
        voice_override="Table-forward; every row ties to a named baseline.",
        hero_art_hint="Data-viz aesthetic, comparison bars.",
        diagram_style="sequence",
        pillar_preference=Pillar.systems,
    )
)

_add(
    PostFormat(
        name="paper_to_code",
        length_min=1500,
        length_max=2500,
        required_sections=_CORE_GOALS
        + [
            "Include a short reference implementation snippet (< 30 lines) or a link to one",
            "Describe where your reproduction numbers diverged from the paper",
        ],
        optional_sections=[],
        voice_override="Code-first, explain invariants before the snippet.",
        hero_art_hint="Schematic, IDE-adjacent, monospace accents.",
        diagram_style="state",
        pillar_preference=Pillar.research,
    )
)

_add(
    PostFormat(
        name="opinion_with_receipts",
        length_min=800,
        length_max=1500,
        required_sections=_CORE_GOALS
        + [
            "State the take in the first paragraph, not the last",
            "Address the strongest counterargument you can name and offer a falsifier",
        ],
        optional_sections=[],
        voice_override="Argued, first-person, every claim has a [cite:].",
        hero_art_hint="Editorial, bold type treatment.",
        diagram_style="timeline",
        max_per_10=1,
    )
)

_add(
    PostFormat(
        name="field_notes",
        length_min=600,
        length_max=1000,
        required_sections=_CORE_GOALS
        + [
            "Be explicit about sample size and hedging ('n = 1', 'only on macOS')",
            "End with at least one open question worth another experiment",
        ],
        optional_sections=[],
        voice_override="Short, hedged, honest about n=",
        hero_art_hint="Sketch / notebook, minimal color.",
        diagram_style="flowchart",
    )
)

_add(
    PostFormat(
        name="under_the_hood",
        length_min=1800,
        length_max=3000,
        required_sections=_CORE_GOALS
        + [
            "Describe the instrumentation before showing the trace",
            "Give the reader a durable mental model, not just observations",
        ],
        optional_sections=[],
        voice_override="Systems engineer; latency, memory, and failure mode language.",
        hero_art_hint="Cutaway, layers, server-room palette.",
        diagram_style="sequence",
        pillar_preference=Pillar.systems,
    )
)

_add(
    PostFormat(
        name="dialogue",
        length_min=1200,
        length_max=2000,
        required_sections=_CORE_GOALS
        + [
            "Let both voices disagree with receipts, not just vibes",
        ],
        optional_sections=[],
        voice_override="Socratic; both voices grounded in [cite:].",
        hero_art_hint="Two-column or split, dialogue-appropriate.",
        diagram_style="block",
        max_per_10=1,
    )
)

_add(
    PostFormat(
        name="retrospective",
        length_min=1500,
        length_max=2200,
        required_sections=_CORE_GOALS
        + [
            "Be specific about what aged well and what aged badly, with dates",
            "Name the next bottleneck, not just the last one",
        ],
        optional_sections=[],
        voice_override="Reflective, timeline-aware, cite sources of change.",
        hero_art_hint="Timeline strip, dated milestones.",
        diagram_style="timeline",
        pillar_preference=Pillar.foundations,
    )
)

_add(
    PostFormat(
        name="failure_mode_map",
        length_min=1200,
        length_max=2000,
        required_sections=_CORE_GOALS
        + [
            "For each failure mode, give symptom, root cause, detector, and fix",
            "Call out which gaps are still open, not just solved ones",
        ],
        optional_sections=[],
        voice_override="Postmortem style; no blame, all mechanisms.",
        hero_art_hint="Red/amber risk strip or fault-tree suggestion.",
        diagram_style="state",
        pillar_preference=Pillar.applied,
    )
)


POSTFORMAT_NAMES: tuple[str, ...] = tuple(FORMATS.keys())
