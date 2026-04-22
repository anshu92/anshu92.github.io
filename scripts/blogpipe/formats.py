"""PostFormat catalog: section templates, length bands, voice hints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Optional

from .models import Pillar


@dataclass
class PostFormat:
    name: str
    length_min: int
    length_max: int
    required_sections: list[str]  # H2 text prefix or exact match for lint
    optional_sections: list[str]
    voice_override: str
    hero_art_hint: str
    diagram_style: str
    pillar_preference: Optional[Pillar] = None
    max_per_10: int = 3  # cap rare formats

    @property
    def length_band(self) -> tuple[int, int]:
        return (self.length_min, self.length_max)

    def section_template_markdown(self) -> str:
        return "\n".join(f"## {s}\n" for s in self.required_sections if s)


FORMATS: dict[str, PostFormat] = {}


def _add(fmt: PostFormat) -> None:
    FORMATS[fmt.name] = fmt


_add(
    PostFormat(
        name="deep_dive",
        length_min=2000,
        length_max=3500,
        required_sections=[
            "TL;DR",
            "Why this matters",
            "Why this is hard",
            "What others tried",
            "Approach",
            "Implementation",
            "Results: metrics vs baseline",
            "What did not work",
            "Limitations and boundary conditions",
            "Where this shows up in AEC",
            "Related posts on this site",
            "What to steal",
            "References",
        ],
        optional_sections=["Appendix: deep dive"],
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
        required_sections=[
            "TL;DR",
            "The situation",
            "The constraint",
            "What we tried first",
            "Why it broke",
            "What actually worked",
            "What we would do differently",
            "What did not work",
            "Limitations and boundary conditions",
            "Where this shows up in AEC",
            "Related posts on this site",
            "What to steal",
            "References",
        ],
        optional_sections=["Appendix: deep dive"],
        voice_override="First-person where helpful; narrative, but still number-backed.",
        hero_art_hint="Situational / building-site adjacent without stock clichés.",
        diagram_style="block",
        pillar_preference=Pillar.aec,
    )
)

_add(
    PostFormat(
        name="benchmark_shootout",
        length_min=900,
        length_max=1800,
        required_sections=[
            "TL;DR",
            "The contenders",
            "The rig",
            "The metrics that matter",
            "Head-to-head",
            "Surprises",
            "When to pick which",
            "What did not work",
            "Limitations and boundary conditions",
            "Where this shows up in AEC",
            "Related posts on this site",
            "What to steal",
            "References",
        ],
        optional_sections=["Results chart"],
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
        required_sections=[
            "TL;DR",
            "The claim",
            "The equation (if any)",
            "The 20-line reference implementation",
            "Running it",
            "Where my numbers diverged",
            "Reading the paper differently afterwards",
            "What did not work",
            "Limitations and boundary conditions",
            "Where this shows up in AEC",
            "Related posts on this site",
            "What to steal",
            "References",
        ],
        optional_sections=["Appendix: deep dive"],
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
        required_sections=[
            "TL;DR",
            "The take",
            "Why most people get it wrong",
            "The receipts",
            "The counterarguments I respect",
            "The falsifier",
            "What did not work",
            "Limitations and boundary conditions",
            "Where this shows up in AEC",
            "Related posts on this site",
            "What to steal",
            "References",
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
        required_sections=[
            "TL;DR",
            "Observation",
            "What I tried",
            "What I noticed",
            "Open questions",
            "What did not work",
            "Limitations and boundary conditions",
            "Where this shows up in AEC",
            "Related posts on this site",
            "What to steal",
            "References",
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
        required_sections=[
            "TL;DR",
            "The surface behavior",
            "The interesting question",
            "Instrumentation",
            "What the trace shows",
            "The mental model you should keep",
            "What did not work",
            "Limitations and boundary conditions",
            "Where this shows up in AEC",
            "Related posts on this site",
            "What to steal",
            "References",
        ],
        optional_sections=["Appendix: deep dive"],
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
        required_sections=[
            "TL;DR",
            "Dialogue",
            "What did not work (and what the dialogue hid)",
            "Limitations and boundary conditions",
            "Where this shows up in AEC",
            "Related posts on this site",
            "What to steal",
            "References",
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
        required_sections=[
            "TL;DR",
            "The state of play 2 years ago",
            "What changed",
            "What aged well",
            "What aged badly",
            "The next bottleneck",
            "What did not work",
            "Limitations and boundary conditions",
            "Where this shows up in AEC",
            "Related posts on this site",
            "What to steal",
            "References",
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
        required_sections=[
            "TL;DR",
            "The system",
            "The failure taxonomy",
            "Each mode: symptom, root cause, detector, fix",
            "The gaps still open",
            "What did not work",
            "Limitations and boundary conditions",
            "Where this shows up in AEC",
            "Related posts on this site",
            "What to steal",
            "References",
        ],
        optional_sections=[],
        voice_override="Postmortem style; no blame, all mechanisms.",
        hero_art_hint="Red/amber risk strip or fault-tree suggestion.",
        diagram_style="state",
        pillar_preference=Pillar.applied,
    )
)

POSTFORMAT_NAMES: ClassVar[tuple[str, ...]] = tuple(FORMATS.keys())
