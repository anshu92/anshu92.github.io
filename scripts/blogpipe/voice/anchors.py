"""Voice anchors used by `build_prompt` to shape the writer's voice.

Four options grounded in real, current technical writers:
- A_YAN     : Eugene Yan engineer-practitioner (Why -> What -> How -> Steal-this)
- B_WENG    : Lilian Weng researcher-synthesizer (Taxonomy -> Mechanism -> Cross-domain)
- C_RASCHKA : Sebastian Raschka code-driven analyst (Code/pseudo -> Falsifier)
- D_HYBRID  : Yan structure + Raschka skepticism (DEFAULT)

The chosen anchor's `voice_guide` is copied into `EditorialBrief.voice_guide`,
its `opener_hook` is copied into `EditorialBrief.opener_hook`, and its
`exemplar_block` is appended directly to the writer system prompt.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class VoiceAnchor:
    key: str
    name: str
    source: str
    opener_hook: str
    voice_guide: str
    exemplar_block: str


# --- A : Eugene Yan ----------------------------------------------------------
A_YAN = VoiceAnchor(
    key="A",
    name="Eugene Yan engineer-practitioner",
    source="eugeneyan.com",
    opener_hook="single_concrete_pain_point",
    voice_guide=(
        "Voice: senior engineer-practitioner reading a paper for a working team. "
        "Plain English first; jargon goes in parentheses on first use. "
        "Each section answers one of: WHY this is hard / WHAT the paper does / "
        "HOW they show it works / WHAT I'd do with this on Monday. "
        "First-person only for judgment, never for the paper's experiments. "
        "Close every section with a one-sentence applied takeaway."
    ),
    exemplar_block=(
        "WRITING EXEMPLAR (imitate the SHAPE, never reproduce surface words):\n"
        "GOOD (this is what an expert applied paragraph looks like):\n"
        "    The picky-about-layers trick. Instead of adapting all 36 transformer\n"
        "    blocks (the standard LoRA recipe), the authors fit only 13 - chosen by\n"
        "    treating hidden-state trajectories as polylines and picking the\n"
        "    breakpoints. The 13-layer adapter beats full 36-layer adaptation by\n"
        "    2.4 points on MMLU-Math (81.7 vs 79.3) and crushes a random 13-layer\n"
        "    baseline (75.6). For my own fine-tunes I would steal exactly one habit:\n"
        "    stop adapting every layer by default; pick a structural reason for the\n"
        "    layers I keep.\n"
        "WEAK (this is the failure mode this voice rejects):\n"
        "    UniMesh marks a significant departure from traditional approaches by\n"
        "    integrating tasks that were previously handled in isolation, paving the\n"
        "    way for more holistic 3D vision systems and unlocking novel\n"
        "    capabilities in iterative editing.\n"
        "Why WEAK fails: marketing verbs (marks, paves, unlocks), zero numbers, no\n"
        "named alternative, no Monday-morning takeaway."
    ),
)

# --- B : Lilian Weng ---------------------------------------------------------
B_WENG = VoiceAnchor(
    key="B",
    name="Lilian Weng researcher-synthesizer",
    source="lilianweng.github.io",
    opener_hook="taxonomy_or_framing_distinction",
    voice_guide=(
        "Voice: ML researcher synthesising a sub-field. Open with a taxonomy or "
        "framing distinction, not a problem statement. Reach outside the paper for "
        "ONE cross-domain analogy (signal processing, neuroscience, classical "
        "statistics) - never decorative, must clarify mechanism. "
        "Multiple citations per paragraph; opinion is reserved for one explicit "
        "'What I think' subsection near the end. Third-person for the synthesis."
    ),
    exemplar_block=(
        "WRITING EXEMPLAR (imitate the SHAPE, never reproduce surface words):\n"
        "GOOD:\n"
        "    Layer-selection methods for parameter-efficient fine-tuning fall into\n"
        "    three families: uniform (the original LoRA), magnitude-driven (e.g.\n"
        "    rank pruning), and geometry-driven, of which RDP-LoRA is the recent\n"
        "    example. The geometric framing borrows from cartography: the\n"
        "    Ramer-Douglas-Peucker algorithm simplifies a polyline by keeping only\n"
        "    the points whose removal would most distort the shape. Applied to a\n"
        "    36-layer hidden-state trajectory through Qwen3-8B, the same procedure\n"
        "    yields 13 'load-bearing' layers; adapting only those reaches 81.67%\n"
        "    MMLU-Math vs. 79.32% for full-layer adaptation [cite: primary] - a\n"
        "    small but consistent gap I read as evidence the trajectory has real\n"
        "    geometric structure, not a fitting artefact.\n"
        "WEAK:\n"
        "    The authors propose a novel method for layer selection that achieves\n"
        "    state-of-the-art results on a challenging benchmark.\n"
        "Why WEAK fails: no taxonomy, no analogy, no numbers, no citation, no\n"
        "named family of comparable methods."
    ),
)

# --- C : Sebastian Raschka ---------------------------------------------------
C_RASCHKA = VoiceAnchor(
    key="C",
    name="Sebastian Raschka code-driven analyst",
    source="sebastianraschka.com",
    opener_hook="reproduction_or_falsifier",
    voice_guide=(
        "Voice: applied LLM analyst who would rather sketch the code than describe "
        "it. The load-bearing argument in each section is a code block, pseudo, or "
        "small numeric table - not prose. Compare to >=2 named alternatives by "
        "parameter count, training cost, AND accuracy. Skepticism appears as "
        "'what would falsify this', never as hedging. If the paper does not release "
        "code, name exactly what is missing to reproduce."
    ),
    exemplar_block=(
        "WRITING EXEMPLAR (imitate the SHAPE, never reproduce surface words):\n"
        "GOOD:\n"
        "    The selection rule is two lines of NumPy applied to per-layer\n"
        "    hidden-state means H of shape 36 x d:\n"
        "        idx = rdp(H, eps=tuned)\n"
        "        train_lora(layers=idx)\n"
        "    With eps tuned for k=13 on Qwen3-8B-Base, MMLU-Math reaches 81.67%;\n"
        "    LoRA-rank pruning at the same parameter budget gets 78.9%. What would\n"
        "    falsify the geometric story: if randomising the 13 indices but matching\n"
        "    their depth distribution closes the gap, the polyline framing is\n"
        "    window-dressing on a depth-distribution heuristic. The paper does not\n"
        "    run this ablation.\n"
        "WEAK:\n"
        "    The proposed method demonstrates strong empirical performance and is\n"
        "    likely to inspire future work in this direction.\n"
        "Why WEAK fails: no code, no named alternative with parameter count, no\n"
        "falsifier, no honest acknowledgement of missing ablations."
    ),
)

# --- D : Hybrid (DEFAULT) ----------------------------------------------------
D_HYBRID = VoiceAnchor(
    key="D",
    name="Yan structure + Raschka skepticism (hybrid)",
    source="eugeneyan.com + sebastianraschka.com",
    opener_hook="single_concrete_pain_point_with_falsifier",
    voice_guide=(
        "Voice: pragmatic engineer-analyst who reads papers like code reviews. "
        "Skeleton is WHY -> WHAT -> HOW -> STEAL-THIS (Eugene Yan); evidence "
        "handling is adversarial (Sebastian Raschka). "
        "Plain English first, jargon parenthetical second. "
        "Every section that names a method must also name an alternative + a "
        "rough parameter-count or cost comparison. "
        "The 'Limitations' section is reframed as 'What would falsify this' - "
        "name the cheapest experiment that, if it succeeded, would invalidate the "
        "paper's main claim. "
        "First-person sparingly and only for judgment ('I'd test', 'I read this "
        "as'); never as 'we' for the paper's experiments. "
        "Close every section with a one-sentence applied takeaway."
    ),
    exemplar_block=(
        "WRITING EXEMPLAR (imitate the SHAPE, never reproduce surface words):\n"
        "GOOD (the hybrid voice in one paragraph):\n"
        "    The picky-about-layers trick - and the cheap test that would prove it\n"
        "    wrong. Adapting only 13 of 36 layers, picked by Ramer-Douglas-Peucker\n"
        "    on hidden-state trajectories, beats full LoRA by 2.4 MMLU-Math points\n"
        "    (81.7 vs 79.3) at one-third the trainable parameters [cite: primary].\n"
        "    The honest comparison is to LoRA-rank pruning at the same parameter\n"
        "    budget, which the paper does not run; the falsifier is even cheaper -\n"
        "    randomise the 13 indices but match their depth distribution. If that\n"
        "    closes the gap, the geometry is decorative. For my own fine-tunes I'd\n"
        "    swap in this kind of structural-rule selection before adding more\n"
        "    trainable parameters.\n"
        "WEAK (failure modes this voice rejects):\n"
        "    UniMesh marks a significant departure from traditional approaches by\n"
        "    integrating 3D understanding and generation, paving the way for more\n"
        "    holistic 3D vision systems. While there are some limitations such as\n"
        "    increased computational complexity, the benefits of a unified\n"
        "    framework outweigh the drawbacks.\n"
        "Why WEAK fails: (1) marketing verbs (marks, paves), (2) no numbers, (3)\n"
        "names no alternative method, (4) 'limitations' are speculation about THIS\n"
        "method instead of naming the road not taken, (5) zero applied takeaway."
    ),
)


ANCHORS: dict[str, VoiceAnchor] = {
    "A": A_YAN,
    "B": B_WENG,
    "C": C_RASCHKA,
    "D": D_HYBRID,
}
DEFAULT_ANCHOR: VoiceAnchor = D_HYBRID


def get_anchor(key: str | None = None) -> VoiceAnchor:
    """Return the VoiceAnchor for `key` (A/B/C/D), env BLOGPIPE_VOICE_ANCHOR, else DEFAULT."""
    raw = (key or os.environ.get("BLOGPIPE_VOICE_ANCHOR") or "").strip().upper()
    if raw and raw in ANCHORS:
        return ANCHORS[raw]
    return DEFAULT_ANCHOR
