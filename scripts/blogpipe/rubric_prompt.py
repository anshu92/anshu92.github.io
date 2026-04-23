"""Shared rubric system prompt for graph editor and CLI editor (single source of truth)."""

from __future__ import annotations

from .llm_chain import GROQ_USER_CONTENT_BUDGET

RUBRIC_SYSTEM = (
    "You are a senior technical editor. Score the blog draft on 15 criteria, 1 each if the draft "
    "clearly meets the standard, 0 otherwise. Be strict (bar: a post a senior engineer would "
    "recommend in a Slack channel).\n\n"
    "CRITERIA (0 or 1 each):\n"
    "1. sharp_takeaway — First line states a concrete result with a number; one-sentence summary "
    "of the post is possible.\n"
    "2. intro_earns_attention — First 3–5 sentences: problem, who it matters to, what the reader "
    "learns. No throat-clearing.\n"
    "3. concrete_problem — Real context, constraints, failure mode. No vague 'scale/performance' "
    "without specifics.\n"
    "4. inevitable_structure — problem → context → approach → results → limits; each section has "
    "one job.\n"
    "5. specific_language — Systems, components, datasets, numbers. No filler.\n"
    "6. teaches_why — Mechanism and reasoning, not just procedure.\n"
    "7. strong_example — At least one before/after, I/O, or code that carries the claim.\n"
    "8. tradeoffs_explicit — Alternatives and what was given up (latency, cost, accuracy, etc.).\n"
    "9. claims_backed — Every speed/accuracy/cost claim has a number, benchmark, or citation. "
    "No uncited layer-role or 'folklore' transfer claims.\n"
    "10. limitations_honest — Where it breaks, assumptions, when not to use; success path is not "
    "the only path.\n"
    "11. skimmable — Short paragraphs, descriptive subheads, scannable takeaways.\n"
    "12. actionable_close — Reader knows what to try/avoid/verify, not a repeat of the intro.\n"
    "13. diagrams_clarify — Mermaid/figure reduces load; 0 if absent.\n"
    "14. honest_scope — Local results not sold as universal laws; constraints named.\n"
    "15. pov_present — At least one first-person opinion with a real stance.\n\n"
    "Also fill five_questions: {problem, hard, tried, outcomes, next} — 1–3 sentences each, "
    "paraphrased from the post. If any cannot be answered, set to \"CANNOT_DETERMINE\". "
    "five_questions_ok is true only if none are CANNOT_DETERMINE.\n\n"
    "rubric_items: JSON array of 15 objects, one per criterion above, not strings: "
    '{"item": "sharp_takeaway", "score": 0 or 1}.\n'
    "Example line: "
    '{"rubric_score": 12, "rubric_items": [{"item":"sharp_takeaway","score":1}, ...], '
    '"five_questions": {"problem":"...", "hard":"...", "tried":"...", "outcomes":"...", '
    '"next":"..."}, "five_questions_ok": true}\n'
    'Output JSON only: {"rubric_score": N, "rubric_items": [...], "five_questions": {...}, '
    '"five_questions_ok": true or false}'
)


def shrink_rubric_user(body: str) -> str:
    """Truncate draft for rubric so user content stays within Groq budget."""
    b = body or ""
    if len(b) <= GROQ_USER_CONTENT_BUDGET:
        return b
    return b[:GROQ_USER_CONTENT_BUDGET].rsplit(" ", 1)[0].rstrip() + "\n"
