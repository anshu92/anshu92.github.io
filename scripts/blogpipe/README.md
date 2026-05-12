# Blogpipe Research Radar

Blogpipe is now a metadata-first research radar for Synaptic Radio. It ingests
papers and first-party engineering blogs, stores normalized records in SQLite +
FTS5, ranks them deterministically, builds compact evidence packs, and asks a
single OpenAI-compatible LLM pipeline to produce evidence-grounded Hugo posts.
Generated posts are paper-first technical synthesis blogs planned for a
Principal Machine Learning Engineer at Autodesk evaluating AEC foundation models
and 2D document intelligence. Daily posts now prioritize depth over breadth:
they should build one sharp thesis around 3-4 primary papers, with optional
supporting mentions only when they clarify a tradeoff or adoption decision.

## Commands

```bash
python -m blogpipe ingest --window 72h
python -m blogpipe rank
python -m blogpipe write-daily
python -m blogpipe write-deep-dives --max-new 1
python -m blogpipe render-assets
python -m blogpipe run
```

## Publishing flow

The scheduled `Research radar` workflow does not publish directly. It generates
Hugo posts with `draft: true`, pushes them to a `radar/draft-*` branch, opens a
pull request, and emails the PR link for review.

If validation blocks the draft, the workflow does not publish or open a review
PR. Instead it writes a blocked report under `reports/` and sends a failure
email with the workflow run link and validator summary so the run does not fail
silently.

If repository settings block GitHub Actions from creating pull requests, the
workflow still pushes the branch and emails a GitHub PR creation URL. To restore
fully automatic PR creation, enable "Allow GitHub Actions to create and approve
pull requests" in the repository Actions settings or add a `PR_CREATION_TOKEN`
secret with pull request write access.

Merge the generated PR to approve publication. The
`Publish approved research radar` workflow then flips changed draft posts to
`draft: false`, commits that publish change to `main`, builds Hugo, and deploys
GitHub Pages.

For local fixture checks:

```bash
PYTHONPATH=scripts \
BLOGPIPE_FAKE_SELECTOR_RESPONSE="$(cat tests/fixtures/fake_selector.json)" \
BLOGPIPE_FAKE_OUTLINE_RESPONSE="$(cat tests/fixtures/fake_outline.json)" \
BLOGPIPE_FAKE_LLM_RESPONSE="$(cat tests/fixtures/fake_daily.md)" \
  python -m blogpipe run --fixtures tests/fixtures --dry-run
```

## LLM Configuration

The writer uses one OpenAI-compatible endpoint:

- `BLOGPIPE_LLM_BASE_URL`
- `BLOGPIPE_LLM_API_KEY`
- `BLOGPIPE_LLM_MODEL`
- optional per-step model overrides:
  - `BLOGPIPE_LLM_MODEL_SELECTOR`
  - `BLOGPIPE_LLM_MODEL_OUTLINE`
  - `BLOGPIPE_LLM_MODEL_OUTLINE_REPAIR`
  - `BLOGPIPE_LLM_MODEL_DRAFT`
  - `BLOGPIPE_LLM_MODEL_DRAFT_SECTION`
  - `BLOGPIPE_LLM_MODEL_EDITOR`
  - `BLOGPIPE_LLM_MODEL_REPAIR`
  - workflow convenience knobs: `BLOGPIPE_LLM_MODEL_FAST` and `BLOGPIPE_LLM_MODEL_SMART`
- optional failover chains (comma-separated):
  - `BLOGPIPE_LLM_CHAIN` (global)
  - `BLOGPIPE_LLM_CHAIN_SELECTOR`
  - `BLOGPIPE_LLM_CHAIN_OUTLINE`
  - `BLOGPIPE_LLM_CHAIN_OUTLINE_REPAIR`
  - `BLOGPIPE_LLM_CHAIN_DRAFT`
  - `BLOGPIPE_LLM_CHAIN_DRAFT_SECTION`
  - `BLOGPIPE_LLM_CHAIN_EDITOR`
  - `BLOGPIPE_LLM_CHAIN_REPAIR`
- `BLOGPIPE_LLM_MAX_CALLS`
- `BLOGPIPE_LLM_MAX_TOKENS`
- `BLOGPIPE_DAILY_MIN_WORDS` (default `1200`)
- `BLOGPIPE_DAILY_PRIMARY_PAPERS` (default `4`)
- `BLOGPIPE_DAILY_SUPPORTING_ITEMS` (default `2`)
- `BLOGPIPE_MIN_SIGNAL_SCORE` (default `0.75`)
- `BLOGPIPE_GENERIC_PHRASE_MAX_DENSITY` (default `0.015`)
- `BLOGPIPE_MIN_PAPERS` (default `4`)
- `BLOGPIPE_MAX_BLOGS` (default `2`)
- `BLOGPIPE_PROFILE_RESULTS` (default `40`)
- `BLOGPIPE_SELECTOR_CANDIDATES` (default `24`)
- `BLOGPIPE_SELECTOR_MAX_TOKENS` (default `3200`)
- `BLOGPIPE_OUTLINE_MAX_TOKENS` (default `3200`)
- `BLOGPIPE_OPENREVIEW_VENUES` (comma-separated venue override)

If those are unset, the client falls back to `OPENROUTER_BASE`,
`OPENROUTER_API_KEY`, and `BLOGPIPE_MODEL` for continuity.
When a chain is provided, blogpipe tries models in order and automatically
falls back to the next model on retriable/server/model-availability failures.

## Search and Ranking

arXiv ingest fans out across named profiles for LLM methods, LLM systems,
MLE/evaluation, multimodal geometry, and AEC/CAD/building AI. OpenReview ingest
queries a small best-effort venue list unless overridden. The 72h recency window
is strict for live sources; undated or stale items are dropped before ranking.

The daily flow asks the selector LLM to choose directly from all available paper
titles in the current run for Autodesk/AEC foundation-model and 2D-document
work, rather than pre-filtering with score-ranked candidate slices. The selector
classifies selected items as:

- `primary`: papers that must receive deep treatment in the article.
- `supporting`: optional context used briefly for comparison, implementation
  detail, or an "also watch" mention.

The selector scores candidates for direct AEC/document relevance, transferable
mechanism, experiment strength, engineering actionability, and novelty relative
to prior radar posts. It should prefer a coherent thesis cluster over topic
diversity for its own sake.

## Daily Synthesis Format

For each primary paper, blogpipe builds a structured evidence card with:

- problem statement
- core mechanism or architecture
- objective, metric, or math detail when present
- evaluation setup
- key result text without invented numbers
- limitations or failure modes
- AEC / 2D-document transfer hypothesis

An LLM outline stage then creates thesis-led natural headings instead of public
template sections. Good headings name the technical object or tradeoff:

- `Executable CAD is the evaluation target, not visual plausibility`
- `Layout preservation is a systems problem, not a translation feature`
- `Grounding confidence needs counterfactual visual evidence`

Bad headings are blocked or repaired because they add noise:

- `Navigating the Future`
- `Bridging the Digital Divide`
- `Our Path Forward`
- `AI's Blueprint for AEC`

The writer drafts each section with separate LLM calls and then runs a final
editor pass. Each section should open with a concrete claim, walk through the
mechanism, state the engineering implication, and name a limitation or adoption
blocker. The editor removes corporate transformation language, repeated
`crucial` / `paramount` / `game-changer` phrasing, unsupported first-person
Autodesk claims, and paper-by-paper abstract summaries that lack synthesis.

Validation requires method/objective, experiment, limitation, impact, and
Autodesk/AEC/document relevance coverage, resolved evidence IDs, source links,
supported numbers, at least `BLOGPIPE_DAILY_MIN_WORDS` words, and enough cited
primary papers. It also runs a signal rubric:

- `technical_specificity`: concrete algorithms, objectives, system components,
  or evaluation design.
- `engineering_judgment`: adoption decision, blocker, prototype recommendation,
  benchmark, release gate, or production risk.
- `synthesis`: cross-paper comparison or tradeoff.
- `noise_control`: low density of generic strategy/corporate phrases.
- `primary_depth`: each primary paper has mechanism evidence and a limitation,
  experiment, or objective when available.

Low-signal drafts are blocked, not merely warned. Blocked reports include the
validator errors, rubric scores, and examples of failing text where available.
When a draft contains unsupported numeric claims, blogpipe first rewrites them
into qualitative phrasing before giving up on the run. Frontmatter tags are
derived from the actual selected/cited content rather than applying every global
radar tag.

Generated posts also embed:
- one mermaid flow graph (paper/source map)
- source/topic mix SVG illustrations at `/img/posts/<slug>/source-mix.svg` and
  `/img/posts/<slug>/topic-mix.svg`

## Data Policy

- Durable index: `radar-data/items.sqlite`
- Recent snapshots: `radar-data/daily/*.jsonl.gz`
- Published Markdown: `content/post/`
- Generated assets: `static/img/posts/<slug>/`
- Raw PDFs and built site output are not committed.
