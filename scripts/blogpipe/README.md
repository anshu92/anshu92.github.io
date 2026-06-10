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
python -m blogpipe ingest --window 14d
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

If live source harvesting is temporarily throttled, the daily writer can reuse
recent paper records already present in the SQLite store to satisfy the minimum
paper pool before selector/outliner work begins. The fallback stays inside the
same recency window and still blocks when the store cannot supply enough current
papers.

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
- `OPENROUTER_API_KEY` and optional `OPENROUTER_BASE` for cross-provider
  fallback when an OpenRouter model such as `openrouter/free` appears in a
  failover chain.
- `BLOGPIPE_OPENROUTER_FREE_MODELS`, a comma-separated override for the
  OpenRouter free-model fallback roster.
- When the primary endpoint is Gemini (`generativelanguage.googleapis.com`),
  set `BLOGPIPE_LLM_MODEL_FAST` / `BLOGPIPE_LLM_MODEL_SMART` and per-task
  chains to current models. As of June 2026 the recommended roster is:
  fast tasks → `gemini-3.5-flash`, chain
  `gemini-3.5-flash,gemini-3.1-flash-lite,gemini-2.5-flash`; smart tasks →
  `gemini-3.1-pro-preview`, chain
  `gemini-3.1-pro-preview,gemini-3.5-flash,gemini-2.5-pro`. Do not use
  `gemini-2.0-flash*` (shut down 2026-06-01). `gemini-2.5-*` remain until
  2026-10-16. Constants live in `config.DEFAULT_GEMINI_*`.
- optional per-step model overrides:
  - `BLOGPIPE_LLM_MODEL_SELECTOR`
  - `BLOGPIPE_LLM_MODEL_OUTLINE`
  - `BLOGPIPE_LLM_MODEL_OUTLINE_REPAIR`
  - `BLOGPIPE_LLM_MODEL_DRAFT`
  - `BLOGPIPE_LLM_MODEL_DRAFT_SECTION`
  - `BLOGPIPE_LLM_MODEL_EDITOR`
  - `BLOGPIPE_LLM_MODEL_QUALITY_REVIEW`
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
  - `BLOGPIPE_LLM_CHAIN_QUALITY_REVIEW`
  - `BLOGPIPE_LLM_CHAIN_REPAIR`
- `BLOGPIPE_LLM_MAX_CALLS`
- `BLOGPIPE_LLM_MAX_TOKENS`
- `BLOGPIPE_LLM_MAX_RUNTIME_SECONDS` (workflow default `900`)
- `BLOGPIPE_LLM_FAST_TIMEOUT_SECONDS` (workflow default `45`)
- `BLOGPIPE_LLM_SMART_TIMEOUT_SECONDS` (workflow default `90`)
- `BLOGPIPE_SECTIONWISE_DRAFTING` (default `0`; opt in only when a longer,
  higher-call run is acceptable)
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
Each live LLM request also has a hard wall-clock deadline in addition to HTTP
read timeouts, so a slow fallback provider cannot hold a workflow open for
several minutes past the configured task budget. A model that exceeds that
deadline is skipped instead of being retried repeatedly.
If the primary endpoint is Gemini-compatible and a chain includes an OpenRouter
model name, that model is sent to `OPENROUTER_BASE` with `OPENROUTER_API_KEY`
instead of being sent to the Gemini endpoint.
When `OPENROUTER_API_KEY` is present, the default fallback roster appends these
current zero-priced OpenRouter models after the configured primary models,
ordered for expected Research Radar performance rather than recency:
`qwen/qwen3-next-80b-a3b-instruct:free`,
`nvidia/nemotron-3-ultra-550b-a55b:free`,
`nvidia/nemotron-3-super-120b-a12b:free`,
`nousresearch/hermes-3-llama-3.1-405b:free`, `moonshotai/kimi-k2.6:free`,
`nex-agi/nex-n2-pro:free`, `qwen/qwen3-coder:free`, `poolside/laguna-m.1:free`,
`openai/gpt-oss-120b:free`, `google/gemma-4-31b-it:free`,
`google/gemma-4-26b-a4b-it:free`,
`nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free`,
`meta-llama/llama-3.3-70b-instruct:free`, `openai/gpt-oss-20b:free`,
and `openrouter/free`.

## Search and Ranking

arXiv ingest fans out across named profiles for LLM methods, LLM systems,
MLE/evaluation, multimodal geometry, and AEC/CAD/building AI. OpenReview ingest
queries a small best-effort venue list unless overridden. The 14-day recency window
is strict for live sources; undated or stale items are dropped before ranking.
If arXiv returns persistent `429` responses, blogpipe stops the remaining arXiv
profiles for that run instead of burning retries across every profile. If fewer
than three ranked papers remain afterward, daily drafting blocks before any
selector, outline, or writer LLM calls.

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
- paper-supported claim
- paper-supported limitation
- AEC / 2D-document transfer hypothesis
- open research question when the transfer remains unproven

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
blocker. Daily posts should stay research-deep while reading like engineering
judgment: benchmark design, failure modes, deployment constraints, integration
dependencies, latency/cost tradeoffs, and validation tests should shape the
practical takeaway. The editor removes corporate transformation language, repeated
`crucial` / `paramount` / `game-changer` phrasing, unsupported first-person
Autodesk claims, paper-by-paper abstract summaries that lack synthesis, and
speculative AEC transfer claims that are phrased as proven results.

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
- `evidence_discipline`: transfer hypotheses stay clearly separated from
  paper-supported findings.
- `section_nonredundancy`: repeated same-paper sections do not restate the same
  mechanism without a distinct technical purpose.
- `experiment_detail`: evaluation detail keeps pace with the strength of the
  mechanism claims.

Low-signal drafts are blocked, not merely warned. The rubric is assigned by an
LLM quality-review pass so signal, synthesis, and editorial judgment are scored
in context rather than by brittle keyword heuristics. Blocked reports include
validator errors, rubric scores, and examples of failing text where available.
The review also rejects title/body drift, generic recommendations without an
engineering decision, and synthesis claims that sound stronger than the cited
evidence supports.
When a draft contains unsupported numeric claims, blogpipe first rewrites them
into qualitative phrasing before giving up on the run. Paragraphs that already
cite an evidence ID but omit its same-paragraph source URL are normalized
deterministically before repair, which keeps LLM repair budget focused on
substantive editorial problems. Frontmatter tags are
derived from the final body text rather than applying every selected-paper or
global radar tag. The edited body `# H1` is the canonical publication title and
frontmatter title source.

Generated posts do not include generic Mermaid maps or auto-injected chart
decorations. A visual should appear only when the writing model produces a
source-grounded technical figure that adds real explanatory value.

## Data Policy

- Durable index: `radar-data/items.sqlite`
- Recent snapshots: `radar-data/daily/*.jsonl.gz`
- Published Markdown: `content/post/`
- Generated assets: `static/img/posts/<slug>/`
- Raw PDFs and built site output are not committed.
