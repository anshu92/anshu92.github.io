# Blogpipe Research Radar

Blogpipe is now a metadata-first research radar for Synaptic Radio. It ingests
papers and first-party engineering blogs, stores normalized records in SQLite +
FTS5, ranks them deterministically, builds compact evidence packs, and asks a
single OpenAI-compatible LLM pipeline to produce evidence-grounded Hugo posts.
Generated posts are paper-first technical blogs planned for a Principal Machine
Learning Engineer at Autodesk evaluating AEC foundation models and 2D document
intelligence. Source blogs are used as supporting engineering context rather
than as generic roundup items.

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

The daily flow asks the selector LLM to choose directly from all available
paper titles in the current run for Autodesk/AEC foundation-model and
2D-document work, rather than pre-filtering with score-ranked candidate slices.

An LLM outline stage then creates natural post headings. The writer now drafts
each section with separate LLM calls and then runs a final editor LLM pass to
merge, de-duplicate, and polish the full draft. Validation still requires
method/objective, experiment, limitation, impact, and Autodesk/AEC/document
relevance coverage, resolved evidence IDs, source links, supported numbers, and
at least `BLOGPIPE_DAILY_MIN_WORDS` words. Single-source or generic summaries
are blocked instead of published. When the draft contains unsupported numeric
claims, blogpipe first rewrites them into qualitative phrasing before giving up
on the run. Frontmatter tags are derived from the actual selected/cited content
rather than applying every global radar tag.

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
