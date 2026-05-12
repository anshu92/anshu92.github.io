# Blogpipe Research Radar

Blogpipe is now a metadata-first research radar for Synaptic Radio. It ingests
papers and first-party engineering blogs, stores normalized records in SQLite +
FTS5, ranks them deterministically, builds compact evidence packs, and asks a
single OpenAI-compatible LLM writer to produce evidence-grounded Hugo posts.

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

Merge the generated PR to approve publication. The
`Publish approved research radar` workflow then flips changed draft posts to
`draft: false`, commits that publish change to `main`, builds Hugo, and deploys
GitHub Pages.

For local fixture checks:

```bash
PYTHONPATH=scripts BLOGPIPE_FAKE_LLM_RESPONSE="$(cat tests/fixtures/fake_daily.md)" \
  python -m blogpipe run --fixtures tests/fixtures --dry-run
```

## LLM Configuration

The writer uses one OpenAI-compatible endpoint:

- `BLOGPIPE_LLM_BASE_URL`
- `BLOGPIPE_LLM_API_KEY`
- `BLOGPIPE_LLM_MODEL`
- `BLOGPIPE_LLM_MAX_CALLS`
- `BLOGPIPE_LLM_MAX_TOKENS`

If those are unset, the client falls back to `OPENROUTER_BASE`,
`OPENROUTER_API_KEY`, and `BLOGPIPE_MODEL` for continuity.

## Data Policy

- Durable index: `radar-data/items.sqlite`
- Recent snapshots: `radar-data/daily/*.jsonl.gz`
- Published Markdown: `content/post/`
- Generated assets: `static/img/posts/<slug>/`
- Raw PDFs and built site output are not committed.
