# Blogpipe Core

- Package path: `scripts/blogpipe`; CLI entrypoint is `python -m blogpipe` via `scripts/blogpipe/__main__.py` -> `cli.main`.
- Main orchestration is `pipeline.run_all()`: ingest, rank, write daily draft, optionally write deep dives when runtime budget remains, render assets, write reports.
- Research-generation quality is mainly distributed across `selector.py` (paper/support selection), `outline.py` (structured daily outline), `writer.py` (draft, repair, validation), `topics.py` (ranking/topic signals), `evidence.py` (evidence pack/card materialization), and `models.py` (shared dataclasses).
- Scheduled generation opens a draft PR rather than publishing directly; merge triggers publication workflow.