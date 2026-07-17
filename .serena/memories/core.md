# Core

- Hugo static site at repo root; content posts live in `content/post/`, images/assets in `static/`, theme under `themes/stack/`.
- Research blog automation is the Python package `scripts/blogpipe`; read `mem:blogpipe/core` for pipeline architecture and task-specific entrypoints.
- GitHub Actions workflows live in `.github/workflows/`; research automation is `research-radar.yml`, publication approval is split from generation.
- Tests live in `tests/` and are lightweight pytest files using fixtures under `tests/fixtures/`.