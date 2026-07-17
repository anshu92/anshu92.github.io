# Task Completion

- Run focused pytest for touched behavior, usually with `PYTHONPATH=scripts uv run --with pytest pytest -q ...`.
- For pipeline generation changes, also run the fixture dry-run command from `mem:suggested_commands` when possible.
- Inspect `git diff --stat` and relevant hunks before final response; avoid committing or pushing unless explicitly requested.
- If Serena memories were changed, user can sanity-check with `serena memories check` from repo root.