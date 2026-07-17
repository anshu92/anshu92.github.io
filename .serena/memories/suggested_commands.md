# Suggested Commands

- Focused tests: `PYTHONPATH=scripts uv run --with pytest pytest -q tests/<test_file>.py`.
- Full local tests for blogpipe changes: `PYTHONPATH=scripts uv run --with pytest pytest -q tests`.
- Fixture simulation: `PYTHONPATH=scripts BLOGPIPE_FAKE_SELECTOR_RESPONSE="$(cat tests/fixtures/fake_selector.json)" BLOGPIPE_FAKE_OUTLINE_RESPONSE="$(cat tests/fixtures/fake_outline.json)" BLOGPIPE_FAKE_LLM_RESPONSE="$(cat tests/fixtures/fake_daily.md)" python -m blogpipe run --fixtures tests/fixtures --dry-run`.
- GitHub workflow mirror command: `python -m blogpipe run --window 14d` after installing `./scripts`.