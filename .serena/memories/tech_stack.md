# Tech Stack

- Python package in `scripts/`; install locally with `python -m pip install ./scripts` or run tests with `PYTHONPATH=scripts`.
- Package metadata is minimal setuptools (`scripts/pyproject.toml`); dependency pins are in `scripts/requirements.txt` and `scripts/uv.lock`.
- Test framework is pytest. Existing tests assert behavior through fixtures and string-level workflow checks rather than heavy integration services.
- Site generator is Hugo with Stack theme vendored/submoduled under `themes/stack/`.