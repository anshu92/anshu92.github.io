# Conventions

- Keep blogpipe changes small and compatible with existing CLI/env knobs, report files, Hugo frontmatter, and generated draft PR flow.
- Prefer deterministic ranking/validation helpers around LLM calls; tests often use fake env responses and fixtures.
- Preserve the 30-minute GitHub Actions budget assumptions: daily draft is mandatory, deep dives are optional and budget-gated.
- Generated posts should remain evidence-grounded engineering memos for a Principal ML engineer working on foundation models/AEC/document intelligence, not broad paper roundups.