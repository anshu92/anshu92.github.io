# Blogpipe Agent Swarm

Blogpipe is a technical ML blog generation pipeline for Synaptic Radio. The
current generation path is a controlled agent swarm: it picks the next
foundations-first lesson, curates evidence, writes the technical explanation,
adds implementation detail, plans visuals/tables/components, reviews the result,
and writes a review-gated Hugo draft.

The public generation command is:

```bash
python -m blogpipe swarm run
```

Fixture check:

```bash
PYTHONPATH=scripts python -m blogpipe swarm run --fixtures tests/fixtures --dry-run
```

## Agent Flow

The orchestrator owns state transitions. Agents do not call each other directly;
each one receives validated artifacts and returns a typed report.

1. `RoleMarketScout` surveys Staff/Principal+ AI engineering roles and extracts
   recurring technical expectations as weak catalogue signals.
2. `CatalogueEditor` selects the next prerequisite-safe lesson from the
   foundations-first catalogue.
3. `ResearchLead` curates evidence for that lesson only.
4. `TechnicalExplainer` creates the first-principles concept arc.
5. `ImplementationEngineer` adds the practical how-to path.
6. `SkepticalFactChecker` checks citations, source links, unsupported numbers,
   and overclaims.
7. `PrincipalReviewer` checks Staff/Principal-level usefulness.
8. `VisualExplainer` creates a visual plan.
9. `TableDesigner` turns dense comparisons into compact tables.
10. `ComponentDesigner` adds safe static Hugo-compatible callouts/components.
11. `LayoutReviewer` checks the final look.
12. `ManagingEditor` writes a Hugo draft or a blocked report.

## Catalogue

The catalogue starts with basics and then expands toward frontier systems:

- linear algebra for ML systems
- autodiff and optimization
- neural network architecture foundations
- transformers and attention
- distributed training systems
- inference and serving systems
- evaluation and reliability
- retrieval, grounding, and multimodal systems
- agents and tool-using systems

Autodesk AEC is an occasional application lens, not the primary topic selector.
The primary target is technical depth for a Staff/Principal ML engineer.

## Visual Policy

Generated posts should look finished, but visuals must teach. Allowed visual
forms are Mermaid diagrams, deterministic SVGs under `static/img/posts/<slug>/`,
Markdown tables, fenced code/pseudocode, and small static HTML callouts.

The sanitizer rejects scripts, iframes, remote JS/CSS, inline event handlers,
unsupported shortcodes, and components that were not produced through a
`ComponentSpec`.

## Reports

Each run writes artifacts under `reports/swarm/`:

- `role_market_signal.json`
- `catalogue_decision.json`
- `lesson_brief.json`
- `visual_plan.json`
- `agent_reports.json`
- `review_findings.json`
- `run_report.json`
- `llm_usage.json`

Blocked runs write `reports/swarm/<slug>.blocked.json`. Successful dry runs
write `reports/swarm/<slug>.preview.md`; non-dry runs write a Hugo draft under
`content/post/` with `draft: true`.

## Configuration

Default endpoint:

- `BLOGPIPE_LLM_BASE_URL=https://api.openai.com/v1`
- `BLOGPIPE_LLM_API_KEY` or `OPENAI_API_KEY`

Default model routing:

- fast/planning/review agents: `gpt-5.6-luna`
- writing/synthesis/design agents: `gpt-5.6-terra`
- explicit high-cost override: `gpt-5.6-sol`

Useful overrides:

- `BLOGPIPE_CATALOGUE_LESSON`: force a lesson id.
- `BLOGPIPE_ROLE_MARKET_LIVE=1`: fetch public career pages for role-market
  signals.
- `BLOGPIPE_ROLE_MARKET_FIXTURES=/path/to/role_market.json`: use fixture
  postings instead of live pages.
- `BLOGPIPE_LLM_MAX_CALLS`, `BLOGPIPE_LLM_MAX_TOKENS`,
  `BLOGPIPE_LLM_MAX_RUNTIME_SECONDS`: run budget controls.
