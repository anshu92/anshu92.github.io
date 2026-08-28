---
title: "Start Here"
description: "How to follow Synaptic Radio's public LLM research apprenticeship."
slug: "start-here"
aliases: ["/start-here/"]
---

Synaptic Radio is a public, self-directed apprenticeship in LLM research engineering. The work follows one operating loop:

> Question → hypothesis → objective and data design → implementation → controlled run → diagnosis → credible conclusion → scaled system.

The curriculum is intentionally evidence-first. A course summary, paper explanation, or framework walkthrough may be useful, but it is a **Lab**, not a flagship Research Report. A report is published only after its implementation, controls, retained results, and reproduction path are ready.

## How to follow the work

1. Open the [Curriculum](/page/curriculum/) to see the prerequisite graph and current module.
2. Read [Research Reports](/page/research-reports/) in module order.
3. Use [Labs](/page/labs/) for narrower supporting experiments.
4. Inspect [Artifacts](/page/artifacts/) for the code tag, manifest, raw results, and reproduction command behind a report.
5. Consult the [Reading Map](/page/reading-map/) for the primary resources and the rule governing when production code is consulted.

## What the evidence labels mean

- **Derived:** the governing mechanism and expected failures are documented.
- **Implemented:** a tested reference implementation exists.
- **Validated:** controlled experiments and negative controls support the bounded claim.
- **Scaled:** meaningful multi-GPU or larger-scale evidence exists.
- **Transferred:** the conclusion survives an out-of-distribution or downstream setting.

No label is awarded for ambition. If free compute cannot support a scaling claim, the report remains honestly bounded at the level it earned.

## Authorship contract

The core mechanisms are implemented personally as part of the learning process. AI assistance may help assemble reading packets, challenge experimental designs, review an existing attempt, analyze supplied results, or edit a report. It may not invent measurements, claim an unperformed reproduction, or replace the implementation being studied.
