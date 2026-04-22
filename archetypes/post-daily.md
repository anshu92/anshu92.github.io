---
date: '{{ .Date.Format "2006-01-02" }}'
draft: true
title: "{{ replace .Name "-" " " | title }}"
description: ""
categories: ["Machine Learning"]
tags: []
math: true
mermaid: true
one_sentence_takeaway: ""
image: /img/posts/{{ .Name }}/hero.png
rubric_score: 0
---

<!--
Structure note: posts do NOT follow a fixed template.
Open with one sentence containing the takeaway and a concrete number (no heading).
Then invent your own ## H2 headings specific to this post's story.
Avoid generic headings like Introduction / Background / Overview / Summary / Conclusion.
Include one ```mermaid``` block, one results table with | Method | Metric | Baseline |,
at least one honest limitation, and one first-person opinion ("What I find is..." / "In my view...").
-->

One-sentence takeaway goes here with a concrete number (first line of body).
