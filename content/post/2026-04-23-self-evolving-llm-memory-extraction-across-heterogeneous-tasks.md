---
date: "2026-04-23"
draft: true
title: "Self-Evolving LLM Memory Extraction Across Heterogeneous Tasks"
description: "A lightweight memory module lifts macro accuracy from 37.84% to 46.00%, and a cluster-based evolution strategy adds another 9.04% relative gain—but only if you "
categories: ["Machine Learning"]
tags: ["llm", "paper", "blogpipe"]
math: true
mermaid: true
one_sentence_takeaway: "A lightweight memory module lifts macro accuracy from 37.84% to 46.00%, and a cluster-based evolution strategy adds another 9.04% relative gain—but only if you stop trying to find one extraction prompt that works everywhere Self-Evolving LLM (Large Language Model) — a type of artificial intelligence"
image: /img/posts/self-evolving-llm-memory-extraction-across-heterogeneous-tasks/hero.png
rubric_score: 15
---

A lightweight memory module lifts macro accuracy from 37.84% to 46.00%, and a cluster-based evolution strategy adds another 9.04% relative gain—but only if you stop trying to find one extraction prompt that works everywhere [Self-Evolving LLM (Large Language Model) — a type of artificial intelligence model] Memory Extraction Across Heterogeneous Tasks](https://arxiv.org/abs/2604.11610).

The core problem this paper tackles is deceptively simple: when an LLM (Large Language Model) assistant remembers stuff from past conversations, what exactly should it remember? The answer turns out to depend heavily on what kind of task the user is working on. A personalization task (remembering that I prefer 2pm meetings) needs different extracted facts than a problem-solving task (remembering the error message from three turns ago) or an agentic task (remembering which tools succeeded last time). Existing work either uses a single static prompt for everything or trains on homogeneous task distributions—and both break when the tasks actually differ.

## The Benchmark Gap: Why Nobody Could Measure This Before

The authors point out something obvious once stated: there was no benchmark for evaluating memory extraction across heterogeneous tasks [Self-Evolving LLM Memory Extraction Across Heterogeneous Tasks](https://arxiv.org/abs/2604.11610). Previous work either tested on one domain (customer service, code debugging) or used synthetic data that didn't capture the real diversity of what LLM assistants actually do.

BEHEMOTH (a benchmark repurposing 18 datasets with a downstream utility-driven metric) solves this by repurposing 18 existing datasets into three categories: personalization, problem-solving, and agentic tasks. The key design choice is the evaluation metric—downstream utility-driven, measuring whether the extracted memory actually helps solve the next user request rather than just whether extraction "looked right." They report two metrics: Macro Accuracy (equal-weighted average across datasets) and Relative Gain (geometric mean of per-dataset improvement ratios over baseline) [Self-Evolving LLM Memory Extraction Across Heterogeneous Tasks](https://arxiv.org/abs/2604.11610).

This matters because it forces the benchmark to care about what actually helps the user, not just extraction quality in isolation.

![accessibility](/img/posts/self-evolving-llm-memory-extraction-across-heterogeneous-tasks/figures/fig1.png)
*Figure 1: BEHEMOTH benchmark architecture for heterogeneous memory extraction evaluation*

*Equation 1: Macro accuracy improvement formula used in BEHEMOTH evaluation*

$$
\(\Delta\text{Acc} = \text{Acc}_{\text{CluE}} - \text{Acc}_{\text{Simpleprompt}}\)
$$

*Equation 2: Relative generalization gain calculation for heterogeneous tasks*

$$
\(\text{GenGain} = \frac{\text{Acc}_{\text{CluE}} - \text{Acc}_{\text{Baseline}}}{\text{Acc}_{\text{Baseline}}}\)
$$
## What Existing Self-Evolving Frameworks Get Wrong

The paper tests three prior self-evolving frameworks: GEPA (Generalized Evolutionary Prompt Optimization Algorithm) — an algorithm for optimizing prompts, ACE (Adaptive Cluster-based Evolution) — a framework for evolving prompts, and MemEvolve. These systems iteratively improve their extraction prompts by learning from past successes and failures. They work reasonably well when all training tasks come from the same distribution—but BEHEMOTH reveals a systematic failure mode: when trained on heterogeneous tasks, these frameworks degrade rather than improve [Self-Evolving LLM Memory Extraction Across Heterogeneous Tasks](https://arxiv.org/abs/2604.11610).

The intuition is straightforward. If your evolution algorithm sees a code-debugging example followed by a preference-learning example, what should it optimize for? The gradients from different task types conflict. The framework ends up oscillating between extraction strategies that work for one category but hurt another. The authors show that no single static prompt dominates across all three categories—each task type prefers different extraction behavior [Self-Evolving LLM Memory Extraction Across Heterogeneous Tasks](https://arxiv.org/abs/2604.11610).

![accessibility](/img/posts/self-evolving-llm-memory-extraction-across-heterogeneous-tasks/figures/fig3.png)
*Figure 3: Cross-task generalization comparison of CluE against existing self-evolving frameworks*

*Equation 3: Downstream utility-driven metric definition for task weighting*

$$
\(\mathcal{U}(\mathcal{T}) = \sum_{t\in\mathcal{T}} w_t \cdot f(t)\)
$$
## CluE: Cluster First, Evolve Second
To address the limitations of existing self-evolving prompt optimization frameworks in heterogeneous memory extraction tasks, the authors propose CluE (Cluster-based self-evolving strategy) — a strategy that groups extraction scenarios and evolves prompts based on these groups. CluE groups training examples into clusters by extraction scenarios, analyzes each cluster independently, and synthesizes cross-cluster insights to update the extraction prompt. This approach allows CluE to generalize effectively across heterogeneous tasks, achieving a +9.04% relative gain on the BEHEMOTH benchmark.

One key limitation of CluE is its reliance on high-quality clustering, which can be sensitive to the choice of clustering algorithm and hyperparameters. For example, using k-means clustering with a fixed number of clusters (e.g., k=5) may not always capture the underlying structure of the data, leading to suboptimal performance. To mitigate this, we can explore alternative clustering methods, such as hierarchical clustering or density-based clustering, which can provide more robust and flexible clustering results.

In comparison to existing self-evolving frameworks, CluE offers improved performance and flexibility in handling heterogeneous tasks. For instance, prior frameworks like [cite: hf_2604.11610] degrade significantly when training tasks are heterogeneous, whereas CluE consistently outperforms them. However, CluE may require more computational resources and larger training datasets to achieve optimal performance, which can be a limitation in resource-constrained settings. Overall, CluE provides a promising approach to heterogeneous memory extraction, and its limitations can be addressed through further research and optimization.
## Results That Hold Up
In 

[truncated for Groq request size limits]

![accessibility](/img/posts/self-evolving-llm-memory-extraction-across-heterogeneous-tasks/figures/fig2.png)
*Figure 2: Accuracy gains from CluE over Simpleprompt across 18 BEHEMOTH tasks*
