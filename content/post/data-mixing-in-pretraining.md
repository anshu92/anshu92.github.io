---
date: "2025-09-22"
draft: true
title: "Data Mixing and KL Penalty for Domain Adoption in LLM Pretraining"
description: "A practical playbook for adopting LLMs into new domains during continued pretraining using data mixing schedules and KL guardrails."
categories: ["Machine Learning", "NLP", "LLM"]
tags: ["pretraining", "data-mixing", "domain-adaptation", "kl-divergence", "distillation", "rlhf"]
math: true
mermaid: true
---

## TL;DR

This post is about **pretraining and continued pretraining** for domain adoption.

1. **Data mixing** is the ratio of data sources seen during training. It is where your gradient budget goes.
2. **KL penalty** is a drift-control term to keep the adapted model near a reference behavior.
3. Push domain data too hard without KL and you often get brittle specialization.
4. Push KL too hard and specialization stalls.
5. The job is to tune both knobs together, not win one benchmark.

## A short scene from inside a pretraining run

Imagine two copies of the same model at step ~320k of continued pretraining.

Run A uses data mixing:

- a Python pull-request diff with tests  
- then generic web prose  
- then API documentation  
- then a Stack Overflow thread

Run B uses no mixing (domain-only): nearly every batch is code, diffs, and technical docs.

The model is not making strategic choices; it only follows token distributions and gradient updates.

After enough steps, the difference is predictable:

- **Run A (mixed)** becomes domain-strong while keeping broader instruction behavior.
- **Run B (no mixing)** becomes code-sharp but domain-narrow, with weaker performance on mixed/general prompts.

You can see the contrast quickly with the same instructions:

Instruction: `Fix this Python function so it handles empty input and add tests.`  
Run A likely reply: patched function + tests + short edge-case explanation.  
Run B likely reply: patched function + tests, but thinner reasoning about edge cases and tradeoffs.

Now add a KL tether to a trusted reference checkpoint:

- in **Run A**, it stabilizes adaptation and reduces unwanted drift.
- in **Run B**, it can limit damage, but it cannot fully replace missing base-data exposure.

That is the core contrast: data mixing controls what the model learns; KL controls how far it drifts while learning it.

## Setup and scope

This write-up is intentionally narrow.

A lot of domain-adoption effort goes to later stages like SFT/RLHF, and that makes sense because those stages visibly change behavior.

But in many programs, pretraining/continued pretraining is the higher-leverage stage for domain capability, because it is where representations and domain priors are actually formed.

SFT/RLHF are still critical, but they usually refine expression, preference alignment, and policy behavior on top of whatever representation quality pretraining already gave you.

## Knob #1: data mixing (where gradients spend their life)

At pretraining time, if you have domain buckets \\(D_1, D_2, \ldots, D_k\\) and weights \\(w_1, w_2, \ldots, w_k\\), the objective is:

$$
\mathcal{L}(\theta) = \sum_{i=1}^{k} w_i \ \mathbb{E}_{x \sim D_i}[-\log p_\theta(x)]
$$

This is not dataset housekeeping. This is policy.

In my worldview, \\(w_i\\) is a product decision encoded as math.  
Every point of weight you add to one domain is training pressure taken from another.

For domain adoption, a convenient abstraction is:

- \\(B\\): broad base distribution
- \\(A\\): target domain distribution
- \\(\eta_t\\): adoption intensity at step \\(t\\)

$$
D_t = (1-\eta_t)B + \eta_t A
$$

And the real objective is usually constrained:

$$
\max_\theta \ \text{DomainScore}(\theta) \quad \text{s.t.} \quad \text{GeneralScore}(\theta) \ge \tau
$$

If you skip that constraint, you can get "great domain numbers" and still ship a worse model.

### What happens if you train only on the domain dataset?

Short-term, this can look great. Long-term, it is usually risky.

1. You often get fast gains in domain vocabulary, jargon, and local style.
2. You often lose breadth: weaker performance on mixed or general prompts.
3. Instruction-following and safety behavior can drift because broad priors get overwritten.
4. If the domain corpus is limited, overfitting shows up quickly (repetition, brittle phrasing, narrow reasoning).
5. Reliability outside the domain boundary drops, which is exactly where production traffic surprises you.

Domain-only training can still be a valid choice if the product is intentionally narrow and you accept that tradeoff. For most general-purpose assistants, a base-data floor plus KL is the safer default.

If the domain is **code**, this often looks like:

1. Better syntax completion and local API usage.
2. Worse performance on non-code tasks (open-ended explanation, broad QA, mixed chat workflows).
3. More brittle behavior when prompts combine code with policy/legal/product context.
4. Greater overfitting risk to narrow repository style when the corpus is repetitive.

### Evidence that this actually matters

- GPT-3 used a deliberate non-proportional mix (not raw web distribution), overweighting higher-quality sources ([Brown et al., 2020](https://arxiv.org/abs/2005.14165)).
- PaLM similarly uses an intentional, heterogeneous mixture at scale ([Chowdhery et al., 2022](https://arxiv.org/abs/2204.02311)).
- The Pile and Dolma reinforce the same lesson: composition quality and coverage matter, not only token count ([Gao et al., 2020](https://arxiv.org/abs/2101.00027), [Soldaini et al., 2024](https://arxiv.org/abs/2402.00159)).
- For adaptation specifically, DAPT/TAPT demonstrated reliable gains from domain-aware continued pretraining ([Gururangan et al., 2020](https://arxiv.org/abs/2004.10964)).
- SciBERT is a classic domain corpus win in scientific language understanding ([Beltagy et al., 2019](https://arxiv.org/abs/1903.10676)).

The short version: strong models are curated mixtures, not accidental scrapes.

## Knob #2: KL penalty (how fast identity is allowed to drift)

In continued pretraining, a common control is:

$$
\mathcal{L}_{t} = \mathbb{E}_{x \sim D_t}[-\log p_\theta(x)] + \beta_t D_{\mathrm{KL}}(p_\theta \,\|\, p_{\text{ref}})
$$

Where \\(p_{\text{ref}}\\) is usually a frozen base checkpoint (or SFT/teacher checkpoint, depending on stage).

Intuition:

- lower \\(\beta\\): move faster toward domain behavior
- higher \\(\beta\\): preserve baseline behavior more strongly

I think teams underestimate how often "faster adaptation" without KL is fake speed.  
You gain a week in training and lose a month in regression cleanup.

### Where we see KL in practice

- InstructGPT uses per-token KL against a reference policy in PPO and also discusses retaining capabilities with mixed objectives ([Ouyang et al., 2022](https://arxiv.org/abs/2203.02155)).
- Distillation work classically uses KL-style distribution matching ([Hinton et al., 2015](https://arxiv.org/abs/1503.02531)).
- MiniLLM reports benefits from reverse-KL framing for LLM distillation in their setup ([Gu et al., 2023](https://arxiv.org/abs/2306.08543)).

Base pretraining from scratch may not explicitly use KL-to-reference.  
Continued pretraining for domain adoption often benefits from it.

## Two concrete examples

### Software engineering domain (code assistant + code review)

Suppose you start with:

- mix: 65% base, 35% code corpus (repos, PRs, docs, issue discussions)
- KL: moderate \\(\beta\\) to base checkpoint

Typical pattern:

1. Code completion and bug-fix suggestions improve quickly.
2. Without KL, non-code assistant behavior can drift more than expected.
3. With KL + base floor, coding quality improves while general reasoning degrades less.

If you need higher domain lift, a safer move is usually ramping \\(\eta_t\\) in stages, not one giant jump.

### Architecture domain (design review + building code Q&A)

Suppose you adapt on standards, specs, plan narratives, and regulation text.

Without KL, models often become jargon-dense and less robust on non-domain prompts.  
With KL and a base-data floor, you usually get a better tradeoff: stronger code/spec recall with more stable general interaction quality.

Same mechanism, different industry.

## Strategy patterns that work

1. Start with a static blend and a hard base-data floor.
2. For higher risk domains, use a progressive ramp in \\(\eta_t\\) instead of a one-shot pivot.
3. Run short sweeps over \\((\eta, \beta)\\) and pick points on the Pareto frontier.
4. For large programs, learn the mixture weights instead of arguing heuristics.

On point 4, methods like DoReMi, Data Mixing Laws, and RegMix are useful because they replace intuition wars with measurable optimization ([Xie et al., 2023](https://arxiv.org/abs/2305.10429), [Ye et al., 2024](https://arxiv.org/abs/2403.16952), [Liu et al., 2024](https://arxiv.org/abs/2407.01492)).

Temperature sampling is still a practical baseline when balancing long-tail domains/languages:

$$
p_i \propto q_i^\alpha
$$

mT5 is a representative example with \\(\alpha = 0.3\\) in the final setup ([Xue et al., 2021](https://arxiv.org/abs/2010.11934)).

## From generic Qwen to a coding model: a concrete pretraining recipe

If your starting checkpoint is a generic Qwen model, think in terms of *continued pretraining specialization*, not a single fine-tune job.

Qwen2.5 base models were trained on up to 18T tokens, while Qwen2.5-Coder reports a coding-focused pipeline with 5.5T code-related tokens and extra repo-level training ([Yang et al., 2024](https://arxiv.org/abs/2412.15115), [Hui et al., 2024](https://arxiv.org/abs/2409.12186)).

### Step 1: Build dataset buckets explicitly

Use three buckets and keep them separate in your dataloader.

1. **Code core (roughly 50-70%)**
   The Stack v2 style sources are a strong baseline: permissively licensed code from Software Heritage, plus GitHub pull requests/commits and Kaggle notebooks ([Lozhkov et al., 2024](https://arxiv.org/abs/2402.19173)). For algorithmic and multi-language depth, you can add Project CodeNet-style data ([Puri et al., 2021](https://arxiv.org/abs/2105.12655)).
2. **Code-adjacent natural language (roughly 20-35%)**
   Stack Exchange / Stack Overflow style technical Q&A, READMEs, API docs, issue discussions, and design docs. CodeSearchNet-style code-docstring pairs are also useful for NL-to-code grounding ([Husain et al., 2019](https://arxiv.org/abs/1909.09436)).
3. **General anchor data (roughly 10-20%)**
   Keep a non-trivial slice of broad text to preserve general instruction behavior and reduce collapse into code-only interaction style.

These ranges are not arbitrary; they mirror what strong coding-model reports keep rediscovering: high code share, but not zero natural language ([Guo et al., 2024](https://arxiv.org/abs/2401.14196), [Lozhkov et al., 2024](https://arxiv.org/abs/2402.19173)).

### Step 2: Run staged continued pretraining

1. **File-level stage**
   Train mostly on code files + code-adjacent text, with next-token objective and optionally FIM formatting.
2. **Repository-level stage**
   Add long-context repository chunks (cross-file dependencies, commits/PR context). Qwen2.5-Coder explicitly uses a repo-level stage (reported as 220B tokens) to improve practical coding ability ([Hui et al., 2024](https://arxiv.org/abs/2409.12186)).
3. **Stability stage**
   Increase anchor/general share slightly and apply KL-to-reference to keep general assistant behavior from drifting too far.

### Step 3: Evaluate like a product, not just a benchmark

Track at least two gates per checkpoint:

1. **Coding gate**: pass@k or task success on code generation/review/fix tasks.
2. **Retention gate**: non-code instruction quality, safety behavior, and mixed-context prompts.

If coding score rises while retention falls, you are not done; you are over-specializing.

## A pragmatic pretraining playbook

1. Define domain slices and risk class before training starts.
2. Define two gates up front: domain lift gate and retention gate.
3. Begin conservative: base floor + moderate KL.
4. Sweep \\((\eta, \beta)\\) with short pilots, then scale only frontier candidates.
5. Checkpoint frequently and predefine rollback criteria.
6. Ship only when both gates pass on production-like traffic.

This is usually the dividing line between "impressive in a notebook" and "reliable in production."

## Common mistakes I keep seeing

1. Treating adaptation as "just add more domain text."
2. Running one giant training job before doing small sweeps.
3. Tracking domain metric gains without retention constraints.
4. Treating KL as optional polish instead of a core control.
5. Shipping before production-like evaluation.

Avoid these, and your odds of a clean domain adoption jump substantially.

## References

1. Brown et al., *Language Models are Few-Shot Learners* (GPT-3), 2020.  
   https://arxiv.org/abs/2005.14165
2. Chowdhery et al., *PaLM: Scaling Language Modeling with Pathways*, 2022.  
   https://arxiv.org/abs/2204.02311
3. Gao et al., *The Pile: An 800GB Dataset of Diverse Text for Language Modeling*, 2020.  
   https://arxiv.org/abs/2101.00027
4. Soldaini et al., *Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research*, 2024.  
   https://arxiv.org/abs/2402.00159
5. Xue et al., *mT5: A massively multilingual pre-trained text-to-text transformer*, 2021.  
   https://arxiv.org/abs/2010.11934
6. Xie et al., *DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining*, 2023.  
   https://arxiv.org/abs/2305.10429
7. Ye et al., *Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance*, 2024.  
   https://arxiv.org/abs/2403.16952
8. Liu et al., *RegMix: Data Mixture as Regression for Language Model Pre-training*, 2024.  
   https://arxiv.org/abs/2407.01492
9. Gururangan et al., *Don't Stop Pretraining: Adapt Language Models to Domains and Tasks*, 2020.  
   https://arxiv.org/abs/2004.10964
10. Beltagy et al., *SciBERT: A Pretrained Language Model for Scientific Text*, 2019.  
   https://arxiv.org/abs/1903.10676
11. Ouyang et al., *Training language models to follow instructions with human feedback* (InstructGPT), 2022.  
   https://arxiv.org/abs/2203.02155
12. Hinton et al., *Distilling the Knowledge in a Neural Network*, 2015.  
   https://arxiv.org/abs/1503.02531
13. Gu et al., *MiniLLM: On-Policy Distillation of Large Language Models*, 2023.  
   https://arxiv.org/abs/2306.08543
14. Yang et al., *Qwen2.5 Technical Report*, 2024.  
   https://arxiv.org/abs/2412.15115
15. Hui et al., *Qwen2.5-Coder Technical Report*, 2024.  
   https://arxiv.org/abs/2409.12186
16. Lozhkov et al., *StarCoder 2 and The Stack v2: The Next Generation*, 2024.  
   https://arxiv.org/abs/2402.19173
17. Guo et al., *DeepSeek-Coder: When the Large Language Model Meets Programming -- The Rise of Code Intelligence*, 2024.  
   https://arxiv.org/abs/2401.14196
18. Puri et al., *Project CodeNet: A Large-Scale AI for Code Dataset for Learning a Diversity of Coding Tasks*, 2021.  
   https://arxiv.org/abs/2105.12655
19. Husain et al., *CodeSearchNet Challenge: Evaluating the State of Semantic Code Search*, 2019.  
   https://arxiv.org/abs/1909.09436
