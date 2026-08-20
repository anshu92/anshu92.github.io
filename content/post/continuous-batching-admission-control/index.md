---
title: "Continuous Batching Needs Admission Control"
date: 2026-07-20
lastmod: 2026-07-20
draft: false
slug: "continuous-batching-admission-control"
author: "Anshuman Sahoo"
image: "/images/continuous-batching-admission-control/cover.svg"
description: "A small simulator shows why token-budgeted batching alone can over-admit requests, trigger KV-cache preemption, and damage tail latency."
summary: "Continuous batching is a compute scheduler and a memory admission problem. A small simulator shows why pricing only the next token chunk can create future KV-cache failures."
archetype: "Systems Architect"
series: "Frontier Research Engineer Curriculum"
series_order: 9
categories:
  - Inference and serving
  - GPU systems and performance engineering
  - Reliability and observability
toc_category: "Infrastructure & Scaling"
tags: ["llm-inference", "vLLM", "continuous-batching", "kv-cache", "scheduling"]
competencies: ["admission control", "KV-cache capacity", "tail latency", "serving simulators"]
prerequisites: ["autoregressive decoding", "continuous batching", "KV cache"]
current_role_tracks: ["CR2", "CR5"]
frontier_tracks: ["FR3", "FR5"]
math_level: "intermediate"
code_level: "executable"
---

Continuous batching is often described as a scheduling win: instead of waiting for an entire static batch to finish, the server inserts new requests between decoding steps. That description is correct, but incomplete. A production scheduler is not merely choosing which tokens to compute next. It is also deciding which requests are allowed to become memory obligations.

That distinction matters because the compute budget and the KV-cache budget evolve differently. A scheduler may have room for another prompt chunk in the current iteration while lacking enough cache capacity to carry the request through its full input and subsequent decode. Admitting it can look locally efficient and still create a system-level failure: cache exhaustion, preemption, recomputation, and sharply worse tail latency.

This article reconstructs that failure with a deterministic discrete-event simulator. The simulator is deliberately small. It does not estimate GPU kernel time, network overhead, or exact vLLM internals. It isolates one architectural question:

> What information must an admission decision reserve before a waiting request joins the running set?

## One scheduler step has two budgets

Consider a server with block-granular KV-cache allocation. At every engine step, the scheduler must respect at least two limits:

1. **A token execution budget.** This caps how many prompt or decode tokens may be processed in the next model invocation.
2. **A persistent cache budget.** This caps how many KV blocks the running requests may own after the invocation.

vLLM exposes related controls including `max_num_batched_tokens`, `max_num_seqs`, a full-input reservation option, and a free-block watermark. Its documentation explicitly describes full-input reservation as protection against over-admission and cache thrashing under chunked prefill. It also describes preemption and recomputation when KV-cache space becomes insufficient.

The important asymmetry is this: the token budget resets every step; allocated KV state persists across steps.

Suppose a request has a 320-token prompt and the cache block size is 16 tokens. The first prefill chunk might consume only 64 scheduler tokens, but completing the input requires 20 cache blocks. An admission check that asks only whether the first chunk fits treats a 20-block obligation as a one-block decision.

The request is cheap *now* and expensive *over its lifetime*.

![Throughput remains similar across policies while cache behavior changes](/images/continuous-batching-admission-control/throughput.svg)

## The mechanism model

The package includes `code/simulate_admission.py`. Each synthetic request has:

- an arrival step;
- prompt length;
- requested output length;
- prefilled and decoded token counters;
- block-granular KV ownership;
- timestamps for admission, first token, and completion;
- a preemption counter.

Each scheduler step performs five operations:

1. Move newly arrived requests into the waiting queue.
2. Admit waiting requests while sequence slots and the admission policy permit.
3. Spend the token budget on one-token decode work first.
4. Spend the remaining budget on chunked prefills.
5. If physical cache use exceeds capacity, preempt the newest running requests and reset their computed state.

The fifth operation models recomputation-based recovery. It is intentionally punitive because the request loses its previously computed prefix. That is not an arbitrary penalty; it is the direct cost of treating temporary scheduler progress as durable capacity.

Three policies are compared:

| Policy | Admission test |
|---|---|
| `naive` | Admit when one new block and a sequence slot are available |
| `reserve` | Admit only when the full prompt can fit |
| `reserve_watermark` | Reserve the full prompt and retain 12.5% cache headroom |

All three use the same FCFS workload, token budget, cache size, sequence limit, and random seed. The outputs are synthetic observations from this model, not claimed measurements of vLLM or any GPU.

## What the synthetic run shows

The committed output in `outputs/metrics.json` reports:

| Policy | Completed | Preemptions | P95 TTFT | Throughput |
|---|---:|---:|---:|---:|
| naive | 30 | 1923 | 393 steps | 0.060 req/step |
| reserve | 30 | 115 | 399 steps | 0.060 req/step |
| reserve + watermark | 30 | 77 | 416 steps | 0.060 req/step |

![Preemptions by policy](/images/continuous-batching-admission-control/preemptions.svg)

The naive policy admits more aggressively because it prices only the first cache block. During bursts, several long prompts enter the running set together. Their later chunks expand cache ownership until the set no longer fits. The simulator then preempts recently admitted work, returns it to the queue, and recomputes it later.

The guarded policies appear more conservative at admission time, but they prevent the running queue from accumulating obligations the cache cannot satisfy. In this workload, that reduces repeated work and improves the stability of time-to-first-token.

![P95 TTFT by policy](/images/continuous-batching-admission-control/p95-ttft.svg)

The watermark adds a different kind of protection. Full-input reservation asks, “Can these prompts fit?” A watermark asks, “Should the server consume the last available blocks even when a fit is technically possible?” Headroom gives decode growth, allocator granularity, and workload estimation error somewhere to go.

The model therefore overturns a common intuition: maximizing the number of active sequences is not equivalent to maximizing useful throughput. A sequence that will soon be preempted is not productive concurrency.

## Admission control is a contract, not a heuristic

A robust admission decision needs an explicit resource contract. For a request \(r\), define:

- \(P_r\): prompt tokens not yet represented in cache;
- \(O_r\): remaining output-token allowance or estimate;
- \(B\): cache block size;
- \(K_{\text{free}}\): currently free blocks;
- \(W\): reserved watermark blocks.

A minimum prompt-safe admission condition is:

\[
\left\lceil \frac{P_r}{B} \right\rceil \le K_{\text{free}} - W
\]

This does not reserve the full decode lifetime. Doing so from `max_tokens` can be far too conservative, especially when clients specify loose upper bounds. But the equation makes the policy boundary visible: the scheduler commits to completing the input without relying on future eviction.

A production policy can extend the contract in several directions:

- reserve the full input but let decode grow incrementally;
- estimate output length from request class or historical data;
- route long-context requests to a separate capacity pool;
- cap concurrent long prefills;
- reject requests that can never fit;
- expose queueing and preemption as service-level signals.

The correct choice depends on the service objective. Interactive chat prioritizes TTFT and fairness differently from offline generation. A single FCFS queue may be acceptable for one and harmful for the other.

## What to measure before changing the policy

Admission control should not be tuned from average throughput alone. At minimum, record:

- waiting and running queue depth;
- free and used KV blocks;
- admitted prompt-token obligations;
- preemptions and recomputed tokens;
- TTFT and end-to-end latency by prompt-length bucket;
- completion throughput;
- rejection counts;
- cache-watermark violations;
- sequence-slot utilization versus token-budget utilization.

A useful diagnostic is **recomputed tokens per completed request**. When that value rises, the server may still show high GPU utilization while wasting work. Another is the correlation between long-prompt admission bursts and subsequent preemption. The timeline CSVs in this package make both patterns inspectable.

Rollout should begin with shadow decisions: compute what the guarded policy would admit without changing live scheduling. Compare predicted queueing, headroom, and rejected admissions against actual preemptions. Then enable the policy for a small traffic slice with rollback thresholds on P95 TTFT, throughput, and rejection rate.

## The boundary of this experiment

This simulator proves only a mechanism-level claim: under block-limited cache capacity, first-chunk admission can create future memory obligations that trigger recomputation; reserving full prompt capacity prevents that planted failure in the modeled workload.

It does not prove the best vLLM configuration, a universal watermark, or a GPU-level speedup. Real serving adds prefix caching, speculative decoding, heterogeneous attention types, tensor parallelism, asynchronous scheduling, CUDA graphs, host overhead, and model-specific cache geometry. Those factors can change the optimal policy.

The durable lesson is narrower and more useful: continuous batching needs an admission layer that prices persistent state, not only immediate compute. Without that contract, the scheduler can stay busy while the system moves backward.

## References

- Woosuk Kwon et al., [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180).
- vLLM documentation on [scheduler configuration](https://docs.vllm.ai/en/stable/api/vllm/config/scheduler/), including token and sequence limits, full-input reservation, and watermarking.
- vLLM documentation on [optimization and tuning](https://docs.vllm.ai/en/latest/configuration/optimization/), including KV-cache preemption and recomputation under cache pressure.
