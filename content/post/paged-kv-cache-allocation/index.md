---
title: "The KV Cache Is an Allocator Problem"
date: 2026-07-19
lastmod: 2026-07-20
draft: false
slug: "paged-kv-cache-allocation"
author: "Anshuman Sahoo"
image: "/images/paged-kv-cache-allocation/allocator-layout.svg"
description: "A small simulator exposes why contiguous KV-cache reservation wastes memory, what paging fixes, and what block size cannot fix."
summary: "A small allocator simulator separates KV-cache capacity failures from attention kernels, schedulers, and batching effects."
archetype: "Systems Microscope"
series: "Frontier Research Engineer Curriculum"
series_order: 8
categories:
  - Inference and serving
  - GPU systems and performance engineering
  - Data quality and data systems
tags: ["llm-inference", "kv-cache", "paged-attention", "systems", "vLLM"]
---

A serving system can have enough GPU memory for every live token and still reject a new request. The failure is not necessarily attention, model size, or arithmetic intensity. It can be the allocator.

This article rebuilds the memory-management argument behind paged KV caches with a small, executable simulator. The experiment is deliberately narrower than a serving benchmark: it asks how three allocation policies behave under the same arrival times, prompt lengths, generation lengths, and completion times. That separation matters. It lets us test the allocator before attributing throughput changes to kernels, schedulers, or batching.

## One request, two kinds of waste

During autoregressive decoding, each sequence grows one token at a time. Its key-value cache therefore grows dynamically. A simple implementation can reserve one contiguous region sized for the sequence's maximum possible length. This makes addressing easy, but it creates two distinct losses:

1. **Reservation waste:** memory is held for tokens that have not been generated.
2. **External fragmentation:** free memory exists, but not as one sufficiently large contiguous interval.

PagedAttention was introduced to avoid requiring each sequence's KV cache to occupy one contiguous physical region. It divides the cache into fixed-size blocks and uses a block table to map logical token positions to physical blocks. The original vLLM paper reports that reducing fragmentation and enabling sharing increased serving throughput in its evaluated systems, but those end-to-end results combine allocation, batching, kernels, and scheduling. This simulator isolates only the allocation claim.

Consider a cache with 80 token slots. Three requests reserve 24, 16, and 16 contiguous slots. If the middle request finishes, 16 slots are free. A new request needing 20 slots still cannot start: the largest free interval is 16. A block allocator can use free blocks from multiple locations because the logical sequence is no longer tied to one physical interval.

The featured diagram shows this contrast: contiguous allocation can have enough total free capacity while still lacking one large interval, while paged allocation can assemble a logical cache from smaller physical blocks.

The block table adds indirection. A decode kernel must translate a logical token position into a block ID and an offset. That makes memory layout and access order performance concerns. NVIDIA's CUDA guidance emphasizes coalesced global-memory access because scattered warp accesses require more memory transactions. Paging fixes capacity loss; it does not guarantee an efficient kernel.

## The experiment

The simulator compares three policies:

- **Maximum reservation:** reserve one contiguous interval for `prompt + max_new_tokens`.
- **Exact contiguous growth:** extend a sequence contiguously as tokens arrive; relocation is disallowed.
- **Paged growth:** allocate fixed-size blocks on demand.

Every policy receives the same deterministic event trace. A request is admitted only if its initial prompt cache can be allocated. At each decode step, the allocator tries to grow every active request by one token. We record admitted and rejected requests, live tokens, reserved slots, internal waste, free capacity, largest contiguous free interval, and allocation failures despite sufficient total free capacity.

```bash
python code/kv_allocator.py --scenario data/scenario.json --output outputs/results.json
python tests/test_allocator.py
python code/make_figure.py --results outputs/results.json --output figures/allocator-layout.svg
```

The trace is synthetic. It is not evidence for a specific model, GPU, or production throughput number. Its role is to make allocator behavior inspectable and falsifiable.

## What the trace shows

Across the supplied trace, maximum reservation fails first because it holds space for future tokens. Exact contiguous growth uses less reserved memory early, but it eventually encounters external fragmentation: total free capacity is sufficient while the largest free interval is too small. Paged growth with four-slot blocks admits the full trace. Larger blocks fail in this deliberately tight capacity because their tail waste consumes enough slots to block later growth. Internal waste remains bounded by the unfilled tail of each sequence's final block, but that bound can still matter operationally.

| Policy | Rejected requests | Growth failures | Peak reserved slots | Peak waste |
|---|---:|---:|---:|---:|
| Maximum reservation | 2 | 0 | 70 | 33 |
| Exact contiguous growth | 0 | 4 | 31 | 0 |
| Paged, block size 4 | 0 | 0 | 64 | 8 |
| Paged, block size 8 | 0 | 1 | 72 | 13 |
| Paged, block size 16 | 1 | 1 | 80 | 25 |

The key result is not that paging makes memory free. It changes the unit of allocation:

\[
\text{physical address} =
\text{block table}[\lfloor t/B \rfloor] \times B + (t \bmod B),
\]

where \(t\) is a logical token position and \(B\) is the block size.

That mapping removes the requirement that a sequence's physical cache be contiguous. The cost is that block size now controls a trade-off.

Small blocks reduce tail waste but enlarge block tables and increase allocation operations. Large blocks reduce metadata and allocation frequency but increase internal fragmentation. For a sequence with \(L\) cached tokens, a fixed block size \(B\) reserves

\[
B\left\lceil \frac{L}{B}\right\rceil
\]

slots, so tail waste is between zero and \(B-1\) slots per sequence.

The simulator tests block sizes 4, 8, and 16. The four-slot policy completes the trace, while the larger blocks encounter capacity failures caused by greater tail waste. The smallest block size also has the lowest reserved-slot waste in this trace. That does **not** establish that it would have the best serving throughput. Kernel efficiency, metadata traffic, prefix sharing, scheduler overhead, and hardware behavior can reverse the choice.

## What paging does not solve

A block allocator is only one layer of an inference engine.

It does not decide which request should decode next. It does not make attention computation sublinear in context length. It does not ensure that gathered KV reads are coalesced. It does not remove the need for admission control, preemption, prefix-cache policy, or latency objectives.

There are also credible alternatives. vAttention keeps the KV cache contiguous in virtual address space while mapping physical memory dynamically through CUDA virtual-memory mechanisms. Its authors argue that this preserves compatibility with existing attention kernels and avoids some PagedAttention complexity. That comparison separates the goal-dynamic physical allocation-from one particular implementation.

PyTorch's scaled-dot-product-attention interface can dispatch among fused implementations, and its documentation notes backend-specific constraints. A production design therefore has to reconcile allocator layout with the kernels selected for the target shapes and hardware.

## A practical design rule

Treat the KV cache as a memory subsystem with an explicit contract:

- the scheduler owns request admission and lifetime;
- the allocator owns logical-to-physical mapping and reclamation;
- the attention kernel owns efficient reads through that mapping;
- telemetry distinguishes internal waste, external fragmentation, allocation latency, and kernel cost.

Then test each layer separately before running an end-to-end throughput benchmark.

The simulator in this bundle is a mastery artifact contract, not evidence of mastery. A stronger follow-up would implement the same trace against a real serving engine, collect allocator and kernel timelines, and test whether the block size that minimizes waste also minimizes time per output token. The likely answer is “not always”-which is exactly why allocator correctness should be established before performance is interpreted.

## References

- Woosuk Kwon et al., ["Efficient Memory Management for Large Language Model Serving with PagedAttention"](https://doi.org/10.1145/3600006.3613165), SOSP 2023.
- Ramya Prabhu et al., ["vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention"](https://arxiv.org/abs/2405.04437), 2024.
- NVIDIA, [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html).
- PyTorch, [`torch.nn.functional.scaled_dot_product_attention`](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html).
