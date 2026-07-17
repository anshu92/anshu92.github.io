---
title: "A complete meandering journey through the exciting world of Deep Learning"
date: 2026-06-06
description: "Study notes as I refresh and explore corners of the topics from a technical ML Engineering perspective"
tags: ["LLM", "Training", "Transformers", "Inference"]
categories: ["GPU systems and performance engineering", "Distributed training", "Inference and serving", "Deep-learning mechanisms"]
draft: true
---

## GPU Architecture
A core is an individual processing unit in a CPU or GPU that reads a single instruction and executes it. CPUs have fewer, very powerful cores that can quickly process a lot of instructions sequentially. On the other hand, GPUs have thousands of smaller cores grouped into units called **Streaming Multiprocessors(SMs)**.

When a function(called a kernel) is executed on a GPU, the work is organized in a strict hierarchy:
- **Thread:** Smallest unit of execution.
- **Warp:** A group of exactly 32 threads. GPU schedules work at this level - all 32 threads in a warp execute the exact same instruction at the exact same time (this is called **SIMT: Single Instruction, Multiple Threads**).
- **Blocks:** A group of warps that are assigned to the same SM. Threads within the same block can communicate with each other using fast, localized shared memory.
