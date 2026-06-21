---
date: "2026-06-21"
draft: true
title: "Execution-State Capsules: Graph-Bound Execution-State Checkpoint and Restore for Low-Latency, Small-Batch, On-Device Physical-AI Serving - guided learning deep dive"
post_type: deep_dive
categories: ["Machine Learning"]
tags: ["research-radar", "llm", "mle"]
tracks: ["Applied-Research", "ML-Engineering"]
source_count: 1
paper_count: 1
blog_count: 0
math: true
mermaid: false
---
# Execution-State Capsules: Graph-Bound Execution-State Checkpoint and Restore for Low-Latency, Small-Batch, On-Device Physical-AI Serving

## Technical Thesis

In high-throughput, multi-tenant cloud environments, Large Language Model (LLM) serving systems optimize for concurrency by dynamically managing key-value (KV) caches using paged or radix-based memory allocation [E3, E6]. While highly effective for maximizing token throughput, this paradigm introduces significant latency overheads due to block-table indirection, dynamic memory allocation, and CPU-GPU synchronization [E5]. Furthermore, it manages only a single positional fragment of the model's execution state: the KV cache [E3]. Source: https://arxiv.org/abs/2606.20537

For low-latency, small-batch, on-device physical-AI serving—such as interactive LLM agents, real-time speech systems, and robotic policies—the serving requirements are fundamentally inverted [E4]. These workloads repeatedly branch, reset, interrupt, and re-enter execution paths under tight responsiveness budgets [E4]. In this regime, managing only the KV cache is insufficient; stateful architectures rely on a broader, heterogeneous set of execution states, including recurrent states, convolution states, multi-token prediction (MTP) states, and associated metadata [E8]. Source: https://arxiv.org/abs/2606.20537

This deep dive examines **Execution-State Capsules**, a graph-bound checkpoint and restore mechanism implemented in the **FlashRT** runtime [E2, E5]. By binding the complete restorable state of an execution boundary to a closed set of contiguous static buffers and running captured graph plans without block-table indirection, FlashRT achieves sub-millisecond, byte-exact state restoration [E5, E8, E9]. This architecture shifts the serving paradigm from token-addressed KV fragments to static, graph-bound execution-state boundaries, delivering up to a 27x speedup in Time-to-First-Token (TTFT) for long-context on-device serving [E7, E9].

---

## Prerequisites

To fully comprehend the design and implementation of Execution-State Capsules, you should be familiar with the following systems and machine learning concepts:

*   **CUDA Graph Execution**: The process of capturing a sequence of CUDA kernels as a single, executable graph to eliminate CPU launch overhead and kernel-dispatch latency.
*   **Memory Indirection & Paging**: How systems like vLLM use virtual memory-style block tables to map logical KV caches to non-contiguous physical GPU memory blocks, and the latency penalties associated with pointer chasing during kernel execution.
*   **Stateful Deep Learning Architectures**: The operational mechanics of non-pure-Transformer models, specifically those utilizing recurrent layers (e.g., state-space models like Mamba), convolutional layers (e.g., causal convolutions in audio/speech processing), and multi-token prediction (MTP) heads.
*   **On-Device Hardware Constraints**: The shared memory architectures, thermal limits, and latency budgets of edge computing platforms such as NVIDIA Jetson AGX system-on-chips (SoCs).

---

## Problem Framing: The Throughput-Latency Dichotomy in Physical AI

Existing LLM serving frameworks are architected around the assumption of high concurrency. In these cloud-based systems, the primary objective is to maximize aggregate token throughput across hundreds of concurrent requests. To prevent memory fragmentation and out-of-memory (OOM) errors, frameworks employ paged memory management (e.g., PagedAttention) [E3, E6]. 

While paged memory is highly effective for throughput, it introduces several critical limitations when applied to on-device physical AI [E4]: Source: https://arxiv.org/abs/2606.20537

1.  **State Fragmentation**: Paged systems only track and manage the KV cache [E3]. Modern physical-AI agents, however, are rarely pure Transformers; they frequently incorporate recurrent states, convolutional states, and metadata that must remain synchronized [E8].
2.  **Indirection Latency**: Paged attention mechanisms require looking up physical block addresses via a block table during execution [E5]. This indirection prevents the compiler and hardware from executing highly optimized, contiguous memory sweeps, introducing latency penalties that are unacceptable for real-time control loops [E4, E5].
3.  **Dynamic Preemption Overhead**: Physical-AI workloads are highly dynamic. A robot policy or speech agent may need to instantly interrupt its current generation, roll back to a previous decision point, fork a new execution branch to evaluate alternative actions, or reset entirely [E4]. In a paged serving system, these operations require complex block-table manipulation, metadata updates, and potential CPU-GPU synchronization, leading to latency spikes [E4, E5]. Source: https://arxiv.org/abs/2606.20537

```
Traditional Paged Serving (Cloud-Optimized):
[Logical Tokens] -> [Block Table Indirection] -> [Fragmented Physical KV Blocks] (KV Only)

FlashRT Capsule Serving (Edge-Optimized):
[Captured Graph Plan] -> [Contiguous Static Buffers] (KV + Recurrent + Conv + MTP + Metadata)
```

Execution-State Capsules address this gap by optimizing for the opposite regime: **low-latency, small-batch, on-device serving** [E4]. Instead of treating the model's state as a dynamic collection of token-addressed fragments, FlashRT treats the entire execution state at a committed boundary as a single, immutable, graph-bound capsule [E2, E7]. Source: https://arxiv.org/abs/2606.20537

---

## Method Walkthrough: Execution-State Capsules & FlashRT

The Execution-State Capsule methodology is built upon two core components: the **FlashRT** runtime and the **Capsule** state abstraction [E2, E5].

### 1. The FlashRT Runtime
FlashRT is a white-box, backend-facing kernel runtime designed to eliminate runtime overheads [E5]. Its evaluated NVIDIA CUDA backend operates on two core principles:
*   **Captured Graph Plans**: FlashRT captures the entire execution path of the model as a static CUDA Graph. This eliminates CPU-side kernel launch overhead, which is a major bottleneck in low-latency, small-batch serving.
*   **Contiguous Static Buffers**: Unlike paged runtimes, FlashRT runs its captured graph plans over pre-allocated, contiguous static buffers [E5]. There is no block-table indirection [E5]. Every kernel knows the exact, immutable memory address of its inputs, outputs, and intermediate states at compile/capture time. Source: https://arxiv.org/abs/2606.20537

### 2. The Capsule Abstraction
Because the live state of the model is bound to a closed set of named, contiguous static buffers, FlashRT can treat this state as a cohesive unit—an **Execution-State Capsule** [E8]. Source: https://arxiv.org/abs/2606.20537

A capsule represents the complete restorable state of the model at a specific committed boundary [E2]. This state is not limited to the KV cache; it encompasses the entire execution boundary [E8]:
*   **Key-Value (KV) Cache**: The traditional attention history.
*   **Recurrent State**: Hidden states for recurrent layers or state-space models (SSMs).
*   **Convolution State**: Causal delay lines and history buffers for convolutional layers.
*   **Multi-Token Prediction (MTP) State**: States associated with speculative or multi-token generation heads.
*   **Metadata**: Sequence lengths, attention masks, and internal runtime pointers. Source: https://arxiv.org/abs/2606.20537

### 3. Capsule Operations
Because the state is a closed set of named buffers, FlashRT can perform the following operations with sub-millisecond latency directly on the GPU [E8, E9]:

*   **Snapshot**: Copy the contents of the live named buffers to a GPU-resident capsule storage slot.
*   **Restore**: Overwrite the live named buffers with the contents of a stored capsule, instantly resetting the model's execution state to the committed boundary.
*   **Fork**: Clone a capsule to create a new, independent execution branch.
*   **Rollback**: Revert the live state to a previous capsule checkpoint after an interruption or branch evaluation.

### Conceptual Architecture Blueprint

The following conceptual C++ interface illustrates how the FlashRT runtime manages these named buffers and executes capsule operations over static, graph-bound memory:

```cpp
#include <cuda_runtime.h>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>

// Represents a named, contiguous static GPU buffer
struct NamedBuffer {
    std::string name;
    void* d_ptr;
    size_t bytes;
};

// An Execution-State Capsule containing a snapshot of the closed set of buffers
struct Capsule {
    std::unordered_map<std::string, std::vector<uint8_t>> host_backups; // For CPU-side storage (optional)
    std::unordered_map<std::string, void*> device_backups;             // GPU-resident backups
    size_t total_bytes;
};

class FlashRTRuntime {
private:
    // The closed set of named static buffers bound to the CUDA Graph
    std::unordered_map<std::string, NamedBuffer> live_state_buffers_;
    cudaGraphExec_t captured_graph_instance_;
    cudaStream_t execution_stream_;

public:
    FlashRTRuntime(cudaStream_t stream) : execution_stream_(stream), captured_graph_instance_(nullptr) {}

    // Register a static buffer. This must be done prior to CUDA Graph capture.
    void register_state_buffer(const std::string& name, void* d_ptr, size_t bytes) {
        live_state_buffers_[name] = {name, d_ptr, bytes};
    }

    // Capture the execution plan as a static CUDA Graph
    void capture_graph_plan() {
        // In practice, this wraps the forward pass of the model
        // cudaStreamBeginCapture(execution_stream_, cudaStreamCaptureModeGlobal);
        // ... launch kernels over registered static buffers ...
        // cudaStreamEndCapture(execution_stream_, &graph);
        // cudaGraphInstantiate(&captured_graph_instance_, graph, ...);
    }

    // Create a GPU-resident snapshot of the complete execution state
    std::shared_ptr<Capsule> snapshot() {
        auto capsule = std::make_shared<Capsule>();
        capsule->total_bytes = 0;

        for (const auto& [name, buffer] : live_state_buffers_) {
            void* d_backup = nullptr;
            cudaMalloc(&d_backup, buffer.bytes);
            
            // Perform a high-speed, GPU-resident device-to-device copy
            cudaMemcpyAsync(d_backup, buffer.d_ptr, buffer.bytes, 
                            cudaMemcpyDeviceToDevice, execution_stream_);
            
            capsule->device_backups[name] = d_backup;
            capsule->total_bytes += buffer.bytes;
        }
        cudaStreamSynchronize(execution_stream_);
        return capsule;
    }

    // Restore the complete execution state from a capsule
    void restore(const std::shared_ptr<Capsule>& capsule) {
        for (const auto& [name, buffer] : live_state_buffers_) {
            auto it = capsule->device_backups.find(name);
            if (it != capsule->device_backups.end()) {
                // Overwrite the live static buffer with the capsule state
                cudaMemcpyAsync(buffer.d_ptr, it->second, buffer.bytes, 
                                cudaMemcpyDeviceToDevice, execution_stream_);
            }
        }
        // Ensure all transfers are complete before next graph execution
        cudaStreamSynchronize(execution_stream_);
    }

    // Execute the captured graph plan over the current state of the static buffers
    void step() {
        if (captured_graph_instance_) {
            cudaGraphLaunch(captured_graph_instance_, execution_stream_);
        }
    }

    ~FlashRTRuntime() {
        if (captured_graph_instance_) {
            cudaGraphExecDestroy(captured_graph_instance_);
        }
    }
};
```

---

## Mathematical and Objective Interpretation

To understand why Execution-State Capsules are mathematically required for exact state restoration in physical AI, we must model the state transition dynamics of stateful neural networks.

### 1. The State Space of a Physical-AI Model
Let the complete state of a model at time step $t$ be represented by a tuple $\mathbf{S}_t$. In a pure Transformer architecture, this state is composed entirely of the key-value cache:

$$\mathbf{S}_t = \mathbf{S}_{KV}^{(t)}$$

However, in physical-AI agents utilizing hybrid architectures (e.g., Transformers combined with Recurrent, Convolutional, or Multi-Token Prediction layers), the state space is a heterogeneous union of distinct state variables [E8]: Source: https://arxiv.org/abs/2606.20537

$$\mathbf{S}_t = \left\{ \mathbf{S}_{KV}^{(t)}, \mathbf{S}_{Recurrent}^{(t)}, \mathbf{S}_{Conv}^{(t)}, \mathbf{S}_{MTP}^{(t)}, \mathbf{S}_{Metadata}^{(t)} \right\}$$

Where:
*   $\mathbf{S}_{KV}^{(t)}$ is the set of key-value tensors up to step $t$.
*   $\mathbf{S}_{Recurrent}^{(t)}$ represents the hidden state vectors of recurrent layers (e.g., SSMs, LSTMs).
*   $\mathbf{S}_{Conv}^{(t)}$ represents the sliding history buffers (causal delay lines) of convolutional layers.
*   $\mathbf{S}_{MTP}^{(t)}$ represents the internal state of multi-token prediction heads.
*   $\mathbf{S}_{Metadata}^{(t)}$ contains sequence lengths, attention masks, and position indices.

### 2. The State Transition Function
The model's forward pass can be defined as a deterministic state transition function $\mathcal{F}$ that maps the current input $\mathbf{X}_t$ and the prior state $\mathbf{S}_{t-1}$ to the output token distribution $\mathbf{Y}_t$ and the updated state $\mathbf{S}_t$:

$$\left( \mathbf{Y}_t, \mathbf{S}_t \right) = \mathcal{F}\left( \mathbf{X}_t, \mathbf{S}_{t-1}; \mathbf{\Theta} \right)$$

where $\mathbf{\Theta}$ represents the static model weights.

### 3. The Divergence of KV-Only Restoration
If an agent attempts to branch or roll back to a previous state $t = \tau$ using a **KV-only** restoration mechanism (as is typical in radix or paged KV-cache serving systems), the restored state is incomplete [E3, E10]:

$$\mathbf{S}_{\tau}^{\text{partial}} = \left\{ \mathbf{S}_{KV}^{(\tau)}, \mathbf{0}, \mathbf{0}, \mathbf{0}, \mathbf{S}_{Metadata}^{(\tau)} \right\}$$

When the model executes the next step $\tau + 1$ using this partial state, the transition function operates on corrupted state inputs:

$$\left( \mathbf{Y}_{\tau+1}', \mathbf{S}_{\tau+1}' \right) = \mathcal{F}\left( \mathbf{X}_{\tau+1}, \mathbf{S}_{\tau}^{\text{partial}}; \mathbf{\Theta} \right)$$

Because the recurrent and convolutional states were not restored ($\mathbf{S}_{Recurrent}^{(\tau)} \neq \mathbf{0}$ and $\mathbf{S}_{Conv}^{(\tau)} \neq \mathbf{0}$ in the true historical state), the output diverges:

$$\mathbf{Y}_{\tau+1}' \neq \mathbf{Y}_{\tau+1}$$

This mathematical divergence is empirically validated by the paper's **KV-only ablation study**, which demonstrates that omitting the recurrent state causes the model's output to diverge under greedy decoding [E10]. Thus, the recurrent and convolutional states are mathematically **load-bearing** for state restoration [E10]. Source: https://arxiv.org/abs/2606.20537

### 4. Byte-Exact and Token-Identical Equivalence
By capturing the closed set of named buffers, Execution-State Capsules guarantee that the restored state $\mathbf{S}_{\tau}^{\text{capsule}}$ is byte-exact to the original state $\mathbf{S}_{\tau}$ [E8, E9]:

$$\mathbf{S}_{\tau}^{\text{capsule}} \equiv \mathbf{S}_{\tau}$$

This byte-exact equivalence guarantees that for any sequence of subsequent inputs $\mathbf{X}_{\tau+1}, \dots, \mathbf{X}_{\tau+k}$, the generated outputs are token-identical to those generated from the original state under greedy decoding [E9]: Source: https://arxiv.org/abs/2606.20537

$$\mathcal{F}\left( \mathbf{X}_{\tau+1}, \mathbf{S}_{\tau}^{\text{capsule}}; \mathbf{\Theta} \right) \equiv \mathcal{F}\left( \mathbf{X}_{\tau+1}, \mathbf{S}_{\tau}; \mathbf{\Theta} \right)$$

---

## Empirical Evaluation & Experiment Analysis

The performance and correctness of Execution-State Capsules were evaluated across multiple hardware platforms, including the NVIDIA RTX 5090, Jetson AGX Thor, and DGX Spark [E9]. Source: https://arxiv.org/abs/2606.20537

### 1. Correctness and Fidelity
On all evaluated platforms (RTX 5090, Jetson AGX Thor, and DGX Spark), the capsule restore mechanism demonstrated two fundamental correctness properties [E9]:
*   **Byte-Exact State Restoration**: Bit-wise comparison of the GPU memory buffers before snapshot and after restore confirmed $several\%$ identity at the stored-state level [E9].
*   **Token-Identical Generation**: Under greedy decoding, the model's output sequence following a restore operation was identical to the sequence generated during the initial run [E9]. Source: https://arxiv.org/abs/2606.20537

### 2. The Load-Bearing Nature of Non-KV State
To isolate the importance of non-KV state variables, the researchers conducted a **KV-only ablation study** [E10]. In this experiment, only the KV-cache buffers were restored, while recurrent and convolutional states were initialized to zero or left un-restored. Source: https://arxiv.org/abs/2606.20537

*   **Result**: The model's generation path diverged from the reference path [E10]. This divergence proves that for modern physical-AI models, the recurrent and convolutional states are highly load-bearing and must be preserved to maintain generation fidelity [E10]. Source: https://arxiv.org/abs/2606.20537

### 3. Latency and Speedup Metrics
The primary performance advantage of Execution-State Capsules is the elimination of the "cold prefill" phase when branching or resetting. 

*   **Snapshot/Restore Latency**: On the RTX 5090, GPU-resident snapshot and restore operations are **sub-millisecond** [E9]. Because the operations consist of high-bandwidth device-to-device memory copies over contiguous buffers, they bypass the host CPU and avoid kernel launch overhead [E5, E9].
*   **Time-to-First-Token (TTFT) Speedup**: The speedup of capsule-based restoration over a cold prefill (re-computing the prefix from scratch) scales dramatically with context length [E9]: Source: https://arxiv.org/abs/2606.20537

| Context Length (Tokens) | TTFT Speedup (vs. Cold Prefill) |
| :--- | :--- |
| **2,000 (2k)** | **3.9x** [E9] |
| **16,000 (16k)** | **several.0x** [E9] | Source: https://arxiv.org/abs/2606.20537

```
TTFT Speedup Scaling:
  30x |                                                     * (27x)
  25x |                                                    
  20x |                                                   
  15x |                                                 
  10x |                                                
   5x |                       * (3.9x)
   0x +-----------------------+-----------------------------
     0k                      2k                            16k
                             Context Length (Tokens)
```

### Analysis of the Speedup Curve
The non-linear scaling of the TTFT speedup (from 3.9x to 27x) is a direct consequence of the computational complexity of the prefill phase. 

In a standard prefill, computing the KV cache for a prefix of length $N$ scales quadratically $\mathcal{O}(N^2)$ for standard attention, or linearly $\mathcal{O}(N)$ with a high constant factor for linear attention/SSMs. 

In contrast, the capsule restore operation is a memory-bandwidth-bound copy operation that scales linearly $\mathcal{O}(M)$ with the size of the state $M$. Because the state size $M$ is tightly bound and does not scale quadratically, the relative speedup of restoring a capsule versus re-running the prefill grows larger as the context length increases [E9]. Source: https://arxiv.org/abs/2606.20537

---

## Reproduction Notes & Architectural Blueprint

To implement an Execution-State Capsule system similar to FlashRT on an NVIDIA CUDA backend, engineers should follow this architectural blueprint:

### 1. Memory Layout: Contiguous Static Buffers
To eliminate block-table indirection, you must bypass dynamic allocation frameworks during active inference.
*   **Pre-allocation**: Allocate a single, contiguous block of GPU memory for each state variable (KV, recurrent, convolution, MTP, metadata) at startup.
*   **No Paging**: Avoid using page tables or virtual block mapping. The physical memory addresses must remain static throughout the lifetime of the execution graph.

### 2. CUDA Graph Capture
To achieve sub-millisecond execution, the inference pass must be captured as a CUDA Graph.
*   **Capture Phase**: Warm up the model kernels using the pre-allocated static buffers. Begin CUDA stream capture (`cudaStreamBeginCapture`), execute the forward pass, and end capture (`cudaStreamEndCapture`).
*   **Instantiation**: Instantiate the captured graph (`cudaGraphInstantiate`). This bakes the exact physical memory pointers of the static buffers directly into the kernel launch parameters, completely eliminating CPU-side argument serialization and launch overhead during the inference loop.

### 3. GPU-Resident Snapshot/Restore Kernels
To maintain sub-millisecond performance, state copy operations must never round-trip to host (CPU) memory.
*   **Device-to-Device Copies**: Use `cudaMemcpyAsync` with `cudaMemcpyDeviceToDevice` on the same stream as the inference execution.
*   **Memory Pool**: Maintain a pre-allocated pool of "Capsule Slots" in GPU memory. A snapshot operation is simply an asynchronous copy from the active static buffers to an idle capsule slot in the pool.
*   **Stream Alignment**: Ensure that the restore copy operations are queued on the same CUDA stream as the graph launch to prevent the need for heavy CPU-side stream synchronization.

---

## Limits & Boundary Conditions

While Execution-State Capsules offer massive latency advantages for physical AI, they are not a universal replacement for cloud-scale serving architectures [E1]. Source: https://arxiv.org/abs/2606.20537

### 1. Memory Footprint and Allocation Rigidity
Because capsules rely on contiguous static buffers, they lack the memory efficiency of paged memory systems [E5]. 
*   **Over-allocation**: Static buffers must be sized for the maximum supported sequence length and batch size upfront. This leads to low memory utilization if the average sequence length is much smaller than the Source: https://arxiv.org/abs/2606.20537
