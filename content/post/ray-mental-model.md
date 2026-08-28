---
title: "Ray mental model: tasks, actors, objects, scheduling, placement"
date: 2026-08-28
description: "Ray is best understood as a distributed execution substrate for Python programs."
tags: ["Ray", "distributed systems", "tasks", "actors", "scheduling", "placement groups"]
categories: ["Distributed training", "Inference and serving", "Software architecture"]
toc_category: "Infrastructure & Scaling"
draft: true
---

Ray is best understood as a distributed execution substrate for Python programs. It lets an application express stateful and stateless units of work while the runtime handles placement, resource accounting, object references, retries, and cluster membership. Ray Data, Ray Train, Ray Serve, and many RL stacks are higher-level patterns on top of these primitives.

| Primitive | Use it for | Systems implication |
|---|---|---|
| Task | Stateless or short-lived distributed function calls | Cheap parallel fan-out; dependencies are object refs; retries can be safe when work is idempotent. |
| Actor | Stateful, long-lived process such as a model server, environment pool, cache manager, or trainer coordinator | Owns mutable state and resources; lifecycle/failure semantics matter; ideal for GPU-bound stateful services. |
| Object reference / object store | Passing immutable results between tasks and actors | Enables zero/low-copy local sharing where possible, distributed ownership, spilling, and backpressure-sensitive pipelines. |
| Placement group | Reserve/arrange bundles of CPUs/GPUs across nodes | Expresses co-location or anti-affinity constraints; critical when NCCL/NVLink/RDMA topology matters. |
| Resource labels | CPU, GPU, custom accelerator or logical resource quantities | Turns scheduling into explicit resource matching instead of hidden process assumptions. |
| Autoscaling + job/runtime environment | Cluster elasticity and dependency isolation | Useful for bursty data/inference jobs; dangerous if startup/model-loading time is ignored in SLO planning. |

The critical design choice is where state lives. If a unit of work is cheap and recomputable, prefer tasks. If it owns an expensive model, cache, connection pool, simulator, or device context, prefer actors. Once actors own scarce accelerators, placement and backpressure become architecture, not implementation detail.

```python
# Conceptual Ray pattern
@ray.remote(num_gpus=1)
class RolloutWorker:
    def **init**(self, model):
        self.engine = load_inference_engine(model)

    def generate(self, prompts, policy_version):
        return trajectories(prompts, policy_version)

@ray.remote(num_gpus=8)
class Learner:
    def update(self, batch):
        return new_policy_version

# The orchestrator moves references/metadata, not Python objects by value.
```

## References

- [Ray Core walkthrough](https://docs.ray.io/en/latest/ray-core/walkthrough.html)
- [Ray Core key concepts](https://docs.ray.io/en/latest/ray-core/key-concepts.html)
- [Ray tasks](https://docs.ray.io/en/latest/ray-core/tasks.html)
- [Ray actors](https://docs.ray.io/en/latest/ray-core/actors.html)
- [Ray objects](https://docs.ray.io/en/latest/ray-core/objects.html)
- [Ray scheduling](https://docs.ray.io/en/latest/ray-core/scheduling/index.html)
- [Ray resources](https://docs.ray.io/en/latest/ray-core/scheduling/resources.html)
- [Ray placement groups](https://docs.ray.io/en/latest/ray-core/scheduling/placement-group.html)
- [Task fault tolerance](https://docs.ray.io/en/latest/ray-core/fault_tolerance/tasks.html)
- [Actor fault tolerance](https://docs.ray.io/en/latest/ray-core/fault_tolerance/actors.html)
- [Runtime environments](https://docs.ray.io/en/latest/ray-core/handling-dependencies.html)
- [Configuring autoscaling](https://docs.ray.io/en/latest/cluster/vms/user-guides/configuring-autoscaling.html)
