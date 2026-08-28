---
title: "Reading Map"
description: "The deliberately small resource stack used to learn, implement, validate, and compare each mechanism."
slug: "reading-map"
aliases: ["/reading-map/"]
---

No single course or repository covers the complete research-engineering loop. Each resource has one job, and production implementations are consulted only after the corresponding mechanism has been attempted and tested.

## Resource-use rule

For every module:

1. Read the assignment or primary paper.
2. Write the derivation and expected failure signatures.
3. Implement the smallest trustworthy version.
4. Compare it with a production-oriented reference only after its tests pass.
5. Add one controlled experiment that was not supplied by the course or repository.

## Curriculum spine

- [Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu/) supplies the assignment spine.
- [nanochat](https://github.com/karpathy/nanochat) is the minimal end-to-end comparison after the mechanism is understood.
- [OLMo](https://allenai.org/olmo) provides a transparent, production-oriented model lifecycle reference.
- The [Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook) governs hypotheses, baselines, tuning order, and failed-run interpretation.

## Specialized references

- Data: [DataComp-LM](https://arxiv.org/abs/2406.11794), FineWeb, Dolma, and DoReMi.
- GPU systems: the [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/), Nsight, Triton tutorials, and FlashAttention.
- Distributed training: [PyTorch DTensor/FSDP2](https://docs.pytorch.org/docs/stable/distributed.tensor.html), TorchTitan, and Megatron Core.
- Post-training: [The RLHF Book](https://rlhfbook.com/), InstructGPT, DPO, Open-Instruct, and verl.
- Evaluation: HELM, [Inspect AI](https://inspect.aisi.org.uk/), and lm-evaluation-harness.
- Serving: [vLLM](https://docs.vllm.ai/) and the PagedAttention paper, with SGLang used only for a bounded comparison.

The exact versions, revisions, and access dates used by an experiment belong in that experiment's run manifest. The map is a route, not a substitute for the primary record.
