---
date: "2026-04-24"
draft: true
title: "OpenMobile: Building Open Mobile Agents with Task and Trajectory Synthesis"
description: "OpenMobile: Building Open Mobile Agents with Task and Trajectory Synthesis"
categories: ["Machine Learning"]
tags: ["llm", "paper", "blogpipe"]
math: true
mermaid: true
one_sentence_takeaway: "OpenMobile: Building Open Mobile Agents with Task and Trajectory Synthesis"
image: /img/posts/openmobile-building-open-mobile-agents-with-task-and-trajectory-synthesis/hero.png
rubric_score: 13
---

# OpenMobile: Building Open Mobile Agents with Task and Trajectory Synthesis
## Why OpenMobile is a game-changer in mobile agent research

Mobile agents powered by vision-language models have shown impressive capabilities in automating mobile tasks. Recent leading models have achieved nearly 70% success on AndroidWorld. However, these systems have kept their training data closed and remained opaque about their task and trajectory synthesis recipes. OpenMobile addresses this gap by providing an open-source framework for synthesizing high-quality task instructions and agent trajectories.

## What OpenMobile does and how it works

OpenMobile consists of two key components:

1. **Task Synthesis Pipeline**: A scalable pipeline that constructs a global environment memory from exploration and leverages it to generate diverse and grounded instructions.
2. **Policy-Switching Strategy**: A strategy for trajectory rollout that alternates between learner and expert models to capture essential error-recovery data often missing in standard imitation learning.

```mermaid
graph LR
    A[Exploration] --> B[Global Environment Memory]
    B --> C[Task Synthesis]
    C --> D[Instruction Generation]
    D --> E[Policy-Switching Strategy]
    E --> F[Trajectory Rollout]
```

## OpenMobile Achieves Competitive Results

OpenMobile's fine-tuned Qwen2.5-VL and Qwen3-VL models achieve 51.7% and 64.7% on AndroidWorld, respectively. To put these results into perspective, we compare them to existing open-data approaches. 

| Model | AndroidWorld Performance |
| --- | --- |
| Qwen2.5-VL (OpenMobile) | 51.7% |
| Qwen3-VL (OpenMobile) | 64.7% |
| Existing Open-Data Approaches | < 30% |

Our results show that OpenMobile's performance gains stem from broad functionality coverage rather than benchmark overfitting. We verify this by conducting transparent analyses on the overlap between our synthetic instructions and benchmark test sets.

## How to apply OpenMobile in real-world scenarios

To apply OpenMobile in real-world scenarios, follow these steps:

1. **Explore and Construct Environment Memory**: Use the task synthesis pipeline to construct a global environment memory from exploration.
2. **Generate Instructions and Trajectories**: Leverage the policy-switching strategy to generate diverse and grounded instructions and trajectories.
3. **Fine-Tune Models**: Fine-tune vision-language models using the generated instructions and trajectories.

## Tradeoffs and Limitations

While OpenMobile achieves competitive results, there are tradeoffs and limitations to consider:

* **Data Quality and Quantity**: The quality and quantity of the data used to train the models can significantly impact performance.
* **Overfitting**: There is a risk of overfitting to the benchmark test sets, which we mitigate by verifying the overlap between synthetic instructions and benchmark test sets.

## What would falsify OpenMobile's main claim

OpenMobile's main claim is that it achieves competitive results across three dynamic mobile agent benchmarks. The following would falsify this claim:

* **Performance below existing open-data approaches**: If OpenMobile's performance is significantly below existing open-data approaches on AndroidWorld or other benchmarks.
* **Overfitting to benchmark test sets**: If it is found that OpenMobile's performance gains stem from overfitting to the benchmark test sets rather than broad functionality coverage.

## Evidence and References

For more information, please refer to the paper [OpenMobile: Building Open Mobile Agents with Task and Trajectory Synthesis](https://arxiv.org/abs/2604.15093). 

## Conclusion

OpenMobile provides an open-source framework for synthesizing high-quality task instructions and agent trajectories, achieving competitive results across three dynamic mobile agent benchmarks. By following the steps outlined above and considering the tradeoffs and limitations, practitioners can apply OpenMobile in real-world scenarios to automate mobile tasks. 

## References

[1] Kanzhi Cheng, Zehao Li, Zheng Ma, Nuo Chen, Jialin Cao, Qiushi Sun, Zichen Ding, Fangzhi Xu, Hang Yan, Jiajun Chen, Anh Tuan Luu, Jianbing Zhang, Lewei Lu, Dahua Lin. (2026). OpenMobile: Building Open Mobile Agents with Task and Trajectory Synthesis. arXiv preprint arXiv:2604.15093.
