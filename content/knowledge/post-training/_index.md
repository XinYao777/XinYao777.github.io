---
title: "Post-training"
description: "从 SFT、RL、Preference Learning 到 On-policy Distillation 的后训练知识地图。"
---

## Map

[
\text{Post-training}
=
\text{SFT}
+
\text{Preference/RL}
+
\text{Distillation}
]

- **SFT**
  - Instruction Tuning
  - Sequence Distillation
  - Data Mixture
- **RL / Preference Learning**
  - Reward Model
  - Verifier
  - Exploration
  - [On-policy Learning](/knowledge/post-training/on-policy-learning/)
- **Distillation**
  - Offline Distillation
  - On-policy Distillation
  - OPD
  - Black-box Distillation

这里重点记录不同后训练方法改变的到底是 **监督信号、访问到的状态分布，还是策略本身**。
