---
title: "On-policy Learning"
description: "在当前策略实际访问到的状态分布上学习。"
status: evolving
date: 2026-09-20
lastmod: 2026-09-20
type: knowledge
related: ["SFT", "RL", "OPD", "Distribution Shift"]
sources: []
---

## 一句话

**On-policy Learning 的关键不是“标签来自谁”，而是训练数据是否来自当前策略自己实际访问到的状态分布。**

## 核心要点

Offline 训练更像：

[
x \rightarrow \text{fixed data/teacher} \rightarrow y
]

On-policy 训练更像：

[
x \rightarrow \pi_{current} \rightarrow y \rightarrow \text{feedback} \rightarrow \text{update}
]

因此要区分两个问题：

1. supervision 是否正确；
2. state / trajectory 是否由当前 policy 产生。

## 与相邻概念的关系

- **SFT**：通常主要学习离线 demonstrations。
- **RL**：常在当前 policy rollout 上优化 reward。
- **OPD**：把 teacher 信号施加到 student 自己访问到的分布上。

## Open Questions

- On-policy 带来的收益中，多少来自减少 state distribution mismatch，多少来自更好的探索？
