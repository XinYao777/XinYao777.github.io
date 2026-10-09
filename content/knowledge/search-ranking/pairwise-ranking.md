---
title: "Pairwise Ranking"
description: "通过学习候选之间的相对偏序来训练排序模型。"
status: evolving
date: 2026-09-20
lastmod: 2026-09-20
type: knowledge
related: ["Pointwise Ranking", "Listwise Ranking", "Reward Model", "RankNet"]
sources: []
---

## 一句话

**Pairwise Ranking 不直接要求模型预测绝对相关性，而是学习同一 query 下候选之间谁应该排在前面。**

## 核心要点

典型训练样本：

[
(q, d_i, d_j), \quad y_i > y_j
]

模型学习：

[
s(q,d_i) > s(q,d_j)
]

常见目标包括：

- RankNet / logistic pairwise loss
- hinge / margin ranking loss
- preference reward modeling

## 与相邻概念的关系

- **Pointwise**：学习单个候选的绝对标签或分数。
- **Listwise**：直接对整个 candidate set 建模。
- **Reward Model**：很多 preference RM 本质上也是偏序学习。

## Open Questions

- 业务 score 是否需要显式概率化到 ([0,1])？
- Hinge loss 中 margin 应该定义在 raw logit space 还是经过 sigmoid 的 score space？
