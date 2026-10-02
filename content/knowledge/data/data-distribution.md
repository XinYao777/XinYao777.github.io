---
title: "Data Distribution"
description: "训练样本在能力、领域、难度与表征空间中的分布结构。"
status: evolving
date: 2026-09-20
lastmod: 2026-09-20
type: knowledge
related: ["Data Quality", "Data Mixture", "Coverage", "Diversity"]
sources: []
---

## 一句话

**Data Distribution 回答的不是“这条数据好不好”，而是“训练数据覆盖了哪些区域，各区域占多少”。**

## 核心要点

数据问题至少拆成：

[
\boxed{
\text{Distribution}
+
\text{Quality}
+
\text{Composition}
}
]

其中 Distribution 关注：

- Capability / task 分布
- Domain 分布
- Difficulty 分布
- Input representation 分布
- Real-world condition 分布

## 与相邻概念的关系

- **Quality**：单条样本是否正确、可学、值得学。
- **Coverage**：该覆盖的区域是否缺失。
- **Diversity**：已覆盖区域内部是否存在足够变化。
- **Mixture**：不同区域最终如何配比与曝光。

## Open Questions

- 如何从真实样本反推出稳定、正交、可操作的 distribution axes？
- Coverage 与 Diversity 的边界如何形式化？
