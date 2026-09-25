---
title: "Rubric"
description: "把任务成功条件显式拆成可判断、可评分的标准。"
status: evolving
date: 2026-09-20
lastmod: 2026-09-20
type: knowledge
related: ["Ontology", "Observable Failure", "LLM-as-Judge", "RAR"]
sources: []
---

## 一句话

**Rubric 的作用是把“一个好答案应该满足什么”显式拆成可判断的条件。**

## 核心要点

可以把 Evaluation System 中两个职责分开：

[
\text{Ontology}
\rightarrow
\text{Don't miss an entire class of failure}
]

[
\text{Rubric}
\rightarrow
\text{What exactly should this answer satisfy?}
]

Ontology 管“别漏掉哪一类问题”，Rubric 管“这一类问题具体怎么判”。

## 与相邻概念的关系

- **Ontology**：定义评估空间。
- **Observable Failure**：把抽象能力问题映射为可观察错误。
- **Judge**：执行 Rubric。
- **RAR**：把需求、回答与评估标准组织进更完整的评估流程。

## Open Questions

- Rubric 应该做到多细，才能兼顾可解释性与 Judge 稳定性？
- 哪些维度应由规则判断，哪些适合 LLM-as-Judge？
