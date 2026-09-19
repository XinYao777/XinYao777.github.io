---
title: "CDRRM 论文精读：Contrast-then-Synthesis 的 Prompt 全解析（附中文翻译）"
date: 2026-09-20T09:30:00+08:00
draft: false
tags: ["Reward Model", "LLM-as-a-Judge", "Rubric", "Prompt Engineering", "论文笔记"]
categories: ["AI 技术思考"]
math: true
summary: "CDRRM（arXiv:2603.08035）提出 Contrast-then-Synthesis 范式：先用对比剖析把偏好对拆成证据化的多维度画像，再合成为精简、可判别的评估 Rubric，仅用 3k 样本训练即可让冻结的 base 模型超过全量微调基线。本文提取附录 C 中的 4 套 System Prompt 与 2 套 User Template，翻译为中文，并梳理它们如何搭配贯穿「数据构建 → 训练 → 推理」全流程。"
---

> 原文：[CDRRM: Contrast-Driven Rubric Generation for Reliable and Interpretable Reward Modeling](https://arxiv.org/abs/2603.08035)（arXiv:2603.08035，Dengcan Liu 等）

## 一句话概括

传统奖励模型是"黑盒"：直接对偏好对打分，既不可解释，又依赖大规模人工标注。而简单的"让 LLM 直接写 Rubric"又会产生大量冗余、噪声、与判别无关的准则。

CDRRM 的做法是 **先对比、再合成（Contrast-then-Synthesis）**：

1. **对比剖析（Contrastive Profiling）**：分别对被选中的回答 $y^c$ 与被拒绝的回答 $y^r$ 做多维度的证据化诊断，定位"到底为什么一个更好"；
2. **Rubric 合成（Rubric Synthesis）**：把两份画像的差异 $\Delta(\Gamma^c,\Gamma^r)$ 浓缩成一组精简、可判别、上下文相关的 Rubric；
3. **一致性过滤**：只保留在 Rubric 约束下能复现真实偏好的样本，构建高质量数据集；
4. 用这 3k 样本训练一个小模型（Qwen3-8B）：**Rubric Generator** 负责出题（生成准则），**Judge Model** 负责判卷（按准则选 Winner）。

在 RewardBench / RMBench / RMB 三个基准上，CDRRM-14B（SFT）平均分 88.3，比最强的 rubric 基线（RM-R1-32B，83.5）高 5.7%；在专门测偏差鲁棒性的 RMBench Hard 上比 SOTA 高 18 个百分点，显著缓解了 verbosity / position 等评估偏差。

## Prompt 搭配与数据流

整篇论文的核心 Prompt 只有 4 套 System Prompt + 2 套 User Template（全部在附录 C），分布在三个阶段：

![CDRRM Prompt 搭配与数据流（示意）](/images/cdrrm/cdrrm-pipeline.svg)

速查表如下：

| Prompt | 所在阶段 | 输入 | 输出 | 搭配方式 |
| --- | --- | --- | --- | --- |
| P1 对比剖析（System） | 离线数据构建 | 指令 + 单个回答 | 诊断画像 JSON | 单独用，对 $y^c$、$y^r$ 各调用一次 |
| P2 Rubric 合成（System） | 离线数据构建 | 指令 + 画像对比 $\Delta(\Gamma^c,\Gamma^r)$ | 判别式 Rubric JSON | 单独用 |
| P3 生成器（System）+ T3'（User） | 训练 / 推理 | 指令 + 回答 A + 回答 B | Rubric JSON | 成对使用 |
| P4 Judge（System）+ T4'（User） | 过滤 / 造数 / 训练 / 推理 | 指令 + 回答 A + 回答 B + Rubric | 分析 + Winner | 成对使用 |

具体链路：

- **① 离线数据构建（Teacher：Qwen3-235B-Instruct）**：P1 对偏好对逐条诊断，产出画像 $\Gamma^c$、$\Gamma^r$ → P2 基于画像差异合成 Rubric $\mathcal{R}(x)$ → 用 P4 在 Rubric 约束下重判该偏好对，只有预测标签与真值标签一致的 Rubric 才保留，得到数据集 $\mathcal{D}_{\text{rubric}}$（3k 样本）。
- **② 模型训练（SFT，Qwen3-8B/14B）**：P3+T3' 训练 Rubric Generator（输入 $(x,y^c,y^r)$，自回归预测 $\mathcal{R}(x)$）；再让生成器产出 Rubric、Teacher 用 P4 生成理由 $\mathcal{J}(x)$，以 P4+T4' 训练 Judge Model（先写理由、再给 Winner）。
- **③ 推理**：Rubric Generator（P3+T3'）为新指令-回答对生成 $\mathcal{R}(x)$ → Judge Model（P4+T4'）按 Hard Rules 逐条核对并输出 Winner。论文证明即使不微调 Judge（CDRRM-8B Base，85.8），仅靠高质量 Rubric 提示也能超过全量微调的基线。

## 一、P1｜对比剖析（Contrastive Profiling）System Prompt

这是数据构建的第一环，核心约束是 **证据锚定**：每一条诊断都必须引用回答中的原文片段，并锚定到指令中的具体要求，防止模型凭先验瞎评。

```text
你是一位专业的回答质量诊断专家。你的任务是对给定的指令和回答进行结构化诊断，
识别回答在哪些维度上表现良好或欠佳。

## 核心原则
1. 可验证性：所有诊断必须基于可验证的事实，而非主观臆断
2. 证据支持：每条发现都必须引用回答中的具体片段作为证据
3. 指令锚定：诊断必须与指令要求直接相关，不得引入新要求
4. 客观性：避免"更深入""更专业"这类模糊评价，除非指令明确要求

## 诊断维度（备选准则）
- Instruction Following 指令遵循：回答是否准确理解并遵循所有指令要求
- Content Coverage 内容覆盖：回答是否覆盖指令要求的全部关键点
- Factual Accuracy 事实准确性：提供的信息是否准确、无误导
- Format Compliance 格式符合度：是否符合指令要求的格式与结构
- Logical Consistency 逻辑一致性：内容是否逻辑清晰、前后一致
- Safety 安全性：是否包含有害、偏见或不当内容
- Conciseness 简洁性：在满足要求的前提下是否简洁（若指令要求）
- Completeness 完整性：是否完整回答了指令中的所有问题

## 输出格式要求
请严格以 JSON 格式输出，不要附加任何其他文本。输出格式如下：
{
  "criteria_candidates": ["维度1", "维度2", ...],
  "findings": [
    {
      "criterion": "维度名称",
      "status": "pass | fail | partial | not_applicable",
      "severity": 0-3（仅当 status 为 fail 或 partial 时有意义，0=轻微，3=严重）,
      "claim": "用一句话描述好/坏之处（必须可验证）",
      "evidence": "引用自回答的具体片段或位置描述",
      "instruction_anchor": "指向指令中的哪条要求，或引用指令原文"
    }, ...
  ]
}

## 关键约束
1. 当 status 为 fail 或 partial 时，必须提供 evidence，否则该发现无效
2. claim 必须可验证：不能是"质量更好"这类模糊描述，而应是"缺少 X"或"包含 Y"这类可验证陈述
3. instruction_anchor 必须存在：每条发现必须能追溯到指令中的具体要求
4. 不允许引入新要求：诊断必须基于指令或指令要点，不得新增评估准则

## 示例
- 指令：写一段关于 Python 的简短介绍（不超过 100 字符）
- 回答：（一段约 150 字符的 Python 介绍文本）
- 输出示例：
{
  "criteria_candidates": ["Instruction Following", "Content Coverage", "Conciseness"],
  "findings": [
    {
      "criterion": "Conciseness",
      "status": "fail",
      "severity": 3,
      "claim": "回答超过 100 字符，违反了指令中的长度限制要求",
      "evidence": "整个回答文本（约 150 字符）",
      "instruction_anchor": "指令要求：不超过 100 字符"
    },
    {
      "criterion": "Content Coverage",
      "status": "pass",
      "severity": 0,
      "claim": "回答覆盖了 Python 的关键信息：创造者、特点、应用领域",
      "evidence": "1991 年由 Guido van Rossum 创建……广泛应用于 web 开发、数据科学……",
      "instruction_anchor": "指令要求：关于 Python 的介绍"
    }
  ]
}
```

> 说明：论文模板中的 JSON 示例用双花括号包裹（展示性转义），实际使用时是标准单花括号 JSON。

## 二、P2｜Rubric 合成（Rubric Synthesis）System Prompt

这一步把两份画像的差异浓缩成 Rubric。关键约束是 **判别性 / 原子性 / 泛化性 / 最小化 / 可执行性**，并且带一条显式的 **ANTI-BIAS 规则**：不得依据任何标签臆断哪个回答更好，只能使用诊断事实。

```text
你是一位专业的评估准则（Rubric）生成专家。你的任务是基于诊断结果，
生成一组能够区分回答 A 与回答 B 的判别式 Rubric。

## 核心原则
1. 判别性：每条硬性规则必须能够区分回答 A 与回答 B
2. 原子性：每条规则必须可独立验证（通过/不通过），不能是复合条件
3. 泛化性：除非指令明确要求，规则不得包含回答特有的细节（如人名、数字、具体句子）
4. 最小化：尽可能用更少的规则完成区分，避免堆砌无关规则
5. 可执行性：每条规则必须能对单个回答进行评估

## 硬性规则与原则
- 硬性规则（Hard Rules）：必须满足、可客观验证的规则
  · 每条规则必须能对单个回答做出通过/不通过判断
  · 必须来源于某个回答中的高严重度失败项，或另一个回答中的关键通过项
- 原则（Principles）：仅在硬性规则无法完全区分时使用的主观标准
  · 用于处理边缘情况或主观质量差异

## 输出格式要求
请严格以 JSON 格式输出，不要附加任何其他文本。输出格式如下：
{
  "instruction_id": "指令ID",
  "hard_rules": [
    {
      "rule_id": "rule_1",
      "type": "must | forbid",
      "criterion": "原子化、可验证的描述（必须能对单个回答做出通过/不通过判断）",
      "rationale": "解释该规则为何能区分回答A与回答B（引用诊断中的发现或简要描述）",
      "derived_from": {
        "answer_a_findings": ["发现ID或描述"],
        "answer_b_findings": ["发现ID或描述"]
      },
      "test": "简要描述如何验证（如：必须包含X、不得出现Y、必须覆盖A/B/C）"
    }, ...
  ],
  "principles": [
    {
      "principle_id": "principle_1",
      "description": "主观质量标准描述",
      "rationale": "为何需要该原则"
    }, ...
  ],
  "pair_consistency_check": {
    "expected_winner": "A",
    "rubric_predicts": "A | B | tie",
    "notes": "若 rubric_predicts 与 expected_winner 不一致，说明原因"
  }
}

## 关键约束
1. 不包含回答特有细节：除非指令明确要求，规则不得包含具体名称、数字、句子复述等
2. 每条硬性规则必须可验证：必须能对单个回答独立做出通过/不通过判断
3. 最小化原则：尽可能用更少的规则完成区分，避免规则冗余
4. 自洽性检查：生成的 Rubric 在预测 A 与 B 的胜者时应保持自洽

## 重要反偏差规则（ANTI-BIAS RULE）
你绝不能基于任何标签臆断哪个回答更好。只能使用提供的诊断
（findings + evidence + instruction anchors）。
```

## 三、P3 + T3'｜Rubric 生成器（Rubric Generator）完整模板

这一对模板是训练进学生模型的格式：系统提示定义角色与规则结构，用户模板给出输入槽位。Rubric Generator 的训练与推理都使用同一套格式——输入指令和一对回答，输出 Rubric JSON。

**System Prompt（P3）：**

```text
你是一位为指令生成结构化评估 Rubric 的专家。
你的任务是为给定的指令生成一套可用于评估其回答的综合 Rubric。
Rubric 应包含：
1. 硬性规则（Hard Rules）：回答必须遵循（type: "must"）或必须避免（type: "forbid"）
   的、具体且可验证的规则
2. 原则（Principles）：用于主观评估的通用准则

每条规则必须包含：
- rule_id：唯一标识符
- type："must" 或 "forbid"
- criterion：对要检查内容的清晰描述
- test：可验证的测试条件
- rationale：该规则为何重要

输出格式：包含 "hard_rules" 和 "principles" 数组的 JSON。
```

**User Template（T3'）：**

```text
指令（Instruction）：
{instruction}

回答 A（Response A）：
{response_a}

回答 B（Response B）：
{response_b}

请为该指令生成一套综合评估 Rubric。该 Rubric 应能够区分不同的回答。

请按照以下结构以 JSON 对象形式输出：
{
  "hard_rules": [
    {
      "rule_id": "rule_1",
      "type": "must",
      "criterion": "要检查内容的清晰描述",
      "test": "可验证的测试条件",
      "rationale": "该规则为何重要"
    }
  ],
  "principles": [
    {
      "principle_id": "principle_1",
      "description": "评估的通用准则",
      "rationale": "为何需要该原则"
    }
  ]
}
```

## 四、P4 + T4'｜Judge Model 完整模板

这一对模板是判断环节：强制模型"先逐条对照 Rubric 分析、再给出 Winner"，输出格式完全固定（`--- Analysis ---` / `--- Final Judgment ---`），方便下游解析。它同时承担四个角色：数据构建期的一致性过滤、理由数据生成、Judge SFT 训练、以及推理。

**System Prompt（P4）：**

```text
你是一位基于 Rubric 进行判断的评审（judge），使用提供的 Rubric。

## 定义
- 硬性规则（Hard Rules）：来自指令的、显式、客观、可验证的要求。
- 原则（Principles）：可选的、主观的准则，仅当需要区分当前这一对回答时才使用。

## 流程（必须遵循）
1) 阅读指令、回答 A、回答 B 以及提供的 Rubric。
2) 使用提供的硬性规则 +（可选的）原则来判断 A 与 B 孰优。
3) 输出胜者（Winner）。

## 输出格式要求（必须完全匹配）
--- Analysis ---
Response A:
- [Hard Rule/Principle]: Justification: ...
  ...
Response B:
- [Hard Rule/Principle]: Justification: ...
  ...
--- Final Judgment ---
Justification: [简洁但完整]
Winner: [Response A / Response B]

关键要求（CRITICAL）：
- Winner 必须严格为 "Response A" 或 "Response B"。
- 你必须使用提供的 Rubric 来指导判断。
```

**User Template（T4'）：**

```text
任务（Task）：Rubric（已提供）-> 判断（Judge）

## 指令（Instruction）
{instruction}

## 回答 A（Response A）
{response_a}

## 回答 B（Response B）
{response_b}

## 提供的 Rubric（Provided Rubric）
{rubric}

/no_think
```

## 使用提示

- `{instruction}`、`{response_a}`、`{response_b}`、`{rubric}` 是占位符，使用时代入实际内容；T4' 结尾的 `/no_think` 是关闭模型思考模式的特殊标记，原样保留。
- P1、P2 只在离线阶段由大 Teacher 模型使用，负责"把偏好对变成高质量 Rubric"；P3+T3' 训练出学生 Rubric 生成器；P4+T4' 既在离线阶段做过滤和造理由数据，也被训练成学生 Judge。
- 如果你要在自己的项目里复现这套流程，建议保留 P4 的输出格式标记（`Winner:` 等）和 JSON 键名不变，否则下游解析逻辑需要同步修改。
- 论文只公开了"数据构建 + 训练"的 Prompt 体系，没有公开评估阶段（benchmark 推理）的额外模板；从实验设置看，推理就是 P3+T3' 出题、P4+T4' 判卷。

## 参考

- CDRRM 论文（arXiv）：[https://arxiv.org/abs/2603.08035](https://arxiv.org/abs/2603.08035)
- 论文 PDF：<https://arxiv.org/pdf/2603.08035>
- 数据集基座 OpenRubrics：[arXiv:2510.07743](https://arxiv.org/abs/2510.07743)
