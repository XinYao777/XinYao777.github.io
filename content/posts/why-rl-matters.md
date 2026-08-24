---
title: "Why RL Matters：从 CoT Pattern 到蒸馏与软蒸馏"
date: 2026-08-24T13:55:00+08:00
draft: false
tags: ["RL", "蒸馏", "笔记"]
categories: ["AI 技术思考"]
math: false
summary: "读青稞AI《从 RL 到蒸馏，再到软蒸馏：Why RL Matters？》的归纳。先钉死“蒸馏”的定义（sequence distillation），再梳理 CoT pattern、RL 的两层作用、蒸馏与软蒸馏之间的关系。"
---

> 原文：[《从 RL 到蒸馏，再到软蒸馏：Why RL Matters？》](https://mp.weixin.qq.com/s/S29z51EByrjWJgkvr6vwhQ)（青稞AI，作者 ybq）

这是一篇讨论 RL 在 posttrain 阶段定位的文章。我按「定义先行 → CoT pattern → RL 的两层作用 → 蒸馏 → 软蒸馏」的顺序做归纳，其中最需要先钉死的是**「蒸馏」到底指什么**——不先说清楚，后面整条逻辑都会被这个词误导。

## 先钉死定义：这里的「蒸馏」是 sequence distillation

全文的「蒸馏」**不是**经典 KD，两者必须分清：

- **经典 KD（Hinton 2015，logit / 软标签蒸馏）**：student 去拟合 teacher 输出的**概率分布**（soft logits），用 KL 对齐。需要拿到 teacher 的 logits，是 token 级的分布匹配。
- **本文的蒸馏 = sequence distillation（Kim & Rush 2016）**：teacher 生成完整 response（尤其是 CoT 序列），student 直接拿这些**硬 token 序列做 SFT**。只需要 teacher 的**输出文本**，不需要 logits。一句话概括就是 **“utilize better model response to sft”**。

为什么值得反复强调：正因为是「拿更强模型的 response 做 sft」，才有了后文那个核心矛盾——**你搬来的是 teacher 的成品序列，而不是在对齐概率分布**，于是 student 自身的知识量 / tokenizer 跟这个序列并不匹配。这个矛盾正是 RL 存在的理由。

## CoT Pattern：SFT 的定型职责

SFT 阶段一个关键职责，是给 posttrain 的 **CoT pattern 定型**，这几乎决定了模型在 posttrain 阶段的效果上限。

以 `997 × 1003` 为例，存在三种 pattern：

- **good**：`997 × 1003 = (1000 − 3)(1000 + 3) = 1000² − 3² = 999991`
- **mediocre**：`1003 × (900 + 90 + 7) = 902700 + 90270 + 7021 = 999991`（步骤多、易错、token efficiency 低）
- **bad**：`997 × 1003 = 999991`（不泛化，喂给小模型后很可能「见 997 就答 999991」）

所以好的 CoT pattern 具备**输出更短、准确率更高、更易泛化**的特点。进入 agent 时代，我们绞尽脑汁构造和筛选的 environment trajectories，本质上仍是「reasoning 时代找更优 CoT pattern」这件事的延续。

## RL 的两层作用

**第一层：对 CoT pattern 做「本土化改造」。**

我们拿到的 CoT pattern 大多来自更强的模型或人工编辑，往往「表述极简、思路极优雅」。但要训练的模型没有足够能力直接消化它，需要结合自身能力改造。RL 的关键作用就在于：**在给定一个 CoT pattern 的前提下，通过 explore 足够多的 response，找到与自身「知识量、tokenizer」最契合的表达方式，并强化成专属 pattern。**

- 同样学会 `99 × 99 = 99 × (100 − 1)`，强模型可以跳步直接算 `9900 − 99 = 9801`；弱模型则要把省略的步骤全写出来才算得对。
- reasoning 任务：SFT 只告诉模型「做完反思一下更好」，但反思几次最优最省，要靠 RL 自己摸索。
- agent 任务：SFT 告诉模型有哪些 sub-agent、何时用；但 1T 模型调 2 个就能拿到正确 reward，100B 模型可能要调 5 个——最优 setting 由 RL 摸出来。

这也解释了一个现象：**SFT 起点低的模型，RL 后终点可能更高**。起点低往往是 pattern 与模型能力不适配，而非 SFT 数据本身差。此外 RL 还带一点「基因突变 + 进化」的味道：当算力足够、`rollout_n` 开得足够大，模型有机会 explore 到全新的更优 pattern，此时好的 RL 算法（如 GRPO）需要抓住这条高质量数据，给足学习信号让模型记住它。

**第二层：让模型变得可控（更被低估、也更重要）。**

传统认知里，控制住 SFT 数据分布就能控制输出。但时代变了：**大量合成语料进入 pretrain / midtrain，即使 SFT 数据全都规范，posttrain 也越来越容易冒出乱七八糟的 pattern。** 道理很简单——pretrain 只看数据质量和干净程度，根本不管 posttrain 为体验加的那堆规则；几十 T 训出来的「高质量但格式不优雅」的数据，不是几条 SFT 数据能压制的，何况 SFT 本就不具备打压 pattern 的能力。

而这些让产品头大的 case（多个 `<think>`/`</think>` 切分混乱、中英混杂、输出长度忽长忽短），在 RL 的 reward 惩罚面前格外稚嫩，随手给个惩罚 loss 就不再出现。现在标配的「思考档位 / 思考深度」，也是通过 RL 阶段给不同档位不同上下文窗口轻松实现的。

## RL 与蒸馏：RL 的收益可被蒸馏「窃取」

目前没有直接证据表明 RL 后的模型相比 SFT 后的模型有质的提升——**RL 带来的收益是可以被蒸馏窃取的**。RL 本质仍是在找更好的 pattern，堆算力找到的好 pattern 价值千金，但「寻找过程」本身未必有价值。

于是可以让 RL 后的模型当 teacher，通过蒸馏把它探索到的行为分布迁移到另一个模型上，得到指标接近 RL 模型的 student。业界常用的合版方案之一就是对多个 RL 子模型做 **reject sampling SFT**（另一路线是 OPD）。1T 模型 explore 到的 pattern 比 100B 丰富，把前者的优质 pattern 喂给后者，再用 RL 适应其具体表达（通常会让小模型输出变长），小模型指标就能逼近大模型——同尺寸、同词表时尤其屡试不爽。

蒸馏甚至能让 student 超过 teacher（例如 teacher 的 CoT 中英混出，只保留质量更高的英文 CoT）。但作者提醒：**任何能让 student 超过 teacher 的操作，都应该反过来用去提升 teacher**，而不是自我感动于这个操作多厉害。

一句话总结作者的态度：**认为蒸馏无用，大抵没亲自训过大模型；认为蒸馏是 posttrain 的全部，大抵只想当追赶者。** 蒸馏是「偷看学习笔记」——先拿到高起点的 SFT 模型，再靠 RL 去 adapt pattern、fix bad pattern，一个优秀的 posttrain 模型才闪亮登场。

## 软蒸馏：更高级的「利用更强模型」

distill CoT pattern 是最好用、也最低级的蒸馏。真正高级的做法是**借 sota 模型的能力优化自己的模型**，包括但不限于：

- 用 sota 合成现有模型**能力边界**上的数据；
- 用 sota **诊断**现有模型的 pattern 缺陷；
- 用 sota 优化 RL 阶段的 **verifier**。

软蒸馏普遍存在——任何国内厂商都不敢说优化过程里没有 GPT / Claude 的帮助。**只要国外模型领先，国内模型就有源源不断的进步空间和优化手段。**

## 小结

纯训练阶段已是明牌竞争，比的就是谁更能 scaling：**scaling model parameters、scaling RL data、scaling agent environments**。因此越想领跑的团队越要加大 RL 投入，早日摆脱对 sota 模型的（软）蒸馏依赖；反过来，若只追求某个能力的应用价值，蒸馏就已经是最好的选择。
