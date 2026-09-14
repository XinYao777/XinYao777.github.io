---
title: "如何理解 2026 年的 AI：从模型能力到智能系统的一套统一框架"
date: 2026-09-14T20:00:00+08:00
draft: false
tags: ["Agent", "Harness", "Eval", "RSI", "AI 系统", "笔记"]
categories: ["AI 技术思考"]
math: true
summary: "与其追逐 Language、RL、Agent、Harness、Eval 这些不在同一层级的技术名词，不如建立一套稳定的坐标系。用三个问题理解今天的 AI：模型本身会什么（Capability）→ 系统能完成什么（Task Completion）→ 系统能多快改善自己（Capability Improvement）。这三层几乎可以把当前绝大部分 AI 技术重新放回正确位置，而贯穿其中的主线是——瓶颈的不断迁移。"
---

过去几年，AI 技术的信息密度越来越高。

有人把这一轮 AI 分成 Language、Coding、Multimodal 三种核心能力；有人认为真正决定竞争力的是 Data、Algorithm、Infra；有人讨论 Pretrain、SFT、RL、Distillation；有人认为模型正在让位于 Agent 和 Harness；与此同时，Eval、Memory、Tool Use、Looping、RSI 又不断成为新的关键词。

这些观点大多没有错。

真正的问题是：**它们讨论的根本不是同一层级的问题。**

如果把 Language、RL、Agent Harness、Data、Eval 放在同一张表里比较，就像讨论"发动机、汽油、高速公路、驾驶技术和目的地哪个更重要"。它们彼此相关，却无法直接比较。

因此，与其继续追逐新的技术名词，不如建立一套相对稳定的坐标系。

我更倾向于用三个问题理解今天的 AI：

$$
模型本身会什么？
\rightarrow
系统能够完成什么？
\rightarrow
系统能够多快地改善自己？
$$

也就是：

$$
Capability
\rightarrow
Task\ Completion
\rightarrow
Capability\ Improvement
$$

这三个层次，基本可以把当前绝大部分 AI 技术重新放回正确的位置。

---

## 一、首先把几个经常混在一起的概念拆开

今天我们经常听到以下几组分类：

### Language / Coding / Multimodal

这是在讨论：

> 模型具有哪些能力？

属于 **Capability Space**。

### Data / Algorithm / Infra

这是在讨论：

> 能力是如何被生产出来的？

属于 **Production Factors**。

### Pretrain / SFT / RL / Distillation

这是在讨论：

> 模型参数如何被训练和更新？

属于 **Training Lifecycle**。

### Model / Agent / Harness / Application

这是在讨论：

> 一个完整 AI 系统由什么构成？

属于 **System Architecture**。

### Planning / Tool / Memory / Looping / Reflection

这是在讨论：

> 模型运行时如何组织计算和行动？

属于 **Runtime Mechanism**。

### Eval / Reward / Verifier

这是在讨论：

> 系统如何知道自己做得好不好？

属于 **Feedback System**。

### RSI

这是在讨论：

> AI 是否开始参与生产下一代 AI？

属于 **Improvement Loop**。

一旦按照这个方式拆开，很多争论其实都会消失。

例如：

> "Agent 和 RL 哪个更重要？"

并不是一个特别好的问题。

RL 是一种参数优化方法；

Agent 是一种系统运行形态；

二者甚至可以同时存在于同一个系统中。

---

## 二、第一层：Capability——模型本身到底会什么？

首先看模型本身。

与其把能力简单划分为 Language、Coding、Multimodal，我更倾向于使用一个更稳定的抽象：

$$
Perception
\rightarrow
Cognition
\rightarrow
Action
$$

即：

> **感知 → 认知 → 行动。**

### Perception：感知

模型能看到什么？

包括：

- Text；
- Image；
- Audio；
- Video；
- Sensor。

Multimodal 的核心价值，本质上是在扩大模型的：

$$
Observation\ Space
$$

也就是模型能够感知的世界范围。

### Cognition：认知

模型能不能：

- 理解；
- 推理；
- 分解问题；
- 规划；
- 判断；
- 建立抽象关系。

Reasoning Model 的提升，本质上主要发生在这里。

### Action：行动

模型能否对外部世界产生改变？

例如：

- 写代码；
- 调 API；
- 操作浏览器；
- 修改文件；
- 查询数据库；
- 控制机器人。

这是 Agent 时代极其重要的一次变化：

过去模型主要在生成：

$$
Text
$$

现在越来越多模型开始产生：

$$
Action
$$

---

## 三、为什么 Coding 今天尤其重要？

如果按照上面的框架，Coding 是一个非常特殊的能力。

因为代码同时属于：

$$
Cognition + Action
$$

它既要求推理，又可以直接作用于外部环境。

但 Coding 真正重要的原因，还不只是模型"代码写得好"。

软件世界天然具备五个非常适合 AI 的条件：

$$
\boxed{
可执行
+
可验证
+
可迭代
+
可规模化
+
能够改进AI自身
}
$$

写出代码以后，可以立刻：

$$
Code
\rightarrow
Compile
\rightarrow
Execute
\rightarrow
Test
\rightarrow
Error
\rightarrow
Repair
$$

这意味着软件环境天然给 Agent 提供了一个极好的闭环：

$$
\boxed{
Act
\rightarrow
Observe
\rightarrow
Verify
\rightarrow
Correct
}
$$

相比之下，如果让一个 Agent：

> "制定一个优秀的公司战略。"

什么叫优秀？

什么时候才能知道战略是错的？

三个月？一年？

Reward 极其模糊。

这也是为什么 Coding 可能成为 AI 最早大规模 Agent 化的领域。

OpenAI 在 2026 年的 Harness Engineering 实践中甚至做过一个极端实验：一个内部产品的应用代码、测试、CI、文档和工具全部由 Codex 编写。团队认为，人的工作开始从"直接写代码"，逐渐转向**设计环境、表达意图和建立反馈循环**。

更进一步，OpenAI 公开的内部研究数据显示，到 2026 年 8 月，其研究组织整体已经达到约：

$$
3.1\ Agent\ Workdays
/
1\ Human\ Workday
$$

Coding Agent 已经不仅仅在写业务代码，也开始参与研究基础设施、实验和监控。

因此，Coding 今天坐在"主桌"，背后真正重要的是：

> **它第一次让 Reasoning、Action、Feedback 和 AI R&D 大规模连接在了一起。**

---

## 四、第二层：Task Completion——模型很强，不等于系统能把事情做好

过去，我们习惯用：

$$
M(x)
$$

表示模型输入一个问题、输出一个答案。

但 Agent 时代真正应该研究的是：

$$
P(\text{success}|x)
=
F(M,H,E,B)
$$

这里至少存在四个基本变量：

$$
\boxed{
M,\ H,\ E,\ B
}
$$

### M：Model

模型自身的能力。

例如：

- 推理；
- Coding；
- Multimodal；
- Instruction Following。

### H：Harness

如何组织模型。

包括：

- Context；
- Planning；
- Memory；
- Tool Calling；
- Subagent；
- Retry；
- Reflection；
- Context Compression。

### E：Environment

模型能够：

- 看到什么；
- 操作什么；
- 得到什么反馈。

例如 Coding Agent 的：

- Git repository；
- terminal；
- compiler；
- unit test；
- runtime。

### B：Budget

系统愿意为一个任务投入多少资源：

- Token；
- Compute；
- 时间；
- Tool calls；
- Rollout 数量。

因此，Agent 时代一个非常重要的认知变化是：

$$
\boxed{
Model\ Capability
\neq
System\ Capability
}
$$

同一个模型，放在两个不同 Harness 中，最终系统效果可能差别非常大。

Anthropic 对约 40 万次 Claude Code session 的分析发现，一个非常稳定的协作模式正在出现：

> 人更多负责决定"做什么"，模型更多负责决定"怎么做"。

与此同时，拥有更强领域知识的人，每次给 Agent 一条指令，往往能够让 Agent 完成更多有效工作。

这意味着 AI 并没有简单消灭 expertise。

相反：

$$
Domain\ Expertise
\times
Agent
$$

正在成为一种新的杠杆。

---

## 五、Harness 为什么成为 2026 年的重要关键词？

当模型能力不够的时候，绝大多数问题都只能通过训练更强模型解决。

但是模型越来越强以后，问题逐渐变成：

> 明明模型会，为什么系统仍然做不好？

这时候瓶颈就从：

$$
Model
$$

迁移到了：

$$
Harness
$$

例如，一个模型不会自己天然知道：

- 应不应该搜索；
- 应不应该调用工具；
- 什么时候停止；
- 失败以后如何恢复；
- 哪些信息值得保留在 context；
- 是否应该并行调用多个 Agent。

这些东西并不是模型参数本身，而是：

$$
System\ Design
$$

所以今天出现大量关于：

- Agent Framework；
- Context Engineering；
- Tool Use；
- MCP；
- Skills；
- Memory；
- Subagents；

的讨论，并不奇怪。

本质上都是同一个问题：

> **如何把模型中的 latent capability，稳定地转换成 task completion。**

---

## 六、但 Harness 很可能并不是最终瓶颈

Harness 今天非常重要。

但长期来看，其中相当一部分能力很可能会逐渐基础设施化。

例如：

- Tool Calling；
- Context Compression；
- Retry；
- Memory；
- Subagent；
- Checkpoint；

最终可能会成为类似今天：

- KV Cache；
- Distributed Training；
- Serving；

一样的标准组件。

那么瓶颈会继续往哪里迁移？

我认为下一个极其重要的变量是：

$$
\boxed{
Environment + Verifier
}
$$

原因很简单：

Agent 能行动以后，最难的问题会变成：

> **它怎么知道自己做对了？**

---

## 七、Eval 正在从"考试"变成"控制系统"

过去我们理解 Eval：

$$
Model
\rightarrow
Benchmark
\rightarrow
Score
$$

训练结束以后跑一次 benchmark。

它的作用主要是：

> Measurement。

但未来的 Eval 会越来越不同。

### Eval 1.0：Measurement

$$
M
\rightarrow
Benchmark
\rightarrow
Score
$$

回答：

> 这个模型有多强？

### Eval 2.0：Runtime Verifier

Agent 每执行一步：

$$
Action
\rightarrow
Eval
\rightarrow
Feedback
\rightarrow
Next\ Action
$$

此时 Eval 开始参与运行过程。

### Eval 3.0：Learning Signal

进一步：

$$
Generate
\rightarrow
Evaluate
\rightarrow
Select
\rightarrow
Train
$$

Eval 直接进入后训练。

到这里以后：

$$
Eval
\approx
Reward
\approx
Verifier
$$

它已经不再只是"测试部门"。

而逐渐变成整个 AI 系统中的：

$$
\boxed{
Control\ System
}
$$

Google DeepMind 在 2026 年开始尝试 frontier model 的 double-blind evaluation，背后的动机之一正是 benchmark contamination 和针对 benchmark 优化正在削弱传统评测可信度。

这说明 Eval 的重要性不是下降，而是在升级。

---

## 八、第三层：Capability Improvement——下一代模型到底是怎么来的？

我们再往上一层。

传统模型能力可以粗略写成：

$$
M_{t+1}
=
G(D,A,C,F)
$$

其中：

- \(D\)：Data；
- \(A\)：Algorithm；
- \(C\)：Compute / Infra；
- \(F\)：Feedback / Eval。

过去这四个变量绝大部分由人类生产。

但现在发生了一件本质性的变化：

$$
M_t
$$

开始参与生产：

- Data；
- Code；
- Experiment；
- Evaluation；
- Algorithm ideas。

于是：

$$
M_t
\rightarrow
D,A,C,F
\rightarrow
M_{t+1}
$$

AI 开始进入生产下一代 AI 的生产函数。

这才是理解 RSI 更合理的方式。

---

## 九、RSI 不应该首先理解成"模型自己修改自己的参数"

Recursive Self-Improvement 很容易被理解成一种科幻场景：

> AI 修改自己的源代码，然后突然快速自我进化。

但现实中更有价值的定义其实是：

> **AI 在 AI R&D Loop 中承担了多少工作？**

例如一次模型迭代需要：

$$
Problem
\rightarrow
Idea
\rightarrow
Code
\rightarrow
Experiment
\rightarrow
Eval
\rightarrow
Analysis
\rightarrow
Decision
\rightarrow
Training
$$

如果 AI 开始参与其中越来越多的环节，那么：

$$
R_{\text{self}}
=
\frac{
AI完成的R\&D工作
}{
全部R\&D工作
}
$$

就会不断上升。

2026 年已经出现一些明显的早期信号。

Anthropic 的 Automated Alignment Researchers 已经可以：

- 搜索文献；
- 提出训练方案；
- 生成数据；
- 训练模型；
- 根据 benchmark 结果继续 hill-climb。

在一些被清晰定义的 alignment failure 上，其最佳方法甚至超过了限定时间内的人类研究者方案。

这距离"完全自动科学家"仍然很远，但方向已经非常清楚：

> Coding Agent 正在向 Experiment Agent、Research Agent 演化。

---

## 十、如何衡量这一轮 AI 的真正进步？

因此，我越来越不愿意只看：

- MMLU；
- SWE-bench；
- 某一个 multimodal benchmark。

更值得长期追踪的其实是三个状态变量：

$$
\boxed{
C,\ H,\ R
}
$$

### C：Capability

模型单步能够完成多复杂的问题？

包括：

- Perception；
- Reasoning；
- Coding；
- Action。

### H：Horizon

AI 能够可靠连续工作多久？

例如：

$$
10秒
\rightarrow
10分钟
\rightarrow
1小时
\rightarrow
1天
$$

METR 提出的 Task-Completion Time Horizon 就是在尝试测量类似的问题：

> 一个需要人类专家花费 \(t\) 时间完成的任务，AI 在多长的 \(t\) 上仍然能够保持一定成功率？

相比单点 benchmark，这更接近真实 Agent 能力。

### R：Improvement Rate

AI 能够以多快速度改善 AI？

可以粗略理解成：

$$
R=
\frac{dC}{dt}
$$

如果未来 AI R&D 越来越自动化，那么这个变量可能最终比任何单个 benchmark 更重要。

---

## 十一、最终，可以把今天的 AI 压缩成一张技术地图

整个体系可以理解成：

$$
\boxed{
Capability
\rightarrow
System
\rightarrow
Feedback
\rightarrow
Experience
\rightarrow
Improvement
}
$$

进一步展开：

模型首先获得：

$$
Perception
+
Reasoning
+
Action
$$

然后通过：

$$
Harness
+
Environment
+
Budget
$$

变成能够完成真实任务的 Agent。

Agent 在环境中产生：

$$
Experience
$$

通过：

$$
Eval
+
Verifier
+
Reward
$$

获得反馈。

最后再经过：

$$
SFT
+
RL
+
Distillation
+
其他Post\text{-}training
$$

重新变成更强的模型。

于是形成：

$$
\boxed{
Model
\rightarrow
Agent
\rightarrow
Environment
\rightarrow
Experience
\rightarrow
Training
\rightarrow
Better\ Model
}
$$

---

## 结语：真正的主线是"瓶颈迁移"

如果一定要用一句话总结过去十几年的 AI：

> **AI 的核心瓶颈正在不断迁移。**

早期的问题是：

> 模型不会。

于是主桌是 Architecture、Pretraining、Scaling。

后来变成：

> 模型会，但叫不出来。

于是出现 SFT、RLHF、Reasoning RL、Capability Elicitation。

再后来变成：

> 模型会，但系统不会用。

于是 Agent、Harness、Context Engineering 兴起。

接下来很可能变成：

> Agent 会行动，但不知道自己做得对不对。

所以 Environment、Verifier、Eval、Reward 的重要性开始上升。

再往后：

> AI 能完成任务，但能不能自己找到更好的做法？

于是 Experiment Agent、Research Agent、Automated R&D、RSI 逐渐出现。

因此，理解今天的 AI，可能不需要记住所有热点。

只需要不断追问三个问题：

$$
\boxed{
模型现在会什么？
}
$$

$$
\boxed{
系统现在能可靠完成多长、多复杂的任务？
}
$$

以及：

$$
\boxed{
AI 是否正在加速生产下一代 AI？
}
$$

围绕这三个变量，绝大多数今天看起来杂乱的 AI 技术，都会重新变得清晰。
