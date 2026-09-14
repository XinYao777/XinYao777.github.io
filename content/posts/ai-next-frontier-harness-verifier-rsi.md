---
title: "AI 的下一站是什么？从 Harness、Verifier 到 RSI，以及算法工程师未来 3–5 年怎么下注"
date: 2026-09-14T21:00:00+08:00
draft: false
tags: ["Agent", "Harness", "Verifier", "Environment", "RSI", "职业规划", "笔记"]
categories: ["AI 技术思考"]
math: true
summary: "如果 AI 的技术主线是一系列'瓶颈迁移'，那么预测未来就不该问'下一个火的名词是什么'，而该问'当前瓶颈被解决后，下一个限制系统能力的变量是什么'。沿 Model → Harness → Environment/Verifier → Long-Horizon Agent → Automated R&D → RSI 这条线展开，并落到一个更实际的问题：站在 2026 年，算法工程师未来 3–5 年应该押什么。"
---

如果我们接受一个基本判断：

> AI 的技术主线，本质上是一系列"瓶颈迁移"。

那么预测未来，就不应该主要预测：

> 下一年最火的技术名词叫什么？

而应该问：

> **当今天这个瓶颈被解决以后，下一个限制系统能力的变量是什么？**

沿着这个思路，我目前更倾向于把未来几年的演进压缩成：

$$
\boxed{
Model
\rightarrow
Harness
\rightarrow
Environment/Verifier
\rightarrow
Long\text{-}Horizon\ Agent
\rightarrow
Automated\ R\&D
\rightarrow
RSI
}
$$

再进一步，智能的 Action Space 可能继续从数字世界扩展到科学和物理世界。

这条路线并不意味着前面的技术会消失。

恰恰相反：

> 前一阶段的核心技术，往往会逐渐成为后一阶段的基础设施。

---

## 一、第一步：Harness 会越来越重要，但也会越来越基础设施化

2025–2026 年，一个非常明显的变化是：

大家开始发现，仅仅拥有一个强模型远远不够。

同样一个模型：

```text
简单 Prompt
```

和：

```text
Context
+ Tools
+ Memory
+ Planning
+ Verifier
+ Retry
+ Subagents
```

最终能够完成的任务复杂度完全不同。

因此：

$$
Model\ Capability
\rightarrow
System\ Capability
$$

成为一个重要转变。

OpenAI 在 Harness Engineering 的实践中甚至明确提出：

> 人的主要工作开始从直接编写代码，转向设计环境、指定意图和建立反馈循环。

但这也意味着一个反直觉的判断：

> **Harness 不会永远是最稀缺的东西。**

因为很多 Harness 能力天然具有标准化倾向：

- Tool Calling；
- Context Compression；
- Retry；
- Memory；
- Permission；
- Subagent；
- Checkpoint。

长期来看，这些能力很可能逐渐下沉为模型平台和 Agent Runtime 的默认能力。

就像今天很少有应用公司把：

> "怎么实现 KV Cache"

作为核心竞争优势。

因此真正需要继续追问：

> Harness 之后，瓶颈在哪里？

---

## 二、我认为下一个极可能进入"主桌"的，是 Environment + Verifier

假设：

- 模型足够聪明；
- Harness 也足够成熟；

Agent 为什么仍然经常失败？

一个重要原因是：

> **现实世界缺乏清晰、快速、可信的反馈。**

Agent 可以生成 100 个 Candidate。

但系统真正困难的问题往往是：

$$
\boxed{
Which\ one\ is\ actually\ better?
}
$$

因此未来一个极其重要的能力，是把现实问题改造成：

$$
Observation
\rightarrow
Action
\rightarrow
Feedback
$$

的环境。

---

## 三、Coding 为什么可能只是第一个样板？

Coding 今天非常成功，是因为软件环境天然具备：

$$
Environment + Verifier
$$

例如：

$$
Code
\rightarrow
Compiler
\rightarrow
Unit\ Test
\rightarrow
Runtime
\rightarrow
Benchmark
$$

模型不需要等待几个月，马上就能知道：

> 自己刚才做得对不对。

因此 Coding Agent 能形成：

$$
Generate
\rightarrow
Execute
\rightarrow
Observe
\rightarrow
Verify
\rightarrow
Repair
$$

这是非常理想的智能闭环。

未来最值得寻找的，其实不是：

> Coding 之后哪个领域最热门？

而是：

> **哪个行业能够最先建立自己的 compiler + unit test？**

例如数学：

$$
Proof
\rightarrow
Formal\ Verifier
$$

科学计算：

$$
Hypothesis
\rightarrow
Simulation
\rightarrow
Measurement
$$

芯片设计：

$$
Design
\rightarrow
EDA
\rightarrow
Timing/Power/Area
$$

广告系统：

$$
Policy
\rightarrow
Traffic
\rightarrow
Conversion
$$

教育解题：

$$
Question
\rightarrow
Answer
\rightarrow
Ground\ Truth / Verifier
$$

如果一个领域能够建立稳定的：

$$
Action\rightarrow Feedback
$$

它就具备了被 Agent 化、甚至 RL 化的基础。

---

## 四、因此未来的关键不是"做 Agent"，而是"造 Agent 可以学习的世界"

这是一个我认为经常被低估的区别。

现在很多 Agent 项目关注的是：

- Workflow；
- Planning；
- Tool Calling；
- Multi-Agent。

但长期壁垒可能不在：

> Agent 怎么调用工具。

而在于：

> **你有没有一个其他人拿不到、或者难以复制的 Environment。**

可以形式化成：

$$
E =
(
Observation,\ State,\ Action,\ Feedback
)
$$

一个真正好的 AI Environment 需要回答：

### 模型看见什么？

Observation。

### 模型能做什么？

Action。

### 世界因此发生了什么改变？

State Transition。

### 它怎么知道自己做得好不好？

Feedback / Reward。

如果这四个变量能够被定义清楚：

$$
Agent
\rightarrow
Environment
\rightarrow
Experience
$$

就自然产生了。

而 Experience，最终又可以反过来训练模型。

---

## 五、第二个重要瓶颈：Long-Horizon Reliability

单步模型越来越强以后，一个巨大的问题会暴露出来：

> **每一步都基本正确，不等于长任务能够成功。**

假设一个 Agent 每一步都有：

$$
99\%
$$

的成功率。

一个任务需要 100 个关键步骤，那么极端简化情况下：

$$
0.99^{100}
\approx
36.6\%
$$

也就是说：

> "模型已经很聪明"和"模型可以连续独立工作一天"之间，仍然存在巨大鸿沟。

所以未来 Agent 很重要的一条技术主线不会只是继续提升 IQ。

而是：

- State Management；
- Checkpoint；
- Error Recovery；
- Replanning；
- Verification；
- Memory；
- Long Context；
- Hierarchical Planning。

Anthropic 对真实 Agent 使用的观察已经发现，在最长的一批 Claude Code session 中，Agent 在停下来之前连续自主运行的时间在数月内明显增加。

METR 则直接提出 Task-Completion Time Horizon：

> 用"一个任务需要人类专家多少时间完成"来表示任务复杂度，然后测 AI 在不同任务时间尺度上的成功率。

这个指标比"又提升了几个 benchmark point"更值得长期关注。

因为它回答的是：

$$
\boxed{
AI\ 到底可以独立负责多大的工作单元？
}
$$

---

## 六、再往后：从 Task Agent 到 Research Agent

目前大部分 Agent 的基本模式仍然是：

$$
Goal
\rightarrow
Plan
\rightarrow
Execute
$$

例如：

> 帮我实现这个功能。

但是研究工作的结构并不一样。

Research 更接近：

$$
Hypothesis
\rightarrow
Experiment
\rightarrow
Result
\rightarrow
Analysis
\rightarrow
New\ Hypothesis
$$

这是一个 Open-ended Loop。

因此 Coding Agent 下一阶段最值得关注的变化，可能不是：

> 它能写更多代码。

而是：

> **它开始自己设计并运行实验。**

例如：

```text
发现问题
↓
阅读相关研究
↓
提出5个假设
↓
并行设计实验
↓
实现
↓
训练
↓
评估
↓
分析失败case
↓
设计第二轮实验
```

这实际上已经出现了一些早期形态。

Anthropic 2026 年展示的 Automated Alignment Researchers 已经可以自主：

- 搜索文献；
- 提出方法；
- 构造训练数据；
- 训练模型；
- 根据 benchmark 不断 hill-climb。

在被良好定义的 alignment failure 上，其最佳方案甚至可以超过有限时间的人类研究者基线。

这件事情真正重要的地方不是：

> "AI 已经比科学家强。"

远远不能这么下结论。

而是它第一次比较清楚地展示出：

$$
\boxed{
Experimentation
}
$$

本身也开始进入自动化范围。

---

## 七、AI Research 自动化大概率会经历四个层级

可以粗略把它拆成：

### Level 1：Research Coding

人：

> 跑这个实验。

AI：

> 写代码、Debug、执行。

这一层已经非常明显。

OpenAI 的内部数据已经显示，Coding Agent 正在大幅进入研究基础设施、技术支持和实验工作。

### Level 2：Experiment Agent

人给：

> 比较方案 A/B/C。

AI 自己：

$$
Implementation
\rightarrow
Experiment
\rightarrow
Evaluation
\rightarrow
Report
$$

### Level 3：Research Agent

人只给一个研究目标：

> 降低小模型 reasoning gap。

AI：

$$
Literature
\rightarrow
Hypothesis
\rightarrow
Experiment
\rightarrow
Analysis
\rightarrow
Iteration
$$

### Level 4：Research Organization

进一步不再是一个 Agent。

而可能出现：

$$
Research\ Planner
$$

下面管理：

$$
Agent_1,\ Agent_2,\ ...,\ Agent_n
$$

分别并行探索：

- Data；
- Algorithm；
- Infra；
- Evaluation。

最后由 Reviewer / Meta Researcher 聚合结果。

真正有意义的 Multi-Agent，很可能不是：

> CEO Agent 和工程师 Agent 开会。

而是：

$$
\boxed{
Parallel\ Search
+
Specialization
+
Independent\ Verification
}
$$

它真正扩大的变量是：

$$
Search\ Breadth
$$

---

## 八、这时候 RSI 才真正成为核心问题

RSI，Recursive Self-Improvement，最容易被误解。

如果把它理解成：

> 一个模型突然开始修改自己的源代码。

很容易陷入科幻讨论。

更现实的定义是：

$$
\boxed{
AI参与AI能力提升闭环的比例
}
$$

例如一个模型研发闭环：

$$
Problem
\rightarrow
Idea
\rightarrow
Literature
\rightarrow
Code
\rightarrow
Experiment
\rightarrow
Evaluation
\rightarrow
Analysis
\rightarrow
Decision
\rightarrow
Training
$$

过去：

$$
Human
$$

几乎负责所有环节。

现在 Coding、Debug、Experiment 已经开始被自动化。

未来如果：

- Hypothesis；
- Experiment Design；
- Analysis；
- Data Generation；
- Evaluation；

继续被自动化，那么：

$$
R_{\text{self}}
=
\frac{
AI完成的AI\ R\&D工作
}{
全部AI\ R\&D工作
}
$$

就会持续提高。

我认为这比讨论：

> "模型到底有没有真正修改自己？"

更加有操作意义。

---

## 九、RSI 最后的瓶颈，可能不是 Intelligence，而是 Judgment

这里会出现一个很有趣的问题。

假设未来 Agent 一天可以跑：

$$
1,000,000
$$

组实验。

那么问题会立刻变成：

> **应该跑哪 100 万组？**

如果 Research Direction 错了：

$$
10^6\times Bad\ Hypothesis
$$

只是以更快速度浪费 Compute。

所以技术瓶颈可能继续迁移：

$$
Coding
\rightarrow
Execution
\rightarrow
Experiment
\rightarrow
Judgment
\rightarrow
Taste
$$

OpenAI 当前内部 Agent 使用数据里，一个值得注意的现象是：

> High-level planning 目前仍然只占 Agent 使用中的很小部分。

Anthropic 对 Claude Code 的研究也发现：

> 更强的领域 expertise 仍然能够提高人与 Agent 协作的效果。

因此未来最稀缺的能力可能逐渐从：

> "你能不能做？"

转向：

> **"你知不知道什么值得做？"**

---

## 十、再往后，数字世界的 Agent 范式会向物理世界迁移

前面讨论的绝大多数 Agent，都运行在 Digital Environment。

而物理世界本质上也具有类似结构：

$$
Observation
\rightarrow
Reasoning
\rightarrow
Action
\rightarrow
Feedback
$$

只是困难得多。

因为：

- Action 更贵；
- Feedback 更慢；
- Exploration 可能危险；
- Data 更少；
- Simulation 存在 Sim2Real Gap。

因此 Physical AI 可以理解成：

> **把今天 Coding Agent 的闭环搬到真实世界。**

Google DeepMind 2026 年发布的 Gemini Robotics 2 已经明确把模型从纯理解扩展到 Vision-Language-Action，并探索多步真实世界行动和机器人协作。

所以长期看，可能存在：

$$
Language
\rightarrow
Reasoning
\rightarrow
Coding
\rightarrow
Digital\ Agent
\rightarrow
Scientific\ Agent
\rightarrow
Physical\ Agent
$$

这条路线的本质其实只有一句话：

$$
\boxed{
Action\ Space\ 在不断扩大
}
$$

---

## 十一、那么站在 2026 年，一个 AI 算法工程师应该押什么？

这是比预测行业更实际的问题。

我会用四个变量定义个人技术资产：

$$
V
\approx
S\times L\times C\times T
$$

其中：

- \(S\)：Scarcity，稀缺性；
- \(L\)：Leverage，AI 能否放大你的产出；
- \(C\)：Compounding，是否具有长期复利；
- \(T\)：Transferability，模型和工具换代后是否仍然有效。

按照这个标准，一个非常重要的趋势是：

> **越靠近"具体执行"，未来越容易被 AI 自动化；越靠近"定义问题和判断结果"，越可能升值。**

---

## 十二、第一优先级：成为 AI-native Research Engineer

未来算法工程师最大的变化，不一定是学习更多模型结构。

而是改变自己的生产函数。

传统：

$$
1\ Human
\rightarrow
1\ Experiment
$$

AI-native：

$$
1\ Human
\rightarrow
N\ Agents
\rightarrow
N\times Experiments
$$

你的工作逐渐从：

- 写代码；
- Debug；
- 配环境；

变成：

- 定义问题；
- 设计实验；
- 分配任务；
- 审核结果；
- 决定下一轮。

真正需要提升的指标是：

$$
\boxed{
Experiment\ Throughput
}
$$

以及：

$$
\boxed{
Useful\ Experiment\ Throughput
}
$$

后者更重要。

不是一天跑多少实验。

而是一天能排除多少错误方向、产生多少有效知识。

---

## 十三、第二优先级：Eval / Reward / Verifier

这是我认为未来几年极容易被低估的技术方向。

模型越来越擅长：

$$
Generate
$$

真正困难的是：

$$
Evaluate
$$

未来大量系统都会变成：

$$
Agent
\rightarrow
Action
\rightarrow
Verifier
\rightarrow
Feedback
\rightarrow
Agent
$$

所以 Eval 会越来越靠近系统核心。

值得深入的东西包括：

- Capability Evaluation；
- LLM-as-Judge；
- Reward Model；
- Process Reward；
- Outcome Reward；
- Rule Verifier；
- Rubric；
- Adversarial Evaluation；
- Reward Hacking；
- Scalable Oversight。

OpenAI 在科学计算 Agent 的实践中也观察到类似问题：Agent 可以大幅降低工程工作的成本，但科学有效性的判断仍然大量依赖可靠的外部参考、可测量目标以及人类 judgment。

未来：

> **生成 Candidate 会越来越便宜，可靠判断 Candidate 会越来越值钱。**

---

## 十四、第三优先级：Environment，而不仅仅是 Agent Framework

不要把未来几年大量时间押在：

> 学会某一个 Agent SDK。

Framework 一定会不断变化。

真正长期有价值的是学会：

> 如何把一个领域形式化成 Agent 可以行动和学习的 Environment。

也就是理解：

$$
Observation
$$

$$
State
$$

$$
Action
$$

$$
Feedback
$$

以及它们如何构成：

$$
Experience
$$

一旦 Environment 构建成功：

$$
Agent
\rightarrow
Experience
\rightarrow
Training
$$

自然就能够形成数据和模型飞轮。

---

## 十五、第四优先级：后训练要深入，但不要成为"配方工程师"

SFT、DPO、GRPO、RLOO、OPD 都值得学习。

但不要把自己的长期身份定义成：

> "某个具体 RL 算法专家。"

算法变化太快。

更稳定的抽象其实是：

$$
\boxed{
Experience
\rightarrow
Credit
\rightarrow
Update
}
$$

第一步：

> Experience 从哪里来？

是真实用户？Teacher？Self-play？Agent rollout？Environment rollout？

第二步：

> 如何知道 Experience 的好坏？

GT？Rule？Verifier？RM？LLM-as-Judge？

第三步才是：

> 如何更新 Policy？

SFT？Preference Optimization？RL？Distillation？OPD？

真正长期有价值的是：

$$
Experience\ Design
+
Reward\ Design
+
Optimization
$$

而不是最后一个算法名字。

---

## 十六、第五优先级：Problem Formulation 和 Domain Judgment

这可能最终是价值最高、但最难速成的一类能力。

如果未来越来越多问题变成：

$$
Human:\ Define\ Problem
$$

$$
AI:\ Search\ Solution
$$

那么人的关键价值就变成：

> **什么问题值得搜索？**

例如：

- 当前真正的 failure mode 是什么？
- 是模型能力问题还是系统问题？
- 应该优化哪个 metric？
- Proxy 有没有偏？
- Eval 有没有 contamination？
- 是 Data、Reward 还是 Harness 出问题？
- 提升 2 个百分点究竟有没有业务意义？
- 问题应该建模成 Classification、Ranking、Generation 还是 Agent？

这些事情通常被叫作：

$$
Research\ Taste
$$

听起来很抽象。

但实际上可以训练。

它来自于：

$$
大量真实问题
+
失败经验
+
能力抽象
+
系统归因
$$

---

## 十七、因此 Domain Expertise 不应该被丢掉

一个常见误区是：

> 模型越来越强以后，垂直领域知识会越来越不重要。

现实可能恰恰相反。

通用 Coding、通用 Harness、通用 Model 都越来越便宜以后，真正稀缺的东西反而是：

$$
\boxed{
Domain
+
Environment
+
Verifier
+
Experience
}
$$

Anthropic 对 Claude Code 的研究已经给出了一个早期信号：

领域知识更强的人，能够从 Agent 中获得更高的有效工作产出。

所以未来很好的个人定位，并不是：

> "我只懂 AI。"

而可能是：

$$
\boxed{
某个真实领域
+
AI系统能力
}
$$

---

## 十八、以多模态教育为例：不要只把它看成"拍照解题"

例如一个多模态教育系统，可以重新抽象成：

$$
Multimodal\ Reasoning\ Environment
$$

它天然存在：

### Observation

- 真实图片；
- OCR；
- Layout；
- Diagram；
- Handwriting。

### Reasoning

- 数学；
- 物理；
- 化学；
- 语言理解。

### Action

- Search；
- RAG；
- Python；
- Calculator；
- Tool；
- Direct Solve。

### Feedback

- 标准答案；
- 题库；
- Rule；
- Symbolic Verifier；
- Satisfaction；
- User Feedback。

那么系统就可以形成：

$$
Image
\rightarrow
Agent
\rightarrow
Solve
\rightarrow
Verify
\rightarrow
Repair
\rightarrow
Experience
\rightarrow
Training
$$

这个视角下，它就不再只是：

> 一个搜题产品。

而是：

> **一个真实的大规模 Multimodal Reasoning + Agent + Verifier + Post-training Environment。**

类似的重构方式，可以用于很多行业。

---

## 十九、如果未来三年只有 100 点学习时间，我会这样配置

这是一个粗略比例，而不是绝对标准。

### 25%：AI-native Coding / Research Automation

目标：

$$
1人
\rightarrow
5\sim20个并行Agent
$$

让 AI 深度参与：Coding、Debug、Paper Reproduction、Experiment、Analysis。

### 25%：Eval / Reward / Verifier

重点研究：

> 系统怎样知道自己做对了？

这是未来很多闭环的核心。

### 20%：Post-training / Experience Learning

深入理解：

$$
SFT
\rightarrow
Preference
\rightarrow
RL
\rightarrow
Distillation
$$

但始终围绕：

$$
Experience
\rightarrow
Reward
\rightarrow
Update
$$

### 15%：Agent / Environment

不要只学习 Framework API。

学习：State、Action、Tool、Memory、Recovery、Sandbox、Simulator、Long Horizon。

### 15%：真实 Domain

保留一个自己真正懂的行业或问题空间。

因为：

$$
Domain\ Expertise
\times
AI
$$

很可能比：

$$
Generic\ AI\ Skill
$$

具有更长期的个人壁垒。

---

## 二十、相反，哪些事情不值得重仓？

不是完全不学，而是不应该成为个人核心定位。

例如：

### 某一个 Agent Framework

API 会很快变化。

### Prompt Engineering 技巧

会逐渐成为基础能力。

### 纯手工数据 Pipeline

AI 会大量自动化数据处理。

应该更多研究：

$$
Data\ Value
$$

而不仅仅是：

$$
Data\ Processing
$$

### 只会微调一个开源模型

比如：

> LoRA + 某训练框架 + 调参数。

这是必要技能，但越来越难成为长期壁垒。

### Benchmark Chasing

未来 benchmark contamination、Harness variance 和过拟合问题都会越来越严重。

比 leaderboard 更重要的是：

$$
Capability\ Diagnosis
$$

---

## 二十一、未来算法工程师的角色可能发生一次完整迁移

可以把个人成长路线写成：

$$
Engineer
$$

先升级成：

$$
Research\ Engineer
$$

再升级成：

$$
System\ Builder
$$

最后可能变成：

$$
\boxed{
Loop\ Designer
}
$$

什么叫 Loop Designer？

不是自己亲手完成每个任务。

而是能够定义：

$$
Problem
$$

构造：

$$
Environment
$$

设计：

$$
Verifier
$$

调度：

$$
Agents
$$

获得：

$$
Experience
$$

最后推动：

$$
Improvement
$$

形成：

$$
\boxed{
Problem
\rightarrow
Agent
\rightarrow
Environment
\rightarrow
Verifier
\rightarrow
Experience
\rightarrow
Learning
\rightarrow
Better\ Agent
}
$$

这个闭环。

---

## 结语：不要和 AI 比执行速度，要站到 AI 的杠杆上

过去优秀算法工程师的价值很大一部分来自：

> 我能不能把问题做出来？

未来这个问题会越来越便宜。

因为 AI 会越来越擅长：

$$
Search\ Solution
$$

于是人的价值会逐渐向上迁移：

> 什么问题值得解决？

> 搜索空间应该怎么定义？

> Agent 可以做哪些 Action？

> 什么结果才算真正正确？

> 如何把一次成功变成可复用的 Experience？

> 如何让这些 Experience 最终改善整个系统？

如果只允许我对未来 3–5 年押一个方向，我会押：

$$
\boxed{
Real\ Problem
\rightarrow
Environment
\rightarrow
Agent
\rightarrow
Verifier
\rightarrow
Experience
\rightarrow
Posttraining
\rightarrow
Better\ Agent
}
$$

也就是：

> **找一个真实、高价值、能够验证结果的领域，把它建设成 AI 可以持续行动、获得反馈、产生经验并不断改善的环境。**

因为在那个位置上：

- Model；
- Data；
- Algorithm；
- Agent；
- Harness；
- Eval；
- RL；
- Domain；

第一次真正汇合到同一个闭环中。

而这很可能就是下一阶段 AI 算法工程师个人杠杆最高的位置。