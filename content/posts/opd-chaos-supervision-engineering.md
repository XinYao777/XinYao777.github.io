---
title: "OPD 乱象：从监督信号、数据分布与学习算法正本清源"
date: 2026-09-23T01:30:00+08:00
draft: false
tags: ["OPD", "蒸馏", "强化学习", "Rubric", "后训练"]
categories: ["AI 技术思考"]
math: true
summary: "越来越多方法都被称为 On-Policy Distillation，但它们可能分别使用 teacher logits、rubric、judge、reward 或 step-level supervision。本文不再按论文命名分类，而是从 rollout distribution、supervision signal、credit assignment、objective 和 gradient estimator 五个维度重新拆解 OPD。"
---

最近一段时间，On-Policy Distillation（OPD）开始变成一个越来越宽泛的词。

最早我们理解的 OPD 很具体：

$$
y \sim \pi_S(\cdot \mid x)
$$

Student 自己 rollout，然后 Teacher 在 Student 实际访问到的 prefix 上给出 token distribution：

$$
\pi_T(\cdot \mid x, y_{\lt t})
$$

最后直接做 Teacher–Student 的 distribution matching。

但现在，一些被称为 “Black-box OPD” 的方法已经变成：

$$
\text{Student Rollout}
\rightarrow
\text{Teacher / Judge}
\rightarrow
\text{Rubric or Reward}
\rightarrow
\text{GRPO}
$$

它们当然仍然可以被宽泛地称为 “distillation”：监督信息最终来自 Teacher。

问题在于，如果我们继续只使用 “OPD” 这一个词，很多真正重要的技术差异就被掩盖了。

我更倾向于把问题重新拆开：

$$
\boxed{
\text{Training Method}
=
\text{Rollout Distribution}
\times
\text{Supervision Signal}
\times
\text{Credit Assignment}
\times
\text{Objective}
\times
\text{Gradient Estimator}
}
$$

一句话概括：

> **On-policy 回答样本从哪里来；Distillation 回答知识从哪里来；SFT / KL / RL 回答模型怎么学；Credit Assignment 回答这个信号应该作用到哪里。**

这几个概念本来就不是一回事。

---

## 一、先别问是不是 OPD：Teacher 到底给了什么？

假设有 Teacher $\pi_T$ 和 Student $\pi_\theta$。

Teacher 可以通过不同“带宽”的监督通道向 Student 传递信息。

最简单的是一个完整答案：

$$
x \xrightarrow{\pi_T} y_T
$$

也可以是 Preference：

$$
y^+ \succ y^-
$$

也可以是标量 Reward：

$$
R(x,y)
$$

也可以是一组 Rubric：

$$
\mathcal R_x = \{r_1, r_2, \ldots, r_m\}
$$

如果 Teacher 是白盒模型，还可以直接暴露完整 token distribution：

$$
\pi_T(\cdot \mid s_t)
$$

甚至可以进一步访问 hidden states、attention 或中间表示。

所以所谓“蒸馏”，最先应该问的不是：

> 这是 SFT 还是 OPD？

而是：

> **Teacher 究竟向 Student 暴露了什么训练信息？**

这可以称为 **Supervision Channel**。

---

## 二、黑盒 Teacher 的第一种利用：Sequence Distillation

如果我们只有一个黑盒 API，最自然的做法是让 Teacher 生成答案：

$$
y_T \sim \pi_T(\cdot \mid x)
$$

然后得到训练集：

$$
\mathcal D_T = \{(x, y_T)\}
$$

Student 再用标准最大似然训练：

$$
\mathcal L_{\mathrm{SFT}}
=
-\sum_t
\log
\pi_\theta(y_{T,t} \mid x, y_{T,\lt t})
$$

这类方法可以叫：

- Sequence Distillation
- Response Distillation
- Synthetic-data SFT
- Imitation Learning

2016 年的 [Sequence-Level Knowledge Distillation](https://arxiv.org/abs/1606.07947) 已经非常典型。

从今天的语言看，它其实就是：

$$
\boxed{
\text{Teacher-generated sequence}
+
\text{SFT}
}
$$

这里有一个重要区分：

> **Distillation 是监督知识来源，SFT 是学习算法。**

二者并不冲突。

---

## 三、经典 OPD 真正改变的是训练状态分布

Sequence Distillation 有一个经典问题。

训练时，Student 看到的是 Teacher prefix：

$$
s_t^{T} = (x, y_{T,\lt t})
$$

但推理时，Student 看到的是自己的 prefix：

$$
s_t^{S} = (x, y_{S,\lt t})
$$

于是：

$$
d^{\text{train}}(s) \neq d^{\pi_S}(s)
$$

这就是 autoregressive learning 中长期存在的 distribution mismatch / exposure bias。

2023 年的 [GKD: On-Policy Distillation of Language Models](https://arxiv.org/abs/2306.13649) 抓住了这一点：不要只在 Teacher trajectory 上训练，而是让 Student 自己生成，再让 Teacher 对 Student 真正访问到的状态提供监督。

也就是：

$$
y_S \sim \pi_S(\cdot \mid x)
$$

然后在：

$$
s_t = (x, y_{S,\lt t})
$$

上比较：

$$
\pi_S(\cdot \mid s_t)
\quad\text{与}\quad
\pi_T(\cdot \mid s_t)
$$

例如优化：

$$
D_{\mathrm{KL}}
\left(
\pi_S(\cdot \mid s_t)
\Vert
\pi_T(\cdot \mid s_t)
\right)
$$

所以 OPD 最核心的变化其实不是“出现了 Teacher”。

传统 Knowledge Distillation 本来就有 Teacher。

真正改变的是：

$$
\boxed{
s \sim d^{\pi_T}
\;\longrightarrow\;
s \sim d^{\pi_S}
}
$$

也就是：

> **Student 决定在哪里学习，Teacher 决定在这些状态上往哪个方向调整。**

因此，“On-policy”首先应该被理解为一个 **state / trajectory distribution** 的概念。

---

## 四、On-policy 并没有规定监督信号必须是 KL

这是现在很多讨论开始混乱的地方。

有时大家会不自觉地把：

$$
\text{On-policy}
$$

等同于：

$$
\text{Token-level Teacher KL}
$$

但两者其实完全是两个维度。

On-policy 只说明：

$$
\tau \sim \pi_\theta
$$

即训练 trajectory 来自当前或接近当前的 Student policy。

这条 trajectory 上可以得到不同监督：

### Teacher logits

$$
\pi_T(\cdot \mid s_t)
$$

### Outcome reward

$$
R(\tau)
$$

### Judge score

$$
R_J(x,\tau)
$$

### Rubric

$$
\mathcal R_x = \{r_1,\ldots,r_m\}
$$

### Step correction

$$
\tilde C_k \sim \pi_T(\cdot \mid s_k)
$$

所以：

$$
\boxed{
\text{On-policy}
\not\Rightarrow
\text{Token KL}
}
$$

On-policy 只回答：

> **这批训练样本是谁生成的？**

---

## 五、同样，Distillation 也没有规定必须使用 KL

如果把 Distillation 定义为：

> Student 的训练监督由 Teacher 提供。

那么 Teacher 可以扮演很多不同角色。

### 1. Teacher 是 Target Generator

$$
T(x) \rightarrow y_T
$$

然后 Student 做 SFT。

### 2. Teacher 是 Distribution Provider

$$
T(s_t) \rightarrow \pi_T(\cdot \mid s_t)
$$

然后做 KL matching。

### 3. Teacher 是 Preference Annotator

$$
T(x,y_1,y_2)
\rightarrow
y_1 \succ y_2
$$

然后做 DPO / Preference Optimization。

### 4. Teacher 是 Reward Model

$$
T(x,y) \rightarrow R_T(x,y)
$$

然后做 RL。

### 5. Teacher 是 Rubric Generator

$$
T(x,y_T,y_S)
\rightarrow
\mathcal R_x
$$

再由 Judge 使用这些 rubric 对 Student rollout 评分，然后做 RL。

因此：

$$
\boxed{
\text{Distillation}
=
\text{Knowledge Source}
}
$$

而不是一个固定的 optimizer。

---

## 六、这样再看 ROPD：它更像 Teacher-guided On-policy RL

2026 年的 [Rubric-based On-policy Distillation](https://arxiv.org/abs/2605.07396) 很有代表性。

它想解决的问题是：

> 经典 OPD 依赖 Teacher logits，黑盒 API 拿不到 logits，怎么办？

它的答案并不是去近似 logits。

而是换一个监督通道。

先由 Student 自己 rollout：

$$
y_i \sim \pi_\theta(\cdot \mid x)
$$

然后从 Teacher–Student 对比中生成 prompt-specific rubrics：

$$
\mathcal R_x
$$

再利用这些 rubrics 对 Student rollout 打分：

$$
R_i = R_{\mathcal R_x}(y_i)
$$

最终进行 on-policy policy optimization。

因此，从“知识来源”看，它确实是 Distillation：

$$
\text{Teacher}
\rightarrow
\text{Rubric}
\rightarrow
\text{Student}
$$

但从“训练机制”看，它已经更接近：

$$
\boxed{
\text{Teacher-guided On-policy RL}
}
$$

而不是经典的：

$$
\boxed{
\pi_S(\cdot \mid s)
\approx
\pi_T(\cdot \mid s)
}
$$

两者优化的数学对象已经不同。

经典 logit-based OPD 更接近 distribution matching：

$$
\min_\theta
D(\pi_S,\pi_T)
$$

而 rubric-RL 更接近：

$$
\max_\theta
\mathbb E_{y\sim\pi_\theta}
[R_T(x,y)]
$$

这就是我认为今天 “Black-box OPD” 这个词容易制造误解的地方。

---

## 七、如果什么都能叫 OPD，OPD 就不再是一种算法

如果我们把 OPD 定义放宽为：

$$
\boxed{
\text{Student On-policy Rollout}
+
\text{Any Teacher-derived Signal}
}
$$

那么：

- Teacher logits + KL 是 OPD
- Teacher reward + PPO 是 OPD
- Teacher rubric + GRPO 是 OPD
- Teacher critique + RL 是 OPD
- Teacher step correction + SFT 也是 OPD

这当然不是不允许。

但此时 OPD 已经不再是一种具体优化算法，而只是一个 **Training Paradigm**。

于是研究中只说：

> “我们使用 OPD。”

信息量已经远远不够。

至少还应该说明：

$$
(
\text{Rollout},
\text{Signal},
\text{Credit},
\text{Objective},
\text{Estimator}
)
$$

---

## 八、第四个容易被忽略的轴：Credit Assignment

即使已经知道：

- trajectory 来自 Student；
- supervision 来自 Teacher；

仍然还缺一个问题：

> **这个监督信号到底应该归因给哪些行为？**

这是 Credit Assignment。

### Sequence-level

整条 response 共享一个 advantage：

$$
A_{i,t} = A_i
$$

这是 Vanilla GRPO 最典型的形式。

### Token-level

每个 token 都有自己的局部信号：

$$
A_{i,t} = A(s_{i,t}, a_{i,t})
$$

经典 token-level OPD 可以被理解为极细粒度的局部监督。

### Step-level

如果 trajectory 被切成：

$$
\tau_i = C_{i,1}, C_{i,2}, \ldots, C_{i,K_i}
$$

则可以给每一个 reasoning step 单独 credit：

$$
A_{i,t} = A_{i,k},
\qquad t \in C_{i,k}
$$

2026 年的 [SRaR](https://arxiv.org/abs/2605.17291) 正是在 rubric-RL 中研究这个问题：把 rubric item 归因到具体 reasoning step，而不是把所有 rubric 分数压成一个 sequence scalar。

另一篇 [Step-Level On-Policy Distillation](https://arxiv.org/abs/2608.16333) 则从 OPD 的另一端出发：它认为 token-level teacher correction 太碎，因此让 Teacher 在 Student 已访问到的状态上生成完整 reasoning step。

所以：

$$
\boxed{
\text{Token}
\leftrightarrow
\text{Step}
\leftrightarrow
\text{Sequence}
}
$$

本身就是一个独立的研究维度。

---

## 九、Step-level 也没有想象中简单：跨 rollout 怎么对齐？

SRaR 进一步暴露了一个有意思的问题。

如果同一个 prompt 有多个 rollout：

- Rollout A 有 4 个 reasoning steps；
- Rollout B 有 7 个；
- Rollout C 有 3 个；

那么所谓“同一个 step 做 group normalization”到底意味着什么？

形式上可以写成：

$$
A_{i,k}
=
\operatorname{Norm}_i(R_{i,k})
$$

但这里偷偷假设了：

$$
C_{1,k}, C_{2,k}, \ldots
$$

具有可比性。

现实中并不一定如此。

例如：

- A 的 Step 3 是“建立方程”；
- B 的 Step 3 是“代入计算”；
- C 的 Step 3 已经是“最终验证”。

所以：

$$
\boxed{
k\text{-th step}
\not\Rightarrow
\text{same semantic state}
}
$$

这说明，一旦从 sequence-level GRPO 下沉到 step-level RL，就必须额外回答一个问题：

> **跨 rollout 的 reasoning step 如何进行语义对齐？**

一种更自然的方法，是以 rubric criterion 为 anchor。

假设第 $j$ 个 rubric 表示：

> “是否正确建立了某个关键方程”。

它可以在不同 rollout 中出现在不同位置：

$$
k_{A,j}=2,\qquad
k_{B,j}=4,\qquad
k_{C,j}=1
$$

但它们语义上仍然对应同一个 criterion。

于是可以先沿 rubric 维度比较：

$$
\bar r_{i,j}
=
\operatorname{GroupNorm}_i(r_{i,j})
$$

再把 credit 回填到各自 trajectory 的对应 step：

$$
A_{i,k}
=
\sum_{j:k_{i,j}=k}
\bar r_{i,j}
$$

这个方向比简单的 “Step 3 对 Step 3” 更干净。

它也说明：

> **Credit Assignment 与 Semantic Alignment 本身又是两个不同问题。**

---

## 十、MiniLLM 说明：Objective 和 Optimizer 也不能混为一谈

另一个很常见的问题是：

> “这个方法到底是 KL，还是 RL？”

很多时候，这个问题本身就混淆了不同层次。

例如 [MiniLLM](https://openreview.net/forum?id=5h0qf7IBZZ) 的目标是 sequence-level reverse KL：

$$
D_{\mathrm{KL}}
(
\pi_\theta
\Vert
\pi_T
)
$$

但它使用 policy-gradient machinery 来优化这一目标。

因此完全可能出现：

$$
\boxed{
\text{Objective = KL}
}
$$

同时：

$$
\boxed{
\text{Gradient Estimator = Policy Gradient}
}
$$

所以不能简单地把方法划分成：

- KL 方法
- RL 方法

更严格地应该分开：

1. **你最终想最小化 / 最大化什么？**
2. **你如何估计这个目标的梯度？**

---

## 十一、我更喜欢的统一表示

以后看到任何一个后训练方法，我更愿意先把论文名拿掉，然后写成：

$$
\boxed{
\mathcal A
=
(
q_{\text{rollout}},
\Phi,
C,
J,
G
)
}
$$

其中：

- $q_{\text{rollout}}$：谁产生训练 trajectory；
- $\Phi$：监督信息是什么；
- $C$：credit assignment 粒度与方式；
- $J$：最终优化目标；
- $G$：gradient estimator / policy update。

例如：

### Sequence Distillation

$$
(
\pi_T,
\text{Teacher Response},
\text{Token CE},
\text{MLE},
\text{Backprop}
)
$$

### Vanilla Logit-based OPD

$$
(
\pi_S,
\text{Teacher Logits},
\text{Token},
\text{KL},
\text{Direct Gradient}
)
$$

### MiniLLM

$$
(
\pi_S,
\text{Teacher LogProb},
\text{Reward-to-go},
\text{Sequence Reverse KL},
\text{Policy Gradient}
)
$$

### ROPD

$$
(
\pi_S,
\text{Teacher Rubric},
\text{Sequence},
\text{Expected Reward},
\text{GRPO-style RL}
)
$$

### SRaR

$$
(
\pi_S,
\text{Rubric Judge},
\text{Step},
\text{Expected Reward},
\text{GRPO-style RL}
)
$$

### SOPD

$$
(
\pi_S\text{-visited state},
\text{Teacher Step},
\text{Step},
\text{MLE},
\text{Teacher Forcing}
)
$$

这张表比继续发明一个新的 “xxx-OPD” 名字更有解释力。

---

## 十二、Black-box Teacher 的真正演进：从答案生成器到训练信号生成器

如果只讨论黑盒 API 的利用方式，我觉得技术路线其实很清楚。

最初，Teacher 是一个 **Answer Generator**：

$$
T(x)
\rightarrow
y_T
\rightarrow
\text{SFT}
$$

然后变成 **Preference Annotator**：

$$
T(x,y_1,y_2)
\rightarrow
y_1 \succ y_2
\rightarrow
\text{Preference Optimization}
$$

再后来变成 **Reward Model / Judge**：

$$
T(x,y)
\rightarrow
R(x,y)
\rightarrow
\text{RL}
$$

再进一步，Teacher 开始生成 Rubric：

$$
T
\rightarrow
\mathcal R_x
\rightarrow
\text{Judge}
\rightarrow
\text{RL}
$$

下一步很自然：

$$
T
\rightarrow
\text{Error Attribution}
\rightarrow
\text{Step Credit}
$$

甚至：

$$
T
\rightarrow
\text{Corrected Step}
\rightarrow
\text{Local Repair}
$$

于是 Teacher 的角色发生了一个很重要的变化：

$$
\boxed{
\text{Target Generator}
\rightarrow
\text{Training Signal Generator}
}
$$

我认为这比“Sequence Distillation 进化到 Black-box OPD”更准确。

因为真正进化的是：

> **我们从黑盒 Teacher 中提取训练价值的方式。**

---

## 十三、真正值得研究的可能不是 OPD，而是 Supervision Engineering

如果已经有一个很强的 Teacher API，那么同样的调用预算可以购买不同类型的监督：

### 一个标准答案

$$
y_T
$$

### 一个 Preference

$$
y^+ \succ y^-
$$

### 一个 Reward

$$
R(y_S)
$$

### 一组 Rubric

$$
r_1,\ldots,r_m
$$

### 每个 step 的错误归因

$$
C_k \rightarrow e_k
$$

### 一个局部修正版

$$
C_k \rightarrow \tilde C_k
$$

真正值得问的问题其实是：

$$
\boxed{
\frac{
\text{Model Improvement}
}{
\text{Teacher Query Cost}
}
}
$$

哪一种 supervision channel 的 Training Value 最高？

这比争论：

> “这个方法到底算不算 OPD？”

更接近后训练研究的核心。

我更愿意把它叫做：

$$
\boxed{
\text{Supervision Engineering}
}
$$

---

## 十四、On-policy 的真正价值：把监督预算花在 Student 真正访问的地方

On-policy 为什么重要？

因为训练价值不只取决于监督质量。

还取决于这个监督是不是作用在 Student 当前真正相关的状态上。

可以粗略写成：

$$
\text{Training Value}
\approx
\text{Signal Quality}
\times
\text{State Relevance}
$$

Teacher SFT 数据可能质量极高，但如果这些状态距离当前 Student 很远：

$$
P_{\pi_S}(s) \approx 0
$$

那么对当前模型而言，训练效率未必最高。

On-policy 的核心贡献是：

$$
s \sim d^{\pi_S}
$$

也就是：

> **把监督预算集中在 Student 当前真正会走到、真正会犯错的区域。**

因此，On-policy 是一个非常有价值的思想。

但它没有必要和某一种具体 KL、某一种 RL 算法绑定。

---

## 十五、结语：不要再问“这是不是 OPD”

OPD 这个词正在经历一个典型的术语膨胀过程。

狭义 OPD 很清楚：

$$
\boxed{
\text{Student Rollout}
+
\text{Teacher Token Distribution}
+
\text{Distribution Matching}
}
$$

广义 OPD 则越来越接近：

$$
\boxed{
\text{Student On-policy Data}
+
\text{Teacher-derived Supervision}
}
$$

后一种定义不是错。

但一旦采用这个定义，OPD 就已经不是一个具体算法，而是一类训练范式。

此时真正应该回答的是五个问题：

1. **Rollout 从哪里来？**
2. **Teacher 提供什么监督信号？**
3. **这个信号被归因到 sequence、step 还是 token？**
4. **真正优化的 objective 是什么？**
5. **通过什么 gradient estimator / optimizer 更新参数？**

这五件事情讲清楚以后：

> “它到底是不是 OPD？”

反而变得不重要了。

真正重要的是：

$$
\boxed{
\text{Data Distribution}
\times
\text{Supervision Signal}
\times
\text{Credit Assignment}
\times
\text{Objective}
\times
\text{Gradient Estimator}
}
$$

这才是 OPD 背后真正值得研究的东西。

---

## 参考

- [Kim & Rush, 2016. Sequence-Level Knowledge Distillation](https://arxiv.org/abs/1606.07947)
- [Agarwal et al., 2023. On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://arxiv.org/abs/2306.13649)
- [Gu et al., 2024. MiniLLM: Knowledge Distillation of Large Language Models](https://openreview.net/forum?id=5h0qf7IBZZ)
- [Fang et al., 2026. Rubric-based On-policy Distillation](https://arxiv.org/abs/2605.07396)
- [Xie et al., 2026. Step-wise Rubric Rewards for LLM Reasoning](https://arxiv.org/abs/2605.17291)
- [Sun et al., 2026. Step-Level On-Policy Distillation](https://arxiv.org/abs/2608.16333)

相关阅读：

- [SFT → RL → OPD：从分布视角重新理解大模型后训练](/posts/sft-rl-opd/)
- [Forward KL vs Reverse KL：为什么一个覆盖模式，一个寻找模式？](/posts/forward-reverse-kl/)
