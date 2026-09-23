---
title: "OPD 乱象：从监督信号、数据分布与学习算法正本清源"
date: 2026-09-23T01:30:00+08:00
draft: false
tags: ["OPD", "蒸馏", "强化学习", "Rubric", "后训练"]
categories: ["AI 技术思考"]
math: true
summary: "越来越多方法都被称为 On-Policy Distillation，但它们可能分别使用 teacher logits、rubric、judge、reward 或 step-level supervision。本文不再按论文命名分类，而是从 rollout distribution、supervision signal、supervision construction、credit assignment、objective 和 gradient estimator 六个维度重新拆解 OPD。"
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
\text{Supervision Construction}
\times
\text{Credit Assignment}
\times
\text{Objective}
\times
\text{Gradient Estimator}
}
$$

一句话概括：

> **On-policy 回答样本从哪里来；Distillation 回答知识从哪里来；Supervision Construction 回答监督是否依赖当前 policy 动态生成；SFT / KL / RL 回答模型怎么学；Credit Assignment 回答这个信号应该作用到哪里。**

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

## 六、一个容易漏掉的新轴：监督信号本身是不是 On-policy 的？

前面的五个维度仍然漏掉了一个重要差异。

考虑两个方法：

### Static Rubric RL

先离线构造 rubric：

$
\mathcal R_x
=
G(x, y^+, y^-)
$

或者：

$
\mathcal R_x
=
G(x, y_{\mathrm{ref}})
$

训练时再让当前 Student rollout：

$
Y_S^t \sim \pi_{\theta_t}(\cdot \mid x)
$

并使用固定 rubric：

$
R_i^t
=
V(x,y_i^t,\mathcal R_x)
$

最后做 GRPO。

### Policy-conditioned Rubric RL

rubric 本身也依赖当前 policy 的 rollout：

$
Y_S^t
\sim
\pi_{\theta_t}(\cdot \mid x)
$

$
\mathcal R_x^t
=
G(x,Y_T,Y_S^t)
$

然后：

$
R_i^t
=
V(x,y_i^t,\mathcal R_x^t)
$

再做 GRPO。

这两种方法可能拥有完全相同的：

- rollout distribution；
- supervision type：都是 rubric；
- credit assignment：都是 sequence-level；
- objective：都是 expected reward；
- optimizer：都是 GRPO。

如果只使用原来的五维表示，它们甚至会被写成同一个算法。

但它们显然不是一回事。

真正的差异是：

$
\boxed{
\text{Supervision Construction}
=
\text{Static}
\quad\text{or}\quad
\text{Policy-conditioned}
}
$

也就是不仅要问：

> **监督信号是什么？**

还要问：

> **这个监督信号是在训练前固定好的，还是根据当前 Student 正在犯的错误重新构造的？**

这可以进一步写成：

$
\Phi_t
=
G(x,\pi_{\theta_t})
$

如果：

$
\frac{\partial \Phi_t}{\partial \pi_{\theta_t}}
\neq 0
$

这里不是说真的对 Rubricator 求梯度，而是表示在数据依赖关系上，监督规格会随着当前 policy 的行为发生变化。

因此，真正完整的拆解应该增加一条独立轴：

$
\boxed{
\text{Supervision Signal}
\neq
\text{Supervision Construction}
}
$

前者回答“Rubric / Reward / Logits / Critique 是什么”，后者回答“它是怎么来的、是否随 policy 改变”。

---

## 七、这样再看 ROPD：它更像 Teacher-guided On-policy RL

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

但继续往下拆，会发现 ROPD 真正有价值的地方甚至不是“Rubric + GRPO”。

因为类似的 RubricRL 范式已经可以写成：

$
(x,y^+,y^-)
\xrightarrow{\text{Rubricator}}
\mathcal R_x
\xrightarrow{\text{Verifier}}
R(x,y)
\xrightarrow{\text{GRPO}}
\pi_\theta
$

如果 rubric 在 RL 开始之前已经生成好，那么它属于：

$
\boxed{
\text{Static Rubric RL}
}
$

ROPD 的关键变化是把 contrast 的弱侧替换成当前 policy：

$
\boxed{
\mathcal R_x^t
=
G(
x,
\underbrace{Y_T}_{\text{fixed strong side}},
\underbrace{Y_S^t}_{\text{moving weak side}}
)
}
$

其中：

$
Y_S^t
\sim
\pi_{\theta_t}(\cdot\mid x)
$

因此，它真正应该强调的不是：

$
\text{Rubric}
+
\text{GRPO}
$

而是：

$
\boxed{
\text{Rubric conditioned on current policy failures}
}
$

这可以看成把固定的 chosen-rejected contrast：

$
R_x
=
G(x,y^+,y^-)
$

改成：

$
R_x^t
=
G(x,Y_T,Y_S^t)
$

也就是：

> **negative side is on-policy。**

Teacher responses 本身并不要求每一步重新生成。

完全可以先离线生产：

$
Y_T(x)
=
\{y_{T,1},\ldots,y_{T,m}\}
$

训练中不断复用。

真正必须在线更新的是：

$
\boxed{
\mathcal R_x^t
=
G(x,Y_T(x),Y_S^t(x))
}
$

因为只有：

$
Y_S^t
$

反映了当前 Student 此刻正在犯什么错误。

### 这是不是“Rubric 随训练持续演化”？

这里还需要再严格一点。

如果同一个 prompt 在训练过程中会被反复访问，那么确实可能出现：

$
\mathcal R_x^0
\neq
\mathcal R_x^1
\neq
\mathcal R_x^2
$

因为 Student 的 failure distribution 在变化。

但如果训练只有一个 epoch，大多数 prompt 只被访问一次，那么它更准确的名字是：

$
\boxed{
\text{Policy-conditioned Rubric Generation}
}
$

而不一定是强意义上的：

$
\boxed{
\text{Co-evolving Rubric}
}
$

也就是说，ROPD 保证的是：

> **当前这次 rubric 构造依赖当前 rollout。**

它并不天然保证：

> **同一个 prompt 的 rubric 会在整个训练生命周期中被持续 refresh。**

这两个概念最好分开。

### 代价：Rubricator 进入了 online reward loop

静态 RubricRL 可以把 rubric generation 的成本全部摊到训练前：

$
\text{Offline Rubric}
\rightarrow
\text{Online Verifier}
\rightarrow
\text{RL}
$

而 ROPD 是：

$
\text{Student Rollout}
\rightarrow
\text{Rubricator}
\rightarrow
\text{Verifier}
\rightarrow
\text{RL}
$

Teacher response 可以 cache，但 Rubricator 无法完全 cache，因为它依赖当前 Student rollout。

于是每个 rollout group 的在线成本至少包含：

$
C_{\text{online}}
\approx
C_{\text{rollout}}
+
C_{\text{rubricator}}
+
C_{\text{verifier}}
+
C_{\text{update}}
$

如果 Rubricator / Verifier 都调用强黑盒 API，那么 wall-clock、吞吐和 API dollar cost 都可能成为主要问题。

因此，一个更实用的版本可能不是“每次从零生成完整 rubric”，而是：

$
\boxed{
\mathcal R_x^t
=
\mathcal R_x^{\text{base}}
+
\Delta\mathcal R_x^t
}
$

其中 base rubric 离线生成：

$
\mathcal R_x^{\text{base}}
=
G(x,Y_T)
$

描述长期稳定的任务约束、关键知识点和正确性要求；

动态部分只针对当前 rollout 暴露出的 failure：

$
\Delta\mathcal R_x^t
=
G(
\mathcal R_x^{\text{base}},
Y_T,
Y_S^t
)
$

这样 supervision engineering 就从：

> 每一步重新写一套评分标准

变成：

> **固定任务 rubric + 少量 on-policy failure delta。**

这也给出了一个很自然的新研究问题：

$
\boxed{
\text{Static task specification}
+
\text{Dynamic failure specification}
}
$

到底应该各占多少？

---



## 八、如果什么都能叫 OPD，OPD 就不再是一种算法

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

## 九、Credit Assignment：监督到底应该归因给谁？

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

## 十、Step-level 也没有想象中简单：跨 rollout 怎么对齐？

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

## 十一、MiniLLM 说明：Objective 和 Optimizer 也不能混为一谈

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

## 十二、我更喜欢的统一表示

以后看到任何一个后训练方法，我更愿意先把论文名拿掉，然后写成：

$$
\boxed{
\mathcal A
=
(
q_{\text{rollout}},
\Phi,
\Psi,
C,
J,
G
)
}
$$

其中：

- $q_{\text{rollout}}$：谁产生训练 trajectory；
- $\Phi$：监督信号的类型是什么；
- $\Psi$：监督信号如何构造，是否依赖当前 policy；
- $C$：credit assignment 粒度与方式；
- $J$：最终优化目标；
- $G$：gradient estimator / policy update。

这里新加入的 $\Psi$ 非常重要。

例如两个方法都可能是：

$$
\Phi=\text{Rubric}
$$

但一个是：

$$
\Psi
=
G(x,y^+,y^-)
$$

训练前固定；

另一个是：

$$
\Psi_t
=
G(x,Y_T,Y_S^t)
$$

随当前 Student rollout 改变。

如果不把这条轴单独拿出来，它们在方法分类中会被错误地合并。

例如：

### Sequence Distillation

$$
(
\pi_T,
\text{Teacher Response},
\text{Offline Teacher Generation},
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
\text{On-policy Prefix-conditioned},
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
\text{On-policy Trajectory-conditioned},
\text{Reward-to-go},
\text{Sequence Reverse KL},
\text{Policy Gradient}
)
$$

### Static RubricRL / RaR-style RL

$$
(
\pi_S,
\text{Rubric},
\text{Offline / Fixed Rubric},
\text{Sequence},
\text{Expected Reward},
\text{GRPO}
)
$$

### ROPD

$$
(
\pi_S,
\text{Teacher-derived Rubric},
\text{Policy-conditioned Rubric},
\text{Sequence},
\text{Expected Reward},
\text{GRPO}
)
$$

### SRaR

$$
(
\pi_S,
\text{Rubric Judge},
\text{Rubric-conditioned Step Attribution},
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
\text{On-policy State-conditioned},
\text{Step},
\text{MLE},
\text{Teacher Forcing}
)
$$

这张表比继续发明一个新的 “xxx-OPD” 名字更有解释力。

尤其是加入 $\Psi$ 之后，终于可以把：

$$
\boxed{
\text{on-policy data}
}
$$

和：

$$
\boxed{
\text{on-policy supervision construction}
}
$$

分开讨论。

---

## 十三、Black-box Teacher 的真正演进：从答案生成器到训练信号生成器

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

## 十四、真正值得研究的可能不是 OPD，而是 Supervision Engineering

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

## 十五、On-policy 的真正价值：把监督预算花在 Student 真正访问的地方

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

## 十六、结语：不要再问“这是不是 OPD”

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

此时真正应该回答的是六个问题：

1. **Rollout 从哪里来？**
2. **Teacher 提供什么监督信号？**
3. **监督信号如何构造：训练前固定，还是依赖当前 policy 动态生成？**
4. **这个信号被归因到 sequence、step 还是 token？**
5. **真正优化的 objective 是什么？**
6. **通过什么 gradient estimator / optimizer 更新参数？**

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
\text{Supervision Construction}
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
