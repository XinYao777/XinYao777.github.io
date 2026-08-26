---
title: "以传统搜索排序视角，理解 LLM 时代的 Learning-to-Rank"
date: 2026-08-26T20:00:00+08:00
draft: false
tags: ["Learning to Rank", "Reranker", "Reward Model", "搜索排序", "笔记"]
categories: ["AI 技术思考"]
math: true
summary: "从传统 BERT 排序到 LLM Reranker、Reward Model，排序问题的数学结构并未根本改变。拆开原始标签、raw score、probability、ranking loss 与业务 score，再用两条正交轴——训练目标（Point/Pair/List）× 打分结构（Independent/Joint）——统一理解 Reward Model、GTE/BGE、Qwen-Reranker。"
---

如果过去做过搜索排序，再来看今天的大模型 Reranker、Reward Model、Qwen-Reranker，很容易产生一种错觉：

> 排序方法是不是已经发生了根本变化？Pairwise 为什么好像突然很少被提到？Listwise 是不是成了新的主流？

实际上并没有。

从传统 BERT 排序模型到今天的 LLM Reranker，排序问题最核心的数学结构并没有发生根本变化。真正发生变化的主要有两件事：

1. **排序函数 \(f\) 的实现从 Encoder/Cross-Encoder 扩展到了 Decoder-only LLM 和生成式模型；**
2. **LLM 使"多个候选同时进入模型、候选之间直接交互"的 Joint Candidate Modeling 变得更加现实。**

Pointwise、Pairwise、Listwise 则主要描述的是：

> **监督信号如何组织，以及 loss 在什么范围内定义。**

因此，要理解今天的大模型排序，最重要的是把几个经常混在一起的概念拆开：

* 原始标签；
* 模型 raw score；
* probability；
* ranking loss；
* 最终业务 score。

## 一、先统一符号：排序模型到底在学习什么？

对于一个 Query：

$$
q
$$

以及候选文档：

$$
D=\{d_1,d_2,\dots,d_n\}
$$

最传统的排序系统都可以抽象成：

$$
h_i=f_\theta(q,d_i)
$$

其中 \(h_i\) 是模型产生的语义表示。

然后通过 scoring head：

$$
z_i=g_\phi(h_i)
$$

得到一个标量：

$$
z_i\in\mathbb R
$$

本文统一把 \(z_i\) 称为：

> **raw ranking score / latent utility**

而不是直接叫 probability。这一点非常重要。

模型真正学习的通常首先是 \(z_i\)。至于后续是否计算 \(\sigma(z_i)\) 或者 \(softmax(z_1,\dots,z_n)\)，是由训练目标决定的。

为了避免后文概念混乱，我们统一使用以下符号：

| 符号      | 含义                                 |
| ------- | ---------------------------------- |
| \(y_i\) | 原始人工标签 / supervision               |
| \(h_i\) | 模型 hidden representation           |
| \(z_i\) | raw ranking score / latent utility |
| \(p_i\) | 经过 sigmoid / softmax 后得到的概率        |
| \(S_i\) | 最终业务使用的 score                      |

其中 \(S_i\) 并不一定属于 \([0,1]\)。它可以直接等于 \(S_i=z_i\)，也可以 \(S_i=\sigma(z_i)\)，或者在 0～4 生成式满意度模型中：

$$
S_i=\sum_{k=0}^{4}kP(y=k|q,d_i)
$$

因此：

> **Score 是业务或排序接口概念，不等于 probability。**

## 二、Pointwise：监督单个 Query-Document

Pointwise 最本质的定义不是"二分类"，而是：

$$
L=\sum_i\ell(z_i,y_i)
$$

即每一个 \((q,d_i)\) 都可以独立产生一个 loss。因此 Pointwise 的原始标签 \(y_i\) 可以有很多形式。

### 二分类

$$
y_i\in\{0,1\}
$$

例如：相关 / 不相关；满意 / 不满意；点击 / 未点击。

### 多等级相关性

$$
y_i\in\{0,1,2,3,4\}
$$

例如：4 高度满足；3 满足；2 临界；1 较差；0 完全不满足。

### 连续标签

$$
y_i\in\mathbb R
$$

例如人工给出的连续质量分。所以：

$$
\boxed{\text{Pointwise}\neq\text{Binary Classification}}
$$

Binary relevance 只是 Pointwise 的一个具体实现。

### Pointwise Binary Classification

对于 \(y_i\in\{0,1\}\)，可以定义 \(p_i=\sigma(z_i)\)，使用 BCE：

$$
L_i=-y_i\log p_i-(1-y_i)\log(1-p_i)
$$

这里 \(\sigma(z_i)\) 被训练成了 \(P(y_i=1|q,d_i)\)，因此它具有明确的 absolute prediction 语义。这也是 Pointwise 与 Pairwise/Listwise 一个很重要的区别。

## 三、Pairwise：不学习"几分"，而学习"谁更好"

搜索排序很多时候真正关心的并不是 \(z_i=4\)，而是 \(z_i>z_j\)。因此 Pairwise 直接学习偏序关系。例如原始数据：

```text
qid = Q1

Doc A   label=4
Doc B   label=3
Doc C   label=2
Doc D   label=0
```

可以动态构造 \(A>B\)、\(A>C\)、\(A>D\)、\(B>C\)、\(B>D\)、\(C>D\)。

值得注意的是：

> 原始数据仍然可以保持 0/1/2/3/4 graded label，Pairwise 只是在训练阶段把 graded label 转换成偏序监督。

定义 \(t_{ij}=\mathbf 1(y_i>y_j)\)，还可以进一步构造 pair weight \(w_{ij}=|y_i-y_j|\)。这样 \(4>0\) 可以比 \(4>3\) 具有更大的训练权重。

典型实现是：原始数据按照 qid 组织，同一 Query 下保存多个不同 label 的 document，然后在 batch 内根据 qid 和 label 自动构造 pair；label 差值同时被用作 pair 权重。

## 四、RankNet：Sigmoid 作用的不是单独 score，而是 score difference

RankNet 是理解 Reward Model 最重要的一座桥。首先模型输出 \(z_i=f(q,d_i)\) 和 \(z_j=f(q,d_j)\)，其中 \(z_i,z_j\) 是 raw ranking score。然后定义：

$$
P(i\succ j)=\sigma(z_i-z_j)
$$

因此：

$$
L_{ij}=-\log\sigma(z_i-z_j)=\log(1+\exp(-(z_i-z_j)))
$$

这里 sigmoid 的真正意义是：

$$
\boxed{\sigma(z_i-z_j)=P(i\succ j)}
$$

而不是 \(\sigma(z_i)\) 代表某个绝对满意概率。这是一个非常容易被混淆的地方。

## 五、统一理解 Pairwise 的 Weight、Margin 与 Temperature

Pairwise 排序可以统一表示成：

$$
L_{ij}=w_{ij}\phi\left(\frac{z_i-z_j-m_{ij}}{\tau}\right)
$$

其中三个变量承担完全不同的作用。

### 5.1 Pair Weight：这一对样本有多重要

例如 \(w_{ij}=|y_i-y_j|\)，则 \(4>0\) 可以比 \(4>3\) 获得更大的 loss 权重。这是 **importance weighting**。

### 5.2 Margin：两者应该拉开多远

例如 Margin RankNet：

$$
L_{ij}=-\log\sigma(z_i-z_j-m_{ij})
$$

要求 \(z_i-z_j\) 不仅大于 0，还应该具有一定 separation。Hinge Ranking Loss 则更加直接：

$$
L_{ij}=\max(0,m_{ij}-(z_i-z_j))
$$

它要求 \(z_i-z_j\ge m_{ij}\) 之后 loss 直接变为 0。

### 5.3 Temperature：整个 raw score space 的尺度

可以进一步写成：

$$
P(i\succ j)=\sigma\left(\frac{z_i-z_j-m_{ij}}{\tau}\right)
$$

其中 \(\tau\) 决定 score difference 多大才算"明显"。因此：

* \(w_{ij}\)：这对数据多重要；
* \(m_{ij}\)：这对数据应该拉开多少；
* \(\tau\)：整个 score space 的尺度。

三者是不同概念。

## 六、Raw-score Margin 与 Bounded-score Margin

这里还有一个搜索排序中非常实用的区别。一种做法直接在 \(z_i\in(-\infty,+\infty)\) 上定义 margin：\(z_i-z_j\ge m\)，可以称为 **Raw-score Margin**。Bradley-Terry 模型下 \(P(i\succ j)=\sigma(z_i-z_j)\)，因此 \(z_i-z_j=m\) 可以解释成一个 preference probability。例如：

$$
m=1\Rightarrow P(i\succ j)\approx0.731
$$

$$
m=2\Rightarrow P(i\succ j)\approx0.881
$$

所以 raw-score margin 可以理解为 preference log-odds separation。

另一种做法先定义 \(S_i=\sigma(z_i)\) 使 \(S_i\in(0,1)\)，然后要求 \(S_i-S_j\ge m\)，可以称为 **Bounded-score Margin**。例如 \(m=0.3\) 就具有非常直观的固定尺度。

但需要注意：\(\sigma(z_i)\) 仅仅因为属于 \([0,1]\)，并不意味着它天然等于"80% 满意概率"。如果训练只使用 Pairwise RankNet，那么真正具有概率意义的是 \(\sigma(z_i-z_j)\)，而不是 \(\sigma(z_i)\)。

## 七、一个非常重要的结论：Pairwise 不保证 absolute calibration

假设两组模型输出 \(z_A=10,\ z_B=8\) 和 \(z_A=2,\ z_B=0\)，两者 \(z_A-z_B=2\)，因此 Pairwise RankNet 完全等价。但：

$$
\sigma(10)\approx1,\quad \sigma(8)\approx1
$$

$$
\sigma(2)\approx0.88,\quad \sigma(0)=0.5
$$

完全不同。所以：

$$
\boxed{Pairwise\ Ranking\ 约束的是相对差值，而不是绝对零点}
$$

因此纯 Pairwise 模型最终虽然可以输出 scalar，但 \(\sigma(z_i)\) 并不天然是 calibrated satisfaction probability。这也是"排序"和"是否出卡"虽然相关，却不能完全视为一个问题的原因。

## 八、Listwise：监督整个 Query Group

Listwise 的关键不是"模型有没有 scalar score"。实际上很多传统 Listwise 模型仍然首先计算 \(z_i=f(q,d_i)\)。区别在于：

> Loss 不再独立作用于一个 doc 或一个 pair，而是作用于整个 Query 下的 candidate group。

## 九、Listwise 中 \(y_i\) 到底是什么？

这是很多介绍容易写模糊的地方。首先仍然有原始标签 \(y=[y_1,y_2,\dots,y_n]\)，例如 \(y=[4,3,2,0]\)。但 Listwise Loss 通常不会直接把 4,3,2,0 塞进 cross entropy，它需要将原始 label 转换成一个 **list-level target distribution** \(q=[q_1,q_2,\dots,q_n]\)。

例如可以定义：

$$
q_i=\frac{\exp(g(y_i)/\tau_y)}{\sum_j\exp(g(y_j)/\tau_y)}
$$

最简单 \(g(y)=y\)，于是 \([4,3,2,0]\) 会被映射成一个类似 \([0.66,0.24,0.09,0.01]\) 的目标分布。模型产生 raw score \(z=[z_1,z_2,z_3,z_4]\)，然后：

$$
p_i=\frac{\exp(z_i/\tau)}{\sum_j\exp(z_j/\tau)}
$$

最终优化：

$$
L_{\text{list}}=-\sum_iq_i\log p_i
$$

因此需要明确区分：\(y_i\) 是原始 graded relevance label；\(q_i\) 是从 graded label 转换得到的 listwise target；\(p_i\) 则是模型产生的 listwise probability。

## 十、Swift 中所谓的 Listwise 是一个更简单的特殊情况

ms-swift 当前 Reranker 文档中的 Listwise 主要采用 1 positive + N negatives，例如：

```text
A positive
B negative
C negative
D negative
```

此时原始标签 \(y=[1,0,0,0]\)，目标 distribution 实际上就是 one-hot \(q=[1,0,0,0]\)，模型 \(p=softmax(z)\)，然后 \(L=-\log p_A\)。

因此 Swift 当前的 Listwise 从数学形式上非常接近召回阶段常见的 **InfoNCE / Group Softmax**。它并不代表今天所有 Listwise Ranking 的完整定义，更不意味着 Pairwise 已经过时。

## 十一、Listwise 训练之后当然仍然可以得到 Scalar Score

这是另一个非常容易被误解的问题。Listwise 训练 \(p_i=softmax(z_1,\dots,z_n)_i\)，但是推理时完全可以直接用 \(z_i\) 作为 ranking score，因为：

$$
z_i>z_j\iff softmax(z)_i>softmax(z)_j
$$

甚至还可以定义 \(S_i=\sigma(z_i)\) 作为一个 bounded score。由于 sigmoid 单调，\(z_i>z_j\iff\sigma(z_i)>\sigma(z_j)\)，排序结果仍然不会变化。因此：

$$
\boxed{Pointwise、Pairwise、Listwise 都可以最终输出 scalar}
$$

真正不同的是训练时 probability normalization 发生在哪里。

## 十二、三种训练范式可以统一成一张表

| 方法        | 原始监督                               | 概率空间                        | Loss 作用范围      |
| --------- | ---------------------------------- | --------------------------- | -------------- |
| Pointwise | \(y_i\)                            | \(\sigma(z_i)\) 或分类 softmax | 单个 doc         |
| Pairwise  | \(y_i,y_j\rightarrow t_{ij}\)      | \(\sigma(z_i-z_j)\)         | 一对 doc         |
| Listwise  | \(y_1...y_n\rightarrow q_1...q_n\) | \(softmax(z_1...z_n)\)      | 整个 Query Group |

因此 Pointwise / Pairwise / Listwise 本质描述的是 **监督结构与 loss 定义域**，而不是模型输出的是不是 scalar。

## 十三、从 BERT 到 GPT，真正改变的是 \(f(q,d)\)

传统 BERT Cross-Encoder：

```text
[CLS] Query [SEP] Document [SEP]
                ↓
              BERT
                ↓
             h_[CLS]
                ↓
             Linear
                ↓
               z
```

即 \(z=w^\top h_{CLS}+b\)。

Decoder-only LLM / Reward Model：

```text
Query + Response / Document
              ↓
             LLM
              ↓
      last / EOS hidden state
              ↓
          Scalar Head
              ↓
              z
```

即 \(z=w^\top h_{last}+b\)。从排序模型角度：

$$
\boxed{BERT\ CrossEncoder\rightarrow LLM\ Scalar\ Ranker}
$$

并不存在本质断裂。核心仍然是 \(f(q,d)\rightarrow z\)，变化主要是 backbone 的能力发生了巨大提升。

## 十四、Reward Model 本质也是 Pairwise Ranker

RLHF 中 \(x=\text{prompt}\)，\(d=\text{response}\)，Reward Model 学习 \(z=f(x,d)\)。然后对于 chosen \(d_w\) 和 rejected \(d_l\)，优化：

$$
-\log\sigma(z_w-z_l)
$$

这和 Search RankNet 的数学结构完全一样。所以：

$$
\boxed{Reward\ Model=Preference\ Ranker}
$$

区别不是排序理论，而是"什么东西被排序"。搜索排序是 \(query\leftrightarrow document\)，Reward Model 是 \(prompt\leftrightarrow response\)：一个主要学习 relevance / satisfaction，一个主要学习 generation preference / quality。但从 Pairwise Learning-to-Rank 的角度，它们高度同构。

## 十五、生成式模型带来了一种新的 Scoring Function

LLM 不一定需要增加 scalar head，它本身就有 LM Head。因此可以直接把排序问题转化成"生成一个类别 token"。例如满意度定义 0/1/2/3/4，输入 \(x=(q,d)\)，Decoder 得到 \(h=f_\theta(x)\)，LM Head：

$$
\mathbf z=W_{vocab}h
$$

这里 \(\mathbf z\) 是整个 vocabulary 上的 logits。例如词表大小 \(|V|=150000\)，那么 \(\mathbf z\in\mathbb R^{150000}\)。

### 15.1 从整个 Vocabulary 中截取 Label Token Logits

只取 token "0"、"1"、"2"、"3"、"4" 对应的 vocabulary logits \(z_0,z_1,z_2,z_3,z_4\)。然后不是直接使用整个词表 softmax 的结果，而是在 label token 集合内重新归一化：

$$
p_k=\frac{\exp(z_k)}{\sum_{j=0}^{4}\exp(z_j)}
$$

得到 \(P(y=0),\dots,P(y=4)\)。

## 十六、从 Generative Classification 得到 Ranking Score

有了 \(p_0,\dots,p_4\)，可以定义期望满意度：

$$
\boxed{S_{\text{rank}}=\mathbb E[y|q,d]=\sum_{k=0}^{4}kp_k}
$$

例如 \(P(0)=0.01,\ P(1)=0.02,\ P(2)=0.07,\ P(3)=0.30,\ P(4)=0.60\)，则 \(S_{\text{rank}}=3.46\)。这个连续 score 可以直接进行排序。

如果 3、4 定义为满意，还可以计算：

$$
\boxed{S_{\text{card}}=P(y\ge3)=p_3+p_4}
$$

于是同一个模型可以同时提供：排序 score、满意概率、出卡 threshold。

## 十七、Generative Ranker 与 Scalar Ranker 的区别

因此今天的 LLM Reranker 可以有两类非常不同的 scoring head。

**Scalar Head：** \(h\rightarrow w^\top h\rightarrow z\)，典型如 Reward Model。

**Generative Head：** \(h\rightarrow W_{vocab}h\rightarrow\) vocabulary logits，然后 label logits \(\rightarrow softmax\rightarrow P(y)\rightarrow S\)。

两者最终都可以实现 \(f(q,d)\rightarrow score\)，区别只是 score 的产生方式不同。

## 十八、GTE、Qwen-Reranker、Reward Model 应该如何放在同一张图里？

不要简单把它们看成三套毫无关系的方法。更好的抽象是：

```text
                    Ranking Model
                         │
                  Query + Candidate
                         │
                         ▼
                    Backbone
                         │
           ┌─────────────┴─────────────┐
           │                           │
      Scalar Head                 LM Head
           │                           │
           z                    Vocabulary logits
           │                           │
           │                     Label logits
           │                           │
           │                     Probability
           │                           │
           └─────────────┬─────────────┘
                         ▼
                   Ranking Score
```

* Reward Model 更典型地落在：LLM + Scalar Head + Pairwise Preference Learning。
* 传统 GTE/BGE Cross-Encoder 更接近：Encoder/Cross-Encoder + Relevance Score。
* Qwen-Reranker 则体现了 Decoder-only LLM 下：利用 LM Head，通过 yes/no 等 label token probability 产生 relevance score。

因此它们最本质的区别不在于"是不是排序模型"——它们都是排序模型。区别更多来自：Backbone、Scoring Head、Training Objective、Label / Preference Data。

## 十九、还有一个更容易被忽视的维度：Scorer 是否独立

Pointwise / Pairwise / Listwise 是一条轴，模型 architecture 还有另一条完全独立的轴。传统绝大多数 Reranker 都是：

$$
\boxed{z_i=f(q,d_i)}
$$

也就是说，Doc A 的 raw score 只由 \(q,d_A\) 决定。无论候选集合里还有 B、C、D，\(z_A\) 本身不发生变化。这种模型可以称为 **Independent / Separable Scorer**。

## 二十、传统 Listwise Loss 并不意味着候选之间真的发生了交互

例如 \(z_A=f(q,A)\)、\(z_B=f(q,B)\)、\(z_C=f(q,C)\)，然后 \(softmax(z_A,z_B,z_C)\) 做 Listwise Loss。虽然 loss 是 Listwise，但是 \(z_A\) 计算的时候根本没有看到 \(B,C\)。因此：

$$
\boxed{Listwise\ Loss\neq Joint\ Candidate\ Modeling}
$$

这是理解现代 LLM Ranking 时很重要的一点。

## 二十一、LLM 真正带来的一个新机会：Joint Candidate Modeling

LLM 可以直接输入：

```text
Query

Candidate A
Candidate B
Candidate C
Candidate D
```

于是模型内部通过 Self-Attention，使 \(A\leftrightarrow B\)、\(A\leftrightarrow C\)、\(B\leftrightarrow D\) 发生直接交互。模型变成：

$$
(z_1,\dots,z_n)=F(q,d_1,\dots,d_n)
$$

而不再是 \(z_i=f(q,d_i)\)。这可以称为 **Joint Candidate Modeling / Context-dependent Ranking**。这里真正的新变化不是从 Pairwise Loss 升级到了 Listwise Loss，而是：

$$
\boxed{\text{Independent Scoring}\rightarrow\text{Cross-Candidate Interaction}}
$$

## 二十二、为什么这件事和传统 Listwise 不完全一样？

传统 Listwise \(z_i=f(q,d_i)\)，candidate set 主要影响 loss。现代 Joint Candidate Model \(z_i=F_i(q,D)\)，candidate set 直接进入 scoring function。因此同一个 document \(d_A\) 在不同候选集合 \(D_1\) 和 \(D_2\) 中甚至可以产生不同的 contextual score：

$$
F_A(q,D_1)\neq F_A(q,D_2)
$$

这才是模型 architecture 层面的变化。

当然，推荐系统过去已经存在 slate optimization、diversity ranking 等集合级建模方法，因此这并不是一个完全由 LLM 首创的问题。LLM 真正带来的变化在于：

> 一个通用 Transformer 可以直接通过自然语言上下文和 Attention 完成复杂候选间比较，而不必预先显式设计大量候选间交互特征。

## 二十三、因此排序问题应该用"两条正交轴"理解

**第一条轴：Training Objective。** Pointwise / Pairwise / Listwise，描述监督关系如何组织。

**第二条轴：Scoring Architecture。** Independent Scorer \(z_i=f(q,d_i)\) 与 Joint Candidate Model \((z_1,\dots,z_n)=F(q,D)\)，描述候选是否在模型内部发生直接交互。

因此完全可以出现：Independent + Pointwise、Independent + Pairwise、Independent + Listwise、Joint + Pairwise、Joint + Listwise。不能再简单理解成：

```text
Pointwise → Pairwise → Listwise → LLM
```

这不是一条技术演进链。

## 二十四、API 式排序不是新的排序范式

一个排序模型无论内部是 BERT、RM 还是 Qwen，都可以封装成 `rank(query, candidates)`，返回 `candidate_id / score / rank`。这只是 **系统接口抽象**，并不意味着排序理论发生了变化。真正值得研究的是 API 背后究竟是 \(z_i=f(q,d_i)\) 逐候选独立打分，还是 \((z_1,\dots,z_n)=F(q,D)\) 联合候选建模。

因此：API 是工程封装层；Point/Pair/List 是训练目标层；Scalar/Generative 是 scoring head 层；Independent/Joint 是模型结构层。这几个维度应该彻底拆开。

## 二十五、回到搜题满意度：一个自然的实验体系

对于搜题场景，可以继续保留传统搜索排序非常成熟的数据形式：

```text
qid
query
[
  (doc1, 4),
  (doc2, 3),
  (doc3, 3),
  (doc4, 2),
  (doc5, 1),
  (doc6, 0)
]
```

不要在数据生产阶段提前退化成 pair 或 positive/negative。这份数据可以生成不同 training view。

### Baseline 1：Scalar + Pairwise

模型 \(z_i=f(q,d_i)\)，训练：

$$
L_{pair}=w_{ij}\log\left(1+\exp\left(-\frac{z_i-z_j-m_{ij}}{\tau}\right)\right)
$$

这是传统搜索排序经验最自然的延续。

### Baseline 2：Generative Pointwise

模型输出 vocabulary logits \(\mathbf z\)，截取 \(z_0,\dots,z_4\)，重新归一化 \(p_k=softmax(z_0,\dots,z_4)_k\)，训练 \(L_{point}=-\log p_y\)。排序 \(S_{\text{rank}}=\sum_k kp_k\)，出卡 \(S_{\text{card}}=p_3+p_4\)。

### Experiment 3：Pointwise + Pairwise

在生成式 0～4 模型之上 \(S_i=\sum_kkp_{ik}\)，再增加 \(L_{pair}\)，最终 \(L=L_{point}+\lambda L_{pair}\)。这样 Pointwise 负责学习绝对满意度，Pairwise 负责同一个 Query 下候选排序，两个目标的关系就非常清楚。

### Experiment 4：Graded Listwise

如果后续尝试 Listwise，不应该简单把 0/1/2/3/4 压成 binary positive/negative，而应该保留 graded supervision \([y_1,\dots,y_n]\rightarrow[q_1,\dots,q_n]\)，再和 \(softmax(z_1,\dots,z_n)\) 进行匹配。这样才能真正利用已有满意度等级数据。

## 二十六、大模型时代并没有推翻搜索排序

如果从 2022 年传统搜索排序一路看到今天，其实技术脉络是非常连续的。

过去：

$$
Query+Doc\rightarrow BERT\rightarrow Scalar\rightarrow Point/Pair/List
$$

今天既可以：

$$
Query+Doc\rightarrow LLM\rightarrow Scalar\ Head\rightarrow Point/Pair/List
$$

也可以：

$$
Query+Doc\rightarrow LLM\rightarrow Vocabulary\ Logits\rightarrow Label\ Probability\rightarrow Score
$$

而 LLM 真正额外打开的一条路线是：

$$
Query+[D_1,\dots,D_n]\rightarrow LLM\rightarrow Joint\ Candidate\ Modeling
$$

所以真正值得记住的不是"Pairwise 过时了，Listwise 成为了主流"，而是：

> **排序问题的监督范式基本稳定，真正发生变化的是 Scoring Function 的表达能力，以及候选之间是否能够进入模型内部发生联合建模。**

从这个角度看，Reward Model、GTE/BGE Reranker、Qwen-Reranker，以及未来面向 Agent、RAG、搜索满意度的 Ranking Model，其实都可以被统一放回同一个 Learning-to-Rank 框架中理解。
