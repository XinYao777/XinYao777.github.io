---
title: "把「判断」从生成里拆出来：Jev 非生成式决策模型的数据流全解"
date: 2026-09-24T20:00:00+08:00
draft: false
tags: ["Jev", "决策模型", "非生成式", "分类", "Agent", "蒸馏", "笔记"]
categories: ["AI 技术思考"]
math: true
summary: "Jev 最近被吹成「比 LLM 快 200 倍、便宜 400 倍」，但营销叙事之外，它到底怎么跑起来？我顺着开源复现 NanoJev 的 DATAFLOW 把数据流走了一遍：真正的建模决定只有两条——把标签空间从输出层搬进输入（一个共享 [hdd,1] 给任意候选打分），以及在 choice 上加一层置换等变的 set-attention。其余全是成熟组件的直接组合。本文以数据流实现为主线，背景与冷思考为两端。"
---

> 技术主线（数据流实现）来自开源复现 [TianyuCodings/NanoJev](https://github.com/TianyuCodings/NanoJev)（`docs/DATAFLOW.md`，作者本人阅读、走查后的实践理解）。
> 背景与定位来自 [《Jev 被全网吹爆：凭什么比 LLM 快 200 倍？》（丁师兄）](https://mp.weixin.qq.com/s/vJmGUH2361kLDfFPCHMRdg)，以及其援引的一手资料：TypeSafe AI《Introducing System One Models and Jev》、LangChain《Building a Harness with Jev》、Flavio Copes《A deep dive into Jev》。
> 概率、示意数字均为讲清结构用的占位值，真实数字需下载 checkpoint 实跑。

## 0. 先说清楚：Jev 想解决什么

这几年我们有个习惯：什么问题都先丢给 LLM。但冷静想想，大多数软件真正需要的，其实不是又一个聊天机器人，而是**数不清的小判断**——用户问「这单能退吗」，是退款还是咨询？发来「怎么还没发货」，是催单还是投诉？这些判断很小、很碎，但每天要发生成千上万次。

`tool calling` 和 `structured outputs` 让 LLM 接进软件变简单了，但底层模型仍是生成式的：哪怕答案只有一个词 `billing`，它也得一个 token、一个 token 地吐。放进 agent loop 里问题更被放大——选工具、判断结果、识别风险、看任务完没完……全是「只需判断、根本不用写一段话」的调用。

Jev 的定位就是接管这些调用。用丁师兄文章里的一句话概括：**既然代码本来就知道所有可能的答案，那再让模型去生成一段话，就是用错了工具。** 官方（TypeSafe）把它叫 semantic decision engine，宣传「比同类 LLM workflow 快约 200 倍、便宜约 400 倍」。

但营销归营销。我更想搞清楚的是：**去掉「生成」这一步之后，这东西到底怎么跑起来？** 于是我顺着开源复现 NanoJev 的数据流走了一遍。下面是主线。

## 1. 核心一句话

传统分类的输出层是 `Linear(H, num_classes)`，类别固化在权重里。这个模型只有一个 `Linear(H, 1)`——**把标签空间从输出层搬进了输入**。问题文本、候选文本都作为 token 喂进去，模型只干一件事：读一段「状态 + 问题 + 某个候选」，吐一个标量契合分。于是分类从「固定 N 类」变成「任意问题、任意候选」，而且全程不生成任何 token（`autoregressive_decode_steps = 0`）。

## 2. 一个 JSON 进，一个 JSON 出

一个 `state`（共享上下文）+ 三个 question，分别是 choice / boolean / score：

```json
{
  "states": [{
    "id": "s1",
    "state": "客户同一订单被扣款两次；退款已批准但未到账；服务可正常使用",
    "questions": {
      "team":   { "type": "choice",
                  "instructions": "选择负责的支持团队",
                  "criteria": { "account": "账户与身份验证",
                                "billing": "扣费与退款",
                                "technical": "功能错误" } },
      "paid":   { "type": "boolean",
                  "instructions": "退款是否已实际到账？审批通过不算" },
      "impact": { "type": "score",
                  "instructions": "评估使用影响的严重程度",
                  "criteria": ["功能正常","次要受阻有替代","核心受阻有替代","完全中断无替代"] }
    }
  }]
}
```

程序把 `state × question × 每个候选` 展开成一堆 **leaf**，每个 leaf 是一段独立明文。**同一题的几个 leaf 只有最后 `Candidate:` 那一行不同，前缀完全一样。** 比如 team（choice）展开成 3 个 leaf：

```
State:
客户同一订单被扣款两次；退款已批准但未到账；服务可正常使用
Question type: choice
Question:
选择负责的支持团队
Candidate:
account: 账户与身份验证          ← 只有这一行在变
Decision:                        (末尾还有一个 <eos>)
```

boolean 比较特别，**只展开 1 个 leaf**，候选固定就是「命题为真」这一句：

```
…前缀（Question type: boolean）… Candidate:\nThe proposition is true.\nDecision:
```

score 展开成 4 个 leaf，每个 Candidate 只放该级描述（**不写序号、不看相邻级**）。这个 state 一共 `3 + 1 + 4 = 8` 个 leaf。

输出同样结构，每个 question 换成一个概率分布 + 直接可用的结论：

```json
{ "states": [{ "id": "s1", "answers": {
  "team":   { "type":"choice",  "probabilities":{"account":0.05,"billing":0.88,"technical":0.07}, "choice":"billing" },
  "paid":   { "type":"boolean", "probabilities":{"false":0.93,"true":0.07}, "p_true":0.07, "value":false },
  "impact": { "type":"score",   "probabilities":{"0":0.70,"1":0.20,"2":0.08,"3":0.02}, "score":0.42, "level":0 }
}}]}
```

| 题型 | 进 | 出 |
|---|---|---|
| choice | 候选 `{名:描述}` | 每候选概率 + `choice`（最高者） |
| boolean | 一句判断 | `{false,true}` 概率 + `p_true` + `value` |
| score | 有序等级描述 | 每级概率 + `score`（期望 `Σ i·p_i`）+ `level` |

> 术语对齐：TypeSafe 官方把三原语叫 **Choice / Score / Noul**（Noul 即这里的 boolean 是非题）；NanoJev 复现里统一叫 boolean。下文沿用 NanoJev 的叫法。

## 3. 样本：从 M 行 jsonl 到 P 个 leaf

输入是 **M 行 jsonl**，每行一个 state：`state_i → n_i 个 question → 每个 question 有 k_j 个 term`。`k_j` 的含义随题型变：choice = k 个候选（2–255）；score = k 个有序等级（2–10）；**bool 语义上是 2 个（false/true），但只展开成 1 个 term**。

**定义 leaf：** 一个 leaf = 一个 `(state, question, term)` 三元组拼成的**完整输入序列**：`State + Question` 前缀 + 该 term 的 `Candidate:` 行 + `Decision:` + `<eos>`。叫「叶子」是因为同题所有 term 共享前缀（树干），每个 term 是一条分叉，末端就是叶子。

设平均每 state 有 `l` 个 leaf，一批 M 个样本 → **`P = M·l` 个 leaf 序列**。全篇最关键的是记住这套**三层动态（ragged）结构**：

- `n_i` 逐 state 变（问题数不定）
- `k_j` 逐 question 变（候选数不定）
- leaf 数 ≠ `Σ k_j`，因为 bool 只算 1 → 所以 `P ≤ Σ n_i·kmax`

## 4. 前缀共享：本该 packing，reference 先用 padding 讲清楚

同题 term 共享整段前缀，理论上应该：**(a) sequence packing**——变长序列拼一起、用 `cu_seqlens` 标边界，不 padding；**(b) prefix KV 复用**——causal mask 下前缀 token 不 attend 候选段，前缀 KV 只需算一次、按候选分叉（tree attention）。

reference 实现两者都没做：所有 leaf **padding 到统一 width**、每条重复算整段前缀（`prefix_sharing = False`）。这是为可读性牺牲效率，也是它明确标出的两个优化 TODO。下文按 padding 版讲。

## 5. Forward：`[P, W]` → `[P, hdd]`

```
tokens: [P, W]                       # W = 这批最长 leaf 长度
attn  : [P, W]                       # arange(W) < length，屏蔽 padding
backbone(tokens, attn) → hidden: [P, W, hdd]
leaves = hidden[arange(P), lengths-1]   # [P, hdd] 取每条 leaf 的 valid last token
```

取 last token 是因为 causal 编码下最后一个位置已 attend 整条序列，是该 `(state,question,term)` 的池化表示。**到这一步，transformer 就是个把 P 条序列变成 P 个向量的特征提取器。**

## 6. 关键 reshape：`P` → `(N, kmax)`

扁平 leaf 下标对 backbone 友好，但对**聚合**不友好：softmax 必须在**同一 question 内部**的候选之间做。所以要把 P 重组成 `(问题数 N, 候选数 kmax)`：

```python
h     = zeros(N, kmax, hdd)
valid = zeros(N, kmax, bool)
offset = 0
for i, ex in enumerate(examples):
    n = len(ex.leaf_tokens)                 # 这题的 leaf 数
    h[i, :n] = leaves[offset:offset+n]      # 按 offset 指针切回去
    valid[i, :len(ex.candidate_ids)] = True # 有效候选掩码
    offset += n
```

`kmax` = 这批候选数的最大值，纯粹是把候选维 padding 成矩形的宽度。这里有**两套错位的 raggedness**：backbone 侧填了 `P = Σ n_i` 个格子；valid 侧标了 `Σ |k_j|` 个格子（bool 标 2）——二者对 bool 不一致（见 §8.2）。

## 7. 唯一的预测头：一个 `[hdd, 1]`

```python
z = scalar(norm(h)).squeeze(-1)      # [N, kmax]，scalar = Linear(hdd, 1)
```

**所有题型、所有候选、四个游戏共享这一个标量头**，给每个候选一个 base logit。架构里没有任何维度绑定「类别数」——这就是开放问题空间的根源。

### 7.1 为什么这样就能覆盖所有可能的 question

你可能会问：训练怎么可能覆盖「所有可能的 question」？**答案是：它根本不需要枚举 question，因为输出层里没有任何「类别」维度。**

| | 传统分类头 | 这里 |
|---|---|---|
| 输出层 | `Linear(H, num_classes)` | `Linear(H, **1**)` |
| 类别在哪 | **固化在输出层权重里** | 作为**输入文本**喂进去 |
| 加一个新类 / 新问题 | 改输出维度、重训 | 换一段输入 token，**啥都不动** |
| 候选数量 | 固定 | 2–255 任意（每候选一条独立 forward 行） |

模型学的是一个**通用打分函数** `f(state, question文本, candidate文本) → 一个标量`。它读题、读候选，判断「这个候选和这个 state+question 有多契合」，再对同题候选 softmax。于是：

- **question 是数据，不是标签空间。** 一个从没见过的 question，对模型只是新的输入 token 序列——泛化方式和普通 LLM 读没见过的 prompt 完全一样。
- **训练「覆盖」靠数据多样性，不靠架构枚举。** 喂足够杂的 `(state, question, candidate, teacher概率)` 四元组，让那个共享标量头学会「读文本判契合度」。
- **候选数能 2–255 任意**，因为架构里没有一个维度绑定候选数——`kmax` 只是当前 batch 的 padding 宽度，不是学出来的参数。

这就是全篇最该记住的一句：**把标签空间从输出层搬进输入，分类就从「固定 N 类」变成了「开放、任意 question / 任意候选」。** 一个 0.6B 底座 + 一个 `[hdd,1]` 就能服务四个游戏的所有决策。

## 8. 按 qtype 做 readout

关键：**题型不靠「不同的头」区分，而靠「选行下标」**。只有一个 `[hdd,1]`，差异全在它之上的类型条件化后处理。

### 8.1 choice：set-attention 残差（且只有它用）

choice 的答案是「在集合里挑一个」，好坏是**相对在场其他候选**而言。于是在 base logit 上加一层置换等变的集合注意力：

```python
choice = [i for i,ex in enumerate(examples) if ex.type=='choice']   # 只挑 choice 行
log_k  = valid[choice].sum(-1).log()[:,None,None].expand(-1,kmax,1) # 集合大小特征
u      = set_project(cat([h[choice], log_k], -1))                   # [·, kmax, 128]
mixed  = set_attention(u, u, u, key_padding_mask=~valid[choice])    # 候选之间自注意力
delta  = set_output(tanh(u + mixed)).squeeze(-1)                    # 每候选一个修正
z      = z.index_add(0, choice, delta)                              # 加回 base logit
```

- batch 维 = choice 问题数，序列维 = kmax 候选 → 「同题候选之间」互相 attend；
- `log_k` 让头感知集合大小；`set_output` **初始化为 0** → 起点 delta≡0，等价关掉此头、不扰动 backbone；
- 直觉：base 头知道「这个候选好不好」，set-attention 让它知道「**相对别人**好不好」。

**score/bool 为何不用**：score 是有序标尺、每级独立评估，跨级注意力会破坏「移动一级只改序号」的性质，也让 `Σ i·p_i` 期望语义站不住；bool 只有 1 条 leaf，没有集合可 attend。

### 8.2 bool：`[0, z]` 是 gauge fixing，不是 hack

bool 只出 1 个分数，补零是**规范固定**，干净无损。softmax 对 logits 整体平移不变，固定 false 的 logit = 0、让 true = z：

```python
logits = [0, z]   →   softmax → P(true) = sigmoid(z)
```

标准二元 logistic，`0` 是参照点。至于 §6 里 valid 给 bool 标了 2 格但 h 只填 1 格那个错位——纯粹是 `[N,kmax]` 硬凑的实现残留（多算的 `z[i,1]` 会被丢弃），概念上不存在。

### 8.3 score：独立打分 + 期望读数

```python
level = argmax(p)          # 硬判：最可能的一级
score = Σ i · p_i          # 软读：概率加权的期望等级（靠「下标即顺序」）
```

`score` 有意义正因等级**有序**；choice/bool 无序，加权求和无意义。

**统一视角：一个 `[hdd,1]` 出 base logit，然后 choice 走 set 残差、bool 走 gauge fixing、score 走期望读数——类型分派 = 选行下标，不是换头。**

## 9. 损失与训练

**Loss：候选分布上的交叉熵**（逐 question，再对 batch 取均值）：

```python
target = zeros(N, kmax)
target[i, :k] = teacher_probs        # 软标签：蒸馏老师的完整分布
# 或 target[i, gold_index] = 1        # 硬标签：one-hot
loss = -(target * logits.log_softmax(-1)).sum(-1)   # 无效格 target=0，不污染
```

- **软标签路线是蒸馏**：student 拟合老师的整个概率分布，不只对 argmax；
- bool 是 `[0,z]` 上的二分类 CE，choice/score 是 k 路 CE。

**训练：两段式**（经典 linear-probe → finetune）：(1) 先冻结 backbone、只训 heads 若干步，head lr 拉高；(2) 再解冻整体微调，backbone lr 小（如 `1e-5`）、head lr 稍大（`1e-4`）；(3) 参数 FP32 存、forward BF16 autocast；(4) 多任务按固定权重混合采样。

## 10. 数据流一图流

```
M 行 jsonl (state)
  └─ 展开 n_i 个 question
       └─ 展开 k_j 个 term  ──(bool 只 1 条)──►  P 个 leaf 序列
                                                   │  padding→[P,W]（本可 packing+prefix 复用）
                                                   ▼
                                        backbone(黑盒) → [P, W, hdd]
                                                   │  取 valid last token
                                                   ▼
                                              leaves [P, hdd]
                                                   │  offset 重组(P → N×kmax)
                                                   ▼
                                          h [N, kmax, hdd] + valid 掩码
                                                   │  共享 scalar [hdd,1]
                                                   ▼
                                            base logits z [N, kmax]
                             ┌──────────────┼───────────────┐
                        choice: +set残差   bool: [0,z]     score: 原样
                             └──────────────┼───────────────┘
                                       masked_fill(-1e9)
                                                   │  softmax(每题内)
                                                   ▼
                                        概率分布 → CE(teacher/gold) → mean → loss
```

## 11. 剥掉工程外壳，只有两个真 idea

重复前缀、全 padding、bool 的错位格子——都是 reference 的朴素之处。真正的建模决定只有两条：

1. **标签空间搬进输入**——一个共享 `[hdd,1]` 给任意候选打分，softmax 成分布 → 开放、非生成的分类；
2. **choice 上加一层置换等变的 set-attention** 做相对打分。

其余（bool 的 gauge fixing、score 的期望读数、两段式训练、蒸馏 CE）都是成熟组件的直接组合。这也印证了最朴素的直觉——**机制上就是 cross-encoder 打分，新意在于把它做成一个覆盖 boolean/choice/score 三原语、零解码的统一决策接口。**

## 12. 回到冷思考：它到底能不能替掉 LLM

走完数据流，再回头看官方的宣传，就能分清哪些是真、哪些是话术（这一节主要综合丁师兄文章的判断）。

**「快 200 倍、便宜 400 倍」——底层优势真，但数字要打折。** 不做长 reasoning trace、不生成输出、同一份 state 上的独立问题可以一次性并行打分，这些是架构决定的真优势。但那组数字出自 TypeSafe 自测、挑的是最有利的 best case，当参考看就行。

**「不会幻觉」——只在很窄的定义下成立。** 它确实不会返回 schema 之外的选项（你定义了 billing/technical，它编不出 legal），也不会在等 label 的地方吐一段乱码。但它可以**非常自信地选错那个合法选项**。type safety 防的是形状不合法，不保证判断正确。更准确的说法是：**不会破坏声明的输出 schema，但照样会判断错。**

**最有用的其实是那个概率。** 标签只告诉你「谁赢了」，概率分布才告诉你「赢得有多险」。于是有了很实用的模式：高置信度且后果不严重 → 自动执行；中置信度 → 人工确认或换更强模型；低置信度 → 转人工。而且阈值应该写进代码，才能被 review、被调整。

### RLCD：训练目标为什么是「校准」而不是「准确」

TypeSafe 宣称用 **RLCD（Reinforcement Learning for Calibrated Decisions）** 训练 Jev。以下这段补充主要综合 Sanity 的术语词条。

- **它奖励什么。** RLCD 是一种 RL 后训练目标：奖励信号绑定的是「模型自报的概率是否与它答对的实际频率相符」，而**不是**人类偏好评分（RLHF），也不是可自动验证的正确性（RLVR）。一句话——一个报了 0.8 的决策，在大量同类决策里就该有约 80% 是对的。
- **为什么这个目标合理。** 因为模型的产物本来就是「一个要拿去和阈值比较、让代码分支的概率」，不是给人读的文章。**光有 accuracy 撑不起阈值**：一个 90% 正确但对什么都报「99% 有把握」的分类器，在你的 `if` 语句眼里每个 case 都长一样，没法 gate；反而是一个报 70%、也确实对 70% 的弱模型，能把 case 分成「可无人值守处理」和「该转人」两堆。所以当产物是概率而不是散文时，**校准不是叠在准确率之上的修饰，它本身就是被卖的东西。**
- **和 RLHF / RLVR 的区别。** RLHF 奖励人类偏好的回答（副作用是谄媚、mode dropping），RLVR 奖励程序能机械核验的输出（DeepSeek-R1、Tulu 3 那一路，但它对「该有多确定」只字不提）。RLCD 两者都不要，只盯着「自报概率的诚实度」，输出是带类型的答案 + 分布，而非文本。
- **它不做什么（两个最容易混的点）。** 其一，RLCD **不提升准确率**——校准和准确是两个相互独立的属性，TypeSafe 也没声称它让 Jev 更准。其二，它**不是**防止 off-schema 输出的东西（那是 §7 的架构决定的）。而且校准描述的是**一群预测**，不对任何单条答案给保证——某条 0.93 的含义只在一堆同类决策里才可复原。
- **几个必须打的折。** RLCD 是厂商自造的方法名，不是行业标准：**没有论文、没有公开的 reward function、没有数据集描述、没有放出校准数字**，全部信息都来自 TypeSafe 自己的发布材料，未经独立复现。另外它还是个**缩写撞车**——同样四个字母的 "Reinforcement Learning from Contrastive Distillation"（Yang et al., ICLR 2024, arXiv:2307.12950）是一个完全无关的、属于 RLHF 家族的对齐方法；连 TypeSafe 自己写的是 "for"、TechCrunch 引述 Almeida 时又成了 "from"。

**在 agent 里最好的用法是配合，不是替代。** 要语言、要深推理的活交给 LLM，围绕这些活的高频判断交给 Jev。三个位置特别合适：**模型路由**（给请求打分、挑最便宜又能搞定的模型）、**工具风险闸门**（执行 shell 命令前分类 read-only / reversible / destructive，拿不准就停下等人批）、**验证与监督**（测试过了吗？是不是原地重复同一个动作？输出符合 policy 吗？）。LangChain 的集成就是把它做成 agent 的 middleware。

**适合与不适合。** 适合的用例有三个共同点：答案能提前列全、换个靠谱的人一眼也能判断、这种判断发生得够频繁。不适合的很明确：写回复/总结/代码/推理做不了；算术、计数、比日期、精确匹配交给代码；要从文本里捞出事先不知道的值也不行（得先找好候选再让它挑）。最朴素的一条：**普通代码就能算对的，就留着——一个 `if` 语句比任何模型都快、都便宜、都更好测。**

**别急着上线。** 一个「便宜」的模型，如果它的错误会引来重试、人工 review、线上事故，那它一点都不便宜。稳妥顺序大致是：挑一个边界清楚、低风险的决策 → 先写 rubric 再调模型 → 收集有代表性（含模糊/对抗）的样本 → 让它 shadow mode 跑在现有 workflow 旁边、先不改行为 → 把 accuracy 对 confidence 画出来、用自己的数据定阈值 → 先自动化最安全那条分支 → 把模型版本 / questions / criteria / 阈值都记下来可回放。**questions 本身就是程序的一部分，要像代码一样做版本和 review。**

最后提一句现状：Jev **权重不开源、还在 early access、只支持文本、也缺独立的校准数据**，现在还不到盲信它的时候。但它给出的那个「像软件一样」的模型接口——固定的答案类型、显式的不确定性、由代码掌控的分支——这个思路即便以后 Jev 被别的模型取代，也照样成立。而 NanoJev 的价值，正是让这套接口背后「不写字怎么做分类」的机制，变成了可以逐行读、逐段走查的开源代码。

## 参考资料

- 数据流实现主线：[TianyuCodings/NanoJev](https://github.com/TianyuCodings/NanoJev)（`docs/DATAFLOW.md`，本文的技术走查与代码解读据此复现整理）
- 背景与定位：[《Jev 被全网吹爆：凭什么比 LLM 快 200 倍？》（丁师兄）](https://mp.weixin.qq.com/s/vJmGUH2361kLDfFPCHMRdg)
- RLCD 词条：[What is RLCD (Reinforcement Learning for Calibrated Decisions)?（Sanity Glossary）](https://www.sanity.io/glossary/rlcd-reinforcement-learning-for-calibrated-decisions)
- 一手资料（经上文援引）：TypeSafe AI《Introducing System One Models and Jev》、LangChain《Building a Harness with Jev》、Flavio Copes《A deep dive into Jev》



