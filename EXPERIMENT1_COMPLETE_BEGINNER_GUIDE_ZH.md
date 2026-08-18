# 从 Proposal 到 Experiment 1 全部版本：零 ML 基础完整入门

副标题：这项研究到底想解决什么、第一轮实验为什么这样设计、每个“版本”做了什么、结果究竟意味着什么

更新日期：2026-07-19  
适合读者：完全没有机器学习、强化学习或统计学背景的人；也适合作为团队后续写 Research Doc 时的事实索引。

---

## 读这份文档前，先知道三件事

### 1. 这是核查后的新版，不是旧进度的重复

工作区原来已经有一份 `EXP1_VERSIONS_ZERO_TO_ONE_ZH.md`，它对 pilot 的 Mac、Windows 和 Stage-B seed repeat 讲得比较清楚，但停在 2026-07-18 的状态：当时它仍写着 Experiment 1.5 “正式 CUDA run 尚未开始”。现在这已经不符合事实。

本新版补齐并核查了：

- 正式 proposal 的完整研究逻辑，而不只讲 pilot；
- proposal 与 pilot 的区别；
- pilot 的 Mac / Windows v1 / Windows v2 / Stage-B seed repeat；
- Experiment 1.5 的原计划、v1、v2、v3、float32 测量恢复和完整结果；
- 已冻结但尚未正式执行的 Experiment 1.5.1；
- 必须等团队批准才可启动的 Experiment 1.6；
- 每个版本“为什么存在”“只改了什么”“能说明什么”“不能说明什么”。

### 2. 文中用四种状态标签，防止把计划当结果

- **【正式规格】**：来自 `team_doc/proposal_v3.1_formal.docx`，是项目长线目标。
- **【预注册计划】**：运行前冻结的设计；表示“打算怎么判”，不表示已经跑出结果。
- **【已完成事实】**：已经有 run 目录、JSON/CSV、checkpoint 或安全停止证据。
- **【描述性解释】**：根据已完成数据作出的辅助理解；不是预注册主结论，也不是因果证明。

### 3. 最重要的措辞红线

不能把结果写成“RLVR 降低了模型的一般学习能力”。项目能严谨测量的是：

> 在预先固定的新任务、数据、算法、更新步数和基线下，一个 Stage-A checkpoint 在相同预算内还能改善多少。

这叫 **fixed-budget future adaptability（固定预算未来适应量）**。它是具体可测量的量，不是“学习能力”这个模糊大词。

---

# 第一篇：先用一页看懂全项目

## 1.1 研究问题的大白话版本

想象一个学生分阶段学习：

1. 第一阶段学数学 A；
2. 第二阶段再学逻辑 B；
3. 老师最怕的不是第一阶段成绩差，而是第一阶段看起来一切正常，第二阶段却突然怎么教都不再进步；
4. 如果只能等第二阶段成绩曲线变平后才发现，第二阶段的时间和算力已经花掉了；
5. 团队想在第一阶段进行中给模型做“内部体检”，看体检指标 Q 能否提前预警后面会不会失速；
6. 还要证明这项体检比训练员已经在看的 reward、KL、gradient norm、entropy 等仪表更早或更准。

这就是 proposal 标题 “Seeing the Stall Coming” 的意思：**在失速真正显现以前，看见它要来了。**

## 1.2 一条最短的研究链

```text
Stage A 训练中
    |
    | 每隔一段时间保存 checkpoint
    v
对每个 checkpoint 测内部指标 Q
    |
    | 再从每个 checkpoint 出发，用完全相同预算学习新任务
    v
得到“固定预算适应量”
    |
    | 检查 Q 是否与未来适应量/失速标签有关
    v
再与 reward / KL / grad norm / entropy 比较谁更早预警
```

## 1.3 Experiment 1 在整项研究里的位置

Experiment 1 不是整篇 proposal 的最终实验，而是一个 0.5B 小模型 pilot：

- 先证明代码、数据、指标、存档、适应和分析能端到端工作；
- 看 Q 是否会在训练中移动；
- 粗略看 Q 与固定预算适应量是否相关；
- 找出正式实验之前最危险的工程与统计问题；
- 用本地硬件估算正式实验成本，尽量不消耗 Colab 预算。

它不是：

- 一个已经训练好的 stall detector；
- proposal 要求的 ≥30 个独立 stage-transition 语料；
- 一个能证明因果关系的实验；
- 一个能代表 1.5B–3B 模型、所有任务家族或所有 RLVR 配方的结论。

## 1.4 截至 2026-07-19 的一句话结论

【已完成事实】管线和 Q 测量已经被反复验证；早期中深层 effective-rank 压缩在多条健康 run 中复现；温和剂量下没有观察到持续的固定预算适应力下降；高剂量 `lr=1e-5, beta=0` 会在约 55 步内发生真实策略崩溃；Experiment 1.5 把 outcome 噪声显著压低，但也实锤了 GSM8K→SVAMP 的迁移/天花板混杂。**因此 RQ1 尚未被干净回答，下一步要么密集取证崩溃前 Q（1.5.1），要么寻找“不崩但能留下持久变化”的中间剂量（1.6）。**

---

# 第二篇：Proposal 从零讲起

## 2.1 什么是语言模型

语言模型可以先理解成一台“根据前文猜下一个 token”的机器。token 不是严格等于一个汉字或一个英文单词，而是模型内部切分文本的基本单位。

模型内部有大量参数（parameter，也常叫 weight，权重）。每个参数就是一个数字。Qwen2.5-0.5B 大约有 5 亿个这样的数字。训练不是把答案文件直接塞进模型，而是反复做以下循环：

1. 给模型输入问题；
2. 模型产生输出；
3. 用某种规则判断输出好不好；
4. 算出每个参数应当朝哪个方向改变；
5. 把参数移动一点；
6. 重复很多次。

## 2.2 什么是多阶段训练

现实中的后训练往往不只做一次：

- Stage A：先针对数学推理训练；
- Stage B：再针对逻辑推理训练；
- 未来还可能有 Stage C。

多阶段训练的风险是：Stage A 可能提高了眼前任务表现，却把模型带到一种对后续训练不友好的内部状态。这个状态不能只靠“Stage A reward 还在涨”来排除。

## 2.3 plasticity、forgetting、diversity collapse、stall 不是同一个词

### Plasticity（可塑性）

模型继续从新数据中改变、拟合新目标的能力。它是一个概念，不存在全领域公认的唯一标量。

### Catastrophic forgetting（灾难性遗忘）

学新任务时把旧任务能力忘掉。它问的是“以前会的还会不会”。

### Reduced future adaptability（未来适应量下降）

给相同的新任务和相同预算，某个 checkpoint 的改善量比基线小。它问的是“相同训练预算内还能进步多少”。这是本项目在 pilot 中真正测的东西。

### Diversity collapse（多样性坍缩）

模型对同一道题采样很多次时，解法越来越单一；可以用 pass@k、输出熵等描述。

### Multi-stage stall（多阶段失速）

后续 Stage B 相对自己的起点几乎不再改善，而且明显落后于“从头直接训练 Stage B、同预算”的参考运行。

这些现象可能有关，但不能互相替代。比如 effective rank 下降不自动等于“学习能力下降”，Stage-B delta 变小也可能只是因为起点已经很高、没有上涨空间。

## 2.4 Proposal 的四个研究问题

【正式规格】正式 proposal 把问题分成四层：

1. **存在/相关（RQ1）**：RLVR 过程中是否出现可测的 Q 变化？Q 是否与后续 stall 或固定预算结果有关？
2. **检测（RQ2）**：只用早期信息，Q 能不能在最终准确率暴露问题前预测下一阶段会不会 stall？
3. **正面对决（RQ3）**：Q 是否比 reward plateau、累计 KL、gradient norm 更早或更可靠？这是项目最关键的实用性判断。
4. **机制/有效性（RQ4）**：Q 真的是共同的 capacity-loss 信号，还是只在反映任务专门化、训练步数、难度变化等混杂？需要 mediation、控制实验和 intervention 来区分。

Experiment 1 主要服务 RQ1，而且只是缩小版。

## 2.5 Proposal 的五个任务

### Task 1：造出有 stall 和没有 stall 的 run 语料库

扫描多个旋钮：

- KL 系数 beta：0、小、大；
- 全参数微调 full fine-tune 与 LoRA；
- 短训练与 prolonged 长训练；
- math→logic 与 logic→math 的 stage order；
- 至少 3 个随机种子。

危险角落预计是：小模型、全参数、beta=0、长训练。温和角落预计是：较大 beta、LoRA、短训练。

### Task 2：同时记录 Q 和普通 dashboard

Q 包括 effective rank、dormant fraction；便宜的辅助量包括 weight norm、gradient norm；dashboard 包括 reward 曲线、KL、梯度范数等。所有 Q 必须在固定 probe、eval mode、固定 dtype 和固定层上测。

### Task 3：训练并评估 stall detector

输入是“早期特征 + 最终 stall 标签”，输出是一个预测器。评估时必须按 run 划分训练/测试，不能把同一条 run 的早期 step 放训练集、晚期 step 放测试集，否则会泄漏。

### Task 4：机制和排除混杂

检查 Q 是否中介 beta、训练时长、adaptation 类型对 stall 的影响；再用 task distance、stage order、compute-matched control、equal-accuracy control、relearning battery 排除“只是任务更像/更难”或“只是多训练了几步”。

### Task 5：从预测走向控制

尝试 L2-Init 或 ReDo-style dormant reset 等 capacity-preserving intervention。如果直接保护 Q 后 stall 也减少，才是比相关更强的因果证据。

## 2.6 正式 proposal 如何定义 stall

这是 pilot 与最终论文最容易混淆的一点。

【正式规格】对 Stage B，先计算：

```text
r =（staged run 在 Stage B 结束时的准确率 - 它开始 Stage B 时的准确率）
    /（from-scratch reference 在同预算下的结束准确率 - staged run 的 Stage-B 起点准确率）
```

直觉：分母是“同预算下理论参考能涨多少”，分子是“经过 Stage A 的模型实际涨了多少”。如果只实现了参考提升的一小部分，就可能 stall。

- 主阈值：`r < 0.5` 判 stall；
- 必做敏感性分析：阈值 0.4、0.6；
- from-scratch reference 必须使用相同 Stage-B 预算和设置。

**Pilot 没有实现这个正式 stall label。** Pilot 使用的是每个 checkpoint 的：

```text
svamp_delta = 适应 50 步后的准确率 - 适应前准确率
```

所以 pilot 研究的是 checkpoint-level adaptability proxy，不是正式的二分类 stall detector。

## 2.7 正式 detector 怎么判断“成功”

【正式规格】在未见过的 held-out runs 上：

1. `AUROC(Q) > 0.7`；
2. `AUROC(Q)` 高于最好的 dashboard baseline；
3. 固定 recall=0.8 时，Q 的平均提前量比最佳 dashboard 至少多出 stage 长度的 10%；
4. 还要报告 AUPRC 和 calibration；proposal 规定 held-out run 上 `ECE < 0.1` 才算校准良好。

其中第 3 条“提前多少”是 load-bearing result。若 Q 只和 reward 同时报警，它可能有解释价值，但没有节省算力的核心价值。

## 2.8 Proposal 的正式规模与 pilot 的差距

| 项目 | 正式 proposal | Experiment 1 pilot |
|---|---|---|
| 主要模型 | 1.5B–3B；第二模型家族用于扩展 | Qwen2.5-0.5B |
| Stage 顺序 | GSM8K↔ProntoQA，数学与逻辑双方向 | GSM8K→SVAMP，都是数学文字题 |
| 样本单位 | ≥30 个独立 stage-transition，按 run 留出 | 同一 Stage-A 轨迹上的 5 或 6 个 checkpoint |
| 训练旋钮 | beta、full-FT/LoRA、时长、顺序、seeds | 基本固定 beta=0、full-FT、小模型 |
| outcome | from-scratch 参照下的 stall label | 50-step SVAMP delta |
| 最终分析 | AUROC/AUPRC、校准、lead time、bake-off | Spearman 相关和轨迹描述 |
| 因果性 | mediation + intervention | 没有 |

因此，pilot 的作用是“校准测量仪、找到剂量和噪声问题”，不是提前宣布 proposal 成功或失败。

---

# 第三篇：读懂 Experiment 1 所需的 ML 基础

## 3.1 数据集、train、eval、probe 各是什么

- **Train set**：允许训练算法使用、会影响权重的数据。
- **Eval set**：只用来考试，不用于更新权重。
- **Probe set**：专门拿来测内部激活的固定输入。它不一定要生成答案；只需做前向计算。
- **Frozen split**：题目 ID 在运行前固定并保存。不同 checkpoint 必须面对同一批题，否则差异可能来自题目不同。

Pilot 中：

- Stage A：512 道 GSM8K train + 64 道 held-out eval；
- Q：512 个固定 probe，端点另做 2048 probe sensitivity check；
- Stage B：256 道 SVAMP train；pilot 用 100 eval，Experiment 1.5 升为 300 eval。

## 3.2 什么是 RLVR

RLVR = Reinforcement Learning with Verifiable Rewards，可理解成“答案能自动验对错的强化学习”。

数学题非常适合：解析模型最后答案，与标准数字完全匹配就给 1，否则给 0。它不需要人工给每个回答打分。

## 3.3 GRPO 到底做什么

Pilot 的 Stage A 和 Stage B 都使用 GRPO。对一道题：

1. 模型随机生成 8 个回答；
2. 每个回答得到 0 或 1；
3. 在这 8 个回答内部计算相对好坏；
4. 比组内平均好的生成轨迹被加强；
5. 比组内平均差的生成轨迹被削弱。

关键点：如果 8 个回答全对或全错，组内没有相对差异，这一组就不给 GRPO 有用学习信号。因此正式训练前要做 sparse-reward preflight，检查至少有一些组存在 reward variance。

## 3.4 reward、loss、gradient、optimizer 的关系

- **Reward**：任务层面的好坏分数；本实验是 0/1。
- **Loss**：训练算法把目标变成的可微分数值；优化器实际最小化它。
- **Gradient**：每个参数朝哪个方向变化会让 loss 改善。
- **Optimizer**：根据 gradient、历史动量、学习率等决定实际更新量；这里主要是 AdamW 或 paged AdamW 8-bit。

Reward 上升通常表示模型更常答对，但 reward 正常不保证权重真的有效更新，也不保证未来任务还能学。

## 3.5 learning rate、update、batch geometry

- **Learning rate（lr）**：每次参数更新的步幅尺度。
- **Update**：优化器真正把权重改变一次。
- **Micro-batch**：一次实际放进显存/内存的样本数量。
- **Gradient accumulation**：先累积多个 micro-batch 的梯度，再更新一次。

Mac 用 `8 × 8`，Windows v2 用 `4 × 16`。两者都代表每次更新累计 64 个 completion，因此科学上的有效 batch geometry 对齐，但具体浮点运算顺序和执行轨迹仍可不同。

## 3.6 checkpoint 不是一个新实验

Checkpoint 是训练到某一步时的模型快照。`ckpt-50` 表示 Stage A 完成 50 次更新后的状态。Experiment 1 从同一条 Stage-A 轨迹保存 0/25/50/100/200 五个快照。

然后每个 checkpoint 都被复制成一个新的 Stage-B 起点。它们不是五个独立 Stage-A seeds，而是同一条轨迹上的五个相关时间点。

## 3.7 on-policy 为什么让重复运行分叉

GRPO 是 on-policy：模型用自己当下生成的回答训练自己。生成有随机性：

1. 第一步采样略有不同；
2. 第一步权重更新略有不同；
3. 第二步模型已经不同，采样差异继续放大；
4. 经过几十或几百步形成不同训练轨迹。

所以“同配方、同 seed”跨硬件运行并不保证 bitwise 相同。硬件、精度、库版本、优化器状态恢复和浮点运算顺序都会让轨迹分叉。

## 3.8 seed 能控制什么，不能控制什么

随机种子是伪随机数生成的起点。固定 seed 能增强同环境复现，但不能消灭：

- 不同 GPU/CPU 的运算差异；
- 并行计算顺序差异；
- 低精度舍入；
- 中断后没有恢复 optimizer moments 或 RNG state；
- on-policy 差异被后续步骤放大。

因此需要多个独立 Stage-A seeds，以及每个 checkpoint 多个 Stage-B seeds。

## 3.9 fp32、bf16、autocast 为什么会决定实验真假

- fp32 大约有 7 位十进制有效数字；
- bf16 只有约 2–3 位有效数字，但范围大、显存省；
- `lr=1e-6` 产生的单步变化可能远小于 bf16 在典型权重附近能表示的间隔。

Windows v1 把参数本体直接存成 bf16。结果是大量很小的更新被舍入掉，200 步后权重组最大相对变化只有 `2.4e-8`，reward 和 Q 几乎不动。更准确的说法是“训练在科学意义上近似 no-op”，而不是字面上每一个 bit 都完全没变。

Windows v2 改成：

- fp32 master weights：总账保留高精度；
- bf16 autocast：前向/反向临时使用低精度；
- paged AdamW 8-bit：压缩优化器状态；
- gradient checkpointing：用更多计算换显存。

## 3.10 ceiling / headroom：为什么 delta 小不一定是学不动

假设两个人都考 100 分制：

- A 适应前 50 分，适应后 60 分，delta=+10；
- B 适应前 95 分，适应后 98 分，delta=+3。

不能只凭 delta 说 B 学习能力更差，因为 B 只剩 5 分上涨空间。GSM8K 与 SVAMP 很相似，Stage A 的数学训练可能提前提高 SVAMP 起点，从而压小 Stage-B delta。这就是 Experiment 1.5 最关键的混杂之一。

## 3.11 版本、stratum、seed、checkpoint 的区别

| 词 | 本项目里的准确意思 |
|---|---|
| checkpoint | 同一条 Stage-A 轨迹上的时间快照 |
| seed repeat | 固定其他条件，只换随机种子 |
| execution stratum | 同一科学配方在一种硬件/精度/优化器环境中的完整执行 |
| config revision | 因安全规则或科学参数修正而形成的新冻结配置 |
| experiment number | 一个新的科学问题或新的预注册设计，如 1.5.1、1.6 |
| evidence pack | 结果打包，不是一轮新训练 |

---

# 第四篇：Q 指标和 dashboard 指标

## 4.1 Q 是什么

Q 不是一个已经公认的单一数字，而是“候选 plasticity diagnostic”的统称。正式 proposal 的主要候选是 effective rank，次要候选是 dormant-neuron fraction；weight norm、gradient norm 是便宜辅助量；Local Learning Coefficient 是可选昂贵量。

Pilot 的预注册主 Q 是第 12 层 effective rank，即 `erank_L12`。

## 4.2 激活是什么

模型读一道题时，每一层都会产生内部向量。可以把向量理解成模型对这道题的“内部表示”。

Pilot 对固定 512 个 probe prompt：

1. 模型设为 eval mode；
2. 不做训练；
3. 在第 4、12、22 层截取 hidden state；
4. hidden state 使用每个 prompt 最后一个非 padding token；
5. 512 个向量堆成矩阵；
6. 用同一测量 dtype 和数值累积规则计算指标。

Dormant 指标则对 MLP post-activation 单元，在所有非 padding token 上累计平均绝对激活。

## 4.3 Effective rank 一步一步

假设 512 道题在某层各得到一个 d 维向量，把它们组成矩阵 `A`：

```text
A 的行数 = probe 数量（512）
A 的列数 = hidden dimension
```

然后：

1. 对每一列中心化，去掉共同均值；
2. 对 A 做 SVD，得到奇异值 `sigma_i`；
3. 归一化：`p_i = sigma_i / sum(sigma_j)`；
4. 计算谱熵；
5. `erank = exp(-sum(p_i log p_i))`。

直觉：

- 很多方向都有显著强度 → effective rank 较高；
- 信息集中在少数方向 → effective rank 较低；
- 下降说明表征谱更集中或压缩；
- **但下降本身不等于未来适应力必然下降。** 这正是实验要检验的关系，而不是可以先假定的结论。

## 4.4 为什么还记录 normalized erank、participation ratio、top-k share、anisotropy

- `erank_norm = erank / d`：把维度规模纳入考虑；
- participation ratio：另一种有效维度估计；
- top-1/top-8/top-32 variance share：看最强几个方向解释了多少变化；
- uncentered anisotropy：可能受共同均值方向影响；
- centered anisotropy：中心化后看真正方向集中。

这些是辅助指标，防止只用一个数误读复杂的谱变化。

## 4.5 Dormant-neuron fraction

对每个神经元 i：

```text
s_i = 该神经元在 probe 上的平均绝对激活
      / 同层所有神经元平均绝对激活的平均值
```

如果 `s_i < tau` 就记为 dormant。proposal 要报告 `tau=0.025` 和 `tau=0.1`。

在目前所有有效的 0.5B pilot / exp1.5 测量中，dormant fraction 在这些层和阈值上一直是 0。这表示它在当前设置里没有区分度，不表示模型没有任何形式的容量变化。

## 4.6 Dashboard baselines 是什么

正式 proposal 要 Q 正面对比：

- reward slope / plateau；
- KL-to-reference accumulation；
- gradient norm trajectory；
- pilot 还记录 entropy、completion length、clipped ratio、reward variance。

Experiment 1.5 v2 的真实崩溃告诉我们：

- completion clipping 是慢性红旗；
- entropy 急跌是临近崩溃的急性信号；
- reward variance 连续归零是终局事件；
- loss 和 grad norm 最终也归零。

Q 要成为真正有价值的 early-warning signal，必须比这些已记录信号更早或更可靠。

## 4.7 Leakage rule

任何在 step t 声称可用于预警的特征，只能使用 step t 当时已经存在的信息。

允许：

- 当前 Q 值；
- 最近 20 步 Q 的斜率；
- 到当前为止的最大下探；
- 当前以前的 reward / entropy 窗口。

不允许：

- 用整条 run 的最终均值给早期值标准化；
- 先看到最终崩溃时点，再回头挑最漂亮的 Q 阈值；
- 同一条 run 的晚期数据参与训练、早期数据又当独立测试样本。

---

# 第五篇：Original Experiment 1 的完整设计

## 5.1 一句话配方

【预注册计划】Qwen2.5-0.5B base 在 512 道 GSM8K 上做 200 次 GRPO 更新，保存 0/25/50/100/200；每个 checkpoint 在固定 probe 上测 Q；再分别进行完全相同的 50-step SVAMP GRPO 适应；用 5 个 checkpoint 上的 Spearman 相关检查 `erank_L12` 与 `svamp_delta` 是否同序。

## 5.2 冻结设置

| 项目 | Pilot 设置 |
|---|---|
| 模型 | `Qwen/Qwen2.5-0.5B` base，revision `060db649…` |
| Stage A | GSM8K，512 train，200 updates，lr=1e-6，beta=0 |
| 生成 | 每题 8 个，temperature=0.7，top-p=1.0 |
| checkpoint | 0 / 25 / 50 / 100 / 200 |
| Q probe | 512 prompts；层 4/12/22；有效对比产物实际为 fp32 测量 |
| dormant 阈值 | 0.025 / 0.1 |
| Stage B | SVAMP，256 train / 100 eval，50 GRPO updates，lr=1e-6 |
| 主 outcome | after50 accuracy - before accuracy |
| 主分析 | Spearman(`erank_L12`, `svamp_delta`)，n=5 |
| 默认科学 seed | 42 |

注意：仓库早期 `pilot_config.json` 的 measurement 字段曾保留 `float16` 文字，但两条被用于跨平台比较的有效 metrics JSON 均记录 `model_dtype_requested=float32`、`measurement_contract.model_dtype=torch.float32`。Experiment 1.5 的一次 float16 首测后来也被一致性 gate 拦下并用 float32 重测。跨 run 有效比较以产物里的实际 measurement contract 为准。

## 5.3 四个阶段

### Phase 0：开发验证

测试 reward 解析、数据冻结、metric 数学、MPS 兼容、GRPO 接口、artifact contract。它证明“代码没有明显坏掉”，不产生论文结果。

### Phase 1：GSM8K Stage A

训练并保存 checkpoint，同时记录 reward、grad norm、entropy、completion length、held-out GSM8K accuracy 等。

### Phase 2：测 Q

每个 checkpoint 对同一 probe 做无训练前向，输出 `metrics_ckptN.json`。

### Phase 3：SVAMP 固定预算适应

从每个 checkpoint 独立开始，先考 100 题、训练 50 步、再考同一 100 题，同时记录 10/20/30/40/50 步曲线。

### Phase 4：分析

合并 Q、before、after、delta，运行预注册 Spearman，输出 CSV 和三张图。

## 5.4 三层防呆

### Sparse-reward preflight

训练前先生成小批回答，确认至少有 prompt 的 8 个 reward 不全相同。没有 GRPO 信号就停，不静默改 reward。

### Update-effectiveness sentinel

每 25 步比较权重变化，确认训练真的在更新。这个装置由 Windows v1 的 bf16 no-op 事故直接催生。

### Validator / manifest / hash

确保配置、数据、模型 revision、seed、实际更新步数、评估曲线、基线和 telemetry 对得上。失败 attempt 保留并隔离，不覆盖成“成功”。

---

# 第六篇：Experiment 1 的所有早期版本

## 6.1 版本总览

| 名称 | 性质 | 状态 | 它回答什么 |
|---|---|---|---|
| Phase-0 | 代码/契约验证 | 完成 | 管线能不能工作，不是科学结果 |
| Mac CPU stratum | 第一条完整科学执行 | 完成 | pilot 在 fp32 CPU 上会看到什么 |
| MPS 调查 | 性能路线调查 | 放弃 | Mac GPU 是否真的更快 |
| Windows v1 | 无效执行 / 阴性对照 | 作废但保留 | 低精度是否让训练近似 no-op |
| Windows v2 | 修复后的第二条科学执行 | 完成 | 现象是否跨执行环境复现 |
| Stage-B seed repeats | 只换适应 seed | 部分完成后关闭 | 单 seed 的 delta 排名有多不稳定 |
| Evidence pack | 证据打包 | 已交付 | 让队友离线核查，不是新实验 |

## 6.2 Phase-0：为什么 15 秒测试也重要

【已完成事实】37 个单元/契约测试、迷你 GRPO smoke、8-prompt Q dry run 均通过。`outputs/dry_run_metrics.json` 只代表接口能跑，不能与正式 checkpoint 混用。

它防止三类假结果：

- reward 解析错，把正确答案打成 0；
- activation hook 抓错层或抓错 token；
- 训练器表面运行但没有完成约定更新数。

## 6.3 Mac CPU stratum

【已完成事实】M3 Max CPU、fp32、AdamW、seed 42。Stage A 中途在约 step 67 因进程中断，后续从 checkpoint 恢复并完成。CPU 约 300 秒/update。

主要结果：

- reward 分窗均值 0.430→0.640；
- GSM8K held-out 0.4375→最高 0.5469；
- L22 erank 到 ckpt-200 比基线低约 11.1%；
- L12 早期下探后在 ckpt-200 回到基线附近（+0.7%）；
- dormant fraction 全 0；
- SVAMP delta：+5/+13/+8/+2/+4 pp；
- 主相关 rho=-0.60，p=0.285，n=5。

它意味着：端到端流程可运行，深层谱压缩确实出现；但单条 run 的相关不显著，不能解释成稳定规律。

## 6.4 MPS 调查

Mac 的 GPU 接口 MPS 在裸算力测试中可用，但真实 TRL 生成循环约 265–320 秒/update，与 CPU 接近。逐 token 生成同步开销吃掉加速收益，因此没有切换正式 run。

它的意义是成本工程：先用真实工作负载 benchmark，避免为一个不更快的平台重构实验。

## 6.5 Windows v1：为什么“全部 phase 都跑完”仍然是无效实验

【已完成事实】Windows RTX 4070 Laptop 的第一次版本把参数本体加载为 bf16。四个 phase 工程上跑完，但：

- ckpt-0→200 权重组最大相对变化仅 `2.4e-8`；
- reward 约 0.366→0.380，基本平；
- Q 变化极小；
- Stage-B delta 只在很小范围游走；
- 主相关没有科学解释价值。

原因是 `lr=1e-6` 的微小更新无法在 bf16 master weights 中可靠累积。结论不是“Q 与适应无关”，而是“这条 run 没有有效施加 Stage-A treatment”。

它的意义：

1. 产物齐全不等于实验有效；
2. dashboard 正常不等于权重真的有效移动；
3. update sentinel 是科学有效性检查，不只是工程监控；
4. v1 可作为 near-no-op 阴性对照，但不能进入主科学分析。

## 6.6 Windows v2：修复后的 CUDA stratum

修复为 fp32 master + bf16 autocast + paged AdamW 8-bit + gradient checkpointing。有效 batch 仍是 64 completion/update。

执行限制必须一起记：

- Phase 1 在 25/50/75 附近多次中断；
- 模型权重和调度位置恢复，但 optimizer moments 与 RNG state 不完全恢复；
- 因此它是有效轨迹，但不等价于一次完全不中断的 bitwise run；
- Phase 3 ckpt-50 一次 OOM，失败目录保留，新进程同配方重跑；
- sentinel 33/33 通过，权重最大相对变化 `6.06e-6`。

主要结果：

- reward 0.354→0.582；
- L12 到 ckpt-200 -7.4%；
- L22 到 ckpt-200 -8.3%；
- dormant 全 0；
- SVAMP delta：+6/+11/+3/+5/+12 pp；
- 主相关 rho=+0.50，p=0.391，n=5。

与 Mac 比较：

- 两边都真实训练；
- 两边都有中深层早期压缩；
- L22 后期都持续低于基线；
- 两边的适应 delta 都非负；
- 但主相关符号从 -0.60 变 +0.50。

这说明单条 n=5 轨迹的相关方向很不稳定。

## 6.7 Stage-B seed repeats：只换适应随机性

固定 Windows v2 的五个 Stage-A checkpoint，一个字节不改，只重跑 Stage B：

- seed 42：+6/+11/+3/+5/+12；
- seed 43：+4/+8/+9/+6/+5；
- seed 44：只完成 ckpt-0，delta=0；其余四点未跑。

seed 42 和 43 的 checkpoint delta 排名 Spearman 约 -0.50。把 seed 43 代入同一个 Q，描述性主相关变成 -0.50。

它意味着：

- 只换 Stage-B seed 就足以翻转 checkpoint 排名；
- Mac vs Windows 的相关符号差异不需要用“某台机器坏了”解释；
- 多个 Stage-B seeds 必不可少；
- 这些 repeats 共用同一条 Stage-A 轨迹，不能当作独立 Stage-A 样本。

seed-44 后续被正式关闭，原因是 Experiment 1.5 已用 300 题×3 seeds 的更强设计取代它；未完成的三 seed 汇总分析从未生成。

## 6.8 Evidence pack

`experiment 1/pilot_evidence_pack/` 把 Mac、Windows v2、seed 43/44 的小型可审计产物、配置、测量、适应 summary、分析和 compute log 打包。它是可复核快照，不是一轮训练。

---

# 第七篇：Original Pilot 到底说明了什么

## 7.1 可以确定的

1. 有效 metrics 在相同 measurement contract 下跨平台 ckpt-0 高度一致；例如 L12 erank 均约 231.7567。
2. 早期 L12/L22 谱压缩在两条有效 stratum 都出现。
3. L22 到后期仍低于基线，幅度约 8–11%。
4. dormant fraction 在当前层和阈值下没有变化。
5. 完成的 16 次固定预算适应是 15 正、1 零、0 负。
6. n=5 单 seed Spearman 对执行轨迹和 Stage-B seed 很敏感。
7. bf16 master-weight no-op 是足以让整轮实验失效的真实风险。

## 7.2 不能确定的

- 不能说 RLVR 普遍降低未来学习能力；
- 不能说 erank 下降就是 plasticity collapse；
- 不能说 rho 的正负方向稳定；
- 不能说 SVAMP 是合适的“新任务家族”；
- 不能比较 Q 与 dashboard 的 lead time，因为没有足够 stall 正样本和密集 Q；
- 不能把 checkpoint 当成独立样本估计普遍规律。

## 7.3 Pilot 暴露的三个核心问题

1. **剂量不足**：`lr=1e-6 × 200` 没有制造持续适应力下降。
2. **outcome 噪声大**：100 题×1 seed 的 delta 排名不稳定。
3. **任务太近**：GSM8K 训练可能直接提高 SVAMP 起点，污染 delta。

Experiment 1.5 就是针对前两项的预注册升级，并进一步验证第三项。

---

# 第八篇：Experiment 1.5 为什么出现

## 8.1 原冻结计划

【预注册计划，2026-07-16】核心目标是：让 Stage A 更强，同时让 Stage-B outcome 更稳。

| 旋钮 | Pilot | 1.5 原计划 | 目的 |
|---|---:|---:|---|
| Stage-A lr | 1e-6 | 1e-5 | 提高剂量 |
| Stage-A updates | 200 | 500 | 覆盖 reward 饱和后区间 |
| Q checkpoint | 5 个 | 8 个：0/25/50/100/200/300/400/500 | 看更细轨迹 |
| Stage-B eval | 100 | 300 | 降低二项评估噪声 |
| Stage-B seeds | 1 | 3：42/43/44 | 降低 rollout/训练随机性 |
| 适应 checkpoint | 5 个 | 6 个：0/50/100/200/300/500 | 预算内提高 n |

保持不变：模型、数据、reward、beta=0、8 generations、temperature、probe、层、Stage-B 50 步预算和大部分执行 profile。

## 8.2 原预注册的 MC1 / MC2

### MC1：Q 是否被剂量明显推动

```text
|erank_L12(500) / erank_L12(0) - 1| >= 10%
```

### MC2：是否出现相对 ckpt-0 的固定预算适应下降

至少一个后期 checkpoint 的三 seed 平均 delta 比 ckpt-0 低 ≥5pp。

### 2×2 解释

| | MC2 通过 | MC2 失败 |
|---|---|---|
| MC1 通过 | Q 与 outcome 的相关可作为有效 RQ1 读数 | Q 动、适应不降：任务太近或 Q 不跟踪损伤 |
| MC1 失败 | 适应下降但端点 erank 无信号：反对该 erank 定义 | 剂量仍不足 |

重要：这些规则是在结果前冻结的，避免看到数据后才改判定。

---

# 第九篇：Experiment 1.5 的 v1 / v2 / v3

## 9.1 为什么 1.5 有三个版本

它们不是为了挑一个“好看结果”。每一版都由前一版的已保存安全事件触发，并写了 amendment；旧 run 不覆盖。

| 版本 | Stage-A lr | 终点 | 性质 |
|---|---:|---|---|
| v1 `e737…` | 1e-5 | step 7 停 | clipping hard-stop 与冻结计划不一致，保留现场 |
| v2 `dd5f…` | 1e-5 | step 55 停 | clipping 改诊断后，捕获真实 policy collapse |
| v3 `c7cc…` | 1e-6 | 500/500 | 单变量降回已验证 lr，完整完成 |

## 9.2 v1：step 7 为什么停

【已完成事实】连续 5 步 completion clipping 超过 10% 触发 hard stop；step 7 的 clipped ratio 为 0.5625。此时：

- loss 和 grad norm 有限；
- reward mean 0.171875、std 0.380；
- entropy 0.244；
- GRPO 仍有组内 reward variance。

冻结计划/Windows guide 原本没有把 clipping 列为硬停条件，因此 Amendment v2 把它改为“继续记录但不因它单独停”。v1 不能回答高剂量最终会怎样，但证明长回答顶到 512 token 很早就成为危险信号。

## 9.3 v2：本项目第一次真实捕获 policy collapse

【已完成事实】同样 `lr=1e-5`，clipping 只作诊断后：

- preflight 通过；
- step-25、step-50 sentinel 通过，证明权重在动；
- step 37–47 还有零星正确 reward；
- completion clipping 逐渐接近 1.0；
- mean completion length 接近 512 token 上限；
- entropy 从约 0.20 降到约 0.05；
- step 51–55 连续 5 步组内 reward variance=0；
- loss=0、grad norm=0；
- step 55 安全停。

这叫 policy collapse：模型不是因为 GPU 坏了或数值 NaN，而是生成策略退化到没有可用 reward 差异，GRPO 无法继续学习。

它对项目非常重要，因为这是真正的“需要提前预警的坏事件”。但它的 checkpoint 只有 0/25/50，崩溃窗口太稀，无法判断 Q 是否比 entropy 更早报警。

## 9.4 v3：为什么又把 lr 降回 1e-6

Amendment v3 只改 Stage-A learning rate `1e-5→1e-6`，其他科学与执行条件保持。原因：

- 1e-5 在 55 步内真实崩溃；
- beta=0 保持不变，避免同时改 lr 和 KL 后无法判断哪个防止崩溃；
- 1e-6 已在 pilot v2 完成 200 步。

所以 v3 的真实含义是：**把训练时长从 200 延长到 500，并保留 300 题×3 seed 的降噪设计。它不再是原计划的 10× lr 高剂量实验。**

## 9.5 v3 Phase 2 的 float16→float32 测量恢复

第一次 Phase 2 用 float16 测量，ckpt-0 与 pilot fp32 reference 比较时被 gate 拦下：

- L4 erank 差 0.391；
- L12 差 0.294；
- L22 差 0.031。

权重 norm 一致，说明是 measurement dtype mismatch，不是模型错了。处理：

1. 原 float16 测量完整归档；
2. 不重训 Stage A；
3. 用 float32 重测 8 个 checkpoint；
4. ckpt-0 identity gate 通过后才继续适应。

它证明“固定 dtype”不是文档装饰，而是会实际影响 erank 可比性的硬合同。

## 9.6 v3 Stage A 的真实轨迹

【已完成事实】500/500 步完成，无 safety stop，sentinel 全过。

每 100 步窗口的训练 reward / entropy：

| 步数 | reward 均值 | entropy 均值 | clipping 均值 |
|---|---:|---:|---:|
| 1–100 | 0.444 | 0.200 | 0.009 |
| 101–200 | 0.566 | 0.148 | 0.014 |
| 201–300 | 0.624 | 0.107 | 0.009 |
| 301–400 | 0.675 | 0.086 | 0.005 |
| 401–500 | 0.681 | 0.084 | 0.005 |

约 300 步后 reward 进入平台，但训练保持健康，没有出现 v2 的 clipping=1、reward variance=0 的崩溃形态。

## 9.7 v3 Q 轨迹

| ckpt | erank L4 | erank L12 | erank L22 |
|---:|---:|---:|---:|
| 0 | 225.14 | 231.76 | 354.19 |
| 25 | 223.47 | 217.32 | 335.02 |
| 50 | 223.42 | 215.15 | 323.94 |
| 100 | 222.41 | 225.12 | 323.60 |
| 200 | 225.02 | 235.77 | 322.90 |
| 300 | 226.16 | 228.43 | 327.51 |
| 400 | 225.68 | 228.87 | 326.71 |
| 500 | 225.65 | 229.50 | 327.42 |

读法：

- L4 基本稳定；
- L12 在 ckpt-50 暂时下探约 7.2%，之后恢复，500 时只比基线低约 0.98%；
- L22 早期下探后维持约 7.5–8.8% 压缩；
- dormant fraction 所有点、所有层、两个阈值都为 0；
- 2048-probe 端点 sensitivity check 已完成。

## 9.8 v3 的 18 个固定预算适应

6 checkpoint × 3 seeds = 18 个适应格全部通过 validator。一次 seed42/ckpt-100 OOM 被保留，随后新进程同配方重跑；最终结果完整。

| ckpt | SVAMP 适应前 | 三 seed 平均 delta | seed SD |
|---:|---:|---:|---:|
| 0 | 53.0% | +5.7pp | 2.7pp |
| 50 | 62.3% | -2.9pp | 1.3pp |
| 100 | 61.0% | +0.2pp | 0.8pp |
| 200 | 54.7% | +6.9pp | 3.4pp |
| 300 | 53.3% | +8.6pp | 2.2pp |
| 500 | 56.0% | +6.8pp | 1.8pp |

## 9.9 预注册 Phase-4 判定

【已完成事实】

- MC1：FAIL。L12 端点相对变化 -0.98%，没有达到绝对值 10%；
- MC2：PASS。ckpt-50 平均 delta 比 ckpt-0 低 8.6pp；ckpt-100 低 5.4pp；
- 主相关：rho=+0.60，p=0.208，n=6；
- per-seed rho：+0.43 / +0.71 / +0.90；
- seed 两两 delta 排名相关：0.55–0.71；
- checkpoint 间方差 0.00202，seed 内方差 0.00049，约 4:1；
- 预注册落格：MC1 fail + MC2 pass，按规则是“适应结果下降但端点 erank_L12 无对应操纵信号”，记为反对该端点 erank 定义的证据。

## 9.10 为什么 MC2 PASS 不能直接叫“plasticity loss”

【描述性解释】MC2 的下降是 V 型，不是持续恶化：

- ckpt-50 最差；
- ckpt-200/300/500 恢复并高于 ckpt-0；
- ckpt-50 恰好有最高适应前准确率 62.3%，比 ckpt-0 高 9.3pp；
- `delta` 与 `svamp_before` 的描述性 Spearman 约 -0.66。

这符合 ceiling/headroom 混杂：GSM8K 早期训练直接迁移到 SVAMP，让起点变高、可涨空间变小。也就是说，MC2 的数学阈值确实通过，但“它代表持久可塑性损伤”的构念解释被任务相似性削弱。

## 9.11 Experiment 1.5 真正带来的四个成果

1. **捕获了真实崩溃正样本**：v2 的 dashboard 前兆完整在档。
2. **证明 300 题×3 seed 能显著改善 outcome 稳定性**：排名相关从 pilot 的负值提升到 0.55–0.71，between/within variance 约 4:1。
3. **证明温和 1e-6×500 仍没有持久端点 Q 或适应力退化**。
4. **把 SVAMP-too-close 从疑虑升级为数据支持的混杂问题**。

---

# 第十篇：Experiment 1.5.1 是做什么的

## 10.1 状态

【预注册计划】截至 2026-07-19，计划和三个 config 已冻结，但正式 run 尚未出现。它不是结果。

## 10.2 它要回答的唯一问题

v2 在 step 55 崩溃，entropy 急跌只提前约 3 步。1.5.1 要把 Q checkpoint 加密到每 5 步，直接问：

> 崩溃前，erank 或 dormant fraction 是否在健康 run 的正常波动范围之外提前变化？如果有，是否至少比 entropy 早 5 步以上？

## 10.3 为什么它不是又做一次 1.5

- 不做 Stage B 适应；
- 不测 future adaptability；
- 只做 Stage-A collapse forensics；
- 目标是 detector lead-time，而不是 RQ1 相关；
- 运行到安全停或 step 80 hard cap 即结束；
- `max_steps` 仍保留 500，确保 learning-rate schedule 与 v2 相同，只用 hard cap 截断。

## 10.4 三条预注册判定

- 事件 E：连续零 reward variance 段的第一步；
- SC1：在 `c <= E-10` 的测量点，L12 erank 相对基线 ≤-12%；
- SC2：任一层、任一阈值 dormant fraction >0；
- SC3：SC1 成立，并且 Q lead 至少比 entropy lead 多 5 步；
- 跨 replicate：至少 2/3 个发生崩溃的 run 成立，才说 pilot 规模支持。

为什么阈值 -12%：健康 run 在前 100 步的 L12 最深下探约 -5% 到 -8.6%；-12% 预先放在健康包络以外，避免看到数据后随意画线。

## 10.5 三个 seed 的意义

seed 42 近似重放 v2，43/44 测崩溃时点和 Q 形态是否依赖轨迹。如果某个 run 到 step 80 仍不崩，它被记为 censored，而不是失败；这本身说明高 lr 崩溃 hazard 有随机性。

## 10.6 四种可能结局

| 结局 | 解释 |
|---|---|
| SC1+SC3 支持 | Q 提前越出健康包络，并早于 entropy，是真正候选预警信号 |
| SC1 支持、SC3 不支持 | Q 有异常，但只是确认信号，不比 dashboard 早 |
| SC1 不支持 | activation Q 没有提前信号，是对核心假设的干净反面证据 |
| 至少 2 个 run 不崩 | 崩溃 hazard 强烈依赖轨迹，需要重新估计发生率/扩大 replicate |

---

# 第十一篇：Experiment 1.6 是做什么的

## 11.1 状态

【预注册计划】计划已冻结，但有硬前置条件：Tommy/团队必须明确批准本地 `lr=3e-6` 探针，launch commit 必须带 `team-ack:`。截至当前没有正式 run 目录，所以不能写成已启动或已有结果。

## 11.2 为什么选 3e-6

现有剂量地图：

- 1e-6×500：稳定，但没有持久 Q/适应下降；
- 1e-5：55 步内崩溃；
- 中间区域没有测。

3e-6 是对数量级的几何中点附近：比 1e-6 强约 3 倍，比 1e-5 弱约 3 倍。它试图寻找“不直接崩溃，但足以留下持久变化”的窗口。

## 11.3 唯一科学变量

相对 exp1.5 v3，只改 Stage-A lr：`1e-6→3e-6`。模型、数据、beta、500 步、checkpoint、float32 Q、300 题×3 seed 适应等保持。

## 11.4 为什么 endpoint-first

18 个适应格约需 21 GPU 小时。1.6 先只跑 checkpoint 0 和 500，各 3 seed，共 6 格，然后门控：

- G-A：late-window `{300,400,500}` 的 L12 erank 均值相对 ckpt-0 位移绝对值 ≥7.5%；
- G-B：ckpt-0 的三 seed mean delta 比 ckpt-500 至少高 5pp；
- G-A 或 G-B 过，才扩展到完整 6-checkpoint grid；
- 两门都失败就停止，把 3e-6 记录为仍不足的剂量点。

这叫“端点先行 + 门控扩格”：先用最小 outcome 成本判断是否值得补中间点。

## 11.5 1.6 仍然解决不了什么

即使 3e-6 找到差异，GSM8K→SVAMP 的 before/headroom 混杂仍存在。预注册要求任何相关读数都同时报告控制 `svamp_before` 的版本。更彻底的解决方案仍是更远的任务家族，例如 proposal 的 ProntoQA 方向。

---

# 第十二篇：所有“版本”放在同一张地图里

## 12.1 科学问题演化

```text
Proposal：Q 能否提前预测正式 Stage-B stall，并胜过 dashboard？
    |
    v
Experiment 1 pilot：小模型管线能否工作？Q 与 50-step SVAMP delta 是否有关？
    |
    +--> Windows v1：训练近似 no-op，发现精度陷阱与 sentinel 必要性
    |
    +--> Mac / Win v2：谱压缩复现，但 n=5 单 seed 相关不稳定
    |
    +--> Stage-B repeats：只换 seed 就翻排名，确认 outcome 噪声问题
    |
    v
Experiment 1.5：提高剂量并用 300 题×3 seed 降噪
    |
    +--> v1：clipping hard-stop 规则修正
    +--> v2：lr=1e-5 在 step 55 真实 policy collapse
    +--> v3：lr=1e-6 跑满 500；降噪成功，但无持久损伤，且暴露任务迁移混杂
    |
    +--> Experiment 1.5.1：密集 Q 取证，问 Q 是否比 entropy 更早（待跑）
    |
    +--> Experiment 1.6：试 3e-6 中间剂量，寻找不崩但有持久变化的窗口（待团队批准）
    |
    v
未来 Experiment 2：更大模型、更远任务、多个独立 Stage-A seeds、正式 stall label 与 detector bake-off
```

## 12.2 三条不同的证据线，不要混在一起

### 线 A：测量可靠性

ckpt-0 跨平台一致、dtype gate 拦截、2048 sensitivity、manifest/validator。结论：仪器本身越来越可信。

### 线 B：训练病理

1e-5 下 clipping、entropy 急跌、reward variance 归零；1e-6×500 健康。结论：已有真实 collapse 正样本和剂量边界。

### 线 C：未来适应 outcome

100 题×1 seed 噪声大；300 题×3 seed 稳定得多；但 SVAMP before 改变造成 ceiling 混杂。结论：量尺更稳，但“测的构念”仍需修正。

---

# 第十三篇：结果应该怎么读，最常见误读是什么

## 13.1 “erank 下降了，所以模型失去塑性”——错

正确表述：某些层的激活谱压缩了。是否代表固定预算未来适应量下降，需要独立 outcome 验证。

## 13.2 “rho=+0.60，所以 Q 有效”——证据不足

n=6、来自同一轨迹、p=0.208，而且 V 型可能由迁移/headroom 共因驱动。它是值得追的形状，不是 detector 成功证明。

## 13.3 “MC2 通过，所以出现 plasticity collapse”——错

MC2 的操作阈值通过，但最差点同时拥有最高适应前准确率，后期又完全恢复。它更像暂态+天花板混杂，不能升级成一般学习能力结论。

## 13.4 “v1 四个 phase 都完成，所以也能算一条 run”——错

科学 treatment 近似没有施加。完成文件不能替代权重变化、reward 学习和 sentinel 证据。

## 13.5 “Mac rho 和 Windows rho 方向相反，说明硬件改变科学规律”——证据不足

Windows v2 有多次 resume，执行 profile 不同，而且只换 Stage-B seed 也能把 rho 从 +0.50 变 -0.50。首先应归因于小样本与 on-policy/适应噪声，而不是硬件规律。

## 13.6 “dormant fraction 恒 0，所以 dormant 指标理论上没用”——过度外推

只能说：在 0.5B、层 4/12/22、当前 post-activation 定义、tau 0.025/0.1 和现有剂量下没有区分度。更大模型、其他层、阈值或训练病理可能不同。

## 13.7 “1.5 失败了”——不准确

原高 lr 计划没能稳定跑到 500，但产生了最有价值的 collapse 正样本；v3 完成了降噪和长时程测试；结果还直接决定了 1.5.1/1.6。它没有验证原假设，但显著提高了后续实验设计质量。

---

# 第十四篇：如果你第一次打开 run 目录，应该看什么

## 14.1 先确认身份

- `config.json`：科学和执行配方；
- `manifest.json`：模型/data/git/config hash；
- `phase*_complete.json`：哪些 phase 真完成；
- `safety_stop.json`：是否以预期外安全事件终止。

## 14.2 再确认训练真的发生

- `dashboard.jsonl`：reward、entropy、grad norm、completion；
- `phase1_update_sentinel.jsonl`：权重是否有效改变；
- `gsm8k_eval.jsonl`：held-out 表现轨迹。

## 14.3 再看 Q

- `measurements/metrics_ckpt*.json`；
- 先检查 `n_probe`、`layers`、`model_dtype_requested`、`measurement_contract`；
- 不同 probe 大小的 absolute erank 不直接混比；
- sensitivity_2048 只与同 probe 大小的端点比较。

## 14.4 最后看适应与主分析

- `adaptation_seed*/ckpt-N/baseline.json`；
- `summary.json`；
- `svamp_eval_curve.jsonl`；
- `analysis/results_table.csv`；
- `analysis/analysis_summary.json`。

永远同时看 before、after、delta，不要只看 delta。

---

# 第十五篇：当前最诚实的知识边界

## 15.1 已经知道

1. 测量和 artifact 管线可跨平台复核，并能拦下 dtype 不可比问题。
2. pure-bf16 master weights + lr=1e-6 可让训练近似 no-op。
3. 0.5B、beta=0、lr=1e-6 的健康 run 中，L12 早期下探和 L22 持续压缩可重复出现。
4. 这些谱变化尚未被证明等价于固定预算适应力下降。
5. `lr=1e-5, beta=0` 在本设置中可在约 55 步内发生策略崩溃。
6. 100 题×单 seed 不足以稳定排列 checkpoint；300 题×3 seed 明显更稳。
7. GSM8K→SVAMP 的直接迁移足以污染 delta/headroom。
8. dormant fraction 在当前设置中没有信息量。

## 15.2 还不知道

1. 崩溃前 Q 是否比 entropy 更早；1.5.1 专门回答。
2. 1e-6 与 1e-5 之间是否存在“不崩但可塑性持续下降”的窗口；1.6 尝试回答。
3. 更远任务家族上是否仍有 V 型和 headroom 混杂。
4. 多个独立 Stage-A seeds 上 Q-outcome 关系是否稳定。
5. 正式 stall label 下 Q 的 AUROC、AUPRC、ECE 和 lead time。
6. Q 能否胜过 reward/KL/grad-norm/entropy dashboard。
7. 保护 Q 的 intervention 是否能因果性地防止 stall。

## 15.3 当前推荐的研究路径

按已经冻结的治理规则：

1. 先跑不需团队批准、成本低、直接利用现有 collapse 的 1.5.1；
2. 1.6 只有团队 ack 后才能启动；
3. 正式 Experiment 2 应优先解决任务距离、独立 Stage-A seeds、密集崩溃窗口采样和正式 stall reference；
4. 保留 dashboard bake-off，不能因为 Q 是项目主题就弱化对照组。

---

# 第十六篇：零基础 FAQ

## Q1：为什么不直接看最终考试分数？

因为最终分数只能在训练花完后看到。项目要的是提前预警，而且最终准确率还会混入任务难度、起点和直接迁移。

## Q2：为什么每个 checkpoint 都要先考一次？

因为不同 Stage-A checkpoint 的 SVAMP 起点不同。没有 before，就不知道 after 高是因为本来就高，还是 Stage B 学得多。

## Q3：既然有 before，为什么还会有天花板混杂？

`after-before` 修正了起点，但没有修正剩余上升空间。例如 95 分最多只能涨 5 分。需要换更远任务、建模 before 协变量或设计等起点对照。

## Q4：为什么只看排名相关，不看直线拟合？

Pilot 只有 5/6 点，且不保证线性。Spearman 只看排序，假设更少。但点太少时它也很不稳定。

## Q5：p>0.05 是不是证明没有关系？

不是。它表示当前样本不足以排除“无关系下的随机结果”。n=5/6 的统计功效非常低；不显著既不是证明有关系，也不是证明绝对没关系。

## Q6：为什么 checkpoint 不是独立样本？

它们来自同一训练轨迹，ckpt-100 包含了走到 ckpt-50 的历史。相邻点高度相关，不能当作六个独立模型世界。

## Q7：为什么要多 Stage-A seed 又要多 Stage-B seed？

Stage-A seed 控制“生成了哪条模型轨迹”；Stage-B seed 控制“从固定 checkpoint 学新任务时的随机结果”。两种噪声不同，必须分层重复。

## Q8：为什么高学习率崩溃也算科学结果？

因为项目研究的就是训练失速/坍缩。只要不是基础设施错误、事件被完整记录、规则预先定义，它就是重要正样本。不能为了跑满步数而静默降低 lr 或改 reward。

## Q9：为什么 1.5.1 不做 Stage B？

它只问崩溃前谁先报警。加 Stage B 会耗费大量算力，却不改善这个 lead-time 问题。

## Q10：为什么 1.6 需要团队批准？

它是 exp1.5 预注册落格后的科学升级路线之一，会消耗约 20–34 GPU 小时；中间 lr、KL arm、更大模型、Colab 等路径需要团队共同排序，Person 4 不能静默决定。

---

# 第十七篇：术语速查

| 术语 | 大白话 |
|---|---|
| parameter / weight | 模型内部可调数字 |
| activation | 模型处理一个输入时产生的内部向量 |
| forward pass | 只让模型计算输出，不改权重 |
| backward pass | 根据 loss 计算梯度 |
| optimizer | 把梯度变成实际权重更新的规则 |
| learning rate | 每次更新的步幅尺度 |
| update / step | 一次优化器权重更新 |
| checkpoint | 某一步保存的模型快照 |
| RLVR | 用可自动验证的 reward 做强化学习 |
| GRPO | 同题多回答、组内相对比较的 RL 方法 |
| beta / KL | 限制策略偏离参考模型的强度；beta=0 表示不加该惩罚 |
| full fine-tune | 更新全部模型参数 |
| LoRA | 只训练小型低秩适配参数，通常更便宜 |
| probe set | 固定用于内部测量的输入集 |
| effective rank | 激活矩阵有效使用多少谱方向的指标 |
| dormant fraction | 相对激活极低的神经元比例 |
| entropy | 输出分布不确定性/多样性的一个量 |
| clipping | 生成撞到最大 token 长度而被截断 |
| sentinel | 检查权重是否真的有效改变的哨兵 |
| fixed-budget adaptation | 对所有起点给完全相同的新任务训练预算 |
| delta | 适应后准确率减适应前准确率 |
| Spearman rho | 两组数的排名一致程度 |
| AUROC/AUPRC | 二分类 detector 的区分能力指标 |
| ECE | 预测概率是否校准的指标 |
| lead time | 预警比可见失败提前多少步 |
| confound | 让观察关系产生其他解释的混杂因素 |
| pre-registration | 结果前冻结设计、阈值和分析 |
| stratum | 一种执行环境下的一条完整 run |
| censored | 到观察上限仍未发生事件，不能当普通失败 |

---

# 第十八篇：事实来源与文档地图

## 18.1 权威顺序

1. `team_doc/proposal_v3.1_formal.docx`：正式 proposal，尤其 Method、Datasets、Metrics、Baselines/Ablations；
2. Tommy 的团队规格；
3. `agent_doc/AGENT_BRIEFING_first_experiment.md`：pilot 浓缩规格；
4. 冻结 config、run 目录 JSON/CSV、amendment、compute log；
5. 汇总报告与本入门文档。

若叙述与原始产物冲突，以原始 config/JSON 和正式 proposal 为准。

## 18.2 关键文件

| 内容 | 路径 |
|---|---|
| 正式 proposal | `team_doc/proposal_v3.1_formal.docx` |
| Pilot briefing | `agent_doc/AGENT_BRIEFING_first_experiment.md` |
| Pilot 配方 | `eaaj-pilot/pilot_config.json` |
| Pilot facts-only 报告 | `PILOT_COMBINED_REPORT_3EXPERIMENTS_ZH.md` |
| Windows v1 诊断 | `eaaj-pilot-win4070/WIN4070_RUN_ANALYSIS.md` |
| Windows v2 报告 | `eaaj-pilot-win4070/WIN4070_V2_FINAL_REPORT_ZH.md` |
| Stage-B repeat 状态 | `eaaj-pilot-win4070/WIN4070_STAGEB_SEED_REPLICATION_STATUS_ZH.md` |
| Pilot evidence pack | `experiment 1/pilot_evidence_pack/` |
| Exp1.5 原计划 | `experiment 1.5/EXPERIMENT_1_5_PLAN_ZH.md` |
| Exp1.5 修正 | `EXPERIMENT_1_5_AMENDMENT_V2.md`、`V3.md`、`MEASUREMENT_RECOVERY.md` |
| Exp1.5 完整分析 | `EXP1_5_RESULTS_ANALYSIS_ZH.md` |
| Exp1.5 v3 主分析 JSON | `eaaj-pilot/outputs/exp15_cuda_grpo_gsm8k_c7cc7a1d02d9/analysis/analysis_summary.json` |
| Exp1.5.1 计划 | `experiment 1.5/EXPERIMENT_1_5_1_PLAN_ZH.md` |
| Exp1.6 计划 | `experiment 1.5/EXPERIMENT_1_6_PLAN_ZH.md` |
| 计算账本 | `eaaj-pilot/compute_log.md` |

## 18.3 最终一句话

Experiment 1 系列到现在最重要的进展，不是“已经证明 Q 能预警”，而是把一个模糊假说拆成了可审计的测量、噪声、剂量、崩溃事件和决策门：**我们已经知道仪器怎样才算可信、哪种运行是假训练、哪里会直接崩、怎样把 outcome 噪声压低、以及下一轮必须如何区分真正 capacity loss 与数学任务迁移。** 这使后续正式 detector 实验有可能给出可信的正结果或可信的反结果。
