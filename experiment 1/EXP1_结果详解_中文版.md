# 实验一 结果详解（事实核验修订版·零基础）

**作者：** Aaron（组内负责“早期预警诊断”）  
**核验日期：** 2026-07-23  
**配套数据：** `exp1_result_aaron.zip` → 解压后的 `pilot_evidence_pack/`  
**配套英文版：** `EXP1_RESULTS_EXPLAINED.md`  
**范围：** 只讲 Experiment 1，不混入 Experiment 1.5 的结果

> 这份版本是逐文件核验后的修订版。最重要的新发现是：Mac 的 checkpoint-25
> 适应实验实际上只跑到 30/50 步，却被旧版 summary 和完成标记误算成了完成。
> 因此，旧文档里的“16 个完成适应”和 Mac 五点主相关都需要降级或更正。

---

## 先看结论：哪些成立，哪些要改

已经被原始工件支持的结论：

1. **第一阶段确实训练了模型。** 两台机器的平均奖励都上升；Windows 的权重变化哨兵
   在每个 25 步窗口都通过。
2. **休眠比例没有量程。** 两个阈值、三个层、五个 checkpoint、两台机器，结果全是 0。
3. **第 22 层有效秩会动。** 训练后下降约 8%–14%，两条执行轨迹都复现了这个方向。
4. **预测关系不稳定。** 同一组 Windows checkpoint，只换 Stage B 随机种子，
   主相关就从 `+0.50` 变成 `−0.50`。

必须更正的地方：

1. **不是 16 个合格的 50 步适应，而是 15 个。** 合格的 15 个里，14 个终点为正、
   1 个持平、0 个为负。
2. **Mac 的 `ρ = −0.60` 不能再叫有效的固定 50 步主结果。** 它用了一个只跑 30 步的点，
   只能作为旧版描述性数字保留。
3. **Q 测量存在预注册偏差。** 共享配方写的是 float16；实际两台机器都使用
   float32 模型/激活累加和 float64 SVD。实际协议彼此一致，但必须披露它与配方不一致。

**最安全的一句话：** 实验一把训练、Q 测量和 Stage B 适应流程走通了，也发现了
有效秩的可测变化；但它既没有证明有效秩能可靠预测未来适应，也没有显示相对
checkpoint 0 可复现的固定预算适应力下降。RQ1 仍然开放。

---

## 第 0 节：我们到底在研究什么

推理模型经常分多个阶段训练：先学任务 A，再学任务 B。项目关心的是：

> 在第一阶段的某个 checkpoint 上测到的内部表征指标 Q，能不能提前预测这个
> checkpoint 在未来任务上、固定训练预算内能进步多少？

项目把“未来任务、训练预算、比较基线”都提前钉死，避免使用模糊的“模型还能不能学”
这种说法。本实验的可测量问题是：

- 未来任务：SVAMP；
- 固定预算：50 个 GRPO 更新；
- 比较方式：每个 checkpoint 的适应后准确率减去它自己的适应前准确率；
- 主要问题：`erank_L12` 与这个进步值的 checkpoint 排序是否稳定相关。

“可塑性坍缩”是项目研究的现象名称，但目前没有一个被普遍接受的单一标量能直接代表
“可塑性”。所以，有效秩和休眠比例在这里只是**候选诊断指标**，不是已经被验证的
“可塑性体温计”。

---

## 第 1 节：实验怎么做

### 1.1 模型、任务和 checkpoint

模型是固定 revision 的 `Qwen/Qwen2.5-0.5B` base model。Stage A 用 512 道冻结的
GSM8K 题做 200 个 GRPO 更新：最终答案完全正确给 1 分，否则给 0 分。我们保存了
第 0、25、50、100、200 步的 checkpoint。

第 0 步是未经 Stage A 训练的源模型；后面的 checkpoint 表示经历了不同长度的
GSM8K 强化学习。

### 1.2 Q 是怎么测的

每个 checkpoint 都在同一组 512 个冻结 prompt 上测量，模型处于 eval mode，
层固定为 4、12、22。

- **有效秩（effective rank）**：衡量一组隐藏表示的谱有多少有效方向。它是候选的
  表征容量诊断；“数值下降”本身不等于“未来一定学不动”。
- **休眠比例（dormant fraction）**：统计 MLP 激活分数低于阈值的单元比例，
  阈值为 `0.025` 和 `0.1`。

实际测量工件记录的是：最后一个非 padding token 的隐藏状态、float32 模型与激活累加、
float64 SVD；休眠分数对所有非 padding token 的 MLP post-activation 做平均。

### 1.3 固定预算适应测试

从每个 Stage-A checkpoint 出发：

1. 先在同一组 100 道冻结 SVAMP eval 题上做 greedy 准确率；
2. 用同一组 256 道冻结 SVAMP train 题做 50 个 GRPO 更新；
3. 每 10 步评估一次，计划曲线为 10/20/30/40/50；
4. 记录 `Δ = 适应后准确率 − 适应前准确率`。

评估题、解码方式和预算相同，让不同 checkpoint 尽可能可比。不过 SVAMP 与 GSM8K
很接近，Stage A 本身就可能提高 SVAMP 起点，造成“起点越高、剩余提升空间越小”的
天花板混淆。

---

## 第 2 节：到底跑了几条轨迹

| 证据目录 | Stage A 来源 | Stage B 种子 | 执行方式 | 核验状态 |
|---|---|---:|---|---|
| `mac_run/` | 单独的 seed-42 执行 | 42 | CPU，float32 | 4 个完整 50 步；ckpt-25 只有 30 步 |
| `win_run/adaptation_seed42/` | 一条 Windows seed-42 轨迹 | 42 | RTX 4070，fp32 master + bf16 autocast | 5/5 手工工件核验完整 |
| `win_run/adaptation_seed43/` | 复用同一组 Windows checkpoint | 43 | 同一 Windows profile | 5/5 通过后加的严格 validator |
| `win_run/adaptation_seed44/` | 复用同一组 Windows checkpoint | 44 | 同一 Windows profile | 只完成 ckpt-0 |

Mac 和 Windows 应叫**两个不同的执行层（execution strata）**，不能叫可互换的统计重复：
它们虽然共享模型 revision、种子和逻辑上相同的冻结 split，但硬件、数值精度、
micro-batch 几何和 optimizer 实现不同。

三个 Windows Stage B 种子也不是三条独立 Stage A 样本；它们全部复用同一条
Windows Stage A 轨迹。那条轨迹在 25/50/75 附近经历过修复式续跑，optimizer moments
和 RNG state 没有完整恢复，所以不等同于一条不中断的训练轨迹。

### 2.1 配方和证据包的可审计性

- `recipe/pilot_config.json` 固定了核心模型、任务、seed、LR、beta、温度、checkpoint
  和名义预算。
- 依赖版本和输入哈希在各自 `manifest.json`；机器执行差异在 `config.json`。
- Mac/Windows 的 split 哈希表面不同，是因为 Windows 使用 CRLF 换行；换行归一化后，
  逻辑内容哈希对应。
- Slack zip 没有附上 split 文件本身和源码，只附哈希；队友需要仓库才能从哈希检查
  精确 split 成员。
- Mac manifest 的 `git_sha` 是 null，只有文件哈希，commit 级 provenance 比 Windows 弱。
- 模型权重和 optimizer state 不在 zip 中。这个包能审计表格和日志，不能单靠 zip
  从头复现推理。
- 共享配方写 Q dtype 为 float16，而实际两条轨迹的 metric 文件都记录为
  float32 模型/累加、float64 SVD。这是已经记录但需要公开说明的执行偏差。

---

## 第 3 节：结果一——Stage A 真的改变了模型

| 执行层 | 前 10 → 后 10 步平均 reward | 第 1 → 200 步 entropy | GSM8K 第 0 → 200 步准确率 | 稀疏奖励预检 |
|---|---:|---:|---:|---:|
| Mac | 0.364 → 0.653 | 0.253 → 0.137 | 0.438 → 0.516 | 7/8 组有组内差异 |
| Windows | 0.334 → 0.588 | 0.247 → 0.116 | 0.359 → 0.422 | 8/8 组有组内差异 |

GSM8K 准确率中间有上下波动，并不是单调上升；但两条轨迹的第 200 步都高于第 0 步。
Windows 的八个 25 步权重哨兵窗口全部通过，每窗口相对权重变化从 `6.58e-6`
逐渐降到 `3.23e-7`，都高于 `1e-8` 的报警阈值。

所以可以说“Stage A 更新是有效的”，但不能由此说“两个机器跑出了同一条训练轨迹”。

---

## 第 4 节：结果二——Stage B 完成度审计

| Stage-A checkpoint | Mac s42 | Windows s42 | Windows s43 | Windows s44 |
|---:|---:|---:|---:|---:|
| 0 | +0.05 | +0.06 | +0.04 | +0.00 |
| 25 | **+0.13†** | +0.11 | +0.08 | — |
| 50 | +0.08 | +0.03 | +0.09 | — |
| 100 | +0.02 | +0.05 | +0.06 | — |
| 200 | +0.04 | +0.12 | +0.05 | — |

† **不能用于固定 50 步比较。** Mac checkpoint-25 的 dashboard 只到 step 30，
学习曲线只有 10/20/30，最后 trainer checkpoint 也是 30。旧 `summary.json` 里的
`budget_updates: 50` 只是“请求跑 50 步”，不是“实际完成 50 步”的证明。

因此必须区分两个计数：

- 压缩包里存在 16 份 endpoint summary：15 正、1 平、0 负；
- 真正可以核实为 50 步的只有 15 份：**14 正、1 平、0 负**。

15 个合格 cell 中没有负的终点 delta；而且没有哪个 checkpoint 相对 checkpoint 0
在不同执行层和 Stage B 种子中都稳定更差。最准确的说法是：

> 本 pilot 没有观察到可复现的、相对 checkpoint 0 的固定预算 SVAMP 适应力下降。

这不等于证明“可塑性坍缩在任何地方都不存在”。

还要注意：证据包里的 `phase3_complete.json` 和 README 也把这个 30 步 cell 算成了完成，
所以“存在完成标记”不能替代逐文件检查。严格 completion validator 是后来为 seed repeat
加上的；旧 seed-42 工件必须手工审计。

---

## 第 5 节：结果三——一个指标恒定，一个指标会动

### 5.1 休眠比例

在所有 checkpoint、层 4/12/22、两个执行层、两个阈值下，休眠比例全部是 `0.0`。
常数列没有 Spearman 相关，所以 `spearman_table.csv` 中对应空白是“未定义”，不是漏填。

结论不是“模型没有任何休眠现象”，而是：

> 按这次实现和这两个阈值，休眠比例没有可用动态范围。

### 5.2 第 22 层有效秩

| checkpoint | 0 | 25 | 50 | 100 | 200 |
|---|---:|---:|---:|---:|---:|
| Mac L22 | 354.2 | 321.9 | 306.3（−13.5%） | 311.4 | 314.7（−11.1%） |
| Windows L22 | 354.2 | 321.3 | 325.7 | 317.0 | 324.9（−8.3%） |

这说明 Stage A 在晚层留下了可重复测到的谱变化。它是一个**表征诊断现象**，
不是“未来一定学不动”的证据。

### 5.3 512 与 2048 prompt 敏感性

| 执行层 | 512-prompt L22：ckpt 0 → 200 | 2048-prompt L22：ckpt 0 → 200 |
|---|---:|---:|
| Mac | −11.15% | −12.12% |
| Windows | −8.26% | −7.71% |

probe 数量会显著改变有效秩的绝对数值，但在这里 0→200 的相对变化比较接近。
因此，只有在 probe 成员与数量、测量层/位置、pooling、dtype 和预处理都相同的时候，
绝对值才可直接比较。

checkpoint 0 的 `erank_L12` 在两台机器上四舍五入到四位小数都是 `231.7567`。
这是一项很强的**实现一致性检查**：同一个源模型、同一实际协议，两个环境得到同一读数。
但它不等于“有效秩已经被验证能预测适应力”。

---

## 第 6 节：结果四——主相关不稳定

| 分析 | `rho(erank_L12, delta)` | p 值 | 审计状态 |
|---|---:|---:|---|
| Mac seed 42 | −0.60 | 0.285 | **旧版描述；固定 50 步网格无效** |
| Windows seed 42 | +0.50 | 0.391 | 五点描述性结果有效 |
| Windows seed 43 | −0.50 | 原 analyzer 未预注册计算 | seed-repeat 描述性结果有效 |
| Windows seed 44 | — | — | 只有 checkpoint 0，不能算相关 |

最有信息量的是 Windows seed 42 和 seed 43 的对比，因为它们固定了同一组 Stage A
checkpoint，只改变 Stage B 随机种子：

- seed 42 的最佳 delta 是 checkpoint 200：`+0.12`；
- seed 43 的最佳 delta 是 checkpoint 50：`+0.09`；
- 主相关从 `+0.50` 变成 `−0.50`。

这说明在 `n = 5`、每个 checkpoint 每个 seed 只有一条 Stage B 轨迹、eval 只有固定
100 题的条件下，checkpoint 排序不稳定。结果不是“RQ1 已被否定”，而是：

> 现在的证据强度不足以把有效秩当作可靠预测器；RQ1 仍然开放。

---

## 第 7 节：附带发现——纯 bf16 的近似 no-op 执行

更早的 Windows v1 用纯 bf16 参数和 `lr = 1e-6`。从 checkpoint 0 到 200，
任何已记录参数组的最大相对范数变化只有 `2.45e-8`，reward 基本不涨，Q 也几乎不动。
这些证据与“大多数很小的更新被 bf16 量化吞掉”一致。

准确说法是“**这条执行在实践中接近 no-op**”，而不是“每一个标量更新都严格等于 0”。
它可以作为一次意外的执行诊断案例，但不是随机化、预注册的科学阴性对照，不能拿来
回答 RQ1。

Colab notebook 01 **现在已经修好**：使用 float32 master weights、bf16 autocast，
并加了权重更新哨兵。团队行动项是保留并测试这个数值契约，而不是继续说 notebook
还没有修。

---

## 第 8 节：能说什么，不能说什么

本实验支持：

- 训练、Q 测量、Stage B 适应和工件记录链路基本可用；
- 第 22 层有效秩会随 Stage A 改变；
- 休眠比例按当前阈值没有量程；
- 五点、单 Stage-B seed 的 endpoint 相关对随机种子不稳定。

本实验不支持：

- “RLVR 降低了模型的一般学习能力”；
- 一个有效的 Mac 五点固定预算主相关；
- “有效秩已经是可靠预测器”；
- “相对 checkpoint 0 存在可复现的固定预算适应力下降”；
- 正式 proposal 级别的 stall detector 结果。本 pilot 没有正式 stall label，
  也没有完成 Q 对 dashboard baseline 的 lead-time bake-off。

---

## 第 9 节：下一步怎么修

1. 跑完 Windows seed 44，让每个 checkpoint 都有跨 seed 的误差范围。
2. 所有分析先过 completion gate：必须有实际 50 步、10/20/30/40/50 完整曲线，
   否则自动排除。
3. 增加对单一终点不那么敏感的结果量，例如预先固定的学习曲线 AUC 或 pass@k。
4. 增加 checkpoint 密度；五个点不足以把 rank correlation 当成强证据。
5. 全组冻结一份 Q 协议：激活位置、pooling 单位、probe 大小、dtype 都写死。
6. Mac checkpoint 25 要么从源 checkpoint 全新重跑 50 步，要么永久排除出固定预算分析。

---

## 可以直接发给队友的修订摘要

Experiment 1 把训练、Q 测量和 Stage-B 适应管线走通了，但逐文件审计发现 Mac 的
checkpoint-25 适应只完成 30/50 步，却被旧 summary 和 phase marker 标成完成。
排除它后，共有 15 个可核实的 50-step 适应：14 个终点 delta 为正、1 个持平、0 个为负。
第 22 层 effective rank 下降约 8%–14%，dormant fraction 始终为 0；但在完全相同的
Windows Stage-A checkpoints 上，主相关从 Stage-B seed 42 的 `+0.50` 变成 seed 43 的
`−0.50`。因此 RQ1 仍然开放：当前证据既没有显示相对 checkpoint 0 可复现的固定预算
适应力下降，也没有验证 effective rank 是可靠预测器。

## 外部技术交叉核验

- 固定 revision 的 Qwen config 显示 24 层、hidden size 896：  
  <https://huggingface.co/Qwen/Qwen2.5-0.5B/blob/060db6499f32faf8b98477b0a26969ef7d8b9987/config.json>
- Hugging Face TRL 文档说明了 GRPO 的 `num_generations` 以及 reward、entropy、
  completion length、zero-reward-variance 等日志字段：  
  <https://huggingface.co/docs/trl/grpo_trainer>
- PyTorch `torch.finfo` 文档给出浮点 dtype 的 `eps` 定义；本次 no-op 判断本身仍以
  本地 weight/reward/Q 工件为证据：  
  <https://docs.pytorch.org/docs/stable/type_info.html>
