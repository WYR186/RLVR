# EAAJ Pilot — 三次实验合并报告（中文版，仅陈述事实）

日期：2026-07-14。作者：Aaron Wang。

本报告只陈述已提交、可审计的事实，来自预注册 pilot 的三次执行。不含解释、
不含因果推断、不跨执行环境（stratum）合并数据。凡是为本报告临时计算、
而非来自已提交产物的数字，标注**【描述性】**。

---

## 1. 三次实验共用的预注册设计（完全相同）

单一配方，`eaaj-pilot/pilot_config.json`：

| 项目 | 设置 |
|---|---|
| 模型 | `Qwen/Qwen2.5-0.5B` @ revision `060db649…` |
| Stage A | 在冻结的 512 道 GSM8K 上做 GRPO，exact-answer 二值 reward，200 次更新，lr 1e-6，KL β=0，温度 0.7，top-p 1.0，每题 8 个生成，每次更新 64 个 completion |
| Checkpoint | 0 / 25 / 50 / 100 / 200 次更新 |
| Q 测量 | 冻结的 512-prompt probe 集，eval 模式，固定 dtype；decoder 第 4 / 12 / 22 层；effective rank、dormant-neuron fraction（τ=0.025 和 0.1）、anisotropy 变体 |
| Stage B | 从每个 checkpoint 出发做 SVAMP 适应：冻结的 256 train / 100 eval 题，50 次 GRPO 更新，第 10/20/30/40/50 步评估 |
| 主 outcome | `svamp_delta` = 50 次更新后的准确率 − 该 checkpoint 自己适应前的准确率（所有 checkpoint 用同一套冻结题目） |
| 主分析 | Spearman rho(erank_L12, svamp_delta)，n = 5 个 checkpoint |
| 随机种子 | 42（实验一、二全程用 42；实验三只改 Stage-B 适应阶段的 seed） |

以下所有 outcome 都是固定预算量（恰好 50 次 SVAMP 更新后的准确率变化，
对照每个 checkpoint 各自预先声明的 baseline）。

---

## 2. 实验一 — macOS CPU stratum

运行目录：`eaaj-pilot/outputs/local_grpo_gsm8k_eac028bfcc87`。
执行环境：MacBook M3 Max，CPU，全程 float32，标准 AdamW，
micro-batch 8 × grad-accum 8，不开 gradient checkpointing。

**Stage-A 训练 reward**（`dashboard.jsonl` 的 `reward` 字段分段均值，
按 step 去重）**【描述性】**：
0.430（第 1–25 步）→ 0.518 → 0.575 → 0.582 → 0.640（第 151–200 步）。

**GSM8K held-out 准确率**（固定 64 题）在 ckpt 0/25/50/100/200：
0.4375 / 0.4531 / 0.5156 / 0.5469 / 0.5156。

**Effective rank**（512-probe）：

| ckpt | L4 | L12 | L22 |
|---:|---:|---:|---:|
| 0 | 225.14 | 231.76 | 354.19 |
| 25 | 225.31 | 223.87 | 321.89 |
| 50 | 225.18 | 219.43 | 306.34 |
| 100 | 223.94 | 229.43 | 311.41 |
| 200 | 224.15 | 233.41 | 314.71 |

Dormant-neuron fraction：所有 checkpoint、所有层、两个阈值下均为 0.0。

**固定预算 SVAMP 适应**（seed 42）：

| ckpt | 适应前 | 适应后 | delta |
|---:|---:|---:|---:|
| 0 | 0.53 | 0.58 | +0.05 |
| 25 | 0.48 | 0.61 | +0.13 |
| 50 | 0.61 | 0.69 | +0.08 |
| 100 | 0.63 | 0.65 | +0.02 |
| 200 | 0.67 | 0.71 | +0.04 |

**主分析**（已提交的 `analysis/analysis_summary.json`）：
rho(erank_L12, svamp_delta) = **−0.60**，p = 0.285，n = 5。

---

## 3. 实验二 — Windows CUDA v2 stratum

运行目录：`eaaj-pilot/outputs/local_cuda_grpo_gsm8k_e9b0b52aab6c`。
执行环境：Windows 11，RTX 4070 Laptop GPU（8 GiB），fp32 master 权重 +
bf16 autocast，`paged_adamw_8bit`，gradient checkpointing 开启，
micro-batch 4 × grad-accum 16。

背景：此前的 Windows v1 run（`local_cuda_grpo_gsm8k_6a075c15808e`，
纯 bf16 参数）四个 phase 全部跑完，但 200 次更新中权重组最大相对变化仅
2.4e-8；已记录为无效，保留作阴性对照（`WIN4070_RUN_ANALYSIS.md`）。
v2 是修正后的重跑。

**v2 记录在案的执行事件**（均记录于 `compute_log.md` 和 v2 报告）：
- Phase 1 在第 25、50、75 步附近中断并修复。修复恢复了模型权重、学习率
  调度位置和 resume 元数据；这些边界上的 optimizer moments 和 RNG state
  未被恢复。
- bitsandbytes optimizer state 在 Windows 上保存时挂起或损坏，此后改用
  `save_only_model=True`。
- Phase 3 的 ckpt-50 发生一次瞬时 CUDA OOM；不完整的 attempt 被保留，
  ckpt-50 在新进程中用完全相同的配方从头重跑。
- 全几何 probe 报告 10.809 GiB 峰值 reserved 显存，超过预设的 7.3 GiB
  门槛；已记录为 allocator/统计口径 deviation。

**更新有效性 sentinel**：33/33 窗口通过（Phase 1 有 8 个，Phase 3 有
25 个）。ckpt-0→200 权重组最大相对变化：6.06e-6。

**Stage-A 训练 reward**（方法同 §2）**【描述性】**：
0.354 → 0.443 → 0.462 → 0.538 → 0.582。
（v2 报告中最后一段的 0.5781 对应第 151–199 步窗口；完整 151–200 窗口
为 0.5816。）

**GSM8K held-out 准确率**在 ckpt 0/25/50/100/200：
0.3594 / 0.4688 / 0.4375 / 0.4688 / 0.4219。

**Effective rank**（512-probe）：

| ckpt | L4 | L12 | L22 |
|---:|---:|---:|---:|
| 0 | 225.14 | 231.76 | 354.19 |
| 25 | 219.31 | 215.31 | 321.31 |
| 50 | 223.01 | 211.76 | 325.67 |
| 100 | 222.29 | 208.92 | 317.03 |
| 200 | 223.24 | 214.68 | 324.92 |

对 ckpt-0 与 ckpt-200 做的 2048-prompt 敏感性检查测得 L4 −1.03%、
L12 −7.22%、L22 −7.71%。

Dormant-neuron fraction：所有 checkpoint、所有层、两个阈值下均为 0.0。

**固定预算 SVAMP 适应**（seed 42）：

| ckpt | 适应前 | 适应后 | delta |
|---:|---:|---:|---:|
| 0 | 0.53 | 0.59 | +0.06 |
| 25 | 0.51 | 0.62 | +0.11 |
| 50 | 0.56 | 0.59 | +0.03 |
| 100 | 0.55 | 0.60 | +0.05 |
| 200 | 0.54 | 0.66 | +0.12 |

**主分析**（已提交的 `analysis/analysis_summary.json`）：
rho(erank_L12, svamp_delta) = **+0.50**，p = 0.391，n = 5。

---

## 4. 实验三 — Stage-B 适应 seed 重复（Windows）

位置：`…/local_cuda_grpo_gsm8k_e9b0b52aab6c/adaptation_repeats/`。
计划：`eaaj-pilot-win4070/WIN4070_STAGEB_SEED_REPLICATION_PLAN.md`；
状态：`WIN4070_STAGEB_SEED_REPLICATION_STATUS_ZH.md`。

设计：从**同样的五个 v2 checkpoint** 出发，重跑完全相同的固定预算 SVAMP
适应，**只改适应阶段的随机种子**（42 → 43 → 44）。每个 repeat 的 manifest
记录了源 run 的 config/manifest SHA-256 哈希和 git SHA；配方字段与 §3
完全一致。每个 run 都通过了官方 validator 门槛（50/50 次更新完成、固定
baseline、完整评估曲线、更新有效性 sentinel、无 safety stop）。

**Seed 43 — 已完成（5/5 个 checkpoint，validator 5/5 通过）：**

| ckpt | 适应前 | 适应后 | delta |
|---:|---:|---:|---:|
| 0 | 0.53 | 0.57 | +0.04 |
| 25 | 0.51 | 0.59 | +0.08 |
| 50 | 0.56 | 0.65 | +0.09 |
| 100 | 0.55 | 0.61 | +0.06 |
| 200 | 0.54 | 0.59 | +0.05 |

**Seed 44 — 仅完成 ckpt-0：** 0.53 → 0.53，delta = 0.00。第 10/20/30/40/50
步的评估曲线：0.52 / 0.58 / 0.55 / 0.57 / 0.53。ckpt 200/25/50/100 尚未运行。

事件：seed-43 ckpt-0 的第一次 wrapper attempt 在训练开始前失败
（PowerShell 把 Python/Triton 的 warning 流当成了错误）；失败证据已保留，
之后在新进程中用未改动的配置重跑完成。

已提交状态文档中记录的事实：
- endpoint delta 排名，seed 42：ckpt-200 > 25 > 0 > 100 > 50。
- endpoint delta 排名，seed 43：ckpt-50 > 25 > 100 > 200 > 0。
- seed-42 与 seed-43 两组 delta 排名之间的描述性 Spearman ≈ −0.50。
- ckpt-200 delta 减 ckpt-0 delta：+0.06（seed 42），+0.01（seed 43）。
- 预注册的 seeds-42/43/44 汇总分析**尚未生成**；按计划须等 seed 44
  全部完成。

为本报告另做的描述性计算（不属于预注册分析）**【描述性】**：
- rho(erank_L12, seed-43 deltas) = −0.50，p = 0.391，n = 5。
- rho(erank_L12, seed-42 与 seed-43 delta 的均值) = 0.00，p = 1.0，n = 5。
- ckpt-0 在已跑的三个 seed 下的 delta：+0.06 / +0.04 / 0.00。

---

## 5. 跨实验观察（事实；并列呈现，从不合并）

1. **ckpt-0 处的测量一致性。**两个 stratum 在 ckpt-0 测的是同一个 base
   模型；effective-rank 数值一致到小数点后四位（如 erank_L12 两边都是
   231.7567）。
2. **两个 stratum 中，第 12、22 层的 effective rank 在 ckpt 25–100 都低于
   ckpt-0。**ckpt-200 相对 ckpt-0：L22 为 −11.1%（CPU）和 −8.3%（WIN）；
   L12 为 +0.7%（CPU）和 −7.4%（WIN）。L4 在两个 stratum 的每个
   checkpoint 变化都小于 3%。
3. **迄今完成的固定预算适应共 16 次**（CPU seed-42 5 次、WIN seed-42
   5 次、WIN seed-43 5 次、WIN seed-44 1 次）。endpoint delta：15 个为正、
   1 个为零、0 个为负。
4. **两个 stratum 的每一次测量中，dormant-neuron fraction 均为 0.0**
   （两个阈值下都是）。
5. **已提交的主相关值：**−0.60（CPU，seed 42）和 +0.50（WIN，seed 42）。
   WIN seed 43 的描述性值：−0.50。在 n = 5 下没有一个具有统计显著性。
6. Stage-B 起始准确率（`svamp_before`）跨 checkpoint 的范围：CPU stratum
   为 0.48–0.67，WIN stratum 为 0.51–0.56。

---

## 6. 计算账目（来自 `eaaj-pilot/compute_log.md`）

| 实验 | 记录的 active 时间 |
|---|---|
| 一（CPU） | Phase 1 在 M3 Max 上约 300 秒/更新（前 67 步记录为 5.8 小时）；后续 phase 记录在 run 目录中 |
| 二（WIN v2） | probe <5 分钟；Phase 1 5.18 小时；Phase 2 3.9 分钟；Phase 3 5.26 小时；Phase 4 1.3 分钟 |
| 三（repeats） | smoke 6.4 分钟；seed 43 五个 run 约 4.9 小时；seed 44 ckpt-0 72.2 分钟 |

未消耗任何 Colab 计算单元；以上所有 run 均在本地硬件上执行。Windows 每个
phase 的 GPU telemetry CSV 已随 run 产物一起提交。

---

## 7. 待办事项（按预注册计划）

- Seed 44：ckpt 200 / 25 / 50 / 100（计划顺序）尚未运行。
- Seeds-42/43/44 汇总分析：尚未生成；以 seed-44 完成为前提。
- 记录在案的待团队决策事项：Stage-B 任务家族与 Stage-A 的接近程度
  （GSM8K↔SVAMP）、KL β>0 baseline、base vs Instruct 模型变体，以及
  是否用预留预算跑一条 Colab 参考 stratum。

## 8. 产物索引

| 内容 | 路径 |
|---|---|
| 预注册配方 | `eaaj-pilot/pilot_config.json` |
| 实验一产物 | `eaaj-pilot/outputs/local_grpo_gsm8k_eac028bfcc87/` |
| 实验二产物 | `eaaj-pilot/outputs/local_cuda_grpo_gsm8k_e9b0b52aab6c/` |
| 实验三产物 | `…e9b0b52aab6c/adaptation_repeats/seed-43/`、`…/seed-44/` |
| v1 无效性分析 | `eaaj-pilot-win4070/WIN4070_RUN_ANALYSIS.md` |
| v2 完整报告 | `eaaj-pilot-win4070/WIN4070_V2_FINAL_REPORT_ZH.md` |
| repeat 计划 / 状态 | `eaaj-pilot-win4070/WIN4070_STAGEB_SEED_REPLICATION_PLAN.md`、`…_STATUS_ZH.md` |
| 计算账本 | `eaaj-pilot/compute_log.md` |
