# Experiment 1.6 计划 — 中间剂量搜索（lr=3e-6，端点先行 + 门控扩格，预注册）

日期：2026-07-19。作者：Aaron Wang（Person 4）。
状态：**计划冻结；正式启动需团队拍板**（见 §6——这是 exp1.5 落格后的升级决策，
不是 Person 4 可单方启动的实验）。配置：`exp1_6_config.json`（探针阶段）与
`exp1_6_config_fullgrid.json`（扩格阶段，仅门后使用）。门评估：`exp1_6_gate_eval.py`。
执行手册：`WIN4070_NEXT_RUNS_GUIDE.md`。

## 0. 一句话定位

RQ1 的被预警对象——**持久的固定预算适应力下降**——至今没被干净制造出来：
lr=1e-6 × 500 步伤不到模型（exp1.5 v3，MC1 fail、Q 端点位移 <1%），lr=1e-5 55 步内
政策崩溃（v2）。1e-6 与 1e-5 之间是完全未测的剂量区间。Exp1.6 在**几何中点 3e-6**
放一个探针：**单变量**（只改 lr），复用 exp1.5 的全部代码、数据、测量与降噪设计，
用**端点先行 + 门控扩格**把"剂量又不够"的下行风险钱包压到 ~20 GPU 时。

## 1. 证据 → 设计映射

| 已提交事实 | 推论 | 设计 |
|---|---|---|
| v3（1e-6）：erank_L12 端点 −0.98%，late-window（300/400/500）均值 −1.2%；权重在动、reward 36→50% | 剂量不足以留下持久 Q 位移 | lr **3e-6**（3.3× v3；1/3.3× v2），其余逐字段照抄 v3 |
| v2（1e-5）：55 步崩溃；熵/方差全套现场在档 | 10× 剂量不可行（无 KL 时） | 3e-6 若也崩：安全停即观察，保留现场上报（runner 对 1.6 保持严格契约——**不设** `safety_stop_expected`） |
| v3 的 MC2"下降"被迁移天花板混杂（Δ 与 before 的 ρ=−0.66） | 端点 outcome 门必须配 `svamp_before` 记录 | G-B 只做门控用；任何相关读出必须带 before 协变量（config `open_team_questions` 已写明） |
| 18 格适应 ≈21 h 是大头；Q 测量 <15 min | 先花小钱验"病人是否存在"，再花大钱测量剂量-反应曲线 | **端点先行**：phase 3 预注册只跑 ckpt {0,500} × 3 seeds（6 格 ≈7 h）；全格（+12 格 ≈14 h）锁在扩格门后 |
| run-dir 哈希只含 stage_a/seed/execution | 改 adaptation 网格不换 run dir | `exp1_6_config_fullgrid.json` 与探针 config 解析到**同一** run dir（已验证：`exp15_cuda_grpo_gsm8k_caebbcc73461`），扩格无需重训 |

## 2. 设计详表

**与 exp1.5 v3 逐字段一致**：模型/revision、数据与冻结 split、seed 42、500 步、
checkpoint 网格 [0,25,50,100,200,300,400,500]、批几何、8 生成、温度/top-p、512 上限、
β=0、安全规则（clipping 仅诊断、零方差 patience 5）、preflight、sentinel、
Q 测量（512-probe、层 4/12/22、**float32**——直接采用 v3 恢复后的正确 dtype、
2048 敏感性在 0/500）、适应配方（SVAMP 256 训 / 300 评 / 50 步 / 3 seeds / legacy-100 桥）。

**唯一科学改动**：`stage_a.learning_rate` 1e-6 → **3e-6**。
**唯一流程改动**：phase 3 的预注册网格从 6 checkpoint 起步改为端点 {0,500} 起步 + 扩格门。

## 3. 预注册门（跑前定死，`exp1_6_gate_eval.py` 原样执行）

- **G-A（Q 门）**：|mean(erank_L12 @ {300,400,500}) / erank_L12 @ 0 − 1| ≥ **7.5%**。
  为什么是 late-window 均值而非端点单点：exp1.5 §2.5 教训——单点端点值 run 间散布可达
  ~9%，3 点均值压噪；为什么 7.5%：v3 健康值 −1.2% 的 6 倍以上，且低于 MC1 的 10%，
  使"接近阈值"的剂量仍能拿到 outcome 数据。
- **G-B（outcome 门）**：mean3-Δ(ckpt-0) − mean3-Δ(ckpt-500) ≥ **0.05**（≈2× 均值差 SE，
  与 MC2 同标度）。
- **规则**：G-A **或** G-B 通过 → 用 `exp1_6_config_fullgrid.json` 把 phase 3 扩到
  [0,50,100,200,300,500]（已完成的 6 格自动跳过）；两者皆败 → **停**，产物照常 commit，
  "3e-6 仍亚治疗剂量"作为剂量-反应第三个点上报团队。VERDICT 行由脚本打印
  （EXPAND / STOP / INVESTIGATE），并把 `exp16_gate_eval.json` 写进 run dir。
- 扩格完成后才允许 runner 的 phase 4（对 <4 checkpoint 的 config，phase 4 直接拒绝——
  已写进 runner）。MC1/MC2/主相关沿用 exp1.5 定义原样执行，另加：**任何相关读出必须
  同时报告以 `svamp_before` 为协变量的版本**（v3 的天花板教训，预注册在此）。

## 4. 判定矩阵（预注册解释）

| 结局 | 含义 | 对 experiment 2 的输入 |
|---|---|---|
| G-A 过（Q 动了） | 首次拿到"移动 Q 且不崩"的剂量 | 全格跑完 → 第一条可用的剂量-反应曲线；exp2 剂量下限确定 |
| G-A 败、G-B 过 | outcome 动而 Q 不动 | 反对 erank 的证据加强（与 v3 落格同向）；检查 before 混杂后上报 |
| 双败 | 3e-6 仍亚治疗 | 剂量窗收窄为 (3e-6, 1e-5)；下一步只剩 KL arm / 更大模型 / Colab，纯团队决策 |
| 安全停（崩溃） | 崩溃阈值 < 3e-6 | 剂量窗收窄为 (1e-6, 3e-6)；1.5.1 的取证协议直接适用于该现场 |

四格皆有价值；本实验不存在"白跑"结局。

## 5. 预算与执行顺序

| 步骤 | 内容 | 预计 |
|---|---|---|
| 1 | phase 1（500 步，~93 s/步） | ≈13 h（可分晚，resumable） |
| 2 | phase 2（8 ckpt + 2 敏感性，fp32） | <20 min；随后 `exp15_gates.py ckpt0 --config exp1_6_config.json` |
| 3 | 端点 probe：phase 3 × {0,500} × {42,43,44}（用 `--adapt-checkpoint/--adapt-seed` 逐格） | ≈7 h；首格后跑 bridge gate |
| 4 | `exp1_6_gate_eval.py` → EXPAND / STOP | ~1 min |
| 5 | （仅 EXPAND）fullgrid config 补 12 格 → phase 4 | +≈14 h |

合计：**门前 ≈20 h；全格 ≈34 h**。磁盘与 v3 相同（≥30 GiB 前置检查；适应 trainer
目录自动清理）。sentinel 阈值随 config lr 自动缩放（`exp15_gates.py sentinel` 直接可用）。

## 6. 启动条件（硬性）

1. Exp1.5 结果 + 本计划已发团队 Slack，**Tommy 明确同意**走"本地 3e-6 探针"这一升级
   路径（备选路径 KL arm / 1.5B / Colab 由团队排序）；
2. 同意记录：正式 phase 1 的 launch commit message 必须含 `team-ack:` 行（谁、何时、
   哪条 Slack 消息）；无此行不得开跑（写给 Windows agent 的硬规则，见执行手册）；
3. `--smoke` 验证不受此限制（不产生科学数据）。

## 7. 风险与预案

| 风险 | 预案 |
|---|---|
| 3e-6 崩溃 | 严格契约生效：phase 1 报错退出、现场保留、上报；随后可按 1.5.1 协议对其做取证测量（terminal-stop 容忍使 phase 2 可测已存 ckpt） |
| 门前 13 h 白跑（双败） | 已是设计接受的下限（20 h 换剂量-反应第三点 + 窗口收窄） |
| OOM / 中断 | v3 语义：resume；适应格 OOM → 隔离进程重跑（`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 先例） |
| 扩格后 MC1 仍败 | 按 §3 预注册解释落格，不追加任何未注册分析 |
