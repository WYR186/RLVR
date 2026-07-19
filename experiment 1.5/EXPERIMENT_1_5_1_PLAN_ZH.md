# Experiment 1.5.1 计划 — 崩溃取证：密集 Q 采样的失速前瞻（预注册）

日期：2026-07-19。作者：Aaron Wang（Person 4）。
状态：**计划冻结**。配置：`exp1_5_1_config_seed{42,43,44}.json`（预注册，跑前不改）。
执行手册：`WIN4070_NEXT_RUNS_GUIDE.md`。分析：`analysis_exp1_5_1.py`（阈值同时冻结在
每份 config 的 `analysis` 块里）。

## 0. 一句话定位

Exp1.5 v2 在 lr=1e-5 下于 update 55 真实政策崩溃——这正是本项目标题里要"提前看见"的
失速，但 v2 只存了 3 个 checkpoint，**崩溃前 Q 有没有先动，数据回答不了**。Exp1.5.1
用完全相同的配方重放崩溃 3 次（3 个 Stage-A 种子），把 checkpoint 网格加密到**每 5 步**，
直接回答：**Q（erank / dormant fraction）是否先于 dashboard 急性前兆（熵急跌，v2 中仅
先行 ~3 步）给出预警？** 无论答案是"先"还是"不先"，都是 detector bake-off 的第一份
正样本证据。零 Colab 消耗，预计 6–8 GPU 时。

## 1. 证据 → 设计映射

| 已提交事实 | 推论 | 本实验对应设计 |
|---|---|---|
| v2 崩溃：熵在 step 48 跌破 0.1，先行终局零方差段（51–55）约 3 步；clipping 从 step ~28 起慢性病态（0.6–1.0） | 急性前兆窗口极窄；慢性红旗无时点信息 | checkpoint 每 5 步一存——Q 若要"赢过"熵，必须至少提前 ~10 步可见 |
| v2 的 ckpt 只有 0/25/50，崩溃窗内仅 1 个 Q 采样点 | 核心问题数据缺失 | 密集网格 [0,5,…,80]；**Phase 0 先补测 v2 已存的 ckpt-0/25/50（fp32）**，白捡 3 个真实崩溃前 Q 点 |
| lr 调度随 max_steps 变化（linear decay） | max_steps=80 会改变剂量轨迹，不再是 v2 的复现条件 | **max_steps 保持 500**（调度与 v2 逐步一致），新增 schedule-preserving 硬帽 `hard_stop_step=80`（`HardCapStop` 回调，写 `hard_cap_stop.json`） |
| on-policy 轨迹随机性大（同配方 run 间 Q 端点散布 ~9%） | 单次崩溃时点/形态不可外推 | 3 个 replicate（seed 42=近似复现 v2；43/44=崩溃时点方差）；预注册 ≥2/3 判定规则 |
| 健康 run（lr=1e-6 × 3 个）的 erank_L12 在 ≤100 步内最深下探 −8.6%，dormant 恒 0 | 崩溃信号必须超出健康包络才算数 | SC1 阈值 −12%（1.4× 包络）；dormant>0 即高显著（历史零假阳性） |

## 2. 设计详表

**冻结（与 v2 = `exp1_5_config_v2.json` 完全一致）**：模型/revision、512 GSM8K 冻结题、
exact-answer reward、lr=1e-5、β=0、温度 0.7、top-p 1.0、8 生成、512-token 上限、
batch 几何（execution profile 覆盖为 4×16）、max_steps=500（调度用）、
安全停规则（零方差 patience=5、clipping 仅诊断）、sparse-reward preflight、
update sentinel（每 25 步）、Q 测量管线（512-probe、层 4/12/22、eval 模式）。

**改动（全部预注册在 config）**：

| 旋钮 | v2 | 1.5.1 | 理由 |
|---|---|---|---|
| checkpoint 网格 | [0,25,50,100,…] | **[0,5,10,…,80]**（17 点） | 崩溃取证需要 ≤5 步分辨率 |
| 终止方式 | 跑到 500 或安全停 | 安全停（预期 ~55）或 **step-80 硬帽** | 崩溃后无科学价值；帽子不改调度 |
| Stage-A seed | 42 | **42 / 43 / 44 三个独立 run** | 崩溃时点方差 |
| 测量 dtype | float16（当时） | **float32** | v3 恢复教训：与 pilot 参考同 dtype |
| GSM8K held-out 评估 | 每 25 步 | **每 5 步** | 与 Q 同分辨率的第三通道（64 题，SE±6pp，仅描述性） |
| 运行语义 | 安全停=报错退出 | `safety_stop_expected=true`：**安全停=预注册终态**，phase 1 正常完结、phase 2 只测已存 ckpt | 崩溃是观察对象 |
| Phase 3/4 | 有 | **无**（config 空网格，runner 拒跑 phase 3；phase 4 对 exp1_5_1 直接拒绝） | 取证实验不做适应 |

**Phase 0（新增，先于一切训练）**：对 v2 归档 run（`exp15_cuda_grpo_gsm8k_dd5f54a0e2b7`）
执行 `--phase 2 --config exp1_5_config_v2.json`。新的 terminal-stop 容忍逻辑会测量其
已存的 ckpt-0/25/50、跳过其余，并把 float32 测量写入其 `measurements/`。这是**向 v2
证据目录的加性写入**（不覆盖不删除任何既有文件），与 v3 恢复先例（`postgate_recovery.jsonl`）
同一证据政策；本段即为该偏差的书面记录。ckpt-0 测完立即跑
`exp15_gates.py ckpt0 --run-dir <v2 dir>` 对 pilot 参考做一致性 gate。

## 3. 预注册事件与信号定义（防事后挑选）

- **事件 E**：触发安全停的终局零组内奖励方差段的**首步** = trigger_step − patience + 1，
  并与 dashboard 逐步回溯交叉验证（不一致以 dashboard 为准并记录）。
- **熵前兆 onset**：E 之前最后一段**持续** entropy<0.10 的首步；lead = E − onset。
  （v2 实测 lead=3。）
- **clipping**：只报告慢性画像（≥0.9 的步数占比、首次达标步），**不定义时点前兆**——
  v1/v2 已证明它在 lr=1e-5 下从头病态。
- **SC1（Q 先动，主判定）**：存在测量点 c ≤ E−10，使 erank_L12(c)/erank_L12(0)−1 ≤ **−12%**。
  阈值依据：健康包络最深 −8.6%（pilot WIN v2 ckpt-50）；−12% = 包络 ×1.4。
- **SC2（dormant 苏醒）**：任一 c ≤ E 上任意层、任一 τ 的 dormant fraction > 0。
  全项目 5 个完整 run 历史上恒为 0 → 零假阳性历史。
- **SC3（Q 赢过熵）**：SC1 成立且其 lead ≥ 熵 lead + 5（必须超出网格分辨率才算"更早"）。
- **跨 replicate 判定**：每条 SC 在 ≥2/3 个**发生崩溃的** replicate 中成立 → "pilot 规模上
  支持"。未崩溃（撑到 80 步被硬帽停）的 replicate 记为 censored——**不崩本身就是
  崩溃-hazard 方差的数据**，照常测 Q 并对照包络。
- 其余一切（GSM8K 每 5 步曲线、L4/L22、熵/Q 联合形态）**均为描述性**。

## 4. 执行顺序与预算

| 步骤 | 内容 | 预计 |
|---|---|---|
| 0 | v2 补测量 + ckpt0 gate | ~10 min |
| 1 | replicate A（seed 42）：phase 1 → 安全停或 80 步硬帽 | 预期 ~55 步 ≈ 1.6 h（至多 80 步 ≈ 2.3 h，含每 5 步评估 +~25 min） |
| 2 | A 的 phase 2（≤17 × ~1 min）+ ckpt0 gate | ~20 min |
| 3 | replicate B（seed 43）同 1–2；然后 C（seed 44） | 各 ~2–2.5 h |
| 4 | `analysis_exp1_5_1.py` 三个 run dir 一起跑；产物 commit | ~1 min |

合计 ≈ **6–8 GPU 时**（1–2 晚）。磁盘：每 replicate 至多 17 ckpt × ~1.9 GiB ≈ 32 GiB
+ 滚动 trainer ~4 GiB；**开跑前要求 ≥45 GiB 空闲**；上一 replicate 通过 phase 2 +
gate 后，删除其各 ckpt 的 `model.safetensors`（保留 config/tokenizer 与全部 JSON 证据），
replicate A 若磁盘富余则整套保留。

## 5. 分析与产物

`analysis_exp1_5_1.py RUN_A RUN_B RUN_C [V2_DIR]`：向每个 run dir 写 `forensics.json`
（加性），stdout 输出逐 run 判定行 + 跨 replicate 汇总 JSON。四种预注册结局及其含义：

| 结局 | 含义 | 下一步 |
|---|---|---|
| SC1+SC3 支持 | Q 在失速前 ≥10 步给出超包络信号，且早于熵 | 直接写进 Research Doc；detector bake-off 把 erank 列为候选特征 |
| SC1 支持、SC3 不支持 | Q 有信号但不比熵早 | Q 作为"确认信号"而非"预警信号"记录 |
| SC1 不支持 | 崩溃前 Q 无超包络位移 | 对"activation-Q 预警训练失速"的干净反面证据——同样可发表 |
| ≥2 replicate 不崩（censored） | lr=1e-5 崩溃 hazard 依赖轨迹随机性 | 本身是重要发现；与团队讨论加 replicate 或提步数帽 |

## 6. 风险与预案

| 风险 | 预案 |
|---|---|
| replicate 不崩、撑到 80 | 硬帽停，记 censored；照常测 Q（见 §3/§5） |
| 崩溃早于 step 10 | 网格仍有 0/5 两点 + Phase 0 的 v2 三点；结果标注分辨率受限 |
| OOM / 中断 | 与 v3 相同语义：resume 或按 v3 先例隔离进程重跑；证据保留 |
| 磁盘不足 | §4 的 45 GiB 前置检查 + replicate 间清理协议 |
| 任何 gate STOP | 停手、保留现场、上报，不改配方重试 |

## 7. 沟通条款

本实验不动任何待决团队问题（模型/任务/KL/适应算法全部不涉及），属 Person 4 诊断
职责内的取证性 Stage-A 复现。启动前在 Slack 发一条知会（含本计划链接与预算 6–8 GPU 时）；
Phase 0 对 v2 目录的加性写入在同一条内声明。结果无论落哪格，汇总后与 exp1.5 报告
一起进 Research Doc。
