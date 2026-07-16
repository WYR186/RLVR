# Experiment 1.5 计划 — Stage-A 剂量递增 + 噪声修正（预注册）

日期：2026-07-16。作者：Aaron Wang（Person 4）。
状态：**计划已冻结，等待执行**。配方文件：`exp1_5_config.json`（预注册，
跑之前不再改动；任何执行期偏差记入 run 目录并同步 Research Doc）。

---

## 0. 一句话定位

Pilot（experiment 1）证明了管线可靠，但**没造出要预警的现象**（可塑性
损伤），且 **outcome 噪声≥信号**。Experiment 1.5 是介于 pilot 和正式
实验之间的一次本地实验：**只动两组旋钮——把 Stage-A 剂量加大、把
outcome 噪声压低——其余全部冻结**，检验在"病人存在"的前提下 RQ1 是否
可测。仍然零 Colab 消耗。

## 1. 证据 → 设计的映射（每条改动都可追溯到 pilot 数字）

| Pilot 观察（已提交产物） | 推论 | Exp 1.5 对应改动 |
|---|---|---|
| ckpt-0→200 权重最大相对变化仅 6.06e-6（win v2 sentinel）；16 次适应 15 正 1 零，无一负；dormant fraction 恒 0 | lr 1e-6 × 200 步的 Stage A 伤不到模型——自变量没被制造出来 | **lr 1e-6 → 1e-5（10×）**，**200 → 500 步（2.5×）** |
| 训练 reward 到第 200 步仍在上升（win 0.58 / mac 0.64） | 尚未进入 reward 饱和后的"危险区"；文献中可塑性损耗多出现在饱和后继续训练 | 500 步大概率覆盖饱和后区间；饱和不是失败而是**目标状态** |
| 单种子 delta 排名换种子近乎反转（seed 42 vs 43 排名相关 ≈ −0.50）；ckpt-0 三种子 delta = 0.06/0.04/0.00（SD≈0.031） | svamp_delta 被适应种子噪声主导 | **每个 checkpoint 预注册 3 个适应种子（42/43/44），主 outcome = 三种子均值** |
| 100 题评估集二项噪声 SE ≈ ±5.0pp（√(0.25/100)），与 checkpoint 间 delta 差异（2–13pp）同量级 | 量尺自身抖动接近被测差异 | **评估集 100 → 300 题**（pinned SVAMP test split 全部 300 题；SE 降至 ≈2.9pp），pilot 的 100 题是其严格子集，同时记录 legacy-100 分数作为跨实验桥梁 |
| n=5 时 rho 需 ≈0.9 才显著；±0.5~0.6 的 rho 一至两个位次互换即可产生 | checkpoint 数太少 | **保存 8 个 checkpoint（Q 全测），其中 6 个做适应**（n: 5→6；Q 曲线分辨率 5→8 点） |
| 跨平台 ckpt-0 erank 一致到小数点后 4 位 | 测量管线可靠，不许动 | probe 集、层选择、测量 dtype、metric 代码**逐字节复用** pilot 的 |

## 2. 设计详表

### 2.1 冻结不动的部分（与 pilot 完全一致，保证可比性）

- 模型与 revision：`Qwen/Qwen2.5-0.5B` @ `060db649…`（base，非 Instruct——
  待决项维持便宜默认值）
- Stage-A 数据：同一份冻结的 512 道 GSM8K 训练题（`gsm8k_splits.json`）；
  同一 prompt 模板；exact-answer 二值 reward，代码同一文件
- Q 测量：同一份冻结 512-prompt probe 集（`probe_set_ids.json`）、同层
  （4/12/22）、同 dtype（fp16）、eval 模式、`src/metrics.py` 原样复用；
  2048-prompt 敏感性检查在端点 ckpt（0 和 500）
- Stage-B 配方：GRPO（待决项维持默认）、lr 1e-6、β=0、温度 0.7、
  top-p 1.0、8 生成、**同一份冻结的 256 道 SVAMP 训练题**、50 步固定预算、
  每 10 步记录适应速度曲线
- 执行环境：Windows RTX 4070 Laptop，**pilot v2 execution profile 原样
  导入**（fp32 master 权重 + bf16 autocast + paged_adamw_8bit + gradient
  checkpointing + micro-batch 4 × grad-accum 16 = 每次更新 64 completion）
- 防呆装置全套保留：sparse-reward preflight（no-signal 即停并上报，
  不许静默换 reward）、update sentinel（每 25 步验证权重在动）、
  LocalSafetyCallback（连续 5 步零组内奖励方差 / 非有限 loss 即安全停）、
  runner lock、manifest 哈希链

### 2.2 改动的部分（仅此三组，全部预注册在 config）

| 旋钮 | pilot | exp 1.5 | 一行理由 |
|---|---|---|---|
| Stage-A lr | 1e-6 | **1e-5** | 10× 剂量；权重位移预期 ~6e-5 量级，进入"能造成损耗"的区间，同时仍在 RLVR 小模型常用范围内 |
| Stage-A 步数 / ckpt | 200 步 / 5 个 | **500 步 / 8 个**（0,25,50,100,200,300,400,500） | 覆盖 reward 饱和后区间；前 5 个 ckpt 网格与 pilot 相同（跨实验可对齐） |
| 适应 outcome | 100 题 × 1 种子 | **300 题 × 3 种子**（42/43/44），适应 ckpt 取 6 个（0,50,100,200,300,500） | SE(单次)5.0→2.9pp；SE(三种子均值差)≈2.5pp；曲线仍用 legacy-100（省时且与 exp-1 曲线同尺） |

预算取舍说明：适应只做 6/8 个 checkpoint 是算力约束下的选择——18 次
适应 ≈ 21 小时 GPU 已是上限；被跳过的 25/400 两点仍有完整 Q 测量，
若主分析需要可事后补跑（每点 3.5 GPU 时）。

### 2.3 明确不做、留给团队的（不许静默决定）

- KL β>0 对照 arm：config 里备有 stub（β=0.04、200 步、仅 ckpt-200 做
  3 种子适应，`enabled: false`）。**是否启用是团队决策**，Slack 拍板后
  一条命令可跑（追加 ≈8 GPU 时）。
- base vs Instruct、GRPO vs SFT 适应、SVAMP 是否太近：全部维持 pilot
  默认并已写入 config 的 `open_team_questions`。注意 §4 的 MC2 若失败,
  将成为"SVAMP 太近"的正面证据,直接喂给这个待决项。

## 3. 统计功效的纸面推算（为什么这次能测到东西）

Pilot 的教训是"先跑再发现测不到"；这次先算：

- 单次 delta 的种子噪声（pilot 实测,ckpt-0 三种子）：SD ≈ 0.031。
- 300 题评估把二项分量从 5.0pp 压到 2.9pp,保守仍取 SD ≈ 0.03。
- 三种子均值的 SE ≈ 0.03/√3 ≈ **0.017**；两个 checkpoint 均值之差的
  SE ≈ **0.025**。
- 因此若剂量递增真的造成 ≥5pp 的适应力差异（MC2 阈值）,其信噪比 ≈ 2σ,
  6 点排名大概率被正确排序；若真实效应 <2.5pp,本实验仍测不到——
  这是已知且接受的灵敏度下限,写明于此以免事后争论。

## 4. 预注册分析（跑之前定死,phase 4 代码原样执行）

**门控（manipulation checks）——先问"实验条件是否成立"再看相关：**

- **MC1（剂量动了 Q 吗）**：|erank_L12(ckpt-500)/erank_L12(ckpt-0) − 1|
  ≥ **10%**。阈值依据：pilot 同层最大位移是 −7.4%（win,200 步,lr 1e-6）,
  10% = 明确超出 pilot 已见区间,即"新剂量进入了新区间"。
- **MC2（适应力真的拉开差距了吗）**：存在某个后期 checkpoint,其三种子
  平均 delta 比 ckpt-0 的三种子平均 delta **低 ≥ 0.05**（≈2× 均值差 SE,
  见 §3）。这就是 Madhur framing 里的"fixed-budget adaptability 相对
  checkpoint-0 下降"的操作化。

**主分析**：Spearman rho(erank_L12, 三种子平均 delta),n=6。
**解释规则（2×2,预注册）**：

| | MC2 通过（适应力下降了） | MC2 失败（人人适应都好） |
|---|---|---|
| **MC1 通过**（Q 动了） | rho 是 RQ1 的有效读数,按值解释 | Q 动而适应力不降 → "SVAMP 太近"证据 / Q 可能不跟踪损伤;rho 仅描述性 |
| **MC1 失败**（Q 没动） | 适应力降而 erank 无信号 → **反对 erank_L12 作为预警指标的有效证据** | 剂量仍不足 → 升级决策交团队（1.5B / Colab / 更狠配方） |

**次级（全部描述性）**：per-seed rho、种子两两 delta 排名相关（噪声
复核）、between-checkpoint vs within-checkpoint(种子) 方差分解（直接
喂给正式实验的功效计算）、dormant fraction（预期仍为 0,测量免费）、
legacy-100 桥梁分数、完整 dashboard 信号留存（供日后 detector bake-off）。

**中途处置规则**：若 Stage A 在 lr 1e-5 下触发安全停（reward/熵坍缩、
非有限 loss）——保留全部诊断,已存 checkpoint 照常进入后续 phase,
结果标注 truncated,决策上报团队。**坍缩本身是观察对象,不是要静默
重试的故障。**

## 5. 预算、平台与执行顺序

### 5.1 平台选择：Windows RTX 4070 Laptop（结论）

依据 pilot 实测吞吐：

| | 4070 Laptop（cuda v2） | M3 Max（cpu fp32） |
|---|---|---|
| Stage-A 每步 | ≈93 s（5.18h/200 步） | ≈300 s |
| Phase 1（500 步） | ≈13 h | ≈42 h |
| Phase 2（8 ckpt + 2 敏感性） | <15 min | ~1 h |
| Phase 3（18 次适应,每次≈70 min\*） | ≈21 h | ≈75 h+ |
| **合计 active** | **≈34–35 h,可分 4–6 晚** | **≈120 h,不可行** |

\* 适应每次 ≈62 min（pilot 实测）+ 300 题前后各一次评估的增量 ≈8 min;
曲线评估仍用 100 题所以不增时。

MPS 已被 pilot 验证为 CPU 同速（transformers generate 的逐 token 同步
开销）,不考虑。**mac 只用于 `--smoke` 管线验证。**

### 5.2 磁盘与显存

- 显存：批几何与 pilot v2 完全一致,峰值 ≈7 GiB,8 GiB 卡内;OOM 处置
  沿用 v2 经验（保留失败 attempt,新进程原配方重跑该 ckpt）。
- 磁盘：8 个 fp32 checkpoint ≈15 GiB + 滚动 trainer 状态 ≈4 GiB;
  适应阶段默认**验证通过后自动删除各自 trainer/ 子目录**（科学产物
  全在外面;`--keep-trainer-dirs` 可关闭）。runner 开跑前强制检查
  ≥30 GiB 空闲。

### 5.3 执行顺序（预注册,endpoint-first）

Phase 3 的 checkpoint 顺序：**0 → 500 → 200 → 100 → 50 → 300**
（每个 checkpoint 内 seed 42→43→44 连跑）。理由：前两组跑完即可提前
读出 MC2 的主要信息（两端点的适应力差）,若 500 步端点毫无退化迹象,
可提前把"剂量仍不足"的信号发给团队,不必等全部 18 次跑完。

### 5.4 记账（硬性要求不变）

每晚跑完在 `eaaj-pilot/compute_log.md` 追加条目（日期、GPU、时长、
phase）;GPU telemetry CSV 随 run 产物提交;所有产物随跑随 commit
（沿用 pilot 的 "stageb: record …" 工作流）。

## 6. 风险与预案

| 风险 | 概率 | 预案 |
|---|---|---|
| lr 1e-5 训练不稳（熵坍缩/reward 崩到 0） | 中 | SafetyCallback 自动停;诊断保留;已存 ckpt 照常测量与适应,结果标 truncated;上报团队。坍缩点本身就是 Q 的理想测试场 |
| reward 早饱和,后 300 步"没学新东西" | 中 | 正是想要的危险区;dashboard+sentinel 证明权重仍在动;饱和后 ckpt 是 RQ1 最关键的样本 |
| 500 步后 delta 仍无差异（MC2 失败） | 中 | 2×2 矩阵已预注册该格的解释与升级路径;实验不白跑——直接决定正式实验的设计走向 |
| 断电/中断 | 中 | 与 pilot 相同的 resume 语义（trainer 滚动 ckpt;paged optimizer 下 moments 不恢复,边界记录在案）;phase 3 每个 (ckpt,seed) 原子完成,可 `--adapt-checkpoint/--adapt-seed` 精确续跑 |
| 磁盘写爆 | 低 | 开跑前 30 GiB 检查 + trainer 目录自动清理 |

## 7. 与正式实验（experiment 2）的关系

Exp 1.5 无论结果落在 2×2 哪一格,都直接决定 experiment 2 的三个设计
参数：(a) Stage-A 需要的最小剂量（lr×步数）;(b) 每 checkpoint 需要的
种子数与评估集大小（由 §4 方差分解给出）;(c) Stage-B 任务家族是否
必须换（MC2 格子的证词）。Colab 预算继续原封不动地留给 experiment 2。

## 8. 产物布局与操作手册

Run 目录：`eaaj-pilot/outputs/exp15_cuda_grpo_gsm8k_<confighash>/`
（前缀区别于 pilot;本 runner 从不写 `ACTIVE_RUN.txt`,与 pilot 工具链
互不干扰）。内部布局与 pilot 同构:`measurements/`、
`adaptation_seed{42,43,44}/ckpt-N/`、`analysis/`、dashboard/sentinel/
preflight/phase-complete 标记。

操作命令见 `README.md`;先在任一台机器跑 `--smoke` 验证管线,再在
4070 上按 phase 1→2→3→4 执行。
