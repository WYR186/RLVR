# Windows RTX 4070 Stage-B 多 seed 重复实验计划

**状态：待执行，不是已完成实验。**

**执行机器：Windows 11 + RTX 4070 Laptop 8 GiB。**

**源 Stage-A run：** `eaaj-pilot/outputs/local_cuda_grpo_gsm8k_e9b0b52aab6c`。

**新增 Stage-B seeds：** `43`、`44`；已有 `seed=42` 保持不动。

**预计 GPU active time：** 每个 seed 约 5–6 小时，两个 seed 合计约 10–12 小时，另加约 1 小时 smoke、验证和提交时间。

---

## 0. 给 Windows agent 的任务摘要

不要重跑 GSM8K Stage A，也不要修改或覆盖现有 v2 产物。本次任务只做：

1. 先修复“adaptation 提前停止却仍写成 50-update complete”的完成性漏洞，并加测试。
2. 增加一个独立的 Stage-B repeat runner，使不同 seed 写入互不覆盖的目录。
3. 在 RTX 4070 上先做短 smoke，再固定现有五个 Stage-A checkpoint，分别运行 SVAMP GRPO adaptation seeds 43、44。
4. 每个 checkpoint 都必须是独立进程、完整 50 updates、五个 sentinel window 全部有效。
5. 汇总 seeds 42/43/44 的 outcome 方差和 checkpoint 排序稳定性；不得把这三个 seed 当成三条独立 Stage-A 轨迹。
6. 代码和小型 JSON/CSV/日志频繁 push 到 GitHub；模型权重另行保存在 Windows 本地或外部模型存储，不能假设 GitHub 保存了 `*.safetensors`。

如果任何 hard gate 失败，停止当前阶段、保留失败证据，不得为了跑完而静默改变 LR、batch geometry、optimizer、数据、长度上限或 safety threshold。

---

## 1. 为什么跑这个，而不是重跑完整 Stage A

现有 Windows v2 已经提供五个有效变化的 Stage-A checkpoint，但每个 checkpoint 只有一个 SVAMP adaptation seed。100 题 endpoint 的小幅变化和 GRPO rollout 随机性足以改变五点排序；当前 `+3pp` 到 `+12pp` 的差异尚不能区分稳定效应与单 seed 噪声。

本次实验回答一个窄问题：

> 固定同一组 Windows v2 Stage-A checkpoint 后，50-update SVAMP adaptability 的 checkpoint 排序和“ckpt-200 没有 stall”的观察，在 adaptation seeds 42/43/44 下是否稳定？

本次实验**不能**：

- 修复 Windows v2 Stage-A 在 25/50/75 附近没有完整恢复 optimizer moments/RNG 的限制；
- 把三个 Stage-B seed 当成三个独立模型或三个独立 Stage-A trajectory；
- 证明或否定 RLVR 普遍影响模型的学习能力；
- 将 GSM8K→SVAMP 的数学任务迁移解释成通用 plasticity；
- 私自决定 ProntoQA、SFT、KL baseline、1.5B 等开放团队问题。

它的价值是先量化当前 outcome 的重复性，并同时把 A100 前必须具备的 completion/resume/persistence 逻辑跑硬。

---

## 2. 冻结的科学设置

除 adaptation seed 外，必须与 Windows v2 完全一致：

| 项目 | 固定值 |
|---|---:|
| Source checkpoints | 0 / 25 / 50 / 100 / 200 |
| Stage-B task | SVAMP |
| Train/eval split | 现有冻结的 256 train / 100 eval |
| Algorithm | GRPO |
| Budget | 50 optimizer updates |
| Evaluation cadence | 10 / 20 / 30 / 40 / 50 |
| Learning rate | `1e-6` |
| KL beta | `0.0` |
| Sampling | temperature `0.7`, top-p `1.0` |
| Generations | 8 |
| Prompt/completion cap | 512 / 512 |
| Master parameter dtype | float32 |
| Compute autocast | bfloat16 |
| Optimizer | `paged_adamw_8bit` |
| Gradient checkpointing | enabled |
| RTX 4070 geometry | micro-batch 4 × grad-accum 16 |
| New seeds | 43, then 44 |

不要修改 `pilot_config.json` 中 canonical `seed=42`。repeat runner 应通过显式 CLI 参数传入 seed，并把 seed 写入新的 manifest/summary。

---

## 3. 产物隔离规则

现有 seed-42 目录保持原样：

```text
local_cuda_grpo_gsm8k_e9b0b52aab6c/
  adaptation/ckpt-*/                 # existing seed 42; read-only
```

新增产物必须写入：

```text
local_cuda_grpo_gsm8k_e9b0b52aab6c/
  adaptation_repeats/
    seed-43/
      repeat_manifest.json
      ckpt-0/
      ckpt-25/
      ckpt-50/
      ckpt-100/
      ckpt-200/
    seed-44/
      repeat_manifest.json
      ckpt-0/
      ckpt-25/
      ckpt-50/
      ckpt-100/
      ckpt-200/
  repeat_analysis/
    stageb_seed_results.csv
    stageb_seed_summary.csv
    stageb_seed_correlations.csv
    stageb_seed_analysis.json
```

任何失败 attempt 都不能与成功目录混写。失败后应保留为：

```text
ckpt-50_failed_oom_YYYYMMDD_HHMMSS/
ckpt-25_failed_safety_YYYYMMDD_HHMMSS/
```

随后才能在清空 GPU 的新进程中重新创建标准 `ckpt-N/`。不得把两个 attempt 的 dashboard、curve 或 trainer state 拼接在一起。

---

## 4. 正式运行前必须完成的代码修复

Windows agent 可以实现下列工程修改，但不得顺带更改科学设置。

### 4.1 修复 adaptation completion contract

在 `src/adaptation.py` 中，`trainer.train(...)` 返回后必须读取：

```python
actual_updates = int(trainer.state.global_step)
```

只有 `actual_updates == budget_updates` 时才允许：

- 运行正式 after eval；
- 写 `summary.json`；
- 把 `completion_status` 标为 `complete`。

如果实际步数不足：

- 写 `incomplete.json`，包含 `requested_updates`、`actual_updates`、`reason`、`safety_stop_path`、时间和 run path；
- 不写 `summary.json`；
- 抛出非零异常，使 PowerShell wrapper 失败；
- 不允许上层生成 `phase3_complete` 或 repeat-complete marker。

正式 summary 至少新增：

```json
{
  "requested_updates": 50,
  "actual_updates": 50,
  "completion_status": "complete"
}
```

保留旧字段 `budget_updates` 以兼容已有分析，但新分析必须以 `actual_updates` 和 `completion_status` 为 gate。

### 4.2 safety-stop 后不得自动续跑成“完整科学 run”

如果 output 目录存在 `safety_stop.json` 且没有有效 complete summary，repeat runner 必须拒绝在同一目录 resume。保留目录作为失败 attempt，使用新目录从源 Stage-A checkpoint 重新开始。

普通进程中断只有在 trainer checkpoint 含完整、可加载的 optimizer/scheduler/RNG state 时才允许 resume。`paged_adamw_8bit` 若仍只能 `save_only_model=True`，则本 repeat 视为不可恢复：中断后保留 attempt，从该 source checkpoint 重新开始该 50-step adaptation。

### 4.3 phase-complete 判定

任何 complete marker 都必须逐一验证：

- summary 存在；
- `completion_status == "complete"`；
- `actual_updates == 50`；
- curve steps 恰好为 `[10, 20, 30, 40, 50]`；
- sentinel steps 恰好为 `[10, 20, 30, 40, 50]` 且全部 `updates_effective=true`；
- 不存在 `safety_stop.json`。

不能再以“存在五个 summary 文件”作为唯一完成条件。

### 4.4 新增 repeat runner

建议新增：

```text
eaaj-pilot/scripts/run_stageb_seed_repeat.py
eaaj-pilot-win4070/run_stageb_repeat.ps1
```

Python CLI contract：

```powershell
python scripts\run_stageb_seed_repeat.py `
  --source-run outputs\local_cuda_grpo_gsm8k_e9b0b52aab6c `
  --seed 43 `
  --checkpoint 0
```

要求：

- 一次只运行一个 seed × checkpoint；
- 从 source run 的 `config.json` 读取 Windows v2 execution profile；
- 调用同一个 `run_fixed_budget_adaptation`；
- 输出到 `adaptation_repeats/seed-N/ckpt-M`；
- 创建 `repeat_manifest.json`，记录 source manifest/config hash、git SHA、seed、完整 recipe、Python/package/GPU 版本；
- 已有 complete summary 时只做验证并安全退出；
- partial/failed artifacts 存在时拒绝覆盖；
- 非 RTX CUDA、source checkpoint 缺少 `model.safetensors`、配置不匹配时立即退出。

PowerShell wrapper 要复用现有 keep-awake 和每 60 秒 `nvidia-smi` telemetry 逻辑，日志名必须包含 seed 和 checkpoint，例如：

```text
gpu_20260713_230000_stageb_seed43_ckpt0.csv
```

### 4.5 最低测试要求

不得只靠人工检查。至少添加以下快速测试：

1. fake trainer 在 step 30 结束：写 `incomplete.json`、不写 summary、函数失败。
2. step 50 完成：summary 包含 requested/actual/status。
3. phase/repeat completion validator 拒绝 `actual_updates=30`。
4. seed-43 与 seed-44 输出目录不会互相覆盖，也不会覆盖 seed-42。
5. safety-stop 目录不能原地 resume。
6. curve 或 sentinel 缺少 step 40/50 时验证失败。

跑完整测试：

```powershell
cd D:\algoverse\eaaj-pilot
& D:\algoverse\.conda\envs\eaaj-win4070\python.exe -m pytest tests -v
```

---

## 5. Windows 同步与环境检查

### 5.1 拉取计划并创建工作分支

```powershell
cd D:\algoverse
git status --short --branch
git pull --ff-only origin main
git switch -c codex/win4070-stageb-seed-repeats
```

如果已有同名分支，先检查它是否属于本任务；不要 force-reset，不要覆盖其他 agent 的改动。

### 5.2 固定 Python

```powershell
$Python = "D:\algoverse\.conda\envs\eaaj-win4070\python.exe"
if (-not (Test-Path $Python)) { throw "eaaj-win4070 Python missing" }
& $Python -c "import torch,trl,transformers,bitsandbytes; print(torch.__version__, torch.version.cuda, trl.__version__, transformers.__version__, bitsandbytes.__version__); print(torch.cuda.get_device_name(0))"
```

期望仍是 RTX 4070 Laptop、CUDA build 可用、TRL/Transformers 与 source manifest 同一版本。若包版本漂移，先记录并判断是否需要恢复环境；不得静默在新版本上生成同一个 repeat stratum。

### 5.3 检查 source weights

GitHub 不包含 `*.safetensors`。Windows 本地必须仍有：

```powershell
$Run = "D:\algoverse\eaaj-pilot\outputs\local_cuda_grpo_gsm8k_e9b0b52aab6c"
0,25,50,100,200 | ForEach-Object {
  $P = Join-Path $Run "ckpt-$_\model.safetensors"
  if (-not (Test-Path $P)) { throw "missing source checkpoint weights: $P" }
}
```

同时确认 source `manifest.json` 的 platform 为 Windows、GPU 为 RTX 4070、config hash 为 `e9b0b52aab6c`。缺任何 source weight 就停止，不得从 GitHub 的 config-only checkpoint 假装恢复。

### 5.4 GPU 和机器状态

- 插电并开启 keep-awake；
- 关闭占用独显的游戏、浏览器 GPU 标签页和其他训练进程；
- `nvidia-smi` 确认空闲显存；
- 不允许两个 trainer 并行；
- 每个 checkpoint 使用独立 Python 进程，避免 ckpt-50 曾出现的跨-run allocator/OOM 污染。

---

## 6. Smoke：正式 seed 前的硬 gate

正式 50-step repeat 前，使用单独的 `outputs/smoke_*` 路径完成：

1. 2–5 update 的正常完成 smoke，验证新 summary contract。
2. 一个刻意在较早 step 结束的 fake/controlled smoke，验证 incomplete contract。
3. output collision 测试：同 seed/ckpt partial 目录存在时 runner 必须拒绝覆盖。
4. telemetry 能实时落盘，即使 wrapper 被 Ctrl+C 也保留已有行。
5. Git 只会 stage 代码、小型 JSON/CSV/MD，不会 stage model weights。

Smoke 完成后先提交并 push 工作分支：

```powershell
cd D:\algoverse
git status --short
git add eaaj-pilot\src eaaj-pilot\scripts eaaj-pilot\tests eaaj-pilot-win4070\run_stageb_repeat.ps1
git commit -m "win4070: harden fixed-budget adaptation repeats"
git push -u origin codex/win4070-stageb-seed-repeats
```

未通过 smoke，不得开始 seed 43。

---

## 7. 正式运行顺序

每个 checkpoint 单独启动，跑完立即验证和同步。为尽早比较最关键的 base 与 final checkpoint，顺序固定为：

```text
seed 43: ckpt 0 → 200 → 25 → 50 → 100
人工/agent 审计 seed 43
seed 44: ckpt 0 → 200 → 25 → 50 → 100
```

### 7.1 Seed 43

示例命令；实际使用新 wrapper 的最终参数名，但不得改变本计划的路径与科学设置：

```powershell
cd D:\algoverse\eaaj-pilot-win4070
powershell -ExecutionPolicy Bypass -File .\run_stageb_repeat.ps1 -Seed 43 -Checkpoint 0
powershell -ExecutionPolicy Bypass -File .\run_stageb_repeat.ps1 -Seed 43 -Checkpoint 200
powershell -ExecutionPolicy Bypass -File .\run_stageb_repeat.ps1 -Seed 43 -Checkpoint 25
powershell -ExecutionPolicy Bypass -File .\run_stageb_repeat.ps1 -Seed 43 -Checkpoint 50
powershell -ExecutionPolicy Bypass -File .\run_stageb_repeat.ps1 -Seed 43 -Checkpoint 100
```

### 7.2 Seed 44

Seed 43 五个 checkpoint 全部通过验收并完成 Git 同步后，才运行：

```powershell
cd D:\algoverse\eaaj-pilot-win4070
powershell -ExecutionPolicy Bypass -File .\run_stageb_repeat.ps1 -Seed 44 -Checkpoint 0
powershell -ExecutionPolicy Bypass -File .\run_stageb_repeat.ps1 -Seed 44 -Checkpoint 200
powershell -ExecutionPolicy Bypass -File .\run_stageb_repeat.ps1 -Seed 44 -Checkpoint 25
powershell -ExecutionPolicy Bypass -File .\run_stageb_repeat.ps1 -Seed 44 -Checkpoint 50
powershell -ExecutionPolicy Bypass -File .\run_stageb_repeat.ps1 -Seed 44 -Checkpoint 100
```

不要使用一个长驻 Python 进程循环五个 checkpoint；此前 ckpt-50 的 OOM 说明逐进程释放 CUDA 状态更安全。

---

## 8. 每个 checkpoint 的验收清单

一个 seed × checkpoint 只有同时满足以下条件才算完成：

- [ ] wrapper exit code 为 0；
- [ ] `summary.json` 存在；
- [ ] `completion_status == "complete"`；
- [ ] `requested_updates == 50`；
- [ ] `actual_updates == 50`；
- [ ] `safety_stop.json` 不存在；
- [ ] dashboard 有训练 steps 1–50；
- [ ] curve steps 恰好为 10/20/30/40/50；
- [ ] sentinel steps 恰好为 10/20/30/40/50；
- [ ] 五个 sentinel 均为 `updates_effective=true`；
- [ ] loss/grad norm 无 NaN/Inf；
- [ ] baseline accuracy 与 seed-42 同 source checkpoint 一致；
- [ ] telemetry CSV 非空并已复制到 seed-specific artifact 目录；
- [ ] compute log 已添加 date/GPU/seed/checkpoint/duration/status；
- [ ] 失败 attempt 没有被混入成功目录。

Seed-42 的 expected greedy baselines：

| Source ckpt | Expected SVAMP before |
|---:|---:|
| 0 | 0.53 |
| 25 | 0.51 |
| 50 | 0.56 |
| 100 | 0.55 |
| 200 | 0.54 |

同一 source checkpoint 的 greedy baseline 若变化，先检查 source weights、model dtype、evaluation code和数据 hash；不得继续把它解释成 seed 差异。

---

## 9. 停机与失败处理

立即停止当前 run 的条件：

- sentinel 任一窗口 `updates_effective=false`；
- `actual_updates < 50`；
- safety callback 触发；
- NaN/Inf；
- source baseline 不匹配；
- CUDA OOM；
- telemetry/manifest/output path 指向错误 run；
- 发现 seed-42 或其他 seed 目录被覆盖。

失败后：

1. 不删除证据。
2. 将失败目录改为带原因和 timestamp 的 attempt 目录。
3. 写 compute log deviation。
4. commit/push 小型失败日志和诊断信息。
5. 清空 GPU，启动新进程。
6. 只有“相同 science/config 的 fresh-process retry”可直接重试。

不得未经团队同意采用 micro-batch 2 × grad-accum 32、降低 generation 数、缩短 completion、换 optimizer 或关闭 safety gate。即便 effective batch 保持不变，这些也是 execution deviation，必须先记录并获得确认。

---

## 10. GitHub 频繁同步规则

本任务在工作分支执行，避免长实验期间直接污染 `main`。至少在以下节点 push：

1. completion fix + tests + smoke 完成；
2. seed 43 的 ckpt 0/200 完成；
3. seed 43 全部完成；
4. seed 44 的 ckpt 0/200 完成；
5. seed 44 全部完成；
6. repeat analysis 完成。

每次提交前：

```powershell
cd D:\algoverse
git status --short
git add eaaj-pilot\outputs\local_cuda_grpo_gsm8k_e9b0b52aab6c\adaptation_repeats
git add eaaj-pilot\outputs\local_cuda_grpo_gsm8k_e9b0b52aab6c\repeat_analysis
git add eaaj-pilot\compute_log.md
git status --short
git commit -m "win4070 stageb repeats: seed 43 checkpoint 0"
git push origin codex/win4070-stageb-seed-repeats
```

根据当前 `.gitignore`：

- JSON/JSONL/CSV/MD/config/manifest 应提交；
- `*.safetensors`、optimizer `*.pt` 不会进入 Git；
- Git push 不能替代模型 checkpoint 备份；
- 不得 `git add -f` 大权重；
- 不得 force-push。

如果未来机器会被清空，source Stage-A weights 和任何必须恢复的 trainer state 必须另存 Drive/Hugging Face；先实际完成一次上传→删除临时副本→下载→hash 核对，不能只写“计划上传”。

---

## 11. Repeat 分析口径

所有十个新 runs 完整后再生成 repeat analysis。Seed 42 从现有 `adaptation/ckpt-*` 读取；seeds 43/44 从 `adaptation_repeats/` 读取。

### 11.1 必需表格

`stageb_seed_results.csv`：一行一个 seed × checkpoint，至少包含：

```text
seed, checkpoint, acc_before, acc_after, delta_acc,
acc_step10, acc_step20, acc_step30, acc_step40, acc_step50,
normalized_curve_auc, requested_updates, actual_updates,
completion_status, wall_seconds
```

`normalized_curve_auc` 预先定义为 steps 10/20/30/40/50 accuracy 的算术平均；另可记录相对 baseline 的 `improvement_auc`。不要根据结果临时挑一种 AUC。

`stageb_seed_summary.csv`：每个 checkpoint 聚合 seeds 42/43/44：

- mean/std/min/max of `acc_after`；
- mean/std/min/max of `delta_acc`；
- mean/std of curve AUC；
- ckpt-200 与 ckpt-0 的逐 seed 差值。

### 11.2 相关与稳定性

必须报告：

1. 每个 seed 单独的 `rho(erank_L12, delta_acc)`；
2. 每个 seed 单独的 `rho(erank_L22, delta_acc)`，标记 exploratory；
3. Q 与 across-seed mean delta 的 descriptive Spearman；
4. 三个 outcome vectors 两两之间的 checkpoint-rank Spearman；
5. ckpt-200 的 delta 是否在每个 seed 中低于/高于 ckpt-0；
6. endpoint 和 curve AUC 是否给出相同方向。

不要把 15 行 seed×checkpoint 当成 15 个独立 observation 计算普通 Spearman p 值。Stage-B seeds 共享 source model，checkpoint 又共享 Stage-A trajectory。n=5 的 p 值即使计算也必须标为描述性；优先展示三条 seed 曲线、均值和离散度。

### 11.3 预先定义的解释规则

- 若 ckpt-200 相对 ckpt-0 的 delta 与 AUC 在三个 seeds 中方向一致，且 checkpoint 排序相对稳定，则当前 Windows v2 下“后期 checkpoint 未显示 fixed-budget stall”的观察更可信。
- 若 seeds 之间排序频繁翻转或同一 checkpoint delta 的标准差与 checkpoint 间差异同量级，则当前单-seed相关主要由 Stage-B 噪声驱动，应优先增加 repeats/eval stability，而不是解释 Q 的符号。
- 若 delta 与 final accuracy/AUC 给出不同结论，明确报告 ceiling/headroom 混淆，不挑选最支持假设的 outcome。
- 无论结果如何，本实验仍不能单独证明 Q 是 stall predictor，因为 Stage-A 独立 trajectory 数仍为 1。

---

## 12. 完成定义

本任务只有满足全部条件才算完成：

- [ ] completion contract 已修复并有回归测试；
- [ ] Windows smoke 全部通过；
- [ ] seed 43 五个 checkpoint 均完整；
- [ ] seed 44 五个 checkpoint 均完整；
- [ ] 每个 run 有 manifest、dashboard、curve、sentinel、summary、telemetry；
- [ ] compute log 完整；
- [ ] repeat analysis 四个产物生成；
- [ ] 报告明确区分一个 Stage-A trajectory 与三个 Stage-B seeds；
- [ ] 工作分支已 push，所有小型 artifacts 可从 GitHub 恢复；
- [ ] 权重备份状态被明确记录；
- [ ] 未更改开放团队问题或科学参数。

完成后先由 Aaron/主 agent 审计 raw artifacts，再决定是否合并到 `main`，以及是否据此申请 A100。
