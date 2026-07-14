# Windows RTX 4070 Stage-B 多 seed 重复实验阶段报告

- 日期：2026-07-14
- 源 Stage-A run：`eaaj-pilot/outputs/local_cuda_grpo_gsm8k_e9b0b52aab6c`
- 执行分支：`codex/win4070-stageb-seed-repeats`
- 报告性质：阶段性状态报告，不是最终 repeat analysis

## 摘要

Stage-B 多 seed 重复实验的工程加固、GPU smoke 和 seed 43 五个 checkpoint 已完成；seed 44 目前只完成 ckpt-0，剩余 ckpt-200/25/50/100 尚未运行。因此，计划中的 seeds 42/43/44 汇总分析尚不具备完成条件。

当前最重要的科学观察是：seed 42 与 seed 43 的 endpoint 排名并不稳定。seed 42 中 ckpt-200 和 ckpt-25 的适应增益最大；seed 43 中 ckpt-50 和 ckpt-25 最大。两组五点 delta 排名的描述性 Spearman 约为 `-0.50`。这支持继续完成 seed 44 和预先定义的 repeat analysis，而不支持现在就把单 seed 的 checkpoint 排名解释成稳定效应。

所有已完成的新 run 都保持冻结的科学设置，没有为了速度、显存或结果改变学习率、batch geometry、optimizer、数据、生成数、长度上限或 safety gate。

## 1. 当前完成度

| 项目 | 状态 |
|---|---|
| completion contract 修复与回归测试 | 完成 |
| 独立 seed/checkpoint runner 与 PowerShell wrapper | 完成 |
| 2-update RTX 4070 GPU smoke | 完成 |
| seed 43：ckpt 0/200/25/50/100 | 5/5 完成并已推送工作分支 |
| seed 43 整组官方 validator 审计 | 5/5 通过 |
| seed 44：ckpt 0 | 完成，官方 validator 通过，纳入本次发布 |
| seed 44：ckpt 200/25/50/100 | 未运行 |
| seeds 42/43/44 repeat analysis | 未生成；必须等 seed 44 全部完成 |

当前计划完成度按正式新 run 计为 `6/10`；按 seed 44 计为 `1/5`。

## 2. 冻结的实验设置

| 项目 | 固定值 |
|---|---:|
| Stage-A checkpoints | 0 / 25 / 50 / 100 / 200 |
| Stage-B task | SVAMP，256 train / 100 eval |
| Algorithm | GRPO |
| Budget | 每个 checkpoint 独立 50 optimizer updates |
| Eval/sentinel steps | 10 / 20 / 30 / 40 / 50 |
| Learning rate / KL beta | `1e-6` / `0.0` |
| Sampling | temperature `0.7`，top-p `1.0`，8 generations |
| Prompt/completion cap | 512 / 512 |
| Precision | float32 master + bfloat16 autocast |
| Optimizer | `paged_adamw_8bit` |
| RTX 4070 geometry | micro-batch 4 × grad-accum 16 |

不同 Stage-B seed 共享同一条 Stage-A trajectory，不能当作独立模型或独立 Stage-A 样本。

## 3. 工程修改与执行保护

工作分支相对原 `main` 的主要修改如下：

1. 强化 fixed-budget completion contract：只有 `actual_updates == requested_updates == 50` 才能写 complete summary；提前停止写 incomplete evidence 并失败退出。
2. 增加目录 validator，严格检查 summary、完整训练步、五个 curve 点、五个有效 sentinel、baseline、recipe、seed/checkpoint identity、finite dashboard、safety-stop 缺失和 telemetry。
3. 增加 seed-specific manifest、原子 attempt lock、失败目录保护和成功目录幂等校验，避免不同 seed 或 retry 相互覆盖。
4. 增加 Windows RTX 4070 repeat/smoke wrapper，每个 checkpoint 使用独立进程并附加 GPU telemetry。
5. 修复 PowerShell 将 native Python warning output 误判为 fatal error 的问题；现在以 native process exit code 为准，同时仍保留 stderr 诊断信息。
6. 增加 Stage-B repeat 回归测试；合并前完整复测为 62 passed，真实 GPU 2-update smoke 通过。

上述修改未改变冻结的科学参数。

## 4. 已完成结果

### 4.1 Endpoint 结果

| Stage-B seed | Stage-A ckpt | before | after 50 updates | delta | validator |
|---:|---:|---:|---:|---:|---|
| 42 | 0 | 0.53 | 0.59 | +0.06 | 原 v2 完成产物 |
| 42 | 25 | 0.51 | 0.62 | +0.11 | 原 v2 完成产物 |
| 42 | 50 | 0.56 | 0.59 | +0.03 | 原 v2 完成产物 |
| 42 | 100 | 0.55 | 0.60 | +0.05 | 原 v2 完成产物 |
| 42 | 200 | 0.54 | 0.66 | +0.12 | 原 v2 完成产物 |
| 43 | 0 | 0.53 | 0.57 | +0.04 | 通过 |
| 43 | 25 | 0.51 | 0.59 | +0.08 | 通过 |
| 43 | 50 | 0.56 | 0.65 | +0.09 | 通过 |
| 43 | 100 | 0.55 | 0.61 | +0.06 | 通过 |
| 43 | 200 | 0.54 | 0.59 | +0.05 | 通过 |
| 44 | 0 | 0.53 | 0.53 | 0.00 | 通过 |

seed 43 的五个 run 均满足 50/50 updates、固定 baseline、完整 steps 1–50、curve steps 10/20/30/40/50、五个有效更新 sentinel、无 safety stop、finite dashboard 和非空 telemetry。seed 44 ckpt-0 满足同样门槛。

### 4.2 Seed 43 的完整评估曲线

| ckpt | before | step 10 | 20 | 30 | 40 | 50 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.53 | 0.55 | 0.51 | 0.57 | 0.56 | 0.57 |
| 25 | 0.51 | 0.54 | 0.54 | 0.58 | 0.57 | 0.59 |
| 50 | 0.56 | 0.58 | 0.65 | 0.65 | 0.61 | 0.65 |
| 100 | 0.55 | 0.61 | 0.64 | 0.63 | 0.59 | 0.61 |
| 200 | 0.54 | 0.62 | 0.54 | 0.58 | 0.60 | 0.59 |

seed 44 ckpt-0 的曲线为 `0.53 → 0.52 → 0.58 → 0.55 → 0.57 → 0.53`（before，steps 10/20/30/40/50）。单个 endpoint 回到 baseline，说明只看 endpoint 会受到明显训练/评估波动影响。

## 5. 当前解释

1. seed 43 的所有 checkpoint 最终 delta 均为正，但大小和 seed 42 明显不同。
2. seed 42 的 delta 排名为 ckpt-200 > 25 > 0 > 100 > 50；seed 43 为 ckpt-50 > 25 > 100 > 200 > 0。
3. 两个已完整 seed 的五点排名描述性 Spearman 约为 `-0.50`，目前没有稳定 checkpoint 排名证据。
4. seed 42 中 ckpt-200 比 ckpt-0 多 +6pp delta；seed 43 中 ckpt-200 比 ckpt-0 多 +1pp。方向暂时一致，但幅度不稳定。
5. seed 44 ckpt-0 的 0pp endpoint 进一步说明 Stage-B rollout/eval noise 不可忽略；在其余四点完成前不能判断 seed 44 的 checkpoint 排名。
6. 当前证据仍不能证明或否定 Q 是 stall predictor，也不能把三个 Stage-B seeds 当成三个独立 Stage-A trajectories。

## 6. 异常与处理

- seed 43 ckpt-0 的第一次 wrapper attempt 在进入正式训练前失败，原因是 PowerShell 把 Python/Triton warning stream 当成异常。失败证据被隔离保留；修复 wrapper 后使用完全相同的科学配置 fresh-process 重跑成功。
- 后续五个 seed 43 run 和 seed 44 ckpt-0 均未触发 CUDA OOM 或 safety stop。
- `*.safetensors` 和 optimizer state 继续按 `.gitignore` 留在 Windows 本地；GitHub 中保存的是代码、小型 config/manifest、JSON/JSONL/CSV、trainer state metadata 和报告，不能把 Git push 当作模型权重备份。

## 7. 计算成本

seed 43 五个正式 run 合计约 4.9 GPU active hours；seed 44 ckpt-0 约 1.2 小时。加上 smoke 和初次失败诊断，当前阶段已使用超过 6 小时 RTX 4070 active time。

## 8. 剩余工作

按冻结顺序继续运行：

```text
seed 44: ckpt 200 → 25 → 50 → 100
```

每个 checkpoint 仍须独立进程、官方 validator 通过后单独提交。全部完成后才能生成：

- `repeat_analysis/stageb_seed_results.csv`
- `repeat_analysis/stageb_seed_summary.csv`
- `repeat_analysis/stageb_seed_correlations.csv`
- `repeat_analysis/stageb_seed_analysis.json`

最终分析必须比较 endpoint、预先定义的 curve AUC、ckpt-200 vs ckpt-0 的逐 seed 方向以及三组 checkpoint 排名稳定性。

## 9. 发布说明

本报告对应的是可审计的阶段性快照。合并到 `main` 不代表整个 Stage-B repeat 计划完成；它表示工程加固、seed 43 完整证据和 seed 44 ckpt-0 证据已经可供团队复核。工作区中的 `ACTIVE_RUN.txt`、wrapper 源日志和旧 OOM attempt 不属于本次发布范围，不应被暂存。
