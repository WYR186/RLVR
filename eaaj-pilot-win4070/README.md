# eaaj-pilot-win4070 — Windows RTX 4070 Laptop stratum

Local-CUDA execution package for the **same pre-registered pilot** that lives in
[`../eaaj-pilot/`](../eaaj-pilot/): GRPO on Qwen2.5-0.5B over 512 GSM8K questions
(checkpoints at 0/25/50/100/200 updates), effective-rank + dormant-fraction
measurement at each checkpoint, then an identical fixed-budget SVAMP adaptation
(256 train / 100 eval / 50 updates) from every checkpoint.

**Nothing scientific is defined here.** The single recipe stays in
`eaaj-pilot/pilot_config.json`; this folder only adds the Windows/CUDA execution
profile (`--backend cuda`, v2: fp32 master weights + bf16 autocast + 8-bit
paged AdamW, gradient checkpointing), environment setup, preflight checks, and
the runbook. Read [`WIN4070_EXPERIMENT_PLAN.md`](WIN4070_EXPERIMENT_PLAN.md)
before running anything — it is the detailed plan, VRAM budget, deviation log,
and failure playbook.

**Status 2026-07-09:** the first run (v1,
`outputs/local_cuda_grpo_gsm8k_6a075c15808e`) is **invalidated** — pure-bf16
weights at lr=1e-6 rounded every update to zero; it is preserved as a negative
control. Analysis: [`WIN4070_RUN_ANALYSIS.md`](WIN4070_RUN_ANALYSIS.md).
The v2 rerun (pre-registered run dir
`outputs/local_cuda_grpo_gsm8k_e9b0b52aab6c`) follows
[`WIN4070_RERUN_GUIDE.md`](WIN4070_RERUN_GUIDE.md) — 重跑请从这份指引开始,
里面有第 25 步哨兵检查,15 分钟内就能确认这次训练是真实生效的。

## 快速开始（中文）

在 4070 笔记本（Windows 11，最新 NVIDIA 驱动）上：

```powershell
# 0) 克隆仓库 —— 不要放在 OneDrive 同步目录里！建议 C:\src\
git clone git@github.com:WYR186/RLVR.git C:\src\algoverse
cd C:\src\algoverse\eaaj-pilot-win4070

# 1) 一键环境安装（venv + torch cu128 + 依赖 + 模型/数据预下载 + 单测）
powershell -ExecutionPolicy Bypass -File setup_win4070.ps1

# 2) 预检：先小探针（~2-3 分钟），再全几何探针（一次完整 update，出 VRAM/速度 go/no-go）
.venv\Scripts\python.exe scripts\win_preflight.py --grpo-probe-small
.venv\Scripts\python.exe scripts\win_preflight.py --grpo-probe

# 3) 正式四个阶段（可断点续跑；插电、性能模式）
powershell -ExecutionPolicy Bypass -File run_pipeline.ps1 -Phase 1
powershell -ExecutionPolicy Bypass -File run_pipeline.ps1 -Phase 2
powershell -ExecutionPolicy Bypass -File run_pipeline.ps1 -Phase 3
powershell -ExecutionPolicy Bypass -File run_pipeline.ps1 -Phase 4
```

每个阶段结束后：把 `logs/gpu_*.csv`（本地版"GPU dashboard 截图"）对应的一行记录
追加到 `eaaj-pilot/compute_log.md`，然后 commit + push 运行产物（权重文件默认被
.gitignore 排除，只提交 JSON/JSONL 结果）。

## Files

| File | Purpose |
|---|---|
| `WIN4070_RERUN_GUIDE.md` | **Start here for the v2 rerun**: hash check, probes, step-25 sentinel gate, recording protocol |
| `WIN4070_RUN_ANALYSIS.md` | Analysis of the v1 run (why it was a no-op) + cpu-stratum comparison |
| `WIN4070_EXPERIMENT_PLAN.md` | The detailed plan: contract, VRAM budget, deviations, runbook, playbook, change log |
| `setup_win4070.ps1` | One-shot environment setup (venv, torch cu128, pins, prefetch, tests) |
| `run_pipeline.ps1` | Phase wrapper: keep-awake + nvidia-smi telemetry + `--backend cuda` |
| `requirements-win4070.txt` | Same pins as `eaaj-pilot/requirements.txt` (torch installed separately) |
| `scripts/win_preflight.py` | Environment checks + one-update GRPO probes (small / full geometry) |
| `scripts/prefetch_assets.py` | Pinned model + dataset download into the HF cache |

Authored on the macOS machine on 2026-07-08 (no CUDA available there), so the
CUDA path itself is validated by design + the preflight probes, not by a run.
Run the two probes before spending the 200-update budget.
