# WIN4070 rerun guide (v2) — for the agent on the Windows machine

Audience: the coding agent (or human) driving the RTX 4070 Laptop box.
Mission: rerun the full pilot as the **cuda stratum v2** with the precision
fix, prove within the first 25 updates that training is real this time, and
record everything so the run is auditable.

给 Aaron 的一句话:v1 的 200 个 update 全被 bf16 舍入吃掉了;v2 改成 fp32 主权重 +
bf16 autocast + 8-bit 分页 AdamW,并加了"更新有效性哨兵"——跑到第 25 步就能确认
这次是真的在训练。

## 0. What changed and why (read before running)

v1 (`local_cuda_grpo_gsm8k_6a075c15808e`) trained 200 GRPO updates as a
**no-op**: the model was loaded in bf16, and at lr=1e-6 every AdamW update
(~1e-6) is far below the bf16 ulp of a typical weight (~|w|·2⁻⁸ ≈ 8e-5), so
`w + Δw` rounded back to `w`. Full evidence: [WIN4070_RUN_ANALYSIS.md](WIN4070_RUN_ANALYSIS.md).
Keep the v1 run dir — it is preserved as a negative control; never write into it.

| | v1 (invalidated) | v2 (this rerun) |
|---|---|---|
| Master weights | bf16 | **float32** |
| Compute | bf16 (params) | **bf16 autocast** (`bf16=True`) |
| Optimizer | AdamW (bf16 states) | **`paged_adamw_8bit`** (bitsandbytes; fp32 master fits 8 GiB) |
| Geometry | micro-batch 4 × grad-accum 16 | unchanged (64-completion updates) |
| New instrumentation | — | `UpdateEffectivenessSentinel` → `update_sentinel.jsonl` every 25 steps |
| Run dir | `local_cuda_grpo_gsm8k_6a075c15808e` | **`local_cuda_grpo_gsm8k_e9b0b52aab6c`** |

Scientific knobs (seed, LR, β=0, T, top-p, 512/512 caps, 8 generations,
checkpoint steps, frozen splits, probe set) are untouched. The same fix was
applied to notebook 01 and `src/adaptation.py`, so Colab inherits it.

## 1. Sync and environment

```powershell
cd D:\algoverse           # or wherever the repo lives (never OneDrive)
git pull --rebase origin main
# install the new dependency into WHICHEVER env run_pipeline.ps1 resolves
# (EAAJ_PYTHON > conda env > .venv). Example for the repo conda env:
& .conda\envs\eaaj-win4070\python.exe -m pip install -r eaaj-pilot-win4070\requirements-win4070.txt
& .conda\envs\eaaj-win4070\python.exe -c "import bitsandbytes; print('bitsandbytes', bitsandbytes.__version__)"
```

If bitsandbytes has no working Windows wheel for your Python/CUDA combo, STOP
and flag it — do not substitute another optimizer silently.

## 2. Verify the pre-registered run dir (guards against code drift)

```powershell
cd eaaj-pilot
python -c "import sys; sys.path.insert(0,'scripts'); from run_local_pipeline import *; print(local_run_dir(load_pilot(),'cuda')[0].name)"
```

Expected output: `local_cuda_grpo_gsm8k_e9b0b52aab6c`.
Anything else means the profile or pilot config drifted — stop and ask before
running.

## 3. Probes (repeat even though v1 probes passed — the optimizer changed)

```powershell
cd ..\eaaj-pilot-win4070
python scripts\win_preflight.py                     # env checks
python scripts\win_preflight.py --grpo-probe-small  # ~2-3 min stack check
python scripts\win_preflight.py --grpo-probe        # one full-geometry update
```

The probe now rehearses the real v2 profile (fp32 master + bf16 autocast +
paged_adamw_8bit) and appends to `logs/probe_results.jsonl`. Gates:
peak reserved **< 7.3 GiB**, wall **< 120 s/update**. v1 reference at the same
geometry was 31.97 s / 6.94 GiB; fp32 master + paged states will be somewhat
slower and heavier — that is expected. If OOM: 1) free VRAM (iGPU display),
2) micro-batch 2 × grad-accum 32, 3) stop and flag. Log any rung you take.

## 4. Phase 1 with the step-25 kill-gate

```powershell
powershell -ExecutionPolicy Bypass -File run_pipeline.ps1 -Phase 1
```

**Do not walk away before step 25.** When
`eaaj-pilot/outputs/local_cuda_grpo_gsm8k_e9b0b52aab6c/update_sentinel.jsonl`
gets its first row (~15 min in), check it:

- `updates_effective: true` and `rel_change_window ≥ 1e-8` (healthy fp32
  reference: ~5e-7 per 25-update window) → keep going.
- `updates_effective: false` (v1 measured ~3e-9) → **Ctrl+C now**, do not
  spend the remaining updates. Keep all artifacts, write down what happened,
  and flag to Aaron/team.

Secondary check around step 50: `dashboard.jsonl` reward mean should be
drifting up from the ~0.36 base level (the healthy cpu-fp32 stratum reached
~0.65 by step 200; v1 stayed flat at ~0.37–0.38).

## 5. Remaining phases

```powershell
powershell -ExecutionPolicy Bypass -File run_pipeline.ps1 -Phase 2
powershell -ExecutionPolicy Bypass -File run_pipeline.ps1 -Phase 3   # or -AdaptCheckpoint N per sitting
powershell -ExecutionPolicy Bypass -File run_pipeline.ps1 -Phase 4
```

All phases are resumable (re-run the same command); `runner.lock` prevents
double-running. Phase 3 adaptation runs also write `update_sentinel.jsonl`
per checkpoint (every 10 steps) — same kill-gate logic applies to the first
window of the ckpt-0 adaptation.

## 6. Recording requirements (all of them, every phase)

1. **compute_log.md** — append to the `## 2026-07-09 Windows RTX 4070` style
   section in `eaaj-pilot/compute_log.md` (new dated section for v2):
   `| Phase 1: GSM8K GRPO 200 updates | RTX 4070 Laptop (CUDA fp32-master/bf16-autocast) | X.XX h | ckpts saved; sentinel healthy (rel_change_window ~Ne-7); telemetry <path> |`
2. **GPU telemetry** — the wrapper writes `logs/gpu_*_phaseN.csv`, but
   `eaaj-pilot-win4070/logs/` is gitignored. After each phase, copy the CSV
   into the run dir so it gets committed:
   `Copy-Item logs\gpu_*_phaseN.csv ..\eaaj-pilot\outputs\local_cuda_grpo_gsm8k_e9b0b52aab6c\telemetry\`
3. **Probe evidence** — copy `logs/probe_results.jsonl` into the run dir's
   `telemetry/` as well (it is the VRAM/throughput go-decision record).
4. **Deviations** — any ladder rung, driver issue, or manual intervention gets
   one line in compute_log.md Notes and, if it changes numerics, a line in
   `WIN4070_EXPERIMENT_PLAN.md` §11 change log.
5. The run's `manifest.json` (auto-written) now records the bitsandbytes
   version — no action needed, just don't delete it.

## 7. Commit & push protocol

After each completed phase (or at least after phases 1, 3, 4):

```powershell
git pull --rebase origin main
git add eaaj-pilot\outputs\local_cuda_grpo_gsm8k_e9b0b52aab6c eaaj-pilot\compute_log.md
git commit -m "win4070 cuda v2: phase N artifacts"
git push origin main
```

- Weights (`*.safetensors`, `*.pt`) are auto-ignored; JSON/JSONL/CSV artifacts
  are what gets committed.
- `outputs/ACTIVE_RUN.txt` is untracked and machine-local — never add it.
- Artifact paths are disjoint from the mac strata, so rebases should never
  conflict on run artifacts; if a rebase conflicts on docs, keep both sides'
  information and ask if unsure. Never force-push.

## 8. Success criteria for v2 (what "done" looks like)

- Sentinel: every window `updates_effective: true`.
- Phase 1 reward mean clearly rises (cpu reference: 0.36 → 0.65).
- ckpt-0 → ckpt-200 weight-norm relative change ≥ 1e-6 (v1 was 2.4e-8).
- Q moves measurably (cpu reference: erank_L22 dropped ~13% by step 50).
- All deliverables in `WIN4070_EXPERIMENT_PLAN.md` §10 checked off.

## 9. Do NOT

- Change LR, generations, lengths, seeds, checkpoint steps, splits, or the
  reward — those are pre-registered science.
- Decide open team questions (base-vs-Instruct, GRPO-vs-SFT, SVAMP-vs-ProntoQA,
  β>0) — flag, don't decide.
- Write anything into `local_cuda_grpo_gsm8k_6a075c15808e` (v1, preserved) or
  the mac run dirs (`local_grpo_gsm8k_eac028bfcc87`, `local_mps_*`).
- Continue past a sentinel warning "just to see".
