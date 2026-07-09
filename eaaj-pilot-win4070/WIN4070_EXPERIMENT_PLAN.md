# Windows RTX 4070 Laptop (CUDA) execution plan — eaaj pilot, cuda stratum

Date: 2026-07-08
Scope: the same pre-registered first pilot as `eaaj-pilot/` — GRPO on
Qwen2.5-0.5B over 512 GSM8K questions with exact-answer reward, checkpoints at
0/25/50/100/200 updates, effective-rank + dormant-fraction measurement at every
checkpoint, then the identical fixed-budget SVAMP adaptation (256 train / 100
eval / 50 updates) from every checkpoint.
Status: **designed and code-ready; not yet executed** (authored on the macOS
machine, which has no CUDA — the two preflight probes are the first thing to
run on the Windows box).

## 1. What this is and is not

This folder defines a third **execution stratum** of the one pre-registered
recipe in `eaaj-pilot/pilot_config.json`. It changes *where and how fast* the
pilot runs, never *what* is run:

| Stratum | Machine | Device / dtype | Status | Run dir (`eaaj-pilot/outputs/`) |
|---|---|---|---|---|
| cpu-fp32 | M3 Max (macOS) | CPU float32 | running (Phase 3 in flight) | `local_grpo_gsm8k_eac028bfcc87` |
| mps-fp32 | M3 Max (macOS) | Apple GPU float32 | validated, parked (CPU parity) | `local_mps_grpo_gsm8k_42323d70490c` |
| **cuda-bf16** | **RTX 4070 Laptop (Windows 11)** | **CUDA bfloat16** | **this plan** | **`local_cuda_grpo_gsm8k_c1ea6e11b8ca`** |

The run-dir hash is deterministic from the config, so the cuda directory name
above is **pre-registered**; verify it on the Windows box with:
`python -c "import sys; sys.path.insert(0,'scripts'); from run_local_pipeline import *; print(local_run_dir(load_pilot(),'cuda')[0])"`
(from `eaaj-pilot/`).

Strata are never merged silently (LOCAL_EXPERIMENT_PLAN.md deviation rule).
RQ1 — does Q at a GSM8K checkpoint correlate with **fixed-budget SVAMP
adaptability relative to that run's checkpoint 0** — is computed **within** a
stratum. Cross-stratum agreement (cpu-fp32 vs cuda-bf16 trends) is a labeled
robustness observation, never pooled data.

**Framing guard (Madhur):** every outcome this stratum logs is a fixed-budget
quantity (Δaccuracy and accuracy-vs-update curve after exactly 50 SVAMP
updates, from each checkpoint, against pre-declared baselines checkpoint-0).
No claim of the form "RLVR reduces the model's ability to learn" — only
fixed-budget future adaptability on the held-out task family.

**Tommy's difficulty control is implemented, unchanged:** every checkpoint's
outcome is measured as improvement from **its own** starting SVAMP accuracy
(`baseline.json` / `acc_before` in `src/adaptation.py`), on the **same frozen
256-train/100-eval SVAMP questions** (`data/svamp_splits.json`) with the same
50-update budget — excluding the "checkpoint 200 just starts from a worse
SVAMP position / sees harder questions" confound.

## 2. Scientific contract (identical across strata — do not touch)

From `pilot_config.json`, consumed verbatim by `run_local_pipeline.py`:

- Model `Qwen/Qwen2.5-0.5B` @ `060db649…`; GSM8K @ `740312ad…`; SVAMP @ `5e0bf1e5…`; seed 42.
- Stage A: GRPO, 200 updates, LR 1e-6, micro-batch 8 × grad-accum 8 (64
  completions/update), 8 generations/prompt, β(KL)=0, T=0.7, top-p=1.0,
  prompt/completion caps 512/512, eval every 25 on the frozen 64-question
  GSM8K slice, checkpoints saved at exactly 0/25/50/100/200.
- Q measurement: frozen 512-prompt probe set (`data/probe_set_ids.json`),
  eval mode, layers 4/12/22, erank + normalized erank + participation ratio +
  top-k shares + centered/uncentered anisotropy + dormant fraction at
  τ ∈ {0.025, 0.1}; probe-size sensitivity (2048) at checkpoints 0 and 200.
- Adaptation: identical recipe from **each** checkpoint — GRPO, 50 updates,
  same optimizer/LR/seed/sampling, frozen 256 SVAMP train / 100 eval,
  accuracy at updates 0/10/20/30/40/50.
- Leakage rule: nothing normalizes across a run; artifacts append online.
- Phase-1 gate: `sparse_reward_preflight` must find within-group reward
  variance on the base model **before** any update spend. If it reports no
  GRPO signal: STOP, keep the diagnostic JSON, ask the team (hard rule —
  no shaping reward, no model swap).

## 3. Execution profile (what differs, with one-line justifications)

Profile `EXECUTION_PROFILES["cuda"]` in `eaaj-pilot/scripts/run_local_pipeline.py`:

| Deviation vs. what | Setting | Justification (mirror to Research Doc) |
|---|---|---|
| vs. cpu stratum | device `cuda`, dtype **bfloat16** | Matches the pre-registered Colab recipe (notebook 01 loads bf16 + `bf16=True`); the 4070 (Ada, sm_89) has native bf16. Strata stay separate. |
| vs. Colab notebook | **gradient_checkpointing ON** (non-reentrant) | Execution optimization to fit GRPO activations+logits in 8 GiB VRAM; does not change objective, data, or update count (same precedent as the cpu stratum disabling it for RAM abundance). |
| vs. Colab notebook | trainer checkpoints every 25/10 steps, keep 2, with optimizer state | Resumability for a laptop that may sleep/crash; same policy the cpu stratum already uses. |
| vs. notebook 02 | Q measured in **bf16** (stratum dtype), not float16 | Local-runner convention: phases 2–4 always follow the run's recorded execution profile, so Q dtype is fixed *within* the stratum (comparability holds where it matters). Cross-stratum Q values are never pooled. |
| — | optimizer states in bf16 (params bf16 ⇒ AdamW moments bf16) | Identical to what the notebook-01 recipe produces on Colab; noted because LR=1e-6 with bf16 moments is a known precision trade-off — flagged, not "fixed", to stay recipe-faithful. |
| — | `torch_threads=8`, pinned dataloader memory | Host-side only. |

Also fixed for this stratum (Windows-portability changes to shared code, all
behavior-preserving on macOS/Linux):

1. `acquire_runner_lock` no longer probes pids with `os.kill(pid, 0)` on
   Windows — there that call **terminates** the target process; it now uses
   `OpenProcess`/`GetExitCodeProcess`.
2. `src/callbacks.py` no longer hard-imports the POSIX-only `resource` module;
   on Windows peak RSS comes from `GetProcessMemoryInfo` (psapi).
3. `gradient_checkpointing_kwargs={"use_reentrant": False}` whenever gradient
   checkpointing is enabled (inert for cpu/mps strata, which keep it off).

Open team questions (base-vs-Instruct, GRPO-vs-SFT adaptation, SVAMP too close
to GSM8K, β>0 baseline) are **not** decided by this stratum; it inherits the
same logged defaults as the existing runs.

## 4. VRAM budget (8 GiB, the binding constraint)

Qwen2.5-0.5B ≈ 0.494 B parameters (24 layers, hidden 896, MLP 4864, vocab
151 936, tied embeddings).

| Component | Est. GiB | Notes |
|---|---:|---|
| Weights (bf16) | 0.92 | |
| Gradients (bf16) | 0.92 | |
| AdamW moments (2 × bf16) | 1.84 | follow param dtype, as on Colab |
| Rollout KV cache + generation | 0.2–0.5 | 64 seqs × ≤1024 tok; GQA (2 KV heads) keeps this small |
| Logits / log-prob chunks + checkpointed activations | 1.5–2.5 | vocab 152 k dominates; TRL 1.6 chunks selective log-softmax |
| CUDA context, cuDNN, allocator slack | 0.4–0.7 | `expandable_segments:True` set by the wrapper |
| **Peak estimate** | **≈ 5.8–7.3** | measured, not trusted: the full-geometry probe prints true peaks |

Windows itself holds 0.3–0.8 GiB of VRAM for the desktop if the display runs
on the dGPU. Before Phase 1: close GPU apps, ideally leave the panel on the
iGPU, and check `nvidia-smi` shows ≲0.5 GiB used.

**OOM fallback ladder** (each rung logged in `compute_log.md` + notebook
header; scientific knobs — generations, lengths, LR, update counts — are never
touched):

1. Free VRAM (close apps / iGPU display), retry.
2. Micro-batch 4 × grad-accum 16 — same 64-completion effective update.
3. Micro-batch 2 × grad-accum 32.
4. Last resort, team-flagged first: `bitsandbytes` `paged_adamw_8bit` (new
   dependency; changes optimizer-state numerics — a bigger deviation than 2–3).

## 5. Time & storage budget

No CUDA measurement exists yet, so these are bracketed estimates to be
**calibrated against the full-geometry probe and the first 5 real updates**
(the extrapolation itself is a required deliverable for Tommy):

| Phase | Updates | @25 s/upd | @75 s/upd | Notes |
|---|---:|---:|---:|---|
| 1 — GSM8K GRPO | 200 | 1.4 h | 4.2 h | + 9 evals (64 q, greedy ≤512 tok) ≈ 10–30 min |
| 2 — Q at 5 ckpts | — | ~10 min | ~25 min | prompt-only forwards; incl. 2×2048 sensitivity |
| 3 — 5 × 50 SVAMP | 250 | 1.7 h | 5.2 h | + 10 full evals + 5 baselines ≈ 20–60 min |
| 4 — analysis | — | < 2 min | < 2 min | CPU |
| **Total** | | **≈ 4 h** | **≈ 10.5 h** | one long evening or an overnight run |

Reference points: same recipe measured ~300 s/update on M3-Max CPU fp32;
LOCAL_EXPERIMENT_PLAN.md projects 30–90 s/update for Colab CUDA bf16. A
90–140 W laptop 4070 with thermal throttling should land inside 25–75 s.

Storage: 5 scientific ckpts × ~0.95 GiB (bf16 safetensors) + ≤2 resumable
trainer checkpoints per live phase (~3 GiB each with optimizer state) + Phase-3
equivalents ≈ **30–45 GiB peak**; preflight requires ≥ 80 GB free.

Stop / reassess conditions (LocalSafetyCallback, active on all strata): 3
consecutive updates > 8 min; non-finite loss/grad; 5 consecutive updates with
zero within-group reward variance; 5 consecutive updates > 10 % completion
clipping. Plus cuda-specific: recurring CUDA OOM after ladder rung 3, or
driver resets (see §8). Fallback venue remains Colab with the same config.

## 6. Environment (pinned)

- Windows 11, NVIDIA driver ≥ 560.xx (latest Game Ready/Studio), RTX 4070
  Laptop 8 GiB.
- Python 3.13 preferred (matches macOS manifests; 3.12/3.11 acceptable —
  the per-run `manifest.json` records the actual version either way).
- `torch==2.12.*` from the **cu128** index (cu126 fallback, logged);
  `trl==1.6.0`, `transformers==5.13.0`, `datasets==5.0.0`,
  `accelerate==1.14.0` + the rest of `requirements-win4070.txt` — same pins as
  the macOS venv and `eaaj-pilot/requirements.txt`.
- Attention: library-default SDPA. No flash-attn (unneeded at 0.5B; no clean
  Windows wheels), no `torch.compile`, no TF32 overrides — defaults only.
- Env vars set by the wrappers: `PYTHONUTF8=1` (Chinese-locale consoles are
  cp936 otherwise), `HF_HUB_DISABLE_SYMLINKS_WARNING=1`,
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Determinism: seed 42 everywhere via `set_seed`; cuda sampling RNG differs
  from cpu by construction — trajectories are stratum-specific, which is
  exactly why strata exist. Resume-reproducibility uses trainer state, as on
  the cpu stratum.

Windows gotchas handled or checked automatically: long paths (git config +
registry check), OneDrive (preflight warns if the repo sits in a synced
folder — don't), HF cache symlinks, dataloader workers already 0, sleep
prevention via `SetThreadExecutionState` (caffeinate analog), UTF-8 logs.
Optional manual step: add the repo + `%USERPROFILE%\.cache\huggingface` to
Windows Defender exclusions to speed up many-small-file IO.

## 7. Runbook

On the 4070 box (all resumable; re-run the same command after interruption):

```powershell
git clone git@github.com:WYR186/RLVR.git C:\src\algoverse   # NOT under OneDrive
cd C:\src\algoverse\eaaj-pilot-win4070
powershell -ExecutionPolicy Bypass -File setup_win4070.ps1

.venv\Scripts\python.exe scripts\win_preflight.py --grpo-probe-small   # ~2-3 min
.venv\Scripts\python.exe scripts\win_preflight.py --grpo-probe         # full geometry, go/no-go

powershell -ExecutionPolicy Bypass -File run_pipeline.ps1 -Phase 1     # gate + 200 updates
powershell -ExecutionPolicy Bypass -File run_pipeline.ps1 -Phase 2
powershell -ExecutionPolicy Bypass -File run_pipeline.ps1 -Phase 3     # or -AdaptCheckpoint N per sitting
powershell -ExecutionPolicy Bypass -File run_pipeline.ps1 -Phase 4
```

Notes:
- AC power + Windows "Best performance" mode; laptop raised/cooled — a
  thermally throttled 4070 still finishes, it just drifts toward the 75 s/upd
  bracket. The wrapper keeps the system awake and logs `nvidia-smi` CSV every
  60 s to `logs/`.
- Phase 1 runs the sparse-reward gate automatically before spending updates
  (§2). Phase boundaries are idempotent; `runner.lock` prevents double-running
  a run dir.
- Go/no-go after the full-geometry probe: peak reserved < 7.3 GiB and
  s/update < 120 → go. Otherwise apply the §4 ladder before Phase 1, or move
  to Colab.
- After every phase: append the `compute_log.md` row (date, GPU, phase,
  duration, pointer to the `logs/gpu_*.csv` file — the local substitute for
  Colab dashboard screenshots, units column n/a) and commit.

## 8. Failure playbook

| Symptom | Likely cause | Action |
|---|---|---|
| CUDA OOM in probe/Phase 1 | logits/activation peak | §4 ladder, in order; log the rung |
| `torch.cuda.is_available()` False | driver too old / CPU wheel installed | update driver; reinstall torch from cu128 index |
| Driver reset / screen flash, run dies | WDDM TDR on a long kernel (rare here) | re-run (resumes); if recurring, set `TdrDelay=60` (HKLM…\GraphicsDrivers) and log it |
| Very slow updates (>120 s) sustained | thermal/power cap, or VRAM spill | check `logs/gpu_*.csv` clocks/power; cool the machine; verify no other GPU app |
| HF cache path errors | long paths off | enable LongPathsEnabled=1 (preflight warns) |
| `PermissionError` on checkpoint files | OneDrive/AV lock | repo must live outside OneDrive; add Defender exclusion |
| Zero reward variance 5 updates in a row | sparse-reward stall | safety callback stops the run; keep artifacts, ask team (do NOT add shaping) |

## 9. Two-machine git workflow (conflict-free by construction)

- Run artifacts live under stratum-specific dirs (`local_grpo_gsm8k_…` mac,
  `local_cuda_grpo_gsm8k_c1ea6e11b8ca` win) — disjoint paths, so merges never
  collide on artifacts.
- `outputs/ACTIVE_RUN.txt` is **machine-local and untracked** (as of
  2026-07-08): each machine's phases 2–4 follow its own active run; the
  wrapper additionally refuses phases 2–4 if the pointer isn't the cuda
  stratum on the Windows box.
- Weights (`*.safetensors`, `*.pt`, trainer state) are gitignored everywhere —
  only JSON/JSONL/CSV artifacts, configs, manifests, and logs are committed.
- Cadence on the Windows box: `git pull --rebase` before starting; commit +
  push after each completed phase (artifacts + compute log row).

## 10. Deliverables checklist for this stratum

- [ ] `probe_results.jsonl` (small + full geometry) with peak VRAM and s/update
- [ ] Phase 1: `dashboard.jsonl`, `gsm8k_eval.jsonl`, 5 checkpoints, `sparse_reward_preflight.json`
- [ ] Phase 2: `measurements/metrics_ckpt{0,25,50,100,200}.json` (+ 2048-probe sensitivity at 0/200)
- [ ] Phase 3: per-checkpoint `baseline.json`, `svamp_eval_curve.jsonl`, `dashboard.jsonl`, `summary.json`
- [ ] Phase 4: checkpoint × {erank/layer, dormant/τ, GSM8K acc, SVAMP Δacc & final} table + 3 plots (Q vs updates; reward curve with Q overlay; Q vs fixed-budget outcome scatter with Spearman ρ)
- [ ] `compute_log.md` rows + `logs/gpu_*.csv` per phase; wall-clock extrapolation to the full-experiment sweep
- [ ] Deviation lines from §3 mirrored into the Research Doc
