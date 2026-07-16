# EAAJ Pilot — Combined Report of Three Experiments (Facts Only)

Date: 2026-07-14. Author: Aaron Wang.
This report states committed, auditable facts from three executions of the
pre-registered pilot. It contains no interpretation, no causal claims, and no
pooling across execution strata. Where a number was computed for this report
rather than taken from a committed artifact, it is labeled **[descriptive]**.

---

## 1. Shared pre-registered design (identical across all three experiments)

Single recipe, `eaaj-pilot/pilot_config.json`:

| Item | Value |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B` @ revision `060db649…` |
| Stage A | GRPO on 512 frozen GSM8K questions, exact-answer binary reward, 200 updates, lr 1e-6, KL β=0, temp 0.7, top-p 1.0, 8 generations/prompt, 64 completions/update |
| Checkpoints | 0 / 25 / 50 / 100 / 200 updates |
| Q measurement | Frozen 512-prompt probe set, eval mode, fixed dtype; decoder layers 4 / 12 / 22; effective rank, dormant-neuron fraction (τ = 0.025 and 0.1), anisotropy variants |
| Stage B | SVAMP adaptation from every checkpoint: frozen 256 train / 100 eval questions, 50 GRPO updates, eval at steps 10/20/30/40/50 |
| Primary outcome | `svamp_delta` = accuracy after 50 updates − that checkpoint's own accuracy before adaptation (same frozen questions for all checkpoints) |
| Primary analysis | Spearman rho(erank_L12, svamp_delta), n = 5 checkpoints |
| Seed | 42 (Experiments 1–2 end-to-end; Experiment 3 varies only the Stage-B adaptation seed) |

All outcomes below are fixed-budget quantities (accuracy change after exactly
50 SVAMP updates, against each checkpoint's own pre-declared baseline).

---

## 2. Experiment 1 — macOS CPU stratum

Run dir: `eaaj-pilot/outputs/local_grpo_gsm8k_eac028bfcc87`.
Execution profile: MacBook M3 Max, CPU, float32 end-to-end, standard AdamW,
micro-batch 8 × grad-accum 8, no gradient checkpointing.

**Stage-A training reward** (segment means over the `reward` field of
`dashboard.jsonl`, steps deduplicated) **[descriptive]**:
0.430 (steps 1–25) → 0.518 → 0.575 → 0.582 → 0.640 (steps 151–200).

**GSM8K held-out accuracy** (fixed 64 questions) at ckpts 0/25/50/100/200:
0.4375 / 0.4531 / 0.5156 / 0.5469 / 0.5156.

**Effective rank** (512-probe):

| ckpt | L4 | L12 | L22 |
|---:|---:|---:|---:|
| 0 | 225.14 | 231.76 | 354.19 |
| 25 | 225.31 | 223.87 | 321.89 |
| 50 | 225.18 | 219.43 | 306.34 |
| 100 | 223.94 | 229.43 | 311.41 |
| 200 | 224.15 | 233.41 | 314.71 |

Dormant-neuron fraction: 0.0 at every checkpoint, every layer, both thresholds.

**Fixed-budget SVAMP adaptation** (seed 42):

| ckpt | before | after | delta |
|---:|---:|---:|---:|
| 0 | 0.53 | 0.58 | +0.05 |
| 25 | 0.48 | 0.61 | +0.13 |
| 50 | 0.61 | 0.69 | +0.08 |
| 100 | 0.63 | 0.65 | +0.02 |
| 200 | 0.67 | 0.71 | +0.04 |

**Primary analysis** (committed `analysis/analysis_summary.json`):
rho(erank_L12, svamp_delta) = **−0.60**, p = 0.285, n = 5.

---

## 3. Experiment 2 — Windows CUDA v2 stratum

Run dir: `eaaj-pilot/outputs/local_cuda_grpo_gsm8k_e9b0b52aab6c`.
Execution profile: Windows 11, RTX 4070 Laptop GPU (8 GiB), fp32 master
weights + bf16 autocast, `paged_adamw_8bit`, gradient checkpointing,
micro-batch 4 × grad-accum 16.

Context: a prior Windows v1 run (`local_cuda_grpo_gsm8k_6a075c15808e`,
pure-bf16 parameters) completed all four phases with a maximum relative
weight-group change of 2.4e-8 across 200 updates; it is recorded as invalid
and kept as a negative control (`WIN4070_RUN_ANALYSIS.md`). v2 is the
corrected rerun.

**Execution incidents recorded for v2** (all documented in
`compute_log.md` and the v2 report):
- Phase 1 was interrupted and repaired near steps 25, 50, 75. Repairs restored
  model weights, scheduler position, and resume metadata; optimizer moments
  and RNG state at those boundaries were not restored.
- `save_only_model=True` was adopted after bitsandbytes optimizer-state saves
  hung or produced corrupt state on Windows.
- Phase 3 ckpt-50 hit a transient CUDA OOM; the incomplete attempt was
  preserved and ckpt-50 was rerun from scratch in a fresh process with the
  identical recipe.
- The full-geometry probe reported 10.809 GiB peak reserved memory, above the
  pre-set 7.3 GiB gate; logged as an allocator/accounting deviation.

**Update-effectiveness sentinels**: 33/33 windows passed (8 in Phase 1,
25 in Phase 3). Max relative weight-group change ckpt-0→200: 6.06e-6.

**Stage-A training reward** (same method as §2) **[descriptive]**:
0.354 → 0.443 → 0.462 → 0.538 → 0.582.
(The v2 report's last-segment figure of 0.5781 corresponds to steps 151–199;
the full 151–200 window gives 0.5816.)

**GSM8K held-out accuracy** at ckpts 0/25/50/100/200:
0.3594 / 0.4688 / 0.4375 / 0.4688 / 0.4219.

**Effective rank** (512-probe):

| ckpt | L4 | L12 | L22 |
|---:|---:|---:|---:|
| 0 | 225.14 | 231.76 | 354.19 |
| 25 | 219.31 | 215.31 | 321.31 |
| 50 | 223.01 | 211.76 | 325.67 |
| 100 | 222.29 | 208.92 | 317.03 |
| 200 | 223.24 | 214.68 | 324.92 |

A 2048-prompt sensitivity check on ckpt-0 vs ckpt-200 measured L4 −1.03%,
L12 −7.22%, L22 −7.71%.

Dormant-neuron fraction: 0.0 at every checkpoint, every layer, both thresholds.

**Fixed-budget SVAMP adaptation** (seed 42):

| ckpt | before | after | delta |
|---:|---:|---:|---:|
| 0 | 0.53 | 0.59 | +0.06 |
| 25 | 0.51 | 0.62 | +0.11 |
| 50 | 0.56 | 0.59 | +0.03 |
| 100 | 0.55 | 0.60 | +0.05 |
| 200 | 0.54 | 0.66 | +0.12 |

**Primary analysis** (committed `analysis/analysis_summary.json`):
rho(erank_L12, svamp_delta) = **+0.50**, p = 0.391, n = 5.

---

## 4. Experiment 3 — Stage-B adaptation-seed replications (Windows)

Location: `…/local_cuda_grpo_gsm8k_e9b0b52aab6c/adaptation_repeats/`.
Plan: `eaaj-pilot-win4070/WIN4070_STAGEB_SEED_REPLICATION_PLAN.md`;
status: `WIN4070_STAGEB_SEED_REPLICATION_STATUS_ZH.md`.

Design: rerun the identical fixed-budget SVAMP adaptation from the **same five
v2 checkpoints**, changing **only the adaptation seed** (42 → 43 → 44). Each
repeat's manifest records the source run's config/manifest SHA-256 hashes and
the git SHA; recipe fields are identical to §3. Every run passed the official
validator gates (50/50 updates completed, fixed baseline, full eval curve,
effective-update sentinels, no safety stop).

**Seed 43 — complete (5/5 checkpoints, validator 5/5):**

| ckpt | before | after | delta |
|---:|---:|---:|---:|
| 0 | 0.53 | 0.57 | +0.04 |
| 25 | 0.51 | 0.59 | +0.08 |
| 50 | 0.56 | 0.65 | +0.09 |
| 100 | 0.55 | 0.61 | +0.06 |
| 200 | 0.54 | 0.59 | +0.05 |

**Seed 44 — ckpt-0 only:** 0.53 → 0.53, delta = 0.00. Eval curve at steps
10/20/30/40/50: 0.52 / 0.58 / 0.55 / 0.57 / 0.53. Checkpoints 200/25/50/100
have not been run.

Incident: the first seed-43 ckpt-0 wrapper attempt failed before training
started (PowerShell treated a Python/Triton warning stream as an error); the
failure evidence was preserved and the run was redone in a fresh process with
an unchanged configuration.

Facts recorded in the committed status document:
- Endpoint-delta rank order, seed 42: ckpt-200 > 25 > 0 > 100 > 50.
- Endpoint-delta rank order, seed 43: ckpt-50 > 25 > 100 > 200 > 0.
- Descriptive Spearman between the seed-42 and seed-43 delta rankings ≈ −0.50.
- ckpt-200 delta minus ckpt-0 delta: +0.06 (seed 42), +0.01 (seed 43).
- The pre-registered seeds-42/43/44 repeat analysis has **not** been
  generated; per the plan it requires seed 44 to complete first.

Additional descriptive computations for this report (not part of the
pre-registered analysis) **[descriptive]**:
- rho(erank_L12, seed-43 deltas) = −0.50, p = 0.391, n = 5.
- rho(erank_L12, mean of seed-42 and seed-43 deltas) = 0.00, p = 1.0, n = 5.
- ckpt-0 deltas across the three seeds run so far: +0.06 / +0.04 / 0.00.

---

## 5. Cross-experiment observations (facts; strata reported side by side, never pooled)

1. **Measurement agreement at ckpt-0.** Both strata measured the same base
   model at ckpt-0; effective-rank values agree to four decimal places
   (e.g. erank_L12 = 231.7567 in both).
2. **Effective rank at layers 12 and 22 decreased from ckpt-0 at checkpoints
   25–100 in both strata.** At ckpt-200 vs ckpt-0: L22 is −11.1% (CPU) and
   −8.3% (WIN); L12 is +0.7% (CPU) and −7.4% (WIN). L4 changed by less than
   3% at every checkpoint in both strata.
3. **All completed fixed-budget adaptations to date: 16** (5 CPU seed-42,
   5 WIN seed-42, 5 WIN seed-43, 1 WIN seed-44). Endpoint deltas: 15 positive,
   1 zero, 0 negative.
4. **Dormant-neuron fraction is 0.0 in every measurement** taken in both
   strata, at both thresholds.
5. **Committed primary-correlation values:** −0.60 (CPU, seed 42) and +0.50
   (WIN, seed 42). Descriptive value for WIN seed 43: −0.50. None is
   statistically significant at n = 5.
6. Stage-B starting accuracy (`svamp_before`) across checkpoints spans
   0.48–0.67 in the CPU stratum and 0.51–0.56 in the WIN stratum.

---

## 6. Compute accounting (from `eaaj-pilot/compute_log.md`)

| Experiment | Recorded active time |
|---|---|
| 1 (CPU) | Phase 1 ran at ≈300 s/update on M3 Max (first 67 steps logged at 5.8 h); later phases logged in the run directory |
| 2 (WIN v2) | probes <5 min; Phase 1 5.18 h; Phase 2 3.9 min; Phase 3 5.26 h; Phase 4 1.3 min |
| 3 (repeats) | smoke 6.4 min; seed 43 five runs ≈4.9 h; seed 44 ckpt-0 72.2 min |

No Colab compute units have been consumed; all runs above executed on local
hardware. GPU telemetry CSVs for every Windows phase are committed alongside
the run artifacts.

---

## 7. Pending items (per the pre-registered plans)

- Seed 44: checkpoints 200 / 25 / 50 / 100 (planned order) not yet run.
- Seeds-42/43/44 repeat analysis: not yet generated; gated on seed-44
  completion.
- Team decisions on record as open: Stage-B task family proximity
  (GSM8K↔SVAMP), KL β>0 baseline, base vs Instruct variant, and whether a
  Colab reference stratum should be run with the reserved budget.

## 8. Artifact index

| Content | Path |
|---|---|
| Pre-registered recipe | `eaaj-pilot/pilot_config.json` |
| Exp 1 artifacts | `eaaj-pilot/outputs/local_grpo_gsm8k_eac028bfcc87/` |
| Exp 2 artifacts | `eaaj-pilot/outputs/local_cuda_grpo_gsm8k_e9b0b52aab6c/` |
| Exp 3 artifacts | `…e9b0b52aab6c/adaptation_repeats/seed-43/`, `…/seed-44/` |
| v1 invalidation analysis | `eaaj-pilot-win4070/WIN4070_RUN_ANALYSIS.md` |
| v2 full report | `eaaj-pilot-win4070/WIN4070_V2_FINAL_REPORT_ZH.md` |
| Repeat plan / status | `eaaj-pilot-win4070/WIN4070_STAGEB_SEED_REPLICATION_PLAN.md`, `…_STATUS_ZH.md` |
| Compute ledger | `eaaj-pilot/compute_log.md` |
