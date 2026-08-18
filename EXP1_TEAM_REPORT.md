# Experiment 1 — How It Was Run: Design, Execution History, and Results

Author: Aaron Wang (Person 4 — early-warning diagnostics). Date: 2026-07-19.
Scope: the pre-registered pilot ("experiment 1") only — its design, every
execution attempt, and the committed results. Follow-up experiments are out of
scope for this document. Every number below comes from committed artifacts
(run directories, `compute_log.md`, the evidence pack shared on Slack) and has
been re-verified against the raw JSON/CSV files. Values computed for
reporting rather than taken from a committed analysis file are marked
**[descriptive]**.

---

## 1. Objective

**RQ1:** do activation-based plasticity metrics (Q), measured at checkpoints
of an RL training run (Stage A), correlate with each checkpoint's ability to
adapt to a new task under a fixed budget (Stage B)?

Framing constraint (mentor guidance, Madhur): no claims about a model
"losing the ability to learn." Every outcome is a **fixed-budget quantity** —
accuracy change on a named task, under a pre-declared budget, against a
pre-declared baseline. The pilot's purpose was to validate the full pipeline
end-to-end on local hardware at zero cloud cost before the team commits its
Colab budget, and to observe what the signals look like at small scale.

## 2. Pre-registered design

One sentence: Qwen2.5-0.5B is RL-trained with GRPO on 512 fixed GSM8K word
problems (binary exact-answer reward); at five checkpoints (0/25/50/100/200
updates), activation-based plasticity metrics (effective rank, dormant-neuron
fraction) are measured, and an identical fixed-budget adaptation is then run
from each checkpoint (50 GRPO updates on 256 fixed SVAMP problems), producing
per-checkpoint metric values and an accuracy change on 100 held-out SVAMP
questions — testing whether the metrics predict fixed-budget adaptability.

| Item | Value |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B` @ revision `060db6499f32faf8b98477b0a26969ef7d8b9987` |
| Stage A | GRPO, 512 frozen GSM8K questions, exact-answer binary reward, 200 updates, lr 1e-6, KL β=0, temperature 0.7, top-p 1.0, 8 generations/prompt, 64-completion effective updates |
| Checkpoints | 0 / 25 / 50 / 100 / 200 updates |
| Q measurement | Frozen 512-prompt probe set, eval mode, float32, last-non-padding-token hidden states at decoder layers 4/12/22 → SVD → effective rank (plus participation ratio, top-k variance share, anisotropy); dormant fraction at τ=0.025 and 0.1; 2048-prompt sensitivity check at ckpt 0 and 200 |
| Stage B | From every checkpoint: 50 GRPO updates on the same 256 SVAMP questions; evaluation on the same 100 held-out questions at steps 0/10/20/30/40/50 |
| Primary outcome | `svamp_delta` = accuracy after 50 updates − that checkpoint's own pre-adaptation accuracy |
| Primary analysis | Spearman rho(erank_L12, svamp_delta), n = 5 |
| Seed | 42 end-to-end (later Stage-B-only repeats vary only the adaptation seed: 43, 44) |

Core scientific settings live in one pre-registered `pilot_config.json`.
Machine-specific execution settings (micro-batch geometry, optimizer
implementation, precision profile) are recorded per run in its config and
manifest with SHA-256 hashes — this makes runs auditable, not automatically
equivalent across machines.

## 3. How a run proceeds (pipeline phases and gates)

- **Phase 0 — development verification.** 37 unit/contract tests (metric
  formulas, reward parsing, GRPO API contract), a 1-step tiny GRPO smoke
  test, and an 8-prompt Q-measurement dry run. Establishes code correctness
  before any budget is spent; its outputs are plumbing artifacts, never
  results.
- **Preflight gates.** (a) Sparse-reward check: 8 generations are sampled for
  each of 8 frozen training prompts and within-group reward variance is
  required, because GRPO's advantage signal is zero in constant-reward groups
  (measured: 7/8 groups varied on Mac, 8/8 on Windows — both passed).
  (b) Update-effectiveness sentinel: relative weight change is logged every
  25 updates to `update_sentinel.jsonl`, with the first window at step 25
  serving as a manual stop/go gate (origin of this gate: §4.3).
- **Phase 1 — Stage A.** 200 GRPO updates; per-step trainer signals (reward,
  reward std, entropy, grad norm, loss, completion lengths, clipping stats)
  written to `dashboard.jsonl` (β=0, so no KL term is logged); checkpoints
  saved at 0/25/50/100/200. Resumable from rolling trainer checkpoints.
- **Phase 2 — Q measurement.** As specified in §2, identical contract at
  every checkpoint so values are comparable across checkpoints and machines.
- **Phase 3 — Stage B.** The identical fixed-budget adaptation from each of
  the five checkpoints. For repeat runs, a validator accepts a run only if
  all 50 updates completed, the full eval curve exists, all sentinel windows
  passed, the baseline check matches, summary values are finite, telemetry is
  present, and no safety stop fired.
- **Phase 4 — pre-registered analysis.** Spearman rho(erank_L12,
  svamp_delta), n=5, written to `analysis/analysis_summary.json` alongside
  the results and exploratory-correlation tables. Analysis-relevant
  JSON/JSONL/CSV artifacts are committed to git; model weights and optimizer
  states are intentionally stored outside git.

## 4. Execution history

The pilot was executed multiple times on different hardware. Each attempt is
listed in order, including the one that failed and the one that was
investigated and not adopted — both produced load-bearing lessons.

### 4.1 Run 1 — Mac CPU stratum (run dir `local_grpo_gsm8k_eac028bfcc87`)

MacBook M3 Max, CPU, float32 end-to-end, standard AdamW, micro-batch 8 ×
grad-accum 8. Phase 1 ran at ≈300 s/update (~17 h for Stage A); the first
session was interrupted at step ~67 after 5.8 h and resumed from the rolling
trainer checkpoint. All four phases completed. This is the first complete
execution of the design.

### 4.2 MPS investigation (measured, not adopted)

Because 300 s/update was painfully slow, Apple-GPU (MPS) acceleration was
benchmarked **on the real workload before switching**. Standalone benchmarks
looked excellent, but real GRPO updates measured 265–320 s/update — parity
with CPU. Root cause: per-token synchronization overhead in transformers'
generate loop on MPS. Outcome: not adopted (compat patch retained in
`src/mps_compat.py`); the slowness motivated moving the CUDA stratum to the
Windows RTX 4070 laptop.

### 4.3 Windows v1 — invalidated run, retained as a negative control (run dir `local_cuda_grpo_gsm8k_6a075c15808e`)

First Windows attempt stored the parameters themselves in bf16 to fit the
8 GiB GPU. Engineering-wise it completed all four phases in ~7.3 h and looked
normal on every dashboard signal. Post-hoc weight analysis showed the maximum
relative weight change across all parameter groups over 200 updates was
**2.4e-8** (Mac reference for the same recipe: 3.9e-6): at lr=1e-6, every
update was smaller than bf16's precision and rounded away. The run trained
nothing — training reward stayed flat (0.366 → 0.380), effective rank moved
≤0.14%, SVAMP deltas sat within ±2pp evaluation noise, and the primary
correlation (rho = −0.051, p = 0.935) is an execution artifact, not evidence.

Consequences: the run is recorded as invalid and kept as a negative control
(`WIN4070_RUN_ANALYSIS.md`); the update-effectiveness sentinel and the
step-25 manual kill-gate were added to the pipeline; and the same bf16 hazard
was found in the pre-registered Colab notebook recipe — caught before any
Colab budget was spent.

### 4.4 Windows v2 — corrected CUDA stratum (run dir `local_cuda_grpo_gsm8k_e9b0b52aab6c`)

Fix: **float32 master weights + bf16 autocast** for compute, 8-bit paged
AdamW (`paged_adamw_8bit`) to fit optimizer state in 8 GiB, gradient
checkpointing, micro-batch 4 × grad-accum 16 (same 64-completion effective
update as Mac). All four phases completed; sentinels passed 33/33 windows;
max relative weight change ckpt-0→200 was 6.06e-6.

Execution incidents, all documented in `compute_log.md` and the v2 report:

- Phase 1 was interrupted and repaired near steps 25/50/75. Repairs restored
  model weights, scheduler position, and resume metadata; optimizer moments
  and RNG state at those boundaries were not restored (bitsandbytes
  optimizer-state saves hung or corrupted on Windows, so
  `save_only_model=True` was adopted). The trajectory is valid but not
  equivalent to an uninterrupted run.
- Phase 3 ckpt-50 hit a transient CUDA OOM; the incomplete attempt was
  preserved and the checkpoint was rerun from scratch in a fresh process with
  the identical recipe.
- The full-geometry memory probe reported 10.809 GiB peak reserved, above the
  pre-set 7.3 GiB gate and above physical VRAM; logged as a WDDM /
  allocator-accounting deviation.

### 4.5 Stage-B seed repeats (Windows; `…e9b0b52aab6c/adaptation_repeats/`)

To separate "hardware difference" from "plain randomness," the five v2
checkpoints were held fixed and only the Stage-B adaptation seed was changed
(42 → 43 → 44). Every accepted run passed the full validator (§3). One
wrapper attempt failed before training started (PowerShell treated a
Python/Triton warning stream as an error); the failure evidence was preserved
and the run was redone in a fresh process with unchanged configuration.

Status: seed 43 complete (5/5 checkpoints, ≈4.9 h); seed 44 incomplete
(ckpt-0 only, 72.2 min). The pre-registered 3-seed summary analysis is gated
on seed-44 completion. Note: all repeats share the single v2 Stage-A
trajectory, so they probe Stage-B stochasticity only — they are not
independent Stage-A samples.

## 5. Results

### 5.1 Training evidence (Stage A)

Segment means of the per-step training reward **[descriptive]**:

| Steps | Mac CPU | Windows v2 |
|---:|---:|---:|
| 1–25 | 0.430 | 0.354 |
| 26–50 | 0.518 | 0.443 |
| 51–100 | 0.575 | 0.462 |
| 101–150 | 0.582 | 0.538 |
| 151–200 | 0.640 | 0.582 |

GSM8K held-out accuracy (fixed 64 questions) at ckpts 0/25/50/100/200:
Mac 0.4375 / 0.4531 / 0.5156 / 0.5469 / 0.5156;
Windows 0.3594 / 0.4688 / 0.4375 / 0.4688 / 0.4219.
(64-question single evaluations carry ~±6pp sampling noise; note the two
strata score the same base model differently at ckpt-0 because evaluation
generation is itself temperature-sampled.)

### 5.2 Q metrics

Effective rank (512-prompt probe):

| ckpt | Mac L4 | Mac L12 | Mac L22 | Win L4 | Win L12 | Win L22 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 225.14 | 231.76 | 354.19 | 225.14 | 231.76 | 354.19 |
| 25 | 225.31 | 223.87 | 321.89 | 219.31 | 215.31 | 321.31 |
| 50 | 225.18 | 219.43 | 306.34 | 223.01 | 211.76 | 325.67 |
| 100 | 223.94 | 229.43 | 311.41 | 222.29 | 208.92 | 317.03 |
| 200 | 224.15 | 233.41 | 314.71 | 223.24 | 214.68 | 324.92 |

- The two strata agree on the ckpt-0 base model to four decimal places
  (erank_L12 = 231.7567) — a strong implementation-consistency check for the
  measurement contract.
- Shared pattern: layer 4 barely moves (<3% everywhere); layers 12/22
  contract early. Layer 22 drops 8–13.5% within 25–50 updates and stays below
  baseline on both machines (ckpt-200: −11.1% Mac, −8.3% Win). Layer 12
  recovers to +0.7% by ckpt-200 on Mac but stays at −7.4% on Windows.
- The 2048-prompt sensitivity checks agree in direction with the 512-prompt
  values on both machines (e.g. Windows ckpt-200 vs 0: L12 −7.2%, L22 −7.7%).
- Dormant-neuron fraction: 0.0 at every checkpoint, every layer, both
  thresholds, in both strata — zero variance, hence no predictive
  information at these settings.

### 5.3 Fixed-budget SVAMP adaptation

| ckpt | Mac s42 before→after (Δ) | Win s42 before→after (Δ) | Win s43 before→after (Δ) | Win s44 |
|---:|---|---|---|---|
| 0 | 0.53→0.58 (+.05) | 0.53→0.59 (+.06) | 0.53→0.57 (+.04) | 0.53→0.53 (0.00) |
| 25 | 0.48→0.61 (+.13) | 0.51→0.62 (+.11) | 0.51→0.59 (+.08) | not run |
| 50 | 0.61→0.69 (+.08) | 0.56→0.59 (+.03) | 0.56→0.65 (+.09) | not run |
| 100 | 0.63→0.65 (+.02) | 0.55→0.60 (+.05) | 0.55→0.61 (+.06) | not run |
| 200 | 0.67→0.71 (+.04) | 0.54→0.66 (+.12) | 0.54→0.59 (+.05) | not run |

16 completed adaptations: 15 positive endpoint deltas, 1 flat, 0 negative.

A notable side observation: on the Mac trajectory, **un-adapted** SVAMP
accuracy rose from 0.53 to 0.67 across Stage-A checkpoints — strong direct
GSM8K→SVAMP transfer that raises late checkpoints' starting points and
compresses their room for improvement (ceiling effect). The Windows
trajectory shows almost no such transfer (0.51–0.56). This is direct evidence
for the standing team concern that SVAMP may be too close to GSM8K as a
Stage-B task family.

### 5.4 Primary analysis

| Analysis | rho(erank_L12, svamp_delta) | p | Status |
|---|---:|---:|---|
| Mac, seed 42 | **−0.60** | 0.285 | pre-registered, committed |
| Windows, seed 42 | **+0.50** | 0.391 | pre-registered, committed |
| Windows, seed 43 | −0.50 | 0.391 | descriptive |

Additional descriptive facts: the seed-42 and seed-43 delta *rankings* over
the same five checkpoints correlate at ≈ −0.50 (near-inversion from a seed
change alone); ckpt-0 deltas across the three seeds run so far are
+0.06 / +0.04 / 0.00 (SD ≈ 0.031, i.e. seed noise alone is comparable to the
between-checkpoint differences being ranked).

## 6. What can and cannot be concluded

Strongly supported (reproduced across strata, artifacts committed):

1. The end-to-end pipeline works on both platforms, and the Q-measurement
   contract is cross-machine consistent (ckpt-0 agreement to 4 decimals;
   33/33 sentinel windows).
2. GRPO training is accompanied by an early contraction of the mid/late-layer
   activation spectrum (layer 22: 8–13.5%, persisting to ckpt-200 on both
   machines; confirmed at 2048-prompt probe size). This is the pilot's most
   robust scientific observation.
3. In this regime (0.5B model, 200 gentle updates at lr 1e-6, β=0, near-task
   Stage B), no consistent fixed-budget adaptability loss relative to
   checkpoint 0 was observed: 15/16 adaptations improved, 1 flat, 0 negative;
   individual checkpoints fell below ckpt-0's delta in some runs, but which
   ones did was not reproducible.
4. Dormant-neuron fraction carries no signal at these settings (identically
   zero everywhere).
5. The n=5, single-seed primary correlation is unstable: its sign flips
   across execution strata (−0.60 vs +0.50) and across Stage-B seeds alone
   (+0.50 → −0.50). The current evidence neither establishes effective rank
   as a reliable predictor nor shows a reproducible plasticity collapse.
   RQ1 remains open.

Exploratory only (noted for possible pre-registration later, not findings):

6. Centered-anisotropy correlations kept the same sign in both strata
   (L12/L22 positive, L4 negative) while erank's flipped.
7. GSM8K→SVAMP transfer strength is itself trajectory-dependent and, when
   present, contaminates the primary outcome via the ceiling effect (§5.3).

Open questions explicitly left to the team (implemented with the cheaper
default, logged, not silently decided): base vs Instruct model, GRPO vs SFT
for adaptation, whether SVAMP is too close to GSM8K, KL β>0 baseline arm,
and whether a clean Colab reference stratum should be run.

## 7. Compute accounting

Zero Colab compute units were consumed; all runs were local. From
`compute_log.md`:

| Item | Hardware | Active time |
|---|---|---|
| Phase-0 tests + smoke + dry run | Mac (CPU) | <15 s after model cache warm-up |
| Mac Stage A | M3 Max CPU fp32 | ≈300 s/update (~17 h; first 67 steps logged at 5.8 h) |
| MPS investigation | M3 Max MPS | ~1.5 h (outcome: parity, not adopted) |
| Windows v1 (invalid) | RTX 4070 Laptop, pure bf16 | ~7.3 h total |
| Windows v2 probes | RTX 4070 Laptop | <5 min |
| Windows v2 Phase 1 / 2 / 3 / 4 | RTX 4070 Laptop | 5.18 h / 3.9 min / 5.26 h / 1.3 min |
| Seed-43 repeats (5 runs) | RTX 4070 Laptop | ≈4.9 h |
| Seed-44 ckpt-0 | RTX 4070 Laptop | 72.2 min |

GPU telemetry CSVs for every Windows phase are committed alongside the run
artifacts.

## 8. Artifact index

| Content | Path |
|---|---|
| Pre-registered recipe | `eaaj-pilot/pilot_config.json` |
| Mac run artifacts | `eaaj-pilot/outputs/local_grpo_gsm8k_eac028bfcc87/` |
| Windows v2 artifacts | `eaaj-pilot/outputs/local_cuda_grpo_gsm8k_e9b0b52aab6c/` |
| Seed repeats | `…e9b0b52aab6c/adaptation_repeats/seed-43/`, `…/seed-44/` |
| v1 invalidation analysis | `eaaj-pilot-win4070/WIN4070_RUN_ANALYSIS.md` |
| v2 full report | `eaaj-pilot-win4070/WIN4070_V2_FINAL_REPORT_ZH.md` |
| MPS investigation | `eaaj-pilot/LOCAL_EXPERIMENT_PLAN.md` (§2026-07-08) |
| Repeat plan / status | `eaaj-pilot-win4070/WIN4070_STAGEB_SEED_REPLICATION_PLAN.md`, `…_STATUS_ZH.md` |
| Compute ledger | `eaaj-pilot/compute_log.md` |
| Evidence pack (shared on Slack) | `experiment 1/pilot_evidence_pack/` (+ `exp1_result_aaron.zip`) |
