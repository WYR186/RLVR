# Experiment 2 — GURU Math → Simulation, stage-1/stage-2 plasticity pipeline

**Target machine:** Windows RTX 4070 Laptop (8 GB), CUDA stratum
**Audience:** the agent running on that machine
**Owner:** Aaron Wang (Person 4)
**Status:** pre-registered, not started
**Written:** 2026-08-03

---

## 0. What this is

This implements **Tommy's pipeline spec verbatim** (Slack, 12:15, edited version).
Quoted so there is no drift:

> * for stage 1, save checkpoints (epoch = 0, TOTAL_EPOCHS/n and TOTAL_EPOCHS);
> * we do a baseline of training directly with stage 2;
> * then for each checkpoint from stage 1, we
>   * train using stage 2 datasets, and observe the plasticity metric that is
>     `the difference of reward achievable with stage 2 alone vs. stage 2 following stage 1`
> * To measure the effect of transfer between skills needed in stage 1 (A) vs. stage 2 (B),
>   we evaluate each checkpoint `M_{A,t}` on the stage B test set:
>   * `T_t = Score_B(M_{A,t}) − Score_B(M_0)`, where `M_0` is the original model without any training.
>     * If `T_t` > 0, the stage A is helping task performance on B
>     * if `T_t` < 0, hurting
>     * if `T_t` ≈ 0, little transfer

> each of you should work on a stage1 different from each other, and test on stage2 respectively.
> My current suggestion is GURU: Table reasoning / Math → Simulation. […]
> Use `LLM360/guru-RL-92k` and filter the released examples into:
> * Table: the processed HiTab and MultiHierTT subset;
> * Simulation: the processed CodeI/O subset;
> * Math: the processed OR1, DAPO, DeepScaler subset.

The stated purpose (Tommy, in the 2026-08-03 meeting) is **"to observe the phenomenon
of loss of plasticity"** — i.e. RQ1 *find it*, not yet *predict it*.

**Claimed stage 1: Math** (OR1 + DAPO + DeepScaler). Rationale in §2.1. This must be
posted to the team channel before Phase 1 starts, because Tommy requires each person to
hold a different stage 1.

---

## 1. Design decisions and their justification

Every row is a choice Tommy's spec left open. Each gets one line of justification and is
mirrored to the Research Doc (project rule: deviations are logged, not silent).

| Parameter | Value | Why |
|---|---|---|
| Stage 1 domain | **Math** (OR1/DAPO/DeepScaler) | 8 GB VRAM. Table (HiTab/MultiHierTT) requires serialising hierarchical tables into the prompt, which breaks the 512-token prompt geometry validated on this machine. Math prompts are short and the existing numeric exact-answer reward parser applies. |
| Stage 2 domain | **Simulation** (CodeI/O) | Fixed by Tommy for everyone. |
| Model | `Qwen/Qwen2.5-0.5B` @ pinned revision | 8 GB constraint, and comparability with every run the team has done. |
| Precision | fp32 master weights + bf16 autocast, `paged_adamw_8bit` | Loading params in bf16 makes small-LR updates round to zero. Non-negotiable on this machine. |
| Stage-1 length | **200 updates** | ≈4.7 h at the measured 85 s/update *if* the prompt geometry matches GSM8K. Phase 0 replaces this estimate with a measured one. |
| Stage-1 checkpoints saved | **0, 50, 100, 150, 200** | Saving a 0.5B checkpoint costs seconds. Saving 5 but only *adapting from* 3 means extending the grid later needs no retraining. |
| Stage-1 checkpoints adapted from | **0, 100, 200** | Tommy in the meeting: "start with 3 checkpoints — zero, middle, last", for compute. ckpt-0 **is** the "stage 2 alone" baseline; it is not a separate run. |
| Stage-1 learning rate | **5.5e-6** | The one parameter Tommy's spec omits, and the one that decides whether exp2 finds anything. See §1.1 — 3e-6 was tested and returned a STOP verdict, so using it here would guarantee a null. 5.5e-6 matches the already-frozen exp1.7 dose, which makes exp2 and exp1.7 a same-dose / different-task-pair comparison. Flagged to the team as an open question, not decided unilaterally. |
| Stage-2 budget | **50 updates**, eval at 0/10/20/30/40/50 | Gives both the endpoint ("reward achievable") and the curve for AUC. |
| Stage-2 eval set | **300 held-out Simulation questions**, frozen, committed | 100 was measured to be noise-dominated for this outcome. |
| Seeds | **1 (seed 42) pre-registered.** Seeds 43/44 only as a stretch, and only if the primary shows separation. | 3 seeds triples Phase 3 (≈12–19 h). Does not fit before 23 Aug on one 4070. Logged as a known limitation. |
| KL β | **0.0** | Unchanged from every prior run. A β>0 arm is a separate, still-open team question. |

### 1.1 Why 5.5e-6 and not 3e-6

Tommy's spec fixes the datasets and the protocol but not the dose, and the dose is what
decides whether this experiment can find anything. The measured dose–response map for this
exact model and machine:

| Dose | Stage-A stability | Persistent Q shift | Fixed-budget adaptability |
|---|---|---|---|
| 1e-6 × 500 | healthy | L12 late-window ≈ −1.2% | no degradation |
| **3e-6 × 500** | healthy | L12 late-window **+4.06%** (gate needs ≥7.5%) | ckpt-500 adapted **better** than ckpt-0 (+0.0289 vs +0.0200) |
| 1e-5 | one collapse at ~step 55, **not reproduced** in 3 controlled replicates | not sampled | never reached |

exp1.6 ran 3e-6 × 500 to completion and returned a pre-registered **STOP**: both expansion
gates failed, and the adaptability difference pointed the *opposite* way to plasticity loss.
**Running exp2's stage 1 at 3e-6 would therefore be a pre-determined null.**

5.5e-6 is the log-midpoint of the last confirmed-stable dose and the dose where collapse has
been observed (`sqrt(3e-6 × 1e-5) = 5.48e-6`). It is already the frozen dose of exp1.7, so
exp2 inherits its justification and becomes directly comparable to it — same dose, different
task pair, which is exactly the contrast the team needs to separate "wrong dataset pair"
from "wrong dose".

**Known risk, stated up front:** exp1.5.1 ran three controlled replicates at 1e-5 and none
reproduced the collapse seen once at ~step 55. Collapse hazard on this model depends on
trajectory randomness, not on a clean learning-rate threshold. With a single stage-A seed,
exp2 cannot distinguish "5.5e-6 is safe" from "this trajectory happened to survive". The
safety-stop machinery in Phase 1 is what keeps that from destroying the run, and the
single-seed limitation goes in the write-up rather than being papered over.

---

## 2. Pre-registered outcomes (fixed before any run)

Let `R_end(t)` = mean stage-B eval reward at stage-2 update 50, starting from stage-1
checkpoint `t`. Let `M_0` = the untrained base model.

**P1 — Tommy's primary plasticity metric**

```
ΔR_t = R_end(0) − R_end(t)
```

`R_end(0)` is the "stage 2 alone" baseline. `ΔR_t > 0` means training on stage 1 cost us
stage-2 reward.

**P2 — robustness (same data, no extra compute)**

```
AUC_t = mean over eval points {0,10,20,30,40,50} of stage-B eval reward, starting from t
ΔAUC_t = AUC_0 − AUC_t
```

Reported because the endpoint of a 50-update curve on a 0.5B model is a single noisy
number; the curve mean is not.

**P3 — Tommy's transfer control**

```
T_t = Score_B(M_{A,t}) − Score_B(M_0)      # zero-shot, NO stage-2 training
```

**Pre-registered reading rule.** `ΔR_t` is interpreted **conditional on `T_t`**:

| `T_t` | `ΔR_t > 0` reads as |
|---|---|
| ≈ 0 | candidate plasticity loss — the finding we are looking for |
| ≫ 0 | **confounded**: stage 1 raised the stage-B starting point, so there was less headroom left to gain. Must be reported as confounded, not as plasticity loss. |
| ≪ 0 | stage 1 damaged stage-B ability directly; separate the level effect from the learning effect before claiming plasticity loss |

This rule is why Phase 2 runs **before** Phase 3: `T_t` is cheap and it tells us how to
read the expensive result before we spend the compute.

---

## 3. Phases, gates, and kill conditions

Run in order. **A gate that says STOP means stop and report — do not tune around it.**

### Phase 0 — data, geometry, and smoke (GATE)

Nothing here is training. Everything here is a discovery step whose output is committed.

1. **Schema audit.** Download `LLM360/guru-RL-92k`. Do **not** assume any column names.
   Inspect the actual fields and write `data/guru_schema_audit.json` recording: split
   names, column names, dtypes, one full example per target subset, and how the
   domain/source of each example is identified.
2. **Subset filtering.** Using the field discovered above, filter into:
   - stage 1 (Math): the processed OR1, DAPO, DeepScaler examples
   - stage 2 (Simulation): the processed CodeI/O examples
   Record the resulting counts in the audit file.
3. **Answer-format determination.** For each subset, record what a correct answer looks
   like (numeric / string / structured) and what the released verifier expects. Write it
   into the audit file. The reward function is written *from this*, not from a guess.
4. **Token-length audit.** Under the Qwen2.5-0.5B tokenizer, compute p50 / p95 / p99 / max
   prompt token length for both subsets. Write to `data/token_length_audit.json`.
   - **GATE 0a — STOP** if stage-2 (CodeI/O) p95 prompt length **> 1024 tokens**. 8 GB
     cannot hold that geometry with GRPO group sampling. Report the numbers and escalate
     to the L4 (24 GB) request instead of shrinking the batch until it fits.
5. **Freeze splits.** Write and commit `data/exp2_splits.json`: stage-1 train ids,
   stage-2 train ids, stage-2 eval ids (300, held out from stage-2 train). Seeded.
6. **Smoke.** 2 updates on stage 1 and 2 updates on stage 2, in a throwaway dir.
   Verify: checkpoint written, dashboard row written, safety callback fires, GPU
   telemetry captured, loss and gradient finite.
7. **Sparse-reward preflight** on **both** stages.
   - **GATE 0b — STOP** if every sampled prompt group has constant reward. Preserve the
     diagnostic and ask the team. Do not add a shaping reward and do not switch models.

Phase 0 also produces the **real** per-update timing, which replaces the 85 s/update
estimate everywhere below.

### Phase 1 — stage 1 GRPO

- 200 updates, lr 5.5e-6, β=0, checkpoints at 0/50/100/150/200.
- Dashboard row every update: reward mean/std, fraction of zero-variance groups, entropy,
  completion clipping, loss, grad norm, mean completion length.
- **Kill-gate at update 25** — the update-effectiveness sentinel. If the weight-change
  window is not effective, STOP: the run is a no-op and everything after it is worthless.
- **Safety stop** — 5 consecutive updates with zero within-group reward variance.
- **Safety stop** — 5 consecutive updates over 10% completion clipping.

If a safety stop fires, **preserve everything and stop**. A collapse is a result, not a
failure to be retried at a different learning rate.

### Phase 2 — `T_t`, zero-shot (cheap, runs before Phase 3)

For each of ckpt 0/50/100/150/200: evaluate on the frozen 300-question stage-B eval set,
**no training**. Write `analysis/transfer_T.json` with `Score_B(M_{A,t})` and `T_t`.

Report `T_t` to the channel before Phase 3 starts. It costs minutes and it determines how
Phase 3's result can be read.

### Phase 2b — activation metrics (OPTIONAL, beyond Tommy's spec)

Effective rank + dormant fraction at layers 4/12/22 on the frozen probe set, float32,
eval mode, for each saved checkpoint. Measured cost on this machine: ~6 min for the whole
grid.

**This is not part of Tommy's spec.** It gates nothing and blocks nothing. It exists
because it is this owner's assigned role in the project and because re-measuring later
would require reloading every checkpoint. If it costs more than 15 minutes or throws,
skip it and continue to Phase 3.

Note in the report, do not bury: dormant fraction has been identically 0.0 in every prior
run on this model. It must be reported as *a metric with no resolution in this setting*,
never as evidence that plasticity is preserved.

### Phase 3 — stage 2 GRPO from each checkpoint

Cells: ckpt-0 (**= the "stage 2 alone" baseline**), ckpt-100, ckpt-200. Seed 42.
50 updates each, eval on the frozen 300 at updates 0/10/20/30/40/50.

- **One fresh OS process per cell.** The long-loop allocator failure is known on this
  machine; a single process running all cells hit CUDA OOM in backward.
- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- If a cell OOMs: preserve the partial directory under `<cell>_oom_<date>`, do not
  overwrite, rerun that cell alone in a fresh process.
- Every cell must pass the completion validator (50/50 updates, summary written, no
  safety stop) before Phase 4.

### Phase 4 — analysis

Runs on CPU. Emits `analysis/analysis_summary.json` plus:

- `ΔR_t` and `ΔAUC_t` table
- `T_t` table
- the reading-rule verdict per checkpoint (§2)
- **the figure in Tommy's shape**: stage-B reward curve, one line per stage-1 checkpoint,
  with the ckpt-0 baseline drawn as the reference line

---

## 4. Compute budget on the 4070

Anchors are measured on this machine, from exp1.6's completed 19.57 h run: **85 s/update**
for stage-A GRPO at 512/512 geometry (500 updates in 11.76 h); **≈1.3 h per 50-update
adaptation cell** (6 cells in 7.74 h); float32 Q measurement over 8 checkpoints in 4.6 min.

| Phase | Estimate | Note |
|---|---|---|
| 0 | 30–60 min | no training; dominated by download + tokenisation |
| 1 | **4.7 h** | 200 updates × 85 s — *only if* Phase 0 confirms GSM8K-like geometry |
| 2 | ~25 min | 5 checkpoints × 300 questions, generation-bound |
| 2b | ~6 min | optional |
| 3 | **3.9 h** | 3 cells × 1.3 h |
| 4 | ~2 min | CPU |
| **Total** | **≈ 9–10 h** | one overnight plus a morning |

**This fits the 4070.** The three things that break the estimate:

1. CodeI/O prompts are longer than GSM8K prompts. Phase 0 GATE 0a exists for exactly
   this. If p95 > 1024 tokens the budget is wrong by a large factor and the answer is the
   L4, not a smaller batch.
2. A stage-1 collapse at 5.5e-6 ends Phase 1 early. That shortens the run rather than
   lengthening it, and it is a reportable result — see §7.
3. Stage-1 length in *epochs*. Tommy phrases checkpoints as epochs; 200 updates is
   `200 × per_device_batch × grad_accum / len(stage1_train)` epochs. Record the actual
   epoch count in the manifest so the checkpoint spacing is reportable in his terms.

**Fallback:** L4 24 GB is requestable now (A100s are gone). DM Kevin — Archana is out of
office. Request it if GATE 0a fires, or pre-emptively if the token audit shows p95 above
~768.

---

## 5. Recording requirements (all of them, every phase)

Non-negotiable project rules:

- One row per phase in `eaaj-pilot/compute_log.md`: date, GPU, wall time, phase, outcome.
- GPU telemetry CSV per phase, copied under the run's `telemetry/`.
- Fixed seeds everywhere; pinned dataset revision in the manifest.
- All Q measurements (if Phase 2b runs) in eval mode, float32, fixed layers, frozen probe
  set — otherwise values are not comparable to anything.
- **Leakage rule:** any quantity computed at update `t` may use only information available
  at update `t`. No post-hoc normalisation across the whole run.
- Every deviation from this document gets a one-line justification in the run manifest and
  is mirrored to the Research Doc.

**Framing constraint (mentor, Madhur) — applies to every sentence written about results:**
never claim "RLVR reduces the model's ability to learn". The measurable claim is about
**fixed-budget** adaptability on a held-out task family versus a defined baseline. Task,
budget, and baseline are all defined above; keep them attached to every number reported.

---

## 6. Commit protocol

- Run artifacts under `eaaj-pilot/outputs/exp2_cuda_guru_math_<hash>/`.
- `outputs/ACTIVE_RUN.txt` is machine-local and untracked — do not commit it.
- Commit after each completed phase, message prefix `exp2:`.
- Push to `main`. The Mac side pulls from `main`; do not open a branch for this.
- Never overwrite a preserved failure directory (`*_oom_*`, `safety_stop.json`,
  archived first measurements). Evidence is preserved even when it is embarrassing.

---

## 7. Do NOT

- Do **not** start Phase 1 before the stage-1 claim is posted to the team channel. Tommy
  requires each member to hold a different stage 1; starting first and asking later wastes
  4 hours if Jason has already claimed Math.
- Do **not** change the learning rate to rescue a collapse. A collapse at 5.5e-6 is a
  reportable result — it is the first stable-ish evidence that the transition window has
  been reached — and it changes the team's dose map. Preserve everything and report.
- Do **not** add a shaping reward if GATE 0b fires. Preserve the diagnostic and ask.
- Do **not** run all Phase-3 cells in one process.
- Do **not** load model params in bf16.
- Do **not** substitute a different model size to make something fit. Report the fit
  problem and request the L4.
- Do **not** report `ΔR_t` without `T_t` next to it.

---

## 8. Open questions to raise with the team (do not decide silently)

1. **Stage-1 learning rate.** 5.5e-6, matching exp1.7 (§1.1). Nobody has yet established a
   dose that produces a persistent, survivable adaptability loss on this model: 1e-6 and
   3e-6 are confirmed nulls, and 1e-5 collapsed once in four attempts. If exp2 also nulls
   at 5.5e-6, the next lever is dose or model size — not a fifth dataset pair. This is the
   single most important thing to say in-channel before four people spend three weeks
   picking four new task pairs at doses that have never moved anything.
2. **Stage-1 claim collision.** Math is claimed here. Needs confirmation in-channel.
3. **Base vs Instruct**, **GRPO vs SFT for stage 2**, **KL β>0 arm** — all still open, all
   carried forward at the cheaper default.
4. **Seeds.** 1 seed is a real limitation of this run, forced by the 4070 and 23 Aug.
   Say so in the write-up rather than presenting a single-seed ranking as stable.
