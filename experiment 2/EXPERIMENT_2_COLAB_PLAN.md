# Experiment 2 (Colab variant) — GURU Math → Simulation, ≥7B, LoRA GRPO

**Target machine:** Google Colab (L4 default, escalate to A100 — see Gate C0)
**Audience:** the agent running this notebook set
**Owner:** Aaron Wang (Person 4)
**Status:** pre-registered, not started
**Written:** 2026-08-12
**Supersedes for this run only:** `EXPERIMENT_2_PLAN.md` (the Windows RTX4070 / Qwen2.5-0.5B /
full-parameter-fp32 version). That document is not retracted — it is a valid, independently
useful run on different hardware. This document is a **fork**, not an edit, because the
model-size and precision changes below touch nearly every numeric parameter in it.

---

## 0. What this is, and what changed from the 4070 version

Same pipeline spec, same source (Tommy, Slack 12:15 edited version; quoted in full in
`EXPERIMENT_2_PLAN.md` §0 — not re-quoted here to avoid drift between two copies). Same claim:
**stage 1 = Math** (OR1/DAPO/DeepScaler), **stage 2 = Simulation** (CodeI/O).

Two things changed, both from the 2026-08-12 conversation that produced this document:

1. **Machine: Windows RTX4070 laptop (8 GB) → Google Colab.** Frees the 8 GB ceiling.
2. **Model: Qwen2.5-0.5B → Qwen2.5-7B (base).** Tommy's Slack note, 23:37, quoted verbatim:
   *"Note: expect low accuracy / choose a model that is stronger than Qwen2.5-7B."* Read
   literally this asks for >7B; 7B is the floor stated, not the target. §1.1 explains why this
   document still pins 7B as the committed number and treats "even larger" as a stretch goal,
   not the default.

**Everything downstream of those two changes is different**, because a 7B model does not fit
an 8 GB card at all, and does not fit a 24-40 GB Colab GPU under **full-parameter** fine-tuning
either (bf16 weights alone are ~14 GB; the fp32-master-weights trick the 4070 plan required to
stop small-LR updates rounding to zero on the 0.5B model would need ~28 GB for weights alone,
before optimizer state, activations, or GRPO's generation memory). The fix is **LoRA**
(§1.2) — not just a memory optimization here, but the thing that makes "expect low accuracy /
stronger model" attemptable at all inside a Colab session and a fixed compute-unit budget.

**Scope is smaller than the 4070 plan's, on purpose.** That plan ran the full 5-checkpoint ×
3-seed grid because the extra compute was cheap on an idle personal laptop. On Colab, every
7B×A100 hour draws down a shared, metered budget, and the abstract is due **2026-08-23** (11
days from today). §1.3 and §8 make this a logged, explicit scope decision rather than a silent
one: **the committed deliverable is Tommy's original three-checkpoint minimum, one seed**, with
the full grid as a stretch goal gated on measured Phase-0 throughput. This inverts the 4070
plan's "always run the superset" stance, and the reason is the reason itself is worth stating
plainly — compute got 10-20x more expensive per unit of pipeline coverage, and the deadline did
not move.

---

## 1. Design decisions and their justification

| Parameter | Value | Why |
|---|---|---|
| Stage 1 domain | **Math** (OR1/DAPO/DeepScaler) | Unchanged — same claim as the 4070 plan, same team-facing collision constraint (§7). |
| Stage 2 domain | **Simulation** (CodeI/O) | Fixed by Tommy for everyone. |
| Model | `Qwen/Qwen2.5-7B` **base** (not Instruct), pinned revision **discovered in Phase 0**, not guessed | 7B is Tommy's stated floor. Base, not Instruct, because (a) every prior run in this project is base, and mixing chat-template formatting into the numeric-reward pipeline is a new failure surface with 11 days on the clock, (b) "expect low accuracy" reads as expecting base-model-style struggle, not instruction-following. **Still an open team question** (carried at the cheaper default, same as the 4070 plan logged it) — flagged again in §8. |
| Fine-tuning method | **LoRA** (PEFT), base frozen in bf16 | Full-parameter fine-tuning of a 7B model does not fit a single Colab GPU under this project's own precision constraint (fp32 master weights, required because bf16-param updates at the LRs this project uses round to zero — proven on the 0.5B run, see `WIN4070_RUN_ANALYSIS.md`). fp32 master weights alone would be ~28 GB before anything else. LoRA sidesteps the rounding problem structurally: the frozen base's dtype no longer matters because it is never updated, and the trainable LoRA matrices are a few tens of millions of parameters, cheap to keep at fp32 precision regardless of GPU tier. |
| LoRA adapter dtype | **float32**, base in **bfloat16** | Same rounding-hazard discipline as the 4070 plan, applied to the part that actually receives gradient updates (the base doesn't). Verified empirically anyway — via a LoRA-aware subclass of `eaaj-pilot`'s `UpdateEffectivenessSentinel` that samples **only trainable (LoRA) parameters**: the parent samples every parameter, which under LoRA dilutes the relative-change measurement with 7B frozen weights that never move by construction, and could make the step-25 kill-gate misread a healthy run (or mask a dead adapter). The parent's healthy/broken reference scales were measured on full-parameter 0.5B runs and do NOT transfer; Phase 0's smoke test records the first healthy-LoRA reference value. |
| LoRA config | rank **16**, alpha **32**, dropout **0.05**, target modules `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` | Standard full-attention+MLP LoRA target set for the Qwen2 architecture family. **First-pass default, not validated by any run yet** — Phase 0's smoke test is the first time this project will have trained a LoRA adapter at all, so this is stated as a starting point subject to revision, not a measured choice like the 0.5B dose map was. |
| GPU tier | **L4 (24 GB) first, escalate to A100 (40 GB) if Gate C0 fails** | Same escalate-don't-shrink philosophy as the 4070 plan's GATE 0a, applied to GPU tier instead of batch size — L4 is the cheaper compute-unit draw, so it is the default, not the ceiling. |
| Quantization | **None** (plain bf16 base) by default; NF4 4-bit (bitsandbytes, already pinned in `requirements.txt`) as an explicit fallback if Gate C0 fails even on A100 | Keeps the numerics simple under time pressure — quantization is a second failure surface (calibration, dequantization error interacting with the effective-rank/dormant-fraction metrics) that this project has never exercised. Escalate to it only if plain bf16 genuinely does not fit, and if so, re-run the ckpt-0 Q measurement sanity check (§3, Phase 2b) since quantization can distort activation statistics. |
| Stage-1 length | **200 updates** (unchanged target), checkpoints at **0, 100, 200** (three-checkpoint minimum) | Tommy's original compute concession — "start with three: zero, middle, last" — which the 4070 plan exceeded because it was free to. It is not free here (§0, §1.3). Phase 0's measured s/update replaces this estimate before Phase 1 starts, exactly as in the 4070 plan, and if it implies the schedule will not fit before 2026-08-23, cut in the priority order given in §8. |
| Stage-1 checkpoints adapted from | **0, 100, 200** (all three saved checkpoints) | ckpt-0 **is** the stage-2-alone baseline, not a separate run — same convention as the 4070 plan. |
| Stage-1 learning rate | **2e-5** on the LoRA adapter parameters | Explicitly **not** the 4070 plan's 5.5e-6. That dose was derived from a full-parameter dose-response study on the 0.5B model and is not transferable to a LoRA adapter on a 14x larger base — the mechanism moving the weights is different in kind (a rank-16 subspace update vs. every parameter), so there is no principled way to reuse the old number. 2e-5 is a literature-typical order of magnitude for LoRA-adapter RL fine-tuning, **stated as unvalidated**. This run is therefore **not** a same-dose comparison to exp1.7 or exp1.6 the way the 4070 exp2 was designed to be — that comparability is explicitly given up here, and is listed as a consequence in §8, not hidden. |
| Stage-2 budget | **50 updates**, eval at 0/10/20/30/40/50 | Unchanged — same rationale as the 4070 plan (endpoint + AUC). |
| Stage-2 eval set | **300 held-out Simulation questions**, frozen, committed | Unchanged. |
| Stage-2 seeds | **1 (seed 42) committed; seeds 43/44 are a stretch goal**, not a default | Inverts the 4070 plan's stance (§0, §1.3). A single seed cannot support a defensible correlation claim — this is stated as a known limitation of the committed scope, not smoothed over. If Phase 0 timing leaves headroom before the deadline, add seeds in the same two-pass structure the 4070 plan used. |
| Activation-metric probe layers | **[5, 14, 26]** (of 28 decoder blocks) | Rescaled from the 0.5B plan's `[4, 12, 22]` (of 24 blocks) to the same relative depths (~18%/50%/93%) rather than reusing the literal indices, which would sit at different relative depths in a deeper network. |
| Activation-metric probe size | **4096 prompts**, not 2048 | Qwen2.5-7B's hidden dimension is **3584**, not 896. The 0.5B plan's own reasoning (§1, "hidden dim is 896, so a 512-prompt probe makes effective-rank magnitudes n-dependent") applies again one level up: a 2048-prompt probe would itself sample-truncate a 3584-dimensional activation matrix, which is exactly the failure mode that reasoning warned against. 4096 keeps `n_probe > hidden_dim` with margin. |
| Activation-metric probe source | Frozen at Phase 0 in `exp2_splits.json`: stage-B pool rows disjoint from stage-B train AND eval, topped up from held-out stage-A rows if the CodeI/O pool is too small (per-source counts recorded) | The 300-question eval set is far too small to substitute (n < hidden dim), and probe membership must be committed before any training so every checkpoint is measured on identical prompts. The cross-domain top-up, if triggered, is a logged property of the probe — same pattern as eaaj-pilot's probe superset topping up from held-out GSM8K train. |
| ckpt-0 identity gate | **Dropped as a cross-run check; kept as a within-run sanity check** | The 4070 plan's ckpt-0 gate compares against a *committed pilot reference value* from the 0.5B model — that value is not comparable to a 7B model's effective rank (different hidden dim, different architecture instance) and reusing it would be a category error. Here the gate instead re-measures ckpt-0 twice (once right after Phase 1 starts, once during Phase 2b) and requires the two measurements to match — catches a measurement-contract drift (e.g. a stray dtype or layer-indexing bug) without pretending cross-model comparability that doesn't exist. |
| KL β | **0.0** | Unchanged from every prior run in this project. |

### 1.1 Reading Tommy's "stronger than Qwen2.5-7B" note literally

Tommy's note names Qwen2.5-7B as the threshold to beat, not the target: *"choose a model that
is stronger than Qwen2.5-7B."* This document commits to exactly 7B rather than something
larger (e.g. 14B/32B) for one reason: **budget and time**, not a disagreement with the note.
Every step up in model size roughly doubles memory pressure and per-update latency, and this
project has zero measured throughput data for any LoRA GRPO run at any scale yet — Phase 0 is
the first time this will be tried. Going straight to 14B+ compounds two unknowns (does LoRA
GRPO work at all in this codebase, does it fit/run fast enough) instead of isolating them.
**This is the single most important item to flag in-channel before Phase 1** (§8, item 1):
Tommy may mean 7B is still too weak to satisfy "expect low accuracy" in the way he intends, in
which case the right move is a follow-up run at a larger size once 7B is proven to work
end-to-end, not guessing the target size now.

### 1.2 Why LoRA rather than shrinking the protocol to fit full fine-tuning

The alternative to LoRA would be to keep full-parameter fine-tuning and either (a) drop back to
a smaller model that fits under fp32-master-weights on a single Colab GPU, or (b) use gradient
checkpointing plus 8-bit optimizer state alone. (a) directly contradicts the stated purpose of
this fork (satisfy the ≥7B note); (b) still requires materializing fp32 master weights for
every one of 7B parameters — the arithmetic doesn't work even on a 40 GB A100 once GRPO's
per-group generation memory is added. LoRA is the standard, well-supported way (TRL's
`GRPOTrainer` accepts a `peft_config` argument directly — no custom training loop needed) to
get a ≥7B model training within a single Colab GPU's memory, and it composes cleanly with
everything else in this pipeline: `eaaj-pilot/src/metrics.py`'s activation-collection functions
hook `model.model.layers[l]` and `mlp.down_proj` by module reference, which exist identically
on a PEFT-wrapped model — no metric code needs to change.

### 1.3 Why the committed scope shrank instead of staying a superset

The 4070 plan's superset (5 checkpoints × 3 seeds = 15 stage-2 cells, ~25 h) was justified
there because the extra ~15 h cost nothing beyond electricity on an otherwise-idle personal
machine with weeks of runway. Neither is true here: Colab compute is a shared, metered budget
(`CLAUDE.md`'s ~300-unit figure is for the primary pilot; this run was not part of that
budget and needs its own accounting — flagged in §8), and a 7B model on any Colab tier is
markedly slower per update than a 0.5B model on a laptop GPU, by an amount this project has not
yet measured (§4). Committing to the full grid before that number exists risks spending the
entire remaining runway to 2026-08-23 on a single incomplete run. Tommy's original ask (three
checkpoints, presumably one seed) is the floor that was always sufficient to answer his
question; this document keeps that floor as the default and makes expansion conditional and
explicit rather than assumed.

---

## 2. Pre-registered outcomes (fixed before any run)

**Unchanged from `EXPERIMENT_2_PLAN.md` §2** — same three quantities, same reading rule. Restated
here only because this is the document an agent should be able to run from without
cross-referencing the other one:

Let `R_end(t)` = mean stage-B eval reward at stage-2 update 50, starting from stage-1 checkpoint
`t`. Let `M_0` = the untrained base model (Qwen2.5-7B, no LoRA applied).

**P1 — plasticity metric:** `ΔR_t = R_end(0) − R_end(t)`. `R_end(0)` is the stage-2-alone
baseline. `ΔR_t > 0` means stage 1 cost stage-2 reward.

**P2 — robustness:** `AUC_t = mean over {0,10,20,30,40,50} of stage-B eval reward from ckpt t`;
`ΔAUC_t = AUC_0 − AUC_t`.

**P3 — transfer control:** `T_t = Score_B(M_{A,t}) − Score_B(M_0)`, zero-shot, no stage-2
training.

**Reading rule** (unchanged): `ΔR_t` is interpreted conditional on `T_t` — `T_t ≈ 0` reads as
candidate plasticity loss, `T_t ≫ 0` means confounded (less headroom left), `T_t ≪ 0` means
stage 1 damaged stage-B ability directly. Phase 2 (cheap, zero-shot) runs before Phase 3
(expensive) for exactly this reason.

---

## 3. Phases, gates, and kill conditions

Run in order. **A gate that says STOP means stop and report — do not tune around it.**

### Phase 0 — data, geometry, memory, and smoke (GATE)

Nothing here trains anything beyond a 2-update smoke test. Everything here is a discovery step
whose output is committed.

1. **Schema audit.** Identical to the 4070 plan's Phase 0 step 1 — download
   `LLM360/guru-RL-92k`, do not assume column names, write `data/guru_schema_audit.json`.
   Model choice does not change the dataset, so this step and its output are reusable verbatim
   if the 4070 run's Phase 0 has already produced it — check before re-downloading.
2. **Subset filtering** into stage 1 (Math: OR1/DAPO/DeepScaler) and stage 2 (Simulation:
   CodeI/O), counts recorded in the audit file. Same as the 4070 plan.
3. **Answer-format determination**, written into the audit file. The reward function
   (`experiment 2/src/guru_reward.py`) ships with a best-effort implementation for both the
   boxed-numeric convention typical of curated math RL sets and a normalized-string/structured
   match for CodeI/O-style input/output prediction, **selected and parameterized by what this
   step actually finds** — not assumed. If the discovered format doesn't match either shipped
   extractor, extend `guru_reward.py`; do not force a mismatch through the existing regex.
4. **Token-length audit**, under the **Qwen2.5-7B tokenizer** (not 0.5B's — different
   vocabulary/merges can shift token counts). Same **GATE 0a — STOP if stage-2 p95 prompt
   length > 1024 tokens**, same consequence (escalate `max_prompt_length` / GPU tier rather
   than shrinking until it fits).
5. **Gate C0 — GPU memory calibration (new in this document).** Load the base model in bf16 +
   LoRA on the default tier (L4), run the 2-update smoke test at the provisional batch/
   generation-count defaults in `exp2_colab_config.json`, and record peak allocated memory.
   - **PASS** if peak memory leaves ≥15% headroom on L4 → continue on L4.
   - **ESCALATE to A100** if L4 OOMs or leaves <15% headroom — do not shrink
     `num_generations`/batch below the config defaults to force an L4 fit; a smaller effective
     batch changes the GRPO update semantics the same way the 4070 plan refused to shrink batch
     size below its own defaults.
   - **STOP and flag** if it does not fit on A100 either at the config defaults — this is the
     trigger for the NF4-quantization fallback (§1, Quantization row), which needs a one-line
     deviation note before use, not a silent switch.
6. **Freeze splits.** Same as the 4070 plan: `data/exp2_splits.json`, seeded, committed.
7. **Smoke.** 2 updates on stage 1 and 2 updates on stage 2, in a throwaway dir. Verify:
   LoRA adapter checkpoint written (not full model — confirm `save_pretrained` on the PEFT
   wrapper only serializes the adapter), dashboard row written, `UpdateEffectivenessSentinel`
   fires and shows nonzero relative change (this is the first real check that 2e-5 on a LoRA
   adapter is not itself rounding to zero — do not skip it), GPU telemetry captured, loss and
   grad norm finite.
8. **Sparse-reward preflight** on both stages — same **GATE 0b — STOP** condition as the 4070
   plan (constant reward across every sampled group). Do not add shaping reward, do not switch
   models; this reuses `eaaj-pilot/src/preflight.py` unmodified.

Phase 0 also produces the **real** per-update timing and peak-memory numbers, which replace
every estimate in §4 and directly decide the de-scope question in §8.

### Phase 1 — stage 1 GRPO (Math), LoRA

- 200 updates, LoRA lr 2e-5, β=0, checkpoints (adapter-only) at 0/100/200.
- Same dashboard row and same safety stops (5 consecutive zero-variance updates; 5 consecutive
  >10% clipping) as the 4070 plan, reused unmodified from `eaaj-pilot/src/callbacks.py`. The
  update-25 kill-gate uses the **LoRA-aware sentinel subclass** (trainable-parameters-only
  sampling — §1, "LoRA adapter dtype" row) rather than the parent class.
- If a safety stop fires: preserve everything and stop. A collapse is a result. Do not change
  the LR to rescue it — same rule as the 4070 plan (§7).

### Phase 2 — `T_t`, zero-shot (cheap, runs before Phase 3)

For ckpt 0/100/200: evaluate on the frozen 300-question stage-B eval set with each LoRA
adapter attached, no training. Write `analysis/transfer_T.json`. Report to the channel before
Phase 3 starts — same rule as the 4070 plan, same reason (it decides how Phase 3's result can
be read).

### Phase 2b — activation metrics

Effective rank + dormant fraction at layers **[5, 14, 26]** on the frozen **4096-prompt** probe
set, float32 activation accumulation, eval mode, LoRA adapter attached (measuring the
post-adaptation model, not the frozen base) — mirrors the 4070 plan's Phase 2b exactly except
for the two rescaled numbers justified in §1. ckpt-0 identity gate is the **within-run**
re-measurement check described in §1's table row, not a cross-model comparison.

Beyond Tommy's spec, costs minutes, gates nothing in Phase 3. Report dormant fraction honestly
even if it again reads ~0 for structural reasons (SiLU-gated MLP, same as every Qwen2.5 size).

### Phase 3 — stage 2 GRPO from each checkpoint (Simulation), LoRA

**Committed grid: 3 checkpoints × 1 seed (42) = 3 cells.** 50 updates per cell, eval on the
frozen 300 at 0/10/20/30/40/50. ckpt-0 is the stage-2-alone baseline, not a separate run.

**Stretch goal, gated on Phase 0 timing and remaining runway to 2026-08-23:** add seeds 43/44
(→ 9 cells total), in the same "commit after the readable result, then expand" pattern as the
4070 plan's two-pass structure. Do not start the stretch-goal seeds before the 3-cell committed
result is complete, analyzed, and reported.

Execution rules carried over unchanged from the 4070 plan (all still apply on Colab): one fresh
process per cell where the runtime allows it (Colab notebooks are one process per session —
approximate this with an explicit variable/memory reset between cells, i.e. delete the model
and adapter, `gc.collect()`, `torch.cuda.empty_cache()`, matching what
`run_fixed_budget_adaptation` already does at the end of every call); set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; on OOM, preserve the partial directory
under `<cell>_oom_<date>` and rerun that cell alone; every cell must pass the completion
validator before Phase 4; commit after each pass.

### Phase 4 — analysis

Runs on CPU (local Mac or Colab CPU runtime, either is fine — no GPU needed). Same outputs as
the 4070 plan §3 Phase 4: `ΔR_t`/`ΔAUC_t` table, `T_t` table, the reading-rule verdict per
checkpoint, variance decomposition (**with n=3 checkpoints and, at committed scope, n=1 seed —
state plainly that no within-checkpoint variance estimate exists at committed scope**, rather
than computing a false one), Spearman ρ(erank_L14, Δ) reported as descriptive only (n=3 is even
smaller than the 4070 plan's already-descriptive n=5), and the Tommy-shape figure (stage-B
reward curve per stage-1 checkpoint, ckpt-0 as reference line).

---

## 3.1 Code layout

| Phase | Notebook | Backing code |
|---|---|---|
| 0 (audit, GATE 0a/0b, Gate C0, smoke) | `colab/00_setup_schema_audit.ipynb` | `src/guru_data.py` (`audit_schema`, `token_length_audit`, `build_exp2_splits`), `src/pipeline.py` (`gate_c0_memory_probe`, `guru_sparse_reward_preflight`) |
| 1 (stage-A GRPO) | `colab/01_stage_a_math_grpo.ipynb` | `src/pipeline.py` (`run_stage_a_grpo`, `build_peft_model`) |
| 2 (`T_t`) + 2b (Q metrics) | `colab/02_transfer_T_and_qmetrics.ipynb` | `src/pipeline.py` (`run_transfer_T`, `measure_checkpoint_q`) |
| 3 (stage-B grid) | `colab/03_stage_b_simulation_adaptation.ipynb` | `src/pipeline.py` (`run_stage_b_adaptation`) |
| 4 (analysis) | `colab/04_analysis.ipynb` | pandas/scipy inline, reads the JSONL/JSON artifacts every prior phase wrote |

Reward/answer-extraction lives in `src/guru_reward.py` (unit-tested,
`tests/test_guru_reward.py` — run with `cd "experiment 2" && python -m pytest
tests/ -v` before ever touching a GPU, same Phase-0-development discipline as
`eaaj-pilot`). `src/pipeline.py` reuses `eaaj-pilot/src/metrics.py` and four
callback classes from `eaaj-pilot/src/callbacks.py` by explicit file path
(its module docstring explains why — both directories have a top-level `src`
package, so a normal `sys.path`-based import would be ambiguous between
them). Every notebook cell that calls into `src/pipeline.py` is a thin
driver; the logic worth reviewing lives in the `.py` files, not the
notebooks.

**Every notebook has a placeholder run-hash (`REPLACE_WITH_HASH`) in its
`RUN_DIR`.** Pick one short content hash for this run's config before
notebook 01 (mirrors `eaaj-pilot/src/repro.py:config_hash`'s convention) and
find-and-replace it consistently across notebooks 01-04 — do not let each
notebook invent its own run directory.

---

## 4. Compute budget on Colab

**No number in this section is measured yet.** Unlike the 4070 plan's §4, which anchored
estimates to a completed 19.57 h run on the same hardware/model, this project has never run a
LoRA GRPO job at any scale, so there is no anchor. This is a first-pass, explicitly
low-confidence estimate to be **replaced by Phase 0's real numbers before Phase 1 is allowed to
start** — same discipline as the 4070 plan, stated more strongly because the uncertainty here
is larger.

| Phase | Rough estimate | Confidence | Note |
|---|---|---|---|
| 0 | 45–90 min | low | download + tokenization + Gate C0 memory probe + smoke; the memory probe is new relative to the 4070 plan and its own time cost is unmeasured |
| 1 | **unknown — Phase 0 sets this** | very low | 200 updates on a 7B model; per-update latency depends on GPU tier (L4 vs A100), which Gate C0 decides, and on generation length, which the token-length audit decides |
| 2 | 15–30 min | low | 3 checkpoints × 300 questions, generation-bound |
| 2b | 10–20 min | low | 3 checkpoints, float32, 4096-prompt probe — larger probe and larger hidden dim than the 4070 plan's ~5 min figure |
| 3 (committed, 3 cells) | **unknown — Phase 0 sets this** | very low | 3 × 50-update cells |
| 3 (stretch, +6 cells) | **unknown** | very low | only attempted if committed scope finishes with runway to spare |
| 4 | ~5 min | medium | CPU, same code path as the 4070 plan's Phase 4 |

**De-scope priority order if Phase 0's measured throughput implies the committed 3-checkpoint/
1-seed schedule will not finish with margin before 2026-08-23** (apply in this order, stop as
soon as the schedule fits):

1. Confirm the stretch-goal seeds are not started (they are already gated behind this — verify
   nothing jumped ahead).
2. Reduce stage-1 length from 200 → 100 updates, checkpoints at 0/50/100. Still gives three
   points and still answers Tommy's question, just with less stage-1 exposure per checkpoint.
3. If still short: escalate to A100 if not already there (throughput, not just memory, may
   improve enough to matter) — this trades compute-unit cost for calendar time, which is the
   correct trade this close to the deadline.
4. If still short: drop Phase 2b (activation metrics). It is explicitly this owner's role but
   explicitly *not* Tommy's ask and gates nothing in Phase 3 — the least costly thing to cut.
5. Do **not** cut Phase 2 (`T_t`) — without it, `ΔR_t` cannot be read at all (§2, reading rule).
   Do **not** cut below 3 stage-1 checkpoints — that was Tommy's stated floor, not a target to
   negotiate down from.

**Compute-unit accounting is a genuinely open question for this run specifically** — see §8,
item 6.

---

## 5. Recording requirements (all of them, every phase)

Same non-negotiable project rules as the 4070 plan §5, restated for this environment:

- One row per phase in `eaaj-pilot/compute_log.md`: date, GPU tier (L4/A100), wall time, phase,
  outcome. Include the Colab compute-unit cost per phase if the Colab UI reports it.
- GPU telemetry (Colab's own resource panel, screenshotted per project convention) copied under
  the run's `telemetry/`.
- Fixed seeds everywhere; pinned model revision and dataset revision **as discovered by Phase 0
  in this run** (do not reuse the 4070 plan's config values — the model changed).
- All Q measurements in eval mode, float32 activation accumulation, fixed layers ([5,14,26]),
  frozen probe set (4096 prompts) — otherwise values are not comparable across this run's own
  checkpoints, let alone to the 0.5B pilot.
- **Leakage rule:** unchanged — any quantity computed at update `t` uses only information
  available at update `t`.
- Every deviation from this document gets a one-line justification in the run manifest,
  mirrored to the Research Doc.

**Framing constraint (mentor, Madhur) — unchanged, applies to every sentence written about
results:** never claim "RLVR reduces the model's ability to learn." The claim is about
fixed-budget adaptability on a held-out task family versus a defined baseline, with task,
budget, and baseline all defined above and kept attached to every number reported.

---

## 6. Commit protocol

- Run artifacts saved to **Google Drive** under
  `eaaj-pilot/outputs/exp2_colab_guru_math7b_<hash>/` (mirrors the 4070 plan's naming so the
  two runs are visually distinguishable by the `_cuda_` vs `_colab_` and `math_` vs
  `math7b_` infixes, not just by reading the config).
- LoRA checkpoints saved as PEFT adapters (`save_pretrained` on the wrapped model), **not**
  merged into the base — keeps checkpoints small and keeps the frozen base identical and
  reloadable across cells.
- `outputs/ACTIVE_RUN.txt` stays machine-local/untracked per project convention — on Colab this
  means it lives in the Drive-mounted working copy, not committed to git.
- Commit after each completed phase, message prefix `exp2-colab:` (distinct from the 4070
  plan's `exp2:` prefix, so `git log` alone tells the two runs apart).
- Push to `main`; do not open a branch for this, matching the 4070 plan's protocol.
- Never overwrite a preserved failure directory (`*_oom_*`, `safety_stop.json`, archived first
  measurements).

---

## 7. Do NOT

- Do **not** start Phase 1 before the stage-1 claim (Math) is confirmed in the team channel —
  same collision risk the 4070 plan flagged; if this run's claim has already been posted for
  the 4070 version, it still stands (same claim, same domain), but say explicitly in-channel
  that a second, larger-model run of the *same* claimed stage 1 is starting, so nobody reads
  silence as "Math is now free."
- Do **not** change the LoRA learning rate to rescue a collapse — a collapse at 2e-5 is a
  reportable result about this new architecture/dose combination, not a bug to tune away.
- Do **not** add a shaping reward if GATE 0b fires.
- Do **not** merge the LoRA adapter into the base model as a memory-saving shortcut — it
  defeats the small-checkpoint benefit and makes every checkpoint a full 14 GB save.
- Do **not** quietly go past 7B to a larger model to more literally satisfy Tommy's note without
  posting the reasoning in §1.1 to the channel first — model size is a compute-budget decision,
  not just a modeling one, once Colab units are shared.
- Do **not** report `ΔR_t` without `T_t` next to it.
- Do **not** claim this run is dose-comparable to exp1.6/exp1.7 or to the 4070 exp2 run — the
  LoRA-vs-full-parameter architecture change breaks that comparison; say so explicitly wherever
  the two are discussed together.

---

## 8. Open questions to raise with the team (do not decide silently)

1. **Does "stronger than Qwen2.5-7B" mean exactly 7B is acceptable, or does Tommy want strictly
   larger?** §1.1 commits to 7B as a floor-first step for budget/risk reasons. Flag this before
   spending Colab budget on Phase 1, not after.
2. **LoRA vs full fine-tuning for this experiment's "stage 1" arm at all.** Every other team
   member's stage-1 arm (per Tommy's spec) is presumably full-parameter, at whatever model size
   they chose. If everyone else stays small enough for full fine-tuning, this run's LoRA
   adapter is a second confound on top of "different task pair" — plasticity effects in a
   rank-16 subspace may look nothing like plasticity effects in full parameter space. This is
   arguably the most important methodological question in this document and needs an explicit
   answer, not an assumption.
3. **Stage-1 learning rate (2e-5, LoRA).** Completely unvalidated — no dose-response study has
   been run for LoRA GRPO on this model family, unlike the 4070 exp2's 5.5e-6, which came from
   an actual measured dose map. If Phase 1 nulls or collapses uninformatively, the first
   question is whether 2e-5 was simply the wrong order of magnitude, not whether the task pair
   is wrong.
4. **Base vs Instruct**, again — carried forward at the cheaper default (Base), same open
   status as the 4070 plan.
5. **Committed scope reduction (3 checkpoints, 1 seed) vs the 4070 run's superset (5
   checkpoints, 3 seeds).** This is the owner's call, made for time/budget reasons stated in
   §1.3, not a claim that three points and one seed are sufficient evidence — they are the
   floor Tommy asked for. Say plainly in the eventual write-up that this run's n is smaller than
   the sibling 4070 run's.
6. **Compute-unit budget ownership for this specific run.** `CLAUDE.md`'s ~300-unit figure
   covers the primary GSM8K/SVAMP pilot (`eaaj-pilot/`); this exp2-on-Colab run was not part of
   that allocation. Before burning A100 hours on a 7B model, confirm whether this draws from
   the same pool, a separate one, or needs explicit sign-off — this is a real risk of blocking
   the primary pilot's remaining Colab work if unresolved.
7. **This run is not a same-dose comparison to exp1.6/exp1.7 or to `EXPERIMENT_2_PLAN.md`'s
   4070 run** (§1, §7) — worth surfacing early so nobody in the team's later synthesis
   accidentally plots this run's numbers on the same dose axis as those.
