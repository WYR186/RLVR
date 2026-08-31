# SPEC E1 — metric re-measurement sweep on the existing 7B checkpoints

**Owner:** Aaron Wang (early-warning diagnostics)
**Date:** 2026-08-30
**Status:** spec, not yet executed.
**Parent:** `PROJECT_OVERVIEW_AND_NEXT_EXPERIMENTS.md` §5.1 — read that for why.
**Depends on:** nothing. No retraining, no team decision, no other person.
**Estimated cost:** ~1–2 A100-hours ≈ 7–14 compute units.

---

## 0. One-paragraph summary

The 7B run reports `dormant_frac = 0.0000` at every layer, checkpoint and threshold,
and effective rank moving ≤0.55%. We want to claim from this that **Q has no usable
dynamic range on gated-MLP LLMs**. Before that claim can survive review it has to
rule out the alternative explanation that we operationalized Q badly. E1 re-measures
the same three checkpoints under a grid of defensible alternative operationalizations
— probe distribution, dormancy pooling, dormancy tensor, effective-rank tensor,
threshold, layer, probe size — against the same frozen probe and the same
comparability contract. Either
every variant reads flat (a decisive negative result with a mechanism) or one does
not (a corrected metric, which is a better result). No retraining is involved.

---

## 1. Inputs, all already in hand

| Artifact | Location | Verification |
|---|---|---|
| Stage-A adapters ckpt-0 / 50 / 100 | Drive `MyDrive/eaaj-exp2-checkpoints/{02,03,04}_ckpt-*.tar.gz` | untar, then the ckpt-0 identity gate in §6 |
| Config | same folder, `exp2_colab_config_mvp_instruct.json` | hash `e33527592dd9` |
| Frozen splits | same folder, `exp2_colab_splits_instruct.json` | id-list sha16s: `stage_a_train_ids` a06fe6b80e7d40ca/54251, `stage_b_train_ids` df8623cf009e0690/1132, `stage_b_eval_ids` 8ee975c7089dc72a/300, `probe_stage_a_topup_ids` 1e61252e7b54793e/4096 |
| Reference Q values | `experiment 2/FINDING_Q_METRICS_7B_INSTRUCT.md` | the reference arm must reproduce these exactly |

Model: `Qwen/Qwen2.5-7B-Instruct`, LoRA r=16 α=32 on all seven projections, base
bfloat16 / adapter float32. 28 decoder blocks, hidden 3584, MLP intermediate 18944.

Runtime restore path after a VM recycle is the proven ~15-minute one: notebook 00
cells 1–6 only, then the Drive restore cell. Do not re-run cells 8+ (Phase-0 gates,
~30 min, not needed here).

---

## 2. Reference arm — held fixed, and it is also the gate

Exactly the contract already in force. This arm is re-run first, unchanged, and its
output must match `FINDING_Q_METRICS_7B_INSTRUCT.md` **exactly**:

```
model_eval                = True
model_dtype               = bfloat16 base / float32 LoRA
hidden_pooling            = last_non_padding_token
dormant_pooling           = mean_abs_over_all_non_padding_tokens
dormant_tensor            = down_proj input  ==  act_fn(gate_proj(x)) * up_proj(x)
dormant_score             = s_i = E|h_i| / mean_j E|h_j|      (ReDo normalized form)
max_prompt_tokens         = 512
activation_accumulator    = float32
svd_dtype                 = float64
spectrum_centering        = True
n_probe                   = 4096
layers                    = [5, 14, 26]
batch_size                = 16
taus                      = [0.025, 0.1]
```

Reference values to reproduce:

| layer | erank ckpt-0 | ckpt-50 | ckpt-100 | dormant_score_min (ckpt-0) |
|---|---:|---:|---:|---:|
| 5 | 1127.4155 | 1128.2271 | 1128.1812 | 0.1604 |
| 14 | 1281.0450 | 1287.8799 | 1287.8093 | 0.4148 |
| 26 | 1426.0597 | 1433.8730 | 1432.5480 | 0.1606 |

`dormant_frac` is 0.0 at every cell of that table. Two independent ckpt-0 passes
previously agreed **bit for bit** (delta = 0.00e+00), so any nonzero drift here is a
real environment change and **must be resolved before the sweep is trusted**.

Existing code path: `pipeline.measure_checkpoint_q(...)` →
`eaaj-pilot/src/metrics.py::checkpoint_q_metrics` →
`collect_probe_activations`. All six variants below are modifications of
`collect_probe_activations`; `spectrum_metrics` and `dormant_metrics` are unchanged
pure-numpy functions and stay unit-tested as they are.

---

## 3. The six variants

Every variant is measured at **all three checkpoints** unless stated otherwise, and
writes its own full contract block into its output JSON.

### V1 — probe distribution: prompt-only → prompt + continuation

**Why.** The current probe never contains a generated token. RLVR does not update the
model on prompts; it updates it on completions. If stage A changed generation
behaviour while leaving prompt representations intact, the probe would read flat by
construction. This is the single most likely explanation for "Q did not move" that we
have never tested.

**V1a — comparable.** Generate continuations **once from ckpt-0** and freeze them.
All three checkpoints are then measured on the identical token sequences
(prompt + fixed continuation). The frozen-probe contract survives intact, so V1a is
directly comparable across checkpoints and belongs on the same axes as the reference
arm.

**V1b — on-policy, explicitly not comparable.** Each checkpoint generates its own
continuations. This is the behaviourally relevant probe, but the input distribution
now differs per checkpoint, so it violates the comparability contract. Report it as a
separate, labelled measurement. **Never plot V1b on the same series as V1a or the
reference arm**, and say in the caption why.

**Settings.** Subsample the frozen probe to its **first 512 prompts in stored order**
(deterministic, no RNG), `max_new_tokens=256`, greedy (`do_sample=False`) so no seed
enters. Continuation tokens only are pooled for both erank and dormancy — record
`hidden_pooling=last_continuation_token` and
`dormant_pooling=mean_abs_over_continuation_tokens`.

**n-truncation caveat, mandatory in the write-up.** 512 < hidden dim 3584, so V1
erank magnitudes are **sample-truncated** and are not comparable in level to the
n=4096 reference arm. V1 is a *within-V1 across-checkpoint* comparison only. V5's
n-sweep is what establishes how much of the level is n-driven.

**Cost.** V1a = 1 generation pass + 3 measurement passes. V1b = 3 generation passes +
3 measurement passes. This is the expensive variant; run V1a first and only run V1b
if V1a shows anything.

### V2 — dormancy pooling: mean-over-tokens → per-token and max-over-tokens

**Why.** The current statistic averages |h| over every non-pad token of every prompt
into one scalar per unit. A unit that is silent on 95% of inputs and active on 5%
reads as fully active. That is exactly the kind of dormancy the metric is supposed to
find.

**Variants, all on the same forward passes as the reference arm:**

- **V2a per-token fraction** — compute `s` per (unit, token) against that token's own
  layer mean, then report the fraction of (unit, token) pairs below τ.
- **V2b max-over-tokens** — per unit, take `max_t |h_{i,t}|` instead of the mean, then
  normalize as usual. A unit is dormant only if it is *never* active.
- **V2c per-prompt median** — pool within prompt, then take the median across
  prompts, so a few outlier prompts cannot rescue a unit.

Report all three alongside the reference mean-pooling at the same τ grid.

**Cost.** Near zero — it is extra accumulators on the forward passes already being
run. Implement as additional reductions inside `collect_probe_activations`.

### V3 — dormancy tensor: `act_fn(gate)·up` → `act_fn(gate)` alone

**Why.** The hook currently captures the `down_proj` input, i.e. the product. A
near-zero gate can be rescued by a large `up`, so the product has a structurally
higher floor than the gate. SiLU has a genuine near-zero region for large negative
inputs; the product may be destroying the very range the metric needs.

**Variants:** capture `gate_proj` output (pre-activation), `act_fn(gate_proj(x))`
(post-activation), and `up_proj` output, each via its own forward hook, scored with
the unchanged `dormant_metrics`. Report all three next to the current product.

**Cost.** Three extra hooks on the same forward passes. Negligible.

### V4 — τ sweep and the score distribution

**Why.** "No headroom" is currently an assertion resting on two thresholds. A curve
and a histogram make it a demonstration, and pre-empt the reviewer question directly.

- **τ grid:** logarithmic, `1e-4 … 1.0`, 25 points, plus the two registered values.
- **Deliverable 1:** `dormant_frac(τ)` curve per layer per checkpoint per tensor
  (V3) per pooling (V2). One figure, faceted.
- **Deliverable 2:** the **full per-unit score vector** `s_i` saved to disk (18944
  floats per layer per checkpoint — trivially small), so the histogram is
  reproducible without re-running anything.
- **Deliverable 3:** report `min`, p1, p5, median of `s` per layer per checkpoint, so
  the distance between the distribution and the thresholds is quantified rather than
  described.

**Cost.** Zero GPU. Pure post-processing of the accumulators, provided the per-unit
vectors are saved.

### V5 — sensitivity: layers, probe size, token position

**Why.** Only 3 of 28 layers were measured, and the claim that n=4096 > d=3584 makes
erank non-truncated is asserted but never demonstrated on this model.

- **V5a all 28 layers** — one forward pass with hooks on every block. Establishes
  that the flat reading is not a three-layer accident and gives a depth profile,
  which is a figure in its own right.
- **V5b probe size** `n ∈ {512, 1024, 2048, 4096}` — nested prefixes of the frozen
  probe in stored order, no resampling. Shows the erank-vs-n curve and where it
  saturates, which is what justifies (or corrects) the n=4096 choice and is needed to
  interpret V1's truncated values.
- **V5c token position** — last-non-pad token (reference) vs mean over all non-pad
  positions, for the hidden-state spectrum.

**Cost.** V5a is one extra pass with more hooks; V5b is subsetting an activation
matrix already in memory (cheapest of all — compute the SVD on prefixes of the same
matrix); V5c is one extra reduction.

### V6 — effective-rank tensor: residual stream → MLP activations

**Why — this is the variant we know flips the sign.** On 0.5B run 2, across the same
checkpoints and the same collapse, `erank_mlp_mid` fell **−71.5%** while
`erank_resid_mid` rose **+362%** (parent doc §3.1 C). A metric whose direction
depends on which tensor is read is not a detector yet. The 7B run measured **only the
residual variant** — the one that went *up* in the 0.5B collapse — so our "+0.55%,
flat" reading has never been checked against the variant that actually moved.

**Variants.** Compute the full spectrum block (`erank`, `erank_norm`,
`participation_ratio`, top-k variance shares, both anisotropies) on:

- **V6a** residual-stream hidden state (the reference arm; decoder block output)
- **V6b** MLP post-activation, i.e. the same `down_proj` input tensor dormancy is
  scored on, pooled to one vector per prompt by the same `last_non_padding_token`
  rule
- **V6c** `act_fn(gate_proj(x))` alone, pooled the same way — pairs with V3 so the
  gate is checked on both metrics

**Caveat to record.** The MLP tensor has dimension 18944, far above n_probe = 4096,
so V6b/V6c erank is **sample-truncated by construction** and its *level* is not
comparable to V6a's. Only the across-checkpoint change within a variant is
interpretable, and V5b's n-sweep is what bounds how much of the level is n-driven.
State this in the caption every time a V6b number appears.

**Cost.** Extra reductions on the same forward passes as V2/V3. Negligible.

---

## 4. Execution order

Ordered so the cheap, no-generation work lands first and the expensive variant last.

```
1. reference arm            3 passes    — GATE, must reproduce §2 exactly
2. V2 + V3 + V5a + V5c + V6 3 passes    — all extra hooks/accumulators on one sweep
3. V4 + V5b                 0 passes    — post-processing of what step 2 saved
4. V1a                      1 gen + 3 passes
5. V1b                      3 gen + 3 passes  — only if V1a shows movement
```

Steps 1–3 are the bulk of the value and involve no generation at all. If the session
is cut short, stopping after step 3 still yields a complete, publishable
operationalization sweep; V1 is the upside case.

---

## 5. Output schema

One JSON per (variant, checkpoint) under `measurements/e1_sweep/`:

```
{
  "variant": "V2b",
  "checkpoint": 50,
  "measurement_contract": { ... every field from §2, with the variant's overrides ... },
  "n_probe": 4096,
  "layers": [...],
  "per_layer": {
    "layer5": {
      "erank": ..., "erank_norm": ..., "participation_ratio": ...,
      "top1_var_share": ..., "top8_var_share": ..., "top32_var_share": ...,
      "anisotropy_centered": ..., "anisotropy_uncentered": ...,
      "dormant_frac_by_tau": { "0.0001": ..., ..., "1.0": ... },
      "dormant_score_min": ..., "dormant_score_p1": ...,
      "dormant_score_p5": ..., "dormant_score_median": ...,
      "dormant_score_vector_path": "e1_sweep/scores/V2b_ckpt50_layer5.npy"
    }, ...
  },
  "provenance": {
    "config_hash": "e33527592dd9",
    "splits_sha16": { ... the four id-list hashes ... },
    "adapter_path": "...",
    "reference_arm_gate": "pass"
  }
}
```

Plus one `e1_sweep/summary.csv` with one row per (variant, checkpoint, layer, τ) for
direct plotting, and `e1_sweep/scores/*.npy` holding the per-unit vectors.

---

## 6. Acceptance gates

1. **Reference-arm identity gate.** The reference arm must reproduce the §2 table
   exactly (`delta < 1e-4` on erank, and `dormant_frac == 0.0` throughout). This is
   the same gate notebook 02 cell 9 already implements. If it fails, **stop** — the
   environment drifted and nothing downstream is interpretable. Investigate before
   proceeding; do not "accept and note it."
2. **Contract completeness.** Every output JSON contains its full contract block.
   A variant whose JSON does not record what it changed is not usable evidence.
3. **Provenance.** Config hash and all four split sha16s asserted at load, as in the
   existing restore path.
4. **Per-unit vectors saved.** V4 is not reproducible without them and re-running to
   recover a histogram would waste the session's GPU time.
5. **No silent variant substitution.** If a variant cannot be implemented as
   specified (e.g. a hook point does not exist on this architecture), record it as
   not-run with the reason. Do not substitute a similar-looking measurement.

---

## 7. How to read the results

| Outcome | Reading | Consequence for the paper |
|---|---|---|
| every variant flat / zero | Q has no dynamic range under any defensible operationalization | decisive negative result about a metric imported from RL into LLMs, with a mechanism (§3.1 A of the parent doc) |
| **V3** shows range on `act_fn(gate)` but not the product | the standard operationalization is destroyed by the gating multiply | corrected metric — a positive contribution, and a specific fix for gated-MLP architectures |
| **V2** shows range under per-token or max pooling | mean-over-tokens pooling was hiding conditional dormancy | corrected metric; also implies every prior LLM dormancy measurement using mean pooling is understated |
| **V1** shows movement on continuations but not prompts | the probe was blind to the distribution RLVR actually trains on | the most consequential outcome: it would mean "Q did not move" was a measurement artifact, and it changes what every run in the project measured |
| **V6** shows the MLP variant moving where the residual variant is flat | the 7B "Q did not move" reading is a tensor-choice artifact, exactly as the 0.5B collapse predicted | the strongest single result E1 could return; it would mean every Q number in this project measured the wrong tensor |
| **V5a** shows movement at unmeasured depths | the three-layer choice missed where the change is | cheap fix, but weakens all prior per-layer claims — report honestly |

Note the asymmetry: four of these six outcomes would mean **our own prior
measurements understated the metric**, and we should say so plainly rather than
defend the original contract. The reference arm is preserved unchanged in every case
so the before/after is auditable.

---

## 8. What E1 explicitly does not do

- It does not train anything, and it does not touch Stage B.
- It does not change any registered metric definition for existing results. The
  reference arm stays the registered contract; variants are reported alongside it.
- It cannot test RQ1 — Δ-R is flat across these three checkpoints regardless of how Q
  is measured, so no re-measurement of Q can create variance on the outcome side.
  E1 is about whether **the detector** works, not about the correlation.
- It says nothing about the 0.5B runs. Applying the surviving variants there is a
  follow-up once E2/E3 have an owner.
