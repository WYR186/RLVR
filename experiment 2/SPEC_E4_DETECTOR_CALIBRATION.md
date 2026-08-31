# SPEC E4 — detector calibration: how large an intervention does Q need?

**Owner:** Aaron Wang (early-warning diagnostics)
**Date:** 2026-08-31
**Status:** spec + code complete, not yet executed.
**Parent:** `FINDING_E1_METRIC_REMEASUREMENT.md` §10 — "a future detector test needs
a regime where the intervention has measurable dynamic range."
**Depends on:** nothing. No retraining, no team decision, no other person.
**Estimated cost:** E4-small **0 compute units**. E4-large is **0 units if the
M3 Max can carry it** (7B bf16 is 15.2 GB against 128 GB of unified memory) and
~1–1.5 A100-hours ≈ **7–10 units** otherwise. `08_e4_ruler.py --benchmark N`
decides this from measured seconds-per-batch, not from a guess.

---

## 0. One-paragraph summary

E1 ruled out every measurement-choice explanation for the flat 7B Q trajectory: no
probe distribution, pooling, tensor, threshold, layer or probe size gives Q usable
across-checkpoint range. What E1 did **not** establish is a scale. "erank moved
0.53%" is uninterpretable without knowing what a large move looks like on the same
ruler, which leaves the obvious review open — *your intervention was 68× under-dosed,
so of course nothing moved; you have shown a dose problem, not a metric problem.*
E4 builds the ruler. It measures the same Q, on the same frozen probe, under the same
contract, on models separated by interventions of known and controlled size: a full
instruction-tuning pipeline (Arm R), a swept isotropic weight perturbation (Arm N),
and — in weight space, with no forward pass at all — the actual dose our Stage-A
LoRA applied (Arm W). Arm W's number lands on Arm N's axis, so the three compose
into one figure with our run marked on it.

**Both outcomes are publishable, as in E1.** If Q moves sharply somewhere on the
ladder, we can state the detection threshold and show our dose fell below it — the
dose objection is answered with a number instead of a concession. If Q stays flat
even at a destructive dose, claim (A) upgrades from "no range across our
checkpoints" to "no range across interventions spanning orders of magnitude," which
is a far harder result to wave away.

---

## 1. What this is NOT

Stated first because the framing rule (mentor feedback, Madhur) is the constraint
most easily violated by an experiment shaped like this one.

- **Not a claim that instruction tuning reduces plasticity.** Arm R compares two
  released checkpoints separated by an uncontrolled, undocumented pipeline. It is an
  order-of-magnitude reference point, nothing more. No causal language.
- **Not a model of an RLVR update.** Arm N's noise is isotropic Gaussian. Real
  updates are low-rank and structured, and could move Q more or less per unit norm.
  Arm N calibrates *the detector*, not RLVR.
- **Not a fixed-budget adaptability claim.** E4 measures no outcome. It touches the
  predictor side only.
- **Not a re-opening of E1.** The measurement contract, the frozen probe and every
  reduction are E1's, reused verbatim through `src/e1_sweep.py`. The only thing E4
  varies is which weights are in the model.

---

## 2. The one axis that can be stated in a number

Interventions are compared on **relative Frobenius weight change**:

```
        d(m) = ||W_m - W_m^0||_F / ||W_m^0||_F           per module m
   aggregate = sqrt( sum_m ||dW_m||_F^2 ) / sqrt( sum_m ||W_m^0||_F^2 )
```

The aggregate is a norm ratio over the concatenation of all targeted modules, not
the mean of per-module ratios — a mean would let a tiny module outvote a large one.
Both are reported; the aggregate is what the ladder matches.

The module set is exactly the Stage-A LoRA's targets — `q_proj, k_proj, v_proj,
o_proj, gate_proj, up_proj, down_proj` — so Arm W and Arm N measure the same
quantity over the same parameters.

Arm R has no such number available (the two released checkpoints are not a
controlled perturbation of one another), which is precisely why Arm N exists.

---

## 3. The three arms

| Arm | What it measures | Cost | Where it runs |
|---|---|---|---|
| **W** | `\|\|BA·(α/r)\|\|_F / \|\|W_0\|\|_F` for ckpt-0/50/100 | CPU, minutes | M3 Max |
| **R** | Q(Qwen2.5-7B) vs Q(Qwen2.5-7B-Instruct) | 2 probe passes | M3 Max, else A100 |
| **N** | Q(Instruct + noise) at 6 doses | 6 probe passes | M3 Max, else A100 |

### The three machines

| machine | role | why |
|---|---|---|
| **M3 Max, 40 GPU cores, 128 GB** | all local compute | 7B bf16 is 15.2 GB; it fits with 8× headroom |
| **Mac mini M4, 16 GB** | drives Colab; audits artifacts | 7B does not fit; the audit and report drivers are CPU-only and take seconds |
| **Colab A100** | fallback for E4-large only | ~94 of ~300 units left; spend only what the M3 Max cannot absorb |

Routing between the M3 Max and Colab for E4-large is decided by
`08_e4_ruler.py --benchmark`, which times real probe batches on the real model
and extrapolates the eight passes. Do not commit a machine to a multi-hour run
without it.

### Arm W — the Stage-A dose in weight space

Streams each adapter's `lora_A`/`lora_B` pair, forms `B A (α/r)`, takes its Frobenius
norm, and divides by the base weight's norm read one tensor at a time from the
model's safetensors shards. Nothing larger than a single weight matrix is ever
resident, so the 15.2 GB model is a disk requirement, not a memory one.

**Gate W1.** ckpt-0's adapter has `B = 0` by LoRA initialization, so its dose must be
**exactly 0.0**. Anything else means the wrong checkpoint was read, and the audit
fails on it.

If the adapters are unavailable, Arm W records `relative_dose: not_run` with a
reason, per E1's "never silently substitute" discipline. The `||dW||_F` half needs no
base model and is emitted regardless, so the ratio can be completed later without
re-reading the adapters.

### Arm R — the ruler

`Qwen/Qwen2.5-7B` versus `Qwen/Qwen2.5-7B-Instruct`, both measured on the frozen
probe with no adapter attached.

**The probe text is frozen once and reused verbatim.** The two models do not share a
chat template. Rendering "through each model's own template" would feed them
different text and confound the weight difference with a prompt difference, so
`06_e4_freeze_probe.py` renders the 4096 probe prompts once through the **Instruct**
template — exactly as E1 did — writes them to disk with a SHA-256, and every arm
reads that file.

**Gate R1 (hard).** Both tokenizers must map the frozen text to identical token ids
and expose identical vocabularies. If not, the arms differ in their input as well as
their weights and the comparison is uncontrolled: stop, do not record.

**Gate R2 (recorded, not enforced).** The Instruct arm's eranks are compared with
E1's published ckpt-0 values (`1127.4155 / 1281.0450 / 1426.0597`). On the same
A100 this should reproduce closely, because ckpt-0's adapter is the identity. The
delta is **reported, never gated on**: E1's `1e-4` tolerance is a same-hardware,
same-kernel statement and no other accelerator will meet it. Two things make this
worth recording anyway — a large delta means the wrong model or probe rather than a
precision effect, and the size of the drift is itself a portability statement about
erank that nobody has measured.

### Arm N — the calibration ladder

For each dose `d ∈ {1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1}`: reload Instruct, and for
every targeted weight set

```
   W  ←  W + d · (||W||_F / ||G||_F) · G,      G ~ N(0, I), seeded per (dose, module)
```

The rescaling makes `||ΔW||_F / ||W||_F` equal `d` **by construction**, which is what
puts Arm W's number on the same axis.

The model is reloaded between rungs rather than un-perturbed, so no rounding residue
accumulates down the ladder.

**Gate N1.** The achieved dose is measured back off the parameters *after* the dtype
cast, not assumed from the request. A bf16 parameter cannot represent a 1e-6 relative
nudge, and a rung that missed its target must say so in its own record. The audit
additionally refuses two rungs that achieved the same dose.

---

## 4. Two scales, and why the small one is not a rehearsal

| | E4-small | E4-large |
|---|---|---|
| models | Qwen2.5-0.5B / -0.5B-Instruct | Qwen2.5-7B / -7B-Instruct |
| layers | 4, 12, 22 (depth-matched to 5/14/26 of 28) | 5, 14, 26 |
| hidden | 896 | 3584 |
| dtype | float32 (MPS/CPU) | bfloat16 (E1's contract) |
| cost | 0 units, <1 h on the M3 Max | 0 units on the M3 Max if the benchmark allows, else ~7–10 |

E4-small stays worth running even when E4-large is free: it is a **second data
point at a 14× smaller scale**, not a dry run. Whether
the detector's sensitivity threshold is scale-invariant is a real question, and the
answer connects directly to the 0.5B track, where the pilot's MLP spectrum *did*
move when the 7B's did not (`FINDING_E1_METRIC_REMEASUREMENT.md` §4). It also
de-risks E4-large: the identical code path runs green before any units are spent.

Because the two scales use different dtypes and different models, **erank levels are
never compared across scales** — only each scale's arms against that scale's own
`R_instruct` reference.

---

## 5. Outputs

Per arm, one JSON carrying the full `measurement_contract`, `meta` (including the
gated-MLP hook-check error), `spectra` and `dormancy` blocks in exactly E1's schema.
Plus:

- `probe_manifest.json` — probe id hash, rendered-text SHA-256, truncation flag
- `ruler_table.json` — per-layer signed relative erank change of every arm against
  `R_instruct`, and each arm's max absolute change
- `arm_W_weight_dose.json` — per-module and aggregate dose for the three adapters
- `audit_e4.json` — the strict audit's own recomputation

The headline is a single table: max |Δerank| for each arm, against E1's **+0.7303%**
(down_in L14) / **+0.7227%** (resid L16).

---

## 6. Acceptance gates

| # | Gate | On failure |
|---|---|---|
| P1 | probe id set hashes to `1e61252e7b54793e`, n=4096 | stop; this is not E1's probe |
| R1 | both Arm R tokenizers give identical ids on the frozen text | stop; comparison uncontrolled |
| R2 | Instruct erank vs E1's published ckpt-0 | **record only**, never gate |
| N1 | achieved dose measured post-cast; no two rungs equal | audit fails |
| W1 | ckpt-0 relative dose is exactly 0.0 | audit fails; wrong checkpoint |
| A1 | all arms share probe size, layers and dtype | audit fails; levels incomparable |
| A2 | gated-MLP hook identity verified on every arm | audit fails; wrong tensors hooked |
| A3 | `ruler_table.json` recomputes from the raw records | audit fails |

Gate A2 is E1's `_verify_gated_mlp` check, inherited unchanged: it asserts
`act_fn(gate)·up` reproduces the `down_proj` input, so a transformers version that
reorders Qwen2MLP cannot silently hand us a different tensor.

---

## 7. Execution order

```
1  06_e4_freeze_probe.py         M3 Max,   minutes  → probe_frozen.json + manifest
2  07_e4_weight_dose.py          M3 Max,   minutes  → Arm W (full, --download)
3  08_e4_ruler.py --scale small  M3 Max,   <1 h     → E4-small Arms R + N
4  09_audit + 10_report          any Mac,  seconds  → audit + table + figure
5  08_e4_ruler.py --benchmark 4  M3 Max,   minutes  → ROUTING DECISION for step 6
6  08_e4_ruler.py --scale large  M3 Max or A100     → E4-large Arms R + N
7  09_audit + 10_report          any Mac,  seconds  → audit + table + figure
```

Steps 1–5 need no compute units and no team decision. Step 6 is the only one
that can spend budget, and only if step 5 says the M3 Max cannot carry it.

**Compute accounting.** `compute_log.md` records that GPU-dashboard screenshots were
taken for **no** exp2 session — a standing violation of a hard constraint
(`PROJECT_OVERVIEW_AND_NEXT_EXPERIMENTS.md` §5.4.2). Step 5 takes them: balance
before, balance after, resources panel, and a disconnect confirmation.

---

## 8. Open questions this does NOT decide unilaterally

- Whether E4 belongs in the paper as its own Experiments subsection or as a
  calibration paragraph inside Experiments 3 (E1). *Recommendation:* fold into
  Experiments 3 — it is the same detector under the same contract, and a separate
  section would over-sell a control.
- Whether to add a **structured** low-rank perturbation arm (random rank-16 ΔW at
  matched dose) alongside the isotropic one. That would test whether *rank* rather
  than *magnitude* is what Q responds to, which is the sharper mechanistic question.
  Deferred: it doubles Arm N's cost and Arm N must be shown to move Q at all first.
- Whether E4-small's 0.5B arms should reuse Jason's Stage-A checkpoints for a
  matching Arm W at that scale. Needs his track's artifacts; flagged, not assumed.
