# Experiment 2 — Instruct v10 amendment (DRAFT — NOT AUTHORIZED)

**Status:** **DRAFT prepared 2026-08-16 by Person 4 (early-warning diagnostics)
for team review. Nothing here is registered.** No `exp2_config_4070_instruct_v10.json`
has been created, no runner exists, and no run of any kind is authorized by this
file. It exists so that if the team approves the direction, the registration is
already written and the 2026-08-23 abstract deadline is not spent drafting.

**Requires sign-off from:** Tommy Xie (team lead), because it changes a
pre-registered Stage-A training variable.

**Predecessor:** `exp2_config_4070_instruct_v9.json` /
[`EXPERIMENT_2_4070_INSTRUCT_V9_AMENDMENT.md`](EXPERIMENT_2_4070_INSTRUCT_V9_AMENDMENT.md)
**Evidence:** [`FINDING_V9_PHASE0_COLAB_A100.md`](FINDING_V9_PHASE0_COLAB_A100.md)

---

## Why this would be a new run

V9 raised the Stage-A GRPO group from 3 to 8 to defeat the v8 reward-variance
collapse. On 2026-08-16 that hypothesis was confirmed: the v9 group-8 preflight
passed with 7 of 16 prompt groups showing combined within-group variance,
reproducing the RTX 4070 reference (7/16) exactly on a Colab A100. **The group
question is settled; v10 does not revisit it.**

V9 Phase 0 nonetheless failed, on a different and previously unmeasured axis.
Both Stage-A smoke updates trained cleanly with healthy gradients
(`grad_norm` 0.714 and 1.042, finite loss, no OOM), and then Stage A stopped on
its own pre-registered gate:

```json
{ "reason": "stage_a_smoke_completion_clipping_exceeded_limit",
  "limit": 0.1, "clip_ratios": [0.09375, 0.140625] }
```

At `max_completion_length = 1280`, with observed mean completion lengths of
753.2 and 718.6 tokens, **14.06% of completions on update 2 were truncated**
against a 10% gate. A truncated Math completion cannot emit a well-formed
`\boxed{}` answer, so it scores as incorrect regardless of the reasoning that
preceded it. The clipping contaminates the very reward signal v9 exists to
repair.

V10 is therefore an independent hypothesis test of a second, separate claim:
that the Stage-A completion budget, not the group size, is what now bounds
usable reward signal on this Math population.

---

## The single proposed change

Relative to v9, Stage A would change exactly one field:

- `max_completion_length`: `1280` → **to be set by measurement, not by guess**
  (see the sizing rule below).

Everything else remains v9: model and revision, dataset and revision, frozen
populations and splits, seed 42, `exact_plus_boxed_format_0.1` reward,
`num_generations` 8, `per_device_train_batch_size` 8,
`gradient_accumulation_steps` 8, learning rate, optimizer, dtypes,
`max_prompt_length` 512, beta, temperature, top-p, the 200-update budget, the
checkpoint grid, the measurement probe, and every runtime safety threshold.
Stage B remains the v8 exact-reward CodeIO contract, untouched.

### Sizing rule — set the cap from the distribution, not from a round number

We currently know only the *mean* completion length (≈735 tokens) and the clip
rate at one cap (14.06% at 1280). That is not enough to choose a new cap
responsibly. Before registering a number, run a **generation-only measurement**
(no optimizer step, therefore cheap and not a training run): sample the frozen
Stage-A preflight prompts at the registered temperature/top-p with the
completion cap raised well above any candidate, and record the empirical
completion-length distribution.

Register the cap as **the smallest value in `{1536, 1792, 2048, 2560}` whose
measured truncation rate is ≤ 5%** — half the gate, so that ordinary
update-to-update variation does not immediately re-breach it. If no candidate
reaches ≤5%, escalate rather than picking the largest.

### Cost of each candidate, for the sign-off decision

The training logits tensor is the dominant single allocation and scales linearly
with the cap: `per_device(8) x (cap+1) x vocab(151936) x 4 bytes (fp32)`.

| cap | logits tensor | vs v9 | Stage A 200 updates (upper bound) |
|---|---|---|---|
| 1280 (v9) | 5.80 GiB | — | 3.30 h |
| 1536 | 6.96 GiB | +1.16 GiB | ≤ 3.97 h |
| 1792 | 8.12 GiB | +2.32 GiB | ≤ 4.63 h |
| 2048 | 9.28 GiB | +3.48 GiB | ≤ 5.29 h |
| 2560 | 11.60 GiB | +5.80 GiB | ≤ 6.61 h |

Wall-time figures scale the measured 59.49 s/update linearly with the cap and
are therefore **upper bounds** — generation halts at EOS, so real scaling is
sublinear. Every candidate still fits inside one Colab session.

**Hardware consequence:** v9 already needs ≈27 GiB and OOM'd on a 22 GiB L4.
Every v10 candidate needs strictly more. **No v10 configuration will fit an L4
or the 8 GiB RTX 4070.** v10 is an A100-class run by construction, which makes
the compute-ownership question in the open-questions list a blocking one rather
than a bookkeeping one.

---

## Runner change required

`validate_contract` in `run_exp2_4070_v9.py` permits only
`per_device_train_batch_size` and `num_generations` to differ from v8 and
hard-requires both to equal 8. A v10 runner must widen that allow-list to
include `max_completion_length` **and nothing else**, and must additionally
assert that `num_generations == 8`, `per_device_train_batch_size == 8`,
`gradient_accumulation_steps == 8`, and that eight unique prompts per update are
preserved — i.e. v10 inherits every v9 lock except the one field it is testing.
The v8-vs-v10 diff must remain mechanically checkable.

---

## Phase-0 gates

Unchanged from v9 in every respect, including the 10% per-update Stage-A
clipping limit, the five-update zero-variance stop, and the five-update
>10%-clipping streak stop. **None of these are relaxed.** The point of v10 is to
pass the existing clipping gate honestly by giving completions room to finish,
not to move the gate.

Additionally, v10 Phase 0 must record `torch.cuda.max_memory_allocated()` from
inside the training process. The v9 probe run could not report a VRAM peak
because its `nvidia-smi` sampler was absent, and 5-second polling would have
been too coarse to catch a backward-pass spike in any case.

---

## What would falsify this

If a raised cap brings truncation under the gate but Stage-A reward variance or
exact-correctness does **not** improve relative to v9's preflight, then
completion truncation was not the binding constraint, and the sparse-reward
problem on this Math population is deeper than either group size or completion
budget. That outcome is reportable and must not be worked around by moving a
third variable.

---

## Claim boundary

V10 would test whether the Stage-A completion budget bounds usable reward signal
on this Math population. It is not a continuation of v8 or v9, does not fill in
any v8 or v9 checkpoint, and is not an equivalent execution of the original
Base/full-CodeIO Experiment 2 proposal.
