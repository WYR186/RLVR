# Result — Q (effective rank, dormant fraction) is essentially FLAT across Stage A

**Date:** 2026-08-18
**Run:** `exp2_colab_guru_math7b_instruct_group8_e33527592dd9`
**Owner:** Person 4 (early-warning diagnostics) — this is the tertiary MVP
deliverable and this owner's own assignment.
**Probe:** n = **4096** frozen prompts, layers [5, 14, 26], batch 16.
**Measurement contract (identical at every checkpoint):**
`model_eval=True`, `dtype=bfloat16`, `hidden_pooling=last_non_padding_token`,
`dormant_pooling=mean_abs_over_all_non_padding_tokens`, `max_prompt_tokens=512`,
`activation_accumulator=float32`, `svd_dtype=float64`.

---

## 1. The measurement is exactly reproducible

```
ckpt-0 re-check, layer 5: erank 1127.415476 vs 1127.415476   delta = 0.00e+00
```

Two independent passes over the same checkpoint agree **bit for bit**. Every
difference reported below is therefore a real difference, not measurement noise
— which is what makes it meaningful that the differences are so small.

Also note **n_probe = 4096 > hidden dim 3584**, so effective rank is *not*
sample-truncated here. The config's stated requirement is met, not approximated.

## 2. Effective rank barely moves — and moves *upward*

| layer | ckpt-0 | ckpt-50 | ckpt-100 | Δ(50) | Δ(100) |
|---|---:|---:|---:|---:|---:|
| 5 | 1127.4155 | 1128.2271 | 1128.1812 | +0.072% | +0.068% |
| 14 | 1281.0450 | 1287.8799 | 1287.8093 | +0.534% | +0.528% |
| 26 | 1426.0597 | 1433.8730 | 1432.5480 | +0.548% | +0.455% |

Largest change anywhere: **0.55%**. The direction is *up*, not down. A
plasticity-collapse story predicts effective rank falling; nothing of the kind
happens here.

Note also that ckpt-50 and ckpt-100 are nearly indistinguishable (1287.88 vs
1287.81 at layer 14). Q separates "before any RL" from "after RL" by a hair, and
**does not separate 50 updates from 100 updates at all**.

## 3. The dormant-fraction metric has no dynamic range in this regime

`dormant_frac` is **exactly 0.0 at every layer, every checkpoint, and both
thresholds** (τ = 0.025 and τ = 0.1).

That is not a finding about the model — it is a finding about the metric as
calibrated. The *minimum* dormancy score observed is:

| layer | min dormancy score | loose threshold τ=0.1 |
|---|---:|---:|
| 5 | 0.1604 | 0.1 |
| 14 | 0.4148 | 0.1 |
| 26 | 0.1606 | 0.1 |

The least-active unit in the network sits **1.6× above the loose threshold**. No
unit could have been counted as dormant regardless of what training did, so this
metric would have reported 0.0 for any outcome. **As thresholded, dormant
fraction cannot function as an early-warning signal on this model.** If it is to
be used, τ has to be recalibrated against this model's actual dormancy-score
distribution — the `dormant_score_min` / `dormant_score_median` fields are
already recorded per layer for exactly that purpose.

## 4. Why Q is flat — the honest mechanistic reading

Stage A trained a **LoRA adapter, r=16**, with the 7B base weights frozen. The
adapter moved 1.92% in norm over 100 updates (`update_sentinel.jsonl`), and the
per-layer weight norms confirm the base is untouched — they agree to the sixth
decimal place between ckpt-0 and ckpt-100.

Q is an *activation* statistic, so it does respond to the adapter. But a ~2%
perturbation confined to a rank-16 subspace, on top of frozen base weights, is
simply not going to move the spectrum of a 3584-dimensional activation matrix
much. **The flatness is what this parameterisation predicts.**

This is the load-bearing caveat for the whole early-warning question: it is
evidence about *LoRA at this dose*, and **not** evidence that effective rank or
dormancy fail as plasticity diagnostics in general. Full-parameter RLVR, or a
much larger dose, is where the metric would get its chance.

## 5. Consequence for RQ1, stated before Stage B is spent

RQ1 asks whether Q(checkpoint) predicts fixed-budget adaptability. **The
predictor has almost no variance across the three checkpoints measured here**
(0.07–0.55%, with 50 and 100 effectively tied). Whatever ΔR turns out to be, a
Q-vs-adaptability relationship cannot be established from this run — there is
nothing to correlate against.

That is not an argument against running Stage B: ΔR is the *primary* deliverable
in its own right, and T_t being flat
([`FINDING_TRANSFER_T_7B_INSTRUCT.md`](FINDING_TRANSFER_T_7B_INSTRUCT.md)) makes
it clean to interpret. But the abstract must not promise a Q-vs-adaptability
result, and the config's own `honest_limits` already says a rank correlation
over three points is not evidence. This run confirms that warning empirically
rather than leaving it as a caution.

## 6. What would make Q informative next time

In priority order, cheapest first:

1. **Recalibrate τ** from the recorded dormancy-score distribution instead of
   inheriting 0.025/0.1 from the 0.5B plan. As it stands the metric is
   saturated and carries zero information.
2. **Checkpoint more densely early.** If Q moves at all it will move fastest in
   the first updates; [0, 50, 100] may be sampling a curve that has already
   flattened.
3. **Increase the dose or unfreeze the base.** LoRA r=16 bounds how much the
   representation *can* change, which bounds how much Q *can* report.
