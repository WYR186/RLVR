# Stage B (Phase 3) — running notes

**Run:** `exp2_colab_guru_math7b_instruct_group8_e33527592dd9/stage_b`
**Order:** ckpt-0 -> ckpt-100 -> ckpt-50 (logged deviation, see
`FINDING_Q_METRICS_7B_INSTRUCT.md` §7)

---

## 1. Cross-check passed: the Delta-R baseline agrees with T_t exactly

`ckpt0_seed42/baseline.json` gives `acc_before = 0.1733`, and
`analysis/transfer_T.json` independently gives `scores_by_checkpoint["0"] =
0.17333333`. Two different code paths — `run_transfer_T` and
`run_stage_b_adaptation`'s own pre-adaptation eval — produce the **same number
on the same 300 frozen questions**.

This is worth stating because it is the kind of check that usually is not done:
it confirms the eval harness, the frozen eval split, the adapter-loading path
and the verifier all behave identically across the two entry points. Delta-R is
anchored on a number that has been reproduced independently.

## 2. Watch item — the reward is much sparser here than Phase 0 predicted

Update 4 of ckpt-0:

```
reward 0.2344   reward_std 0.427   frac_reward_zero_std 0.75
completions/mean_length 265.9   clipped_ratio 0.078   max_terminated 604 (cap 640)
```

**`frac_reward_zero_std = 0.75`** — three quarters of prompt groups have zero
within-group reward variance and therefore contribute no gradient. Phase 0's
GATE 0b measured 5 of 8 Stage-B groups as variable (i.e. ~0.375 zero-variance)
on this same population, so the live run is **roughly twice as sparse as the
preflight suggested**.

If that ratio holds, a 30-update budget delivers on the order of 7-8 updates'
worth of usable signal. That does not invalidate the fixed-budget design — the
budget is defined in updates and is identical across all three checkpoints, so
Delta-R remains a fair comparison — but it does mean **Delta-R is being measured
in a regime where each checkpoint gets very little effective learning**, which
shrinks any difference between checkpoints toward zero.

Combined with the n=300 eval resolution of ~6 pp
([`FINDING_TRANSFER_T_7B_INSTRUCT.md`](FINDING_TRANSFER_T_7B_INSTRUCT.md) §5),
the honest expectation to set *now*, before the result arrives, is that
**Delta-R will most likely come out flat within noise**. Recording that
prediction in advance is the point — it stops a null result from being
reinterpreted after the fact, and it stops a small positive one from being
over-read.

**Nothing is being changed in response.** The reward mode, budget, eval set and
gates are all as registered. This is a measurement, not a malfunction.

### CORRECTION (update 16) — the sparsity claim above was wrong

`frac_reward_zero_std` went 0.75 (update 4) -> 0.50 (update 9) -> **0.375**
(update 16). GATE 0b measured 5 of 8 Stage-B groups variable, i.e. exactly
**0.375**. The live run converged to the preflight value; the opening updates
were the anomaly, not the preflight.

So "roughly twice as sparse as the preflight suggested" was an artifact of
reading the first few updates as steady state. The Phase-0 gate was accurate.
Leaving the original claim above visible, corrected here rather than deleted.

### CORRECTION — the clipping alarm was also a transient

`clipped_ratio` went 0.078 -> **0.328** (update 9) -> **0.047** (update 16),
with `mean_length` 266 -> 429 -> 277. No five-update streak formed and the
safety callback never fired.

This is the **third** time in this run that a rising-completion-length excursion
resolved itself (Stage A did it twice, around updates 6 and 29). It should now
be treated as a known characteristic of this recipe — an early length overshoot
that decays — rather than raised as an alarm each time.

## 3. Timing, measured

| stage | rate |
|---|---|
| baseline eval, 300 questions | ~25 min, GPU ~40% (latency-bound) |
| Stage-B GRPO update, cap 640 | **~89 s/update**, GPU ~96% |
| 30 updates | ~45 min |
| per checkpoint (1 baseline + 30 updates + 3 in-training evals) | ~2.5 h |

On projection for ~7.5 h total.


---

## 4. First adaptation point — the ckpt-0 arm is learning

`stageb_eval_curve.jsonl`, ckpt-0 seed 42:

| update | accuracy | n_correct / 300 |
|---:|---:|---:|
| 0 (baseline) | 0.1733 | 52 |
| 10 | **0.2200** | **66** |

**+4.67 pp, +14 questions.** Conservative (unpaired) SE on the difference is
3.24 pp, so this is ~1.44 SE — not significant at 95%, but a clear positive
movement rather than noise around zero. The comparison is *paired* on the same
300 frozen questions, so McNemar is the correct test and the true standard error
is smaller than the bound quoted here.

This is R_B for the **stage-2-alone baseline arm**. Delta-R for ckpt-50 and
ckpt-100 will be measured against this curve.

It also settles a question the flat T_t left open: **the Simulation task is
learnable inside a 30-update budget.** If it were not, Delta-R would be a
comparison of three flat lines and the experiment would answer nothing. The
budget is doing real work.

Measured eval cost: `eval_seconds = 1540.9` (25.7 min) per 300-question eval,
confirming the 7.5 h projection.
