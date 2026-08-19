# Result — Stage B Delta-R, 7B Instruct (LIVE, arms land one at a time)

**Status:** ckpt-0 and ckpt-100 complete — **both endpoint arms are in, so the
headline contrast is readable.** ckpt-50 running (midpoint). Updated as arms land.
**Config:** `exp2_colab_config_mvp_instruct_stageb_v2.json`, hash `bd99ddd2817f`
**Run dir:** `outputs/exp2_colab_guru_math7b_instruct_group8_e33527592dd9/stage_b_v2`
**Hardware:** Colab A100-SXM4-80GB. Artifacts mirrored to Drive per arm.

---

## 1. Results so far

| arm | acc_before | acc_after | **Delta-R** | hits before -> after | updates | status |
|---|---|---|---|---|---|---|
| ckpt-0 | 0.1733 | **0.2767** | **+0.1033** | 52 -> 83 /300 | 30/30 | complete |
| ckpt-100 | 0.1867 | **0.2767** | **+0.0900** | 56 -> 83 /300 | 30/30 | complete |
| ckpt-50 | — | — | — | — | running | — |

### The headline: both arms land on the SAME post-adaptation accuracy

`acc_after` is **0.2767 for both** — 83/300 each. The two arms started 4
questions apart and finished on the same number.

The Delta-R gap is therefore **0.1033 - 0.0900 = 0.0133**, which is exactly the
zero-shot T_t gap between the two checkpoints (`FINDING_TRANSFER_T_7B_INSTRUCT.md`:
ckpt-100 - ckpt-0 = +0.0133). In other words the whole difference in Delta-R is
inherited from the different starting points; the fixed-budget *destination* is
identical.

Against a per-accuracy SE of 2.58 pp at n=300, and the +-6 pp smallest
detectable difference that finding already registered, **1.33 pp is far inside
noise.**

**Stated within the framing constraint** (never "RLVR reduces the ability to
learn"): at a fixed budget of 30 GRPO updates on the held-out Simulation family,
**100 updates of Math RLVR produced no detectable change in fixed-budget
adaptability.** Both checkpoints adapt to the same place. The measurable effect
of Stage A here is a ~1.3 pp shift in the starting point, which is itself not
distinguishable from zero.

### What this means for RQ1, honestly

RQ1 asks whether activation-based Q measured during stage A predicts a later
stall. In this run **neither side of that correlation varies**: effective rank
moved <=0.55% and dormant fraction sat at 0.0
(`FINDING_Q_METRICS_7B_INSTRUCT.md`), and now fixed-budget adaptability is flat
too. So this run **cannot test RQ1** — there is no variance to correlate. It is
consistent with the hypothesis and equally consistent with its negation.

That is a real limitation, not a soft one, and it should be reported as such
rather than dressed up as a negative result about Q. What the run does deliver
is a working, gate-passing, fully instrumented Delta-R pipeline and a clean
null at this scale and budget — which is what the MVP was scoped to produce.

### One caution, and what ckpt-50 is now worth

Both arms landing on *exactly* 83/300 is a stronger coincidence than the
argument needs. At SE ~2.6 pp the chance of an exact tie is roughly 4-5% — low
but unremarkable. Still, an exact tie is also what a bug would look like if
`acc_after` were somehow being computed off a shared or stale model state.

The `acc_before` values argue against that: they differ (0.1733 vs 0.1867) and
each reproduces its own T_t entry exactly, so the per-arm model state is
demonstrably being read correctly at the start of each arm.

**ckpt-50 now discriminates.** If it also returns exactly 0.2767, that pattern
should be treated as a defect to investigate before the result is reported. If
it returns something near but not equal, the convergence is real and noisy.
Either way the third arm has become a check, not just a midpoint.

## 2. ckpt-0 — the first Stage B in exp2 to finish its fixed budget

52/300 -> 83/300, **+31 questions**. Two prior attempts died on the clipping
stop (4070 v8 at update 23; the 640-cap run at update 26). This one ran 30/30
with `clip mean 0.0229, max 0.1094`, and the >10%-for-5-updates streak never
exceeded 1.

**The effect is not noise.** SE of the difference of two proportions at
n=300 is ~3.4 pp, giving z ~ 3.0; the design is paired (same 300 questions
before and after), so the true test is tighter still. For scale, the zero-shot
T_t curve's spread across checkpoints was +1.3 to +1.7 pp against a stated
detectable floor of +-6 pp — i.e. flat. Delta-R here is six times that spread.

**This validates a precondition, not just a number.** If fixed-budget Stage B
produced no learning, RQ1 would be unreadable regardless of what Q did — there
would be nothing for Q to predict. There is now a measurable target.

## 3. acc_before reproduced the T_t curve exactly

`ckpt-0 acc_before = 0.1733`, identical to `FINDING_TRANSFER_T_7B_INSTRUCT.md`'s
ckpt-0 entry (0.1733, 52/300) — two independent measurements of the same
quantity, taken on different VMs, either side of a runtime recycle.

That is an end-to-end check on the whole restore path: frozen splits, adapter
restore, and eval all reproduce. It is also why the eval `batch_size` was left
at 8 (see `FINDING_STAGE_B_CAP_SIZING.md` §8 discussion) — raising it to 32
would have saved ~2.5 h and destroyed exactly this check.

## 4. The 512-token eval cap does not explain this result

`FINDING_STAGE_B_CAP_SIZING.md` §8 flagged the confound: if Stage B shortens
completions, part of Delta-R could be "learned to fit inside the 512-token eval
cap" rather than task learning.

ckpt-0's in-training completion lengths, all 30 updates:

```
[328,165,457,280,318,338,464,230,647,363,317,296,255,369,234,258,379,394,451,
 291,277,544,545,356,497,350,312,485,363,263]
```

Mean ~350, **no trend** — it fluctuates in both directions and stays well below
512 throughout. So there is no systematic length shift for the truncation
mechanism to act on, and the confound cannot account for +10.3 pp.

The confound still stands as a question about the *absolute* value of R (roughly
10-15% of eval completions exceed 512 at this domain's p90 of ~620) and is still
Tommy's call. It no longer threatens this contrast.

**Earlier misreading, corrected:** a mid-run note claimed completions were
shortening and reward was rising. Both were read off the min/max of a window
rather than the series. The full series above shows neither trend. Per-update
samples are 8 prompts x 8 generations = 64 completions, far too few to read a
trajectory from.

## 5. The 2048 cap held on both arms

| arm | clip mean | clip max | >10% streak (stop is 5) | wall |
|---|---|---|---|---|
| ckpt-0 | 0.0229 | 0.1094 | never above 1 | 2:29:45 |
| ckpt-100 | 0.0229 | 0.1562 | never above 1 | 2:27:02 |

Two independent starting policies, 60 updates total, and the streak never got
past 1. The registered 640 measured 9.38% truncation *before training started*.

**A mid-run claim that ckpt-100 was faster and clipping-free is withdrawn.**
Both were read from its first four updates: `clip 0.0000` became 0.0229 (the
same as ckpt-0) and `~150 s/update` became 294 s/update against ckpt-0's 300 —
i.e. the two arms ran at the same speed. Four updates is 32 prompts; nothing
about an arm's character can be read from that.

The offline finding it seemed to confirm still stands on its own evidence:
ckpt-100's completion tail *is* shorter (p99 1441 vs 2048; 0.52% vs 1.56%
truncation at 2048), measured on 192 samples in
`FINDING_STAGE_B_CAP_SIZING.md` §6. The in-training clipping rates do not
contradict it — they are just not the dramatic gap the early window suggested.
