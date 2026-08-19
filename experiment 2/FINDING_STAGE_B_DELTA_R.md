# Result — Stage B Delta-R, 7B Instruct (LIVE, arms land one at a time)

**Status:** COMPLETE — all three arms finished their fixed budget (`STAGE_B_V2_DONE`).
This is the MVP deliverable.
**Config:** `exp2_colab_config_mvp_instruct_stageb_v2.json`, hash `bd99ddd2817f`
**Run dir:** `outputs/exp2_colab_guru_math7b_instruct_group8_e33527592dd9/stage_b_v2`
**Hardware:** Colab A100-SXM4-80GB. Artifacts mirrored to Drive per arm.

---

## 1. Results so far

| arm | acc_before | acc_after | **Delta-R** | hits before -> after | updates | status |
|---|---|---|---|---|---|---|
| ckpt-0 | 0.1733 | **0.2767** | **+0.1033** | 52 -> 83 /300 | 30/30 | complete |
| ckpt-100 | 0.1867 | **0.2767** | **+0.0900** | 56 -> 83 /300 | 30/30 | complete |
| ckpt-50 | 0.1900 | **0.2867** | **+0.0967** | 57 -> 86 /300 | 30/30 | complete |

(rows ordered ckpt-0 / ckpt-100 / ckpt-50 in the table above; by Stage-A
training the sequence is 0 -> 50 -> 100.)

### The exact tie was a coincidence, not a bug — ckpt-50 settled it

The two endpoint arms both returned `acc_after = 0.2767` (83/300 each), which
§"One caution" below flagged as *also what a defect would look like*. **ckpt-50
returned 0.2867 (86/300)** — a different value. The identical pair was chance,
and the per-arm eval is reading per-arm model state correctly.

A third exact reproduction also landed: `ckpt-50 acc_before = 0.1900`, matching
`FINDING_TRANSFER_T_7B_INSTRUCT.md`'s ckpt-50 entry (0.1900, 57/300). All three
`acc_before` values now reproduce their T_t entries exactly.

### Delta-R is flat across the three checkpoints

Net questions gained, in Stage-A order:

```
ckpt-0    52 -> 83   net +31   Delta-R +0.1033
ckpt-50   57 -> 86   net +29   Delta-R +0.0967
ckpt-100  56 -> 83   net +27   Delta-R +0.0900
```

Full range **1.33 pp**. The paired SE of a single Delta-R is ~2.1-2.8 pp
depending on the discordant-pair count, so the SE of a *difference between two
arms* is ~3.0-3.9 pp. The observed 1.33 pp gives z ~ 0.4. Together with the
+-6 pp smallest detectable difference already registered for T_t at this n,
**the three arms are statistically indistinguishable.**

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

### The monotone ordering is NOT a trend — do not report it as one

The three Delta-R point estimates decline in Stage-A order, and by a suspiciously
even step: **+31, +29, +27 questions**, i.e. -2 questions per 50 Stage-A updates.
It is tempting. It should not be reported as a finding, for three independent
reasons:

1. **The whole range is inside noise.** 1.33 pp against a ~3.0-3.9 pp SE for an
   arm-to-arm difference. Nothing here is distinguishable from flat.
2. **Three points make a monotone ordering cheap.** With n=3 checkpoints, a
   monotone sequence (either direction) arises by chance **1 time in 3**.
3. **`acc_after` is not monotone** — 83, 86, 83. The monotone appearance in
   Delta-R comes from subtracting a non-monotone `acc_before` (52, 57, 56).

What it is worth: a **pre-specified hypothesis for a properly powered run**. If
the team wants to test "Delta-R declines with Stage-A training", this run says
what that would cost — detecting a 2-question-per-50-updates slope needs far
more than 300 eval questions and 3 checkpoints, and should carry multiple seeds.
That is a concrete, useful thing to hand Tommy, and it is the honest version of
this observation.

Per the framing constraint, none of this is a claim about the model's ability to
learn — every quantity here is fixed-budget adaptability on a named held-out
family at a named budget.

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

Mean ~350, **no trend within the run** — it fluctuates in both directions and
stays well below 512 throughout.

Now that all three arms are in, the same holds across arms, and the second half
of each run is only slightly longer than the first:

| arm | mean_len | first 15 updates | last 15 | clip mean | clip max |
|---|---|---|---|---|---|
| ckpt-0 | 361 | 338 | 385 | 0.0229 | 0.1094 |
| ckpt-50 | 354 | 297 | 410 | 0.0271 | 0.1094 |
| ckpt-100 | 353 | 302 | 404 | 0.0229 | 0.1562 |

Completions drift *upward* by ~20-35% over a run, not downward, and the means
stay ~30% below the 512 eval cap on every arm. The confound would need lengths
to fall toward the cap; they do the opposite. It cannot account for +10.3 pp.

The confound still stands as a question about the *absolute* value of R (roughly
10-15% of eval completions exceed 512 at this domain's p90 of ~620) and is still
Tommy's call. It no longer threatens this contrast.

**Earlier misreading, corrected:** a mid-run note claimed completions were
shortening and reward was rising. Both were read off the min/max of a window
rather than the series. The full series above shows neither trend. Per-update
samples are 8 prompts x 8 generations = 64 completions, far too few to read a
trajectory from.

## 5. The 2048 cap held on all three arms

| arm | clip mean | clip max | >10% streak (stop is 5) | wall |
|---|---|---|---|---|
| ckpt-0 | 0.0229 | 0.1094 | never above 1 | 3:19:36 |
| ckpt-50 | 0.0271 | 0.1094 | never above 1 | 3:23:04 |
| ckpt-100 | 0.0229 | 0.1562 | never above 1 | 3:17:33 |

Three independent starting policies, **90 updates**, and the streak never got
past 1. The registered 640 measured 9.38% truncation *before training started*
and killed the previous attempt at update 26.

**The adapters genuinely moved** — `update_sentinel.jsonl` reports
`updates_effective: true` on every arm, with relative parameter change 1.15% /
1.31% / 1.23%. The no-op failure mode that invalidated the 4070 v1 run (lr too
small, every update rounding to zero) is ruled out here by measurement, not by
assumption.

**A mid-run claim that ckpt-100 was faster and clipping-free is withdrawn.**
Both were read from its first four updates: `clip 0.0000` became 0.0229 (the
same as ckpt-0) and `~150 s/update` became 294 s/update against ckpt-0's 300 —
i.e. the two arms ran at the same speed. Four updates is 32 prompts; nothing
about an arm's character can be read from that. Final wall times differ by under
3%.

**A second mid-run correction, itself wrong, is reversed.** Seeing the last
training step report `665 s/it`, this file's author concluded the periodic eval
took 6-8 minutes rather than the ~25 minutes estimated from `run_transfer_T`.
`stageb_eval_curve.jsonl` records `eval_seconds` of **1529 / 1546 / 1526** — 25.5
minutes. The 665 figure was tqdm's rate display, not the step's duration, and
the original 25-minute estimate was right.

The offline finding it seemed to confirm still stands on its own evidence:
ckpt-100's completion tail *is* shorter (p99 1441 vs 2048; 0.52% vs 1.56%
truncation at 2048), measured on 192 samples in
`FINDING_STAGE_B_CAP_SIZING.md` §6. The in-training clipping rates do not
contradict it — they are just not the dramatic gap the early window suggested.
