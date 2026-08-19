# Result — Stage B Delta-R, 7B Instruct (LIVE, arms land one at a time)

**Status:** ckpt-0 complete; ckpt-100 running; ckpt-50 scheduled for a second
session. This file is updated as each arm lands.
**Config:** `exp2_colab_config_mvp_instruct_stageb_v2.json`, hash `bd99ddd2817f`
**Run dir:** `outputs/exp2_colab_guru_math7b_instruct_group8_e33527592dd9/stage_b_v2`
**Hardware:** Colab A100-SXM4-80GB. Artifacts mirrored to Drive per arm.

---

## 1. Results so far

| arm | acc_before | acc_after | **Delta-R** | updates | status |
|---|---|---|---|---|---|
| ckpt-0 | 0.1733 | 0.2767 | **+0.1033** | 30/30 | complete |
| ckpt-100 | — | — | — | running | — |
| ckpt-50 | — | — | — | queued | — |

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

## 5. What ckpt-100 decides

ckpt-0's +0.1033 is the baseline arm — adaptation with no prior Math RL. The
research question is whether `Delta-R(ckpt-100)` differs from it, i.e. whether
100 updates of Math GRPO changed fixed-budget adaptability on a held-out task
family. Per the framing constraint this is a statement about fixed-budget
adaptability, never about "ability to learn".

Early sign, not a result: ckpt-100 is running at ~150 s/update against ckpt-0's
~300, with `clip 0.0000` through its first updates. That matches the
pre-measured distribution (0.52% truncation at 2048 vs ckpt-0's 1.56%) and the
shorter-tail finding in `FINDING_STAGE_B_CAP_SIZING.md` §6.
