# Finding — Stage B stopped at update 26 on the clipping gate; the 640 cap was never measured

**Date:** 2026-08-18
**Cell:** `stage_b/ckpt0_seed42` (the stage-2-alone baseline arm)
**Stop:** `LocalSafetyCallback` — *five consecutive updates exceeded 10%
completion clipping* — at step 26 of a 30-update budget.
`validate_stage_b_completion` wrote `incomplete.json` and the whole Stage-B
subprocess exited, so **ckpt-100 and ckpt-50 never started.**

```json
{"step": 26, "reason": "five consecutive updates exceeded 10% completion clipping",
 "completions/clipped_ratio": 0.1406,
 "completions/max_terminated_length": 640.0,
 "completions/mean_length": 342.8,
 "frac_reward_zero_std": 0.75}
```

`max_terminated_length` equals the cap exactly — the terminated distribution is
pressed against the ceiling, the same signature Stage A's base run showed at
1274-against-1280.

---

## 1. My monitoring called this wrong, twice

At update 9 I raised the 32.8% clipping as a risk. At update 16 it had fallen to
4.7% and I recorded it as *"a known characteristic of this recipe — an early
length overshoot that decays — rather than raised as an alarm each time."*

**That framing was too confident and it is what let updates 22-26 go unwatched.**
The first two excursions did decay; the third one completed the five-update
streak and ended the run. "It decayed twice" is not evidence that it always
decays — with a patience-5 gate, only one excursion has to persist.

Corrected posture: an excursion above the gate is a *countdown*, not a
transient, until the streak actually resets. Watch the streak, not the level.

## 2. The real defect: 640 was chosen to match the prompt length, not measured

The config's own deviation record says how 640 was picked:

> "DEVIATION, 384 -> 640, registered 2026-08-16. Cause: the WIN4070 v8
> post-stop Stage-B cell at ckpt100/seed42 safety-stopped at update 23 with
> completions/clipped_ratio rising 0.1094 -> 0.1719 -> 0.2344 ... **640 matches
> stage_b max_prompt_length and the token audit**"

So:

- **This exact failure already happened once** on the WIN4070 track, at almost
  the same point in the budget (update 23 there, update 26 here).
- The remedy was 384 -> 640, and **640 was justified by matching the *prompt*
  cap** — a number about inputs — not by any measurement of how long CodeIO
  *completions* actually are.

That is precisely the error that killed Stage A at 1280, where the cap was also
never derived from a completion-length distribution. **The same mistake has now
been made twice in this config, on both stages, and has cost two runs.**

The config is right about the principle — `clipping_stop_unchanged`: *"The fix
is to give the model enough room to finish, not to stop noticing that it
cannot."* It just never applied a measurement to choose the room.

## 3. What is being done

A generation-only completion-length measurement on the **frozen Stage-B train
population**, from the ckpt-0 adapter (the arm that stopped), 32 prompts x 8
generations, registered temperature/top-p, cap raised to 2048. Trains nothing,
changes no registered variable, needs no sign-off — the same procedure that
produced the working Stage-A cap.

**One calibration lesson carried forward from Stage A:** the init-policy
measurement *under-predicted* in-training clipping by ~4.7x for Instruct
(predicted 2.34% at 1536, observed ~11%). A Stage-B cap chosen on the 5% rule
alone would likely repeat this. The sizing decision must apply that known
under-prediction factor rather than take the raw number at face value.

## 4. What was salvaged from the dead cell

The ckpt-0 arm did produce a partial adaptation curve before it stopped:

| update | accuracy | n_correct / 300 |
|---:|---:|---:|
| 0 | 0.1733 | 52 |
| 10 | 0.2200 | 66 |
| 20 | 0.2333 | 70 |

That is not a Delta-R result — Delta-R needs all three arms at a *complete*
30-update budget — but it does establish that the Simulation task is learnable
inside the budget, which is what makes the experiment worth re-running rather
than abandoning.
