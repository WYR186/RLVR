# Finding — the Stage-B completion cap was never measured; measured now, 640 -> 2048

**Date:** 2026-08-18
**Status:** DEVIATION implemented, logged, and flagged to Tommy — not authorized by him.
Config fork `exp2_colab_config_mvp_instruct_stageb_v2.json`, hash **`bd99ddd2817f`**,
parent `exp2_colab_config_mvp_instruct.json` (`e33527592dd9`).
**Measured on:** Colab A100-SXM4-80GB, Qwen2.5-7B-Instruct + the completed Stage-A
LoRA adapters, 24 frozen `stage_b_train` prompts x 8 generations per arm (n=192),
probe cap 2048.
**Supersedes:** the `max_completion_length_deviation` note in
`exp2_colab_config_mvp.json` (384 -> 640, registered 2026-08-16).

---

## 1. What was wrong with 640

The registered Stage-B cap of 640 carried this justification:

> "640 matches stage_b max_prompt_length and the token audit"

`max_prompt_length` is a **prompt-side** limit. It says nothing about how long the
model's *completions* are. Using it as the completion cap is a category error, and
it was never checked against a measured completion-length distribution — the same
mistake `FINDING_GATE_0A_MEASURES_THE_WRONG_POPULATION.md` §5 documents for
`phase0a_note`, where a config's own filter value was read back as if it were an
audit result. This is now the second registered value in this project whose stated
evidence was a different number that happened to be nearby.

## 2. The measurement

Generation only — no optimizer step, no registered variable changed by the probe
itself. Population re-derived from the **frozen** `stage_b_train_ids` (sha
`df8623cf009e0690`, n=1132), so this is the same pool Stage B trains on.

Measured on the two **actual Stage-B starting policies**, not at init. The Stage-A
cap was sized at init and under-predicted in-training clipping (§4), so this probe
deliberately loads the adapters the arms actually start from.

| | ckpt-0 | ckpt-100 |
|---|---|---|
| mean | 328.8 | 318.8 |
| p50 | 271 | 285 |
| p90 | 634 | 617 |
| p95 | 944 | 836 |
| p99 | **2048** | 1441 |
| max | 2048 | 2048 |

Truncation by candidate cap (worst arm in bold):

| cap | ckpt-0 | ckpt-100 | worst |
|---|---|---|---|
| **640** (registered) | 9.38% | 8.33% | **9.38%** |
| 768 | 7.29% | 6.25% | 7.29% |
| 896 | 5.21% | 4.69% | 5.21% |
| 1024 | 4.69% | 2.60% | 4.69% |
| 1280 | 3.12% | 1.04% | 3.12% |
| 1536 | 2.60% | 0.52% | 2.60% |
| **2048** | 1.56% | 0.52% | **1.56%** |

## 3. This explains the 2026-08-18 Stage-B stop

The pre-registered stop is ">10% completion clipping for 5 consecutive updates."
At 640 the measured truncation is **9.38%** before training starts. The run began
already pressed against its own stop, so the death at update 26
(`FINDING_STAGE_B_CLIPPING_STOP.md`) was not a training pathology — it was the
recipe. Any run at 640 was going to end this way; the only variable was which
update it happened on.

## 4. The sizing rule, and what anchors it

**Rule:** the smallest candidate cap whose truncation on the **worst** arm is
<= 2.34%.

2.34% is not a taste threshold. It is the init-policy truncation of the Stage-A
recipe (Instruct @ 1536) that actually **completed 100/100 updates** under this
same clipping stop. So the rule reads: *no worse than the configuration we have
watched survive.*

The rule also carries a measured init-to-training correction. That Stage-A run's
observed in-training clipping (`dashboard.jsonl`, 100 updates) was:

```
mean completions/clipped_ratio  0.0652
max  completions/clipped_ratio  0.2500     (single update; never 5 in a row)
mean_length range               643 - 1052
```

i.e. **6.52% observed against 2.34% predicted — a 2.79x inflation.** Applying that
factor, cap 2048 projects to `1.56% x 2.79 = 4.4%` in training, against a 0.10
gate.

**Selected: 2048.** The clipping stop is NOT relaxed. The fix is to give the model
room to finish, not to stop noticing that it cannot.

## 5. The uncomfortable part — 1.56% is a floor, not a price

ckpt-0's **p99 equals the probe cap itself**. Those generations are not "long,"
they are **non-terminating**. The distribution is bimodal: a median of 271 tokens
sitting alongside a ~1.5% tail that never emits a stop token.

This is the same pathology that disqualified Qwen2.5-7B **base** on Math (4.69%
non-terminating at a 3072 cap — `FINDING_COMPLETION_LENGTH_MEASUREMENT_7B.md`),
and the reason the Instruct amendment fired. Instruct fixed it on Math (0.39%). It
is **back on Simulation/CodeIO**, milder but present.

Consequence: raising the cap further does not buy this down. 1.56% is the floor
of this model on this domain, and any future cap discussion should treat it as a
constant, not as something more headroom can fix.

## 6. A result, not just a nuisance

**ckpt-100's tail is shorter than ckpt-0's** — p99 1441 vs 2048, truncation at
1536 of 0.52% vs 2.60%.

100 updates of GRPO on **Math** made the model terminate *more* reliably on
**Simulation**, a domain it never trained on. That is the opposite direction from
"RLVR degrades transfer," and — stated carefully, per the framing constraint — it
is not a claim about ability to learn; it is a measured property of the output
distribution at a fixed decoding budget.

It matters because it is the only quantity in this run that moved. Effective rank
changed <= 0.55% across checkpoints and dormant fraction sat at 0.0
(`FINDING_Q_METRICS_7B_INSTRUCT.md`), so RQ1's predictor has essentially no
variance here. Completion-termination behaviour does vary with Stage-A training,
and it is cheap to measure. Worth raising with Tommy as a candidate signal.

## 7. Second registered change: eval points [0,10,20,30] -> [0,30]

De-scope on the eval-point axis the MVP fork already used
(`mvp_descope_note`: 6 -> 4).

**Delta-R is unaffected.** `run_stage_b_adaptation` measures `acc_before` and
`acc_after` outside the eval callback, so the endpoint contrast — the deliverable
— is untouched. What is lost is the *shape* of adaptation inside Stage B, i.e. the
intermediate curve points. This buys back the wall-clock the 640 -> 2048 cap costs.

## 8. Known limitation NOT changed — the eval cap is 512

`pipeline.guru_greedy_accuracy` uses `max_new_tokens=512`, and that is the
function that produces every Delta-R number. Against the distribution measured
above (p90 ~ 620), **roughly 10-15% of eval completions are truncated**, and a
truncated answer scores as wrong.

Left at the registered default deliberately: changing the eval definition changes the
absolute value of R. Delta-R remains a valid contrast because before/after and all
three arms share the identical cap.

**But there is a real confound to state.** If Stage-B training shortens
completions, part of the measured Delta-R is "learned to fit inside 512" rather
than "learned the task." This cannot be separated post-hoc from the current
artifacts. Flagged for Tommy: raising the eval cap to 1024 would remove it at
roughly 2x eval cost, and that is a decision about the outcome measure, not an
implementation detail.

## 9. What Tommy should confirm

1. Cap 640 -> 2048 (§2-§4). The rule and its anchor are stated above; the run
   proceeds under it, and if he rejects it the Stage-B arms are discarded — the
   accepted cost of the implement-log-flag pattern.
2. Eval points [0,10,20,30] -> [0,30] (§7).
3. The eval cap question (§8) — the one item here that is genuinely his call,
   because it changes what R means.
4. Whether the completion-termination signal (§6) is worth adding as a measured
   quantity alongside Q.
