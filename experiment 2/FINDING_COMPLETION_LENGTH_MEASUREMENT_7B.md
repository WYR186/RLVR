# Finding — raising `max_completion_length` does **not** fix the 7B clipping stop

**Date:** 2026-08-18
**Measurement:** generation-only, no optimizer step (the prerequisite
`EXPERIMENT_2_4070_INSTRUCT_V10_AMENDMENT_DRAFT.md` prescribes before any cap
may be registered). Qwen2.5-7B **base** + LoRA at init, frozen Stage-A
population, 32 prompts x 8 generations = 256 completions, registered
temperature 0.7 / top-p 1.0, cap raised to 3072.
**Artifact:** `/content/completion_length_measurement.json`

---

## 1. The distribution

```
n=256  mean=726.0  p50=532  p90=1358  p95=2805  p99=3072  max=3072
```

| candidate cap | measured truncation | v10 rule (<= 5%) |
|---:|---:|:--|
| 1280 (current) | 11.33% | no |
| 1536 | 8.98% | no |
| 1792 | 8.59% | no |
| 2048 | 6.25% | no |
| 2560 | 5.08% | **no — by one completion out of 256** |
| 3072 | 4.69% | yes, but 3072 is not a candidate |

**Calibration check:** the measurement predicts 11.33% truncation at 1280; the
real training run measured 7.8–18.8% (mean ~13%) over 7 updates. The measurement
is calibrated against reality and can be trusted.

## 2. The rule's own escape clause fires

The v10 sizing rule reads:

> "Register the cap as the smallest value in `{1536, 1792, 2048, 2560}` whose
> measured truncation rate is <= 5% ... **If no candidate reaches <=5%, escalate
> rather than picking the largest.**"

No candidate reaches 5%. **Escalate — do not register 2560.**

The honest caveat in the other direction: 2560 misses by 5.08% vs 5.00%, which
is *one completion out of 256*. With n=256 the standard error on a 5% estimate
is ~1.4 percentage points, so 2560 is statistically indistinguishable from the
threshold. If the only question were "does 2560 pass the rule", the answer would
be "too close to call". But that is not the only question — see §3.

## 3. Why a bigger cap is the wrong fix, not merely an insufficient one

Look at the shape, not the summary. `p50 = 532` but `p95 = 2805` and
`p99 = max = 3072`. The distribution is **bimodal**: half of all completions
finish inside 532 tokens, and a distinct tail runs to whatever cap it is given
and gets cut there.

Doubling the budget barely moves the tail:

| cap change | tokens bought | truncation bought |
|---|---:|---:|
| 1280 -> 2048 | +768 | -5.1 pp |
| 2048 -> 2560 | +512 | -1.2 pp |
| 2560 -> 3072 | +512 | **-0.4 pp** |

Returns are collapsing toward zero while cost grows linearly. Extrapolating the
last step, reaching 5% honestly would need a cap in the many thousands, and a
non-trivial fraction of generations appear **never to emit EOS at all** — 4.69%
were still running at 3072.

**This is the textbook pathology of a base (non-instruction-tuned) LM.** Base
models are trained on continuous text and have no strong prior for terminating;
Instruct models are trained to emit EOS at the end of a response. A ~5%
never-terminates rate on Qwen2.5-7B **base** is expected behaviour, not a
recipe bug.

## 4. Cost, which makes this decisive

At cap 1280 the real run measured **162.7 s/update**, i.e. ~4.5 h for 100
updates. Generation dominates that, and the straggler tail sets the wall time
for every batch it appears in — so the cost of a raised cap is driven by exactly
the completions that will still be truncated.

A 2560 cap plausibly lands Stage A at **7–8 h**, which no longer fits one Colab
session, against a 2026-08-23 deadline — and the measurement says it would sit
at ~5% truncation *on the base policy*, which drifted +15% longer over just 7
updates. It would likely re-breach the 10% gate part-way through and lose the
whole run again.

**Spending 7–8 h of A100 on a cap the measurement predicts will re-breach is not
a fix.** That is the reason this stops here rather than restarting Stage A.

## 5. What this points at instead — and why it is NOT being decided here

The non-terminating tail points at the **base-vs-Instruct** question, which
`CLAUDE.md` lists as an open team question that must not be silently decided,
and which both configs already carry a contingency for
(`model_variant_contingency` — currently written to fire only on a `\boxed{}`
format collapse, which did not happen).

A generation-only measurement on **Qwen2.5-7B-Instruct** has been launched
under the identical protocol. It decides nothing and trains nothing; it produces
the second number that makes the choice mechanical instead of a matter of
opinion. `guru_data._render_prompt` already routes prompts through
`tokenizer.apply_chat_template(..., add_generation_prompt=True)`, so the Instruct
variant is a clean drop-in through the same code path.

**One consequence to flag if the variant is ever switched:** `stable_id` is
`sha256(raw_id + rendered_text + ground_truth)`, and the rendered text depends on
the tokenizer's chat template. If the Instruct template differs from base, **every
row id changes and the frozen splits will not match**. Switching variant is
therefore a re-freeze, not a drop-in, at the splits layer. The measurement below
sidesteps this by re-deriving the population from the same filter rule rather
than from frozen ids.

---

## 6. Follow-up (2026-08-18, ~30 min into the Instruct run) — the init-policy measurement under-predicts in-training clipping

The Instruct run's first 9 updates at cap 1536:

| step | clip | mean_len | reward | reward_std |
|---:|---:|---:|---:|---:|
| 2 | 0.000 | 708.6 | 0.3031 | 0.4055 |
| 3 | 0.0625 | 953.2 | 0.3438 | 0.4407 |
| 4 | 0.1563 | 953.8 | 0.1344 | 0.2184 |
| 5 | 0.0781 | 918.8 | 0.2641 | 0.3848 |
| 6 | 0.2656 | 1037.7 | 0.1203 | 0.2234 |
| 7 | 0.000 | 731.7 | 0.4438 | 0.4787 |
| 8 | 0.000 | 852.2 | 0.2875 | 0.3934 |
| 9 | 0.1250 | 891.6 | 0.1656 | 0.2762 |

**The prediction was wrong in magnitude.** §1 measured 2.34% truncation at 1536
on the policy at init; the run is oscillating 0–26.6%, averaging ~11%. Compare
how the two models' predictions held up:

| | predicted (init policy) | observed (in training) | ratio |
|---|---:|---:|---:|
| base @1280 | 11.33% | ~13% | 1.15x |
| instruct @1536 | 2.34% | ~11% | **4.7x** |

So the instrument is well calibrated for the base model and badly calibrated for
Instruct. The mechanism is visible in the table: mean completion length is
already 953–1038 by step 4–6, against 793.8 at init. Instruct drifts longer,
much faster than base did (+15% over 7 updates).

**This is the caveat in §5 of `FINDING_7B_STAGE_A_CLIPPING_STOP.md` coming true,
larger than anticipated.** The 4x margin between 2.34% and the 10% gate was
supposed to absorb drift; roughly all of it has been spent within 9 updates.

### Why the run is nevertheless still alive, and not obviously wrong

The gate needs **five consecutive** updates above 10%. The observed sequence
resets constantly (0.0 at steps 2, 7, 8), so no streak builds. That is not luck
that can be relied on — if mean length keeps climbing, the resets stop.

More importantly, the reward tells a different story from the base run:

| | reward (mean over first updates) | reward_std |
|---|---:|---:|
| base, steps 1–7 | ~0.10 | ~0.22 |
| instruct, steps 2–9 | **~0.26** | **~0.35** |

Reward is roughly **2.6x** the base run's, with correspondingly larger spread.
Read together with the length growth, the most economical reading is that
Instruct is learning to write longer, more complete solutions that earn the
exact-match reward more often — i.e. the length drift is a *symptom of learning*,
not of degeneration. That is a claim this run can test, not one to assume.

### What is NOT being done about it

Nothing. The cap is not being raised again, the gate is not being touched, and
no third variable is moving. If the streak builds and the run stops, that is the
registered protection working and it is the reportable outcome — the alternative
is chasing a moving target across an unbounded number of untracked changes.
