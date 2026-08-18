# Experiment 2 — Colab 7B MVP, Instruct amendment

**Status:** **DEVIATION — implemented, logged, and flagged. Not authorized by
Tommy.** Registered here under `CLAUDE.md`'s implement-the-cheaper-default rule
because the alternative was no 7B deliverable at all before 2026-08-23. **If
Tommy rules against either change below, the run made under it is discarded.**
That is the accepted cost of the pattern, stated up front.
**Date:** 2026-08-18
**Config:** `exp2_colab_config_mvp_instruct.json`, hash **`e33527592dd9`**
(verified with `eaaj-pilot/src/repro.py:config_hash`)
**Predecessor:** `exp2_colab_config_mvp.json`, hash `fc243e587296`

---

## 1. Why a new config exists

The base-model run stopped at update **7 of 100** on the pre-registered
completion-clipping gate — five consecutive updates above 10% truncation
([`FINDING_7B_STAGE_A_CLIPPING_STOP.md`](FINDING_7B_STAGE_A_CLIPPING_STOP.md)).
The gate was correct and was not touched.

The generation-only measurement that the v10 draft requires before any cap may
be registered then showed that **raising the cap does not fix it**
([`FINDING_COMPLETION_LENGTH_MEASUREMENT_7B.md`](FINDING_COMPLETION_LENGTH_MEASUREMENT_7B.md)):
no candidate in `{1536, 1792, 2048, 2560}` reaches the rule's 5% target on the
base model, and 4.69% of generations never terminate at all within 3072 tokens.

## 2. The measurement that decided it

Identical protocol both times: frozen Stage-A population, 32 prompts x 8
generations = 256 completions, registered temperature 0.7 / top-p 1.0, cap 3072,
adapter at init, no optimizer step.

| | p50 | p90 | p95 | p99 | never terminated (@3072) |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-7B **base** | 532 | 1358 | 2805 | 3072 | **4.69%** (12/256) |
| Qwen2.5-7B **Instruct** | 766 | 1194 | 1338 | 1895 | **0.39%** (1/256) |

| candidate cap | base truncation | instruct truncation |
|---:|---:|---:|
| 1280 (registered) | 11.33% | 6.25% |
| **1536** | 8.98% | **2.34%** |
| 1792 | 8.59% | 1.56% |
| 2048 | 6.25% | 0.78% |
| 2560 | 5.08% | 0.39% |

Note the *shape*, which is the actual finding: Instruct's **median is longer**
(766 vs 532) — it writes more thorough answers — but its **tail is far shorter**
(p95 1338 vs 2805). The base model's problem was never that it writes too much
on average; it is that a subpopulation of generations never emits EOS. That is
the standard behaviour of a non-instruction-tuned LM, not a recipe defect.

**Calibration:** the base measurement predicted 11.33% truncation at 1280 and
the real training run measured 7.8–18.8% over 7 updates. The instrument agrees
with reality.

## 3. The two changes, and nothing else

| field | from | to | basis |
|---|---|---|---|
| `model_id` / `model_variant` | `Qwen/Qwen2.5-7B` / base | `Qwen/Qwen2.5-7B-Instruct` / instruct | §2 — no base-model cap satisfies the sizing rule |
| `stage_a.max_completion_length` | 1280 | 1536 | the v10 draft's own rule: smallest of {1536,1792,2048,2560} with truncation <= 5%; Instruct gives 2.34% at 1536 |

**Unchanged:** dataset and revision, seed 42, `exact_plus_boxed_format_0.1`
reward, `num_generations` 8, `per_device_train_batch_size` 8,
`gradient_accumulation_steps` 8, learning rate 2e-5, LoRA geometry, optimizer,
dtypes, `max_prompt_length` 512, beta, temperature, top-p, the 100-update MVP
budget, checkpoints [0, 50, 100], the probe, **every gate threshold** (including
the 10% clipping limit and the five-update patience), and all of Stage B
including its `max_completion_length` of 640 — Stage B was not measured, and
changing what has not been measured is the exact error this episode came from.

## 4. Honest weaknesses of this amendment

1. **The contingency fired on an adjacent condition, not its literal trigger.**
   The registered trigger was "persistent boxed-format failures on the base
   model". What was observed was a non-terminating completion tail. These are
   the same failure from two sides — a completion cut off at the cap cannot emit
   a well-formed `\boxed{}` answer — but a reviewer may reasonably hold that the
   registered trigger did not literally fire. **This is the weakest point of
   this amendment and it is not being hidden.**
2. **Two registered variables change at once**, so if Stage A now succeeds we
   cannot attribute the improvement to variant or to cap separately. The
   measurement in §2 is the evidence that separates them (no base cap qualifies;
   Instruct qualifies at the smallest candidate), but that is measurement, not a
   controlled training comparison.
3. **The cap was sized on the policy at init**, not on a trained policy — the
   step-7 policy was never checkpointed. In-training drift on the base run was
   +15% mean length over 7 updates. The 2.34%-against-a-10%-gate margin is what
   absorbs this.
4. **`base` is what the proposal's Experiment 2 framing assumes.** Switching to
   Instruct moves this arm closer to the WIN4070 track (which went Instruct at
   v1) and away from the original Base plan. That is a real scope statement, not
   a detail.

### Config-transfer integrity

The Colab runtime derives this config by applying the mutation above to
`exp2_colab_config_mvp.json` and then **hard-asserting** the resulting hash
equals `e33527592dd9`, the value computed on the Mac with
`eaaj-pilot/src/repro.py:config_hash`. `config_hash` sorts keys, so key order
cannot cause a spurious mismatch, and any drift between the two copies stops the
cell instead of silently training a different recipe into a differently-named
run directory.

## 5. Splits must be re-frozen — this is not a drop-in

`guru_data` computes `stable_id = sha256(raw_id + chat-template-rendered prompt
+ ground_truth)`, and the rendered prompt comes from
`tokenizer.apply_chat_template`. Base and Instruct templates differ, so **every
row id changes.** `build_exp2_splits` is idempotent and refuses to overwrite a
differing file, so this config freezes to **`exp2_colab_splits_instruct.json`**.

The population *definition* is unchanged — same filters, same seed 42, same
sizes. Only the identifiers change. Cross-variant comparisons must be made at
the population level, never by id.

## 6. What would falsify this

If Stage A now runs to 100 updates under the clipping gate but reward and
exact-correctness are no better than the base run's first 7 updates
(reward ~0.10, reward_std ~0.22), then truncation was not what bounded usable
signal, and switching variant bought nothing but a completed run. That outcome
is reportable and must not be answered by moving a third variable.

## 7. What Tommy is being asked to confirm, after the fact

1. Does firing `model_variant_contingency` on a non-terminating tail rather
   than on literal boxed-format collapse count as the contingency firing?
2. Is 1536 acceptable, given it is derived from the v10 draft's own rule but
   that rule was written for the 0.5B chain and never formally adopted for the
   Colab config?
3. Base-vs-Instruct for the 7B arm is on the open-questions list. This
   amendment answers it *provisionally, under measurement*. Confirm or reverse.
