# Team decisions needed before / during the pilot

Do not silently resolve these in code. The committed default remains the cheaper
briefing recipe: base model, GRPO adaptation, β=0, SVAMP.

## Ready-to-send Slack note

> Pilot implementation is Phase-0 ready: frozen GSM8K/SVAMP splits, exact-answer
> reward, Q metrics, dashboard logging, fixed-budget adaptation, and run-scoped
> analysis artifacts pass local tests. Before the 200-update spend, can we confirm:
> (1) Qwen2.5-0.5B base vs Instruct, (2) GRPO vs SFT for the 50-update SVAMP probe,
> (3) whether SVAMP is acceptable despite being close to GSM8K, and (4) whether
> to add one β>0 baseline? Notebook 01 now saves a no-update sparse-reward
> preflight and stops if the base model produces no within-group exact-reward
> variance. If it stops, should we switch to Instruct, or retain base and approve
> an additional format reward? I will log either change as a deviation.

## Interpretation guardrail

The pilot asks whether Stage-A RLVR checkpoint state predicts **fixed-budget
future adaptability** on the pre-declared SVAMP task, relative to checkpoint 0.
It does not establish that RLVR reduces a model's general ability to learn.
