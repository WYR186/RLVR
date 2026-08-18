# EAAJ Pilot — Evidence Pack

Raw artifacts from three executions of the same pre-registered pilot recipe.
No narrative report is included here — this is the underlying data so you
can check any number yourself. For the write-ups, see the `PILOT_*.md`
reports in the main repo.

## What was run (identical across all three)

GRPO on `Qwen/Qwen2.5-0.5B`, 512 GSM8K questions, exact-answer reward,
200 updates, checkpoints saved at 0/25/50/100/200. At each checkpoint:
measure effective rank + dormant-neuron fraction on a frozen probe set,
then run an identical fixed-budget SVAMP adaptation (256 train / 100 eval
questions, 50 GRPO updates) starting from that checkpoint.
Full recipe: `recipe/pilot_config.json`.

## Folder guide

```
recipe/                    the one shared config both runs used, + the compute/time ledger
mac_run/                    Experiment 1 — macOS (CPU, float32), seed 42
win_run/                    Experiment 2 — Windows (RTX 4070 GPU), seed 42
  adaptation_seed42/         win_run's own Stage-B adaptation (seed 42)
  adaptation_seed43/         Experiment 3 — same win_run checkpoints, only the
  adaptation_seed44/         Stage-B adaptation seed changed (43, then 44)
```

Status note: `adaptation_seed43` is complete (all 5 checkpoints).
`adaptation_seed44` has only `ckpt-0` so far — the remaining four
checkpoints have not been run yet, so those folders are intentionally
absent, not missing.

`mac_run` and `win_run` are **not the same training run** — they're two
independent executions of the identical recipe on different hardware
(mac_run: CPU/float32; win_run: CUDA GPU, fp32 master weights + bf16
autocast — see each folder's `config.json` → `execution` field for exact
settings). Do not merge numbers across them; compare them side by side.

`adaptation_seed43` / `adaptation_seed44` reuse the exact same five
`win_run` checkpoints — only the random seed used during the SVAMP
adaptation step changed. This isolates how much of the result is just
adaptation-noise vs. a real per-checkpoint effect.

## Inside each run folder

| File                                            | What it is                                                                              |
| ----------------------------------------------- | --------------------------------------------------------------------------------------- |
| `config.json`, `manifest.json`                  | exact hyperparameters and data hashes for this run                                      |
| `dashboard.jsonl`                               | per-step Stage-A training log (reward, loss, grad norm, etc.)                           |
| `gsm8k_eval.jsonl`                              | held-out GSM8K accuracy checks during Stage A                                           |
| `sparse_reward_preflight.json`                  | pre-training check that the reward signal isn't degenerate                              |
| `phase{1,2,3}_complete.json`                    | completion markers for each pipeline phase                                              |
| `phase1_update_sentinel.jsonl` *(win_run only)* | proof that weights actually changed each window (guards against a silent no-op run)     |
| `measurements/metrics_ckpt*.json`               | raw effective-rank / dormant-fraction measurements per checkpoint                       |
| `analysis/results_table.csv`                    | the one table with every checkpoint's Q metrics + SVAMP before/after/delta              |
| `analysis/spearman_table.csv`                   | correlation of every Q variant against every outcome                                    |
| `analysis/analysis_summary.json`                | the pre-registered primary result: rho(erank_L12, svamp_delta)                          |
| `analysis/fig_a/b/c_*.png`                      | Q vs. training step; reward vs. Q; the primary scatter plot                             |
| `adaptation*/ckpt-N/summary.json`               | that checkpoint's SVAMP before/after accuracy and delta                                 |
| `adaptation*/ckpt-N/svamp_eval_curve.jsonl`     | accuracy at steps 10/20/30/40/50 during the SVAMP adaptation                            |
| `adaptation*/ckpt-N/baseline.json`              | pre-adaptation SVAMP accuracy for that checkpoint                                       |
| `adaptation*/ckpt-N/update_sentinel.jsonl`      | same no-op guard, for the adaptation run                                                |
| `*/repeat_manifest.json`                        | proves a seed-repeat run reused the exact same source checkpoint/config (hash-verified) |

## Fastest way to see the headline result

Open `mac_run/analysis/analysis_summary.json` and `win_run/analysis/analysis_summary.json`
side by side — each has one number: the pre-registered Spearman correlation
between checkpoint-level effective rank and SVAMP adaptation delta.
