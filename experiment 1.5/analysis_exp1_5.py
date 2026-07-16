"""Phase-4 analysis for experiment 1.5.

Pre-registered readout (EXPERIMENT_1_5_PLAN_ZH.md §4):

  MC1 (dose check)     |erank_L12(ckpt_last)/erank_L12(ckpt_0) - 1| >= 0.10
  MC2 (outcome check)  some later checkpoint's mean-of-3-seed delta sits
                       >= 0.05 BELOW checkpoint-0's mean-of-3-seed delta
  Primary              Spearman rho(erank_L12, svamp_delta_mean3), n = 6.
                       Interpretable as an RQ1 readout ONLY if MC2 passes;
                       otherwise it is recorded as descriptive.

Everything else written here (per-seed rhos, seed-pair rank correlations,
variance decomposition, legacy-100 bridge columns) is descriptive support.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR))

import exp1_5_lib as lib  # noqa: E402  (puts eaaj-pilot on sys.path too)


def run_exp15_analysis(run_dir, cfg: dict) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.stats import spearmanr

    run_dir = Path(run_dir)
    out_dir = run_dir / "analysis"
    out_dir.mkdir(exist_ok=True)
    layers = cfg["measurement"]["layers"]
    all_ckpts = cfg["stage_a"]["checkpoint_steps"]
    adapt_ckpts = cfg["adaptation"]["checkpoints"]
    seeds = cfg["adaptation"]["seeds"]
    checks = cfg["analysis"]["manipulation_checks"]

    # ---- Q table over every saved checkpoint --------------------------------
    q_rows = []
    for n in all_ckpts:
        m = json.loads((run_dir / "measurements" / f"metrics_ckpt{n}.json")
                       .read_text(encoding="utf-8"))
        row = {"ckpt": n}
        for layer in layers:
            q = m["per_layer"][f"layer{layer}"]
            row.update({
                f"erank_L{layer}": q["erank"],
                f"erank_norm_L{layer}": q["erank_norm"],
                f"dormant025_L{layer}": q["dormant_frac_tau0.025"],
                f"dormant100_L{layer}": q["dormant_frac_tau0.1"],
                f"aniso_c_L{layer}": q["anisotropy_centered"],
            })
        q_rows.append(row)
    df_q = pd.DataFrame(q_rows).set_index("ckpt").sort_index()

    # ---- adaptation long table (ckpt x seed) --------------------------------
    long_rows = []
    for n in adapt_ckpts:
        for seed in seeds:
            s = json.loads(
                (run_dir / f"adaptation_seed{seed}" / f"ckpt-{n}" / "summary.json")
                .read_text(encoding="utf-8"))
            long_rows.append({
                "ckpt": n, "seed": seed,
                "svamp_before": s["acc_before"],
                "svamp_after": s["acc_after"],
                "svamp_delta": s["delta_acc"],
                "svamp_before_legacy100": s.get("acc_before_legacy100"),
                "svamp_after_legacy100": s.get("acc_after_legacy100"),
                "svamp_delta_legacy100": s.get("delta_acc_legacy100"),
            })
    df_long = pd.DataFrame(long_rows).sort_values(["ckpt", "seed"])
    df_long.to_csv(out_dir / "results_long.csv", index=False)

    agg = df_long.groupby("ckpt").agg(
        svamp_before=("svamp_before", "first"),
        svamp_delta_mean3=("svamp_delta", "mean"),
        svamp_delta_sd3=("svamp_delta", "std"),
        svamp_delta_legacy100_mean3=("svamp_delta_legacy100", "mean"),
        n_seeds=("seed", "count"))
    df = agg.join(df_q, how="left")
    df.to_csv(out_dir / "results_table.csv")

    # ---- manipulation checks ------------------------------------------------
    first_ckpt, last_ckpt = min(all_ckpts), max(all_ckpts)
    erank0 = df_q.loc[first_ckpt, "erank_L12"]
    erank_last = df_q.loc[last_ckpt, "erank_L12"]
    mc1_rel_change = float(erank_last / erank0 - 1.0)
    mc1_pass = abs(mc1_rel_change) >= checks["mc1_erank_l12_rel_change_min"]

    delta0 = float(df.loc[min(adapt_ckpts), "svamp_delta_mean3"])
    later = [c for c in adapt_ckpts if c != min(adapt_ckpts)]
    drops = {int(c): float(delta0 - df.loc[c, "svamp_delta_mean3"]) for c in later}
    mc2_max_drop = max(drops.values()) if drops else float("nan")
    mc2_pass = bool(drops) and mc2_max_drop >= checks["mc2_mean_delta_drop_vs_ckpt0_min"]

    cell = {
        (True, True): "MC1+MC2 pass: dose moved Q AND adaptability degraded — "
                      "primary rho is a valid RQ1 readout",
        (True, False): "MC1 pass, MC2 fail: Q moved but fixed-budget "
                       "adaptability did not degrade — strengthens the "
                       "SVAMP-too-close concern; rho descriptive only",
        (False, True): "MC1 fail, MC2 pass: adaptability degraded without an "
                       "erank_L12 signature — informative evidence AGAINST "
                       "erank_L12 as the early-warning metric",
        (False, False): "MC1+MC2 fail: dose still insufficient at this scale — "
                        "escalate design decision to the team",
    }[(mc1_pass, mc2_pass)]

    # ---- correlations --------------------------------------------------------
    sub_q = df_q.loc[adapt_ckpts]
    rho, p_value = spearmanr(sub_q["erank_L12"], df["svamp_delta_mean3"])
    per_seed = {}
    for seed in seeds:
        d = df_long[df_long.seed == seed].set_index("ckpt").loc[adapt_ckpts]
        r, p = spearmanr(sub_q["erank_L12"], d["svamp_delta"])
        per_seed[str(seed)] = {"rho": float(r), "p": float(p)}
    seed_pair_rank_corr = {}
    for a, b in itertools.combinations(seeds, 2):
        da = df_long[df_long.seed == a].set_index("ckpt").loc[adapt_ckpts, "svamp_delta"]
        db = df_long[df_long.seed == b].set_index("ckpt").loc[adapt_ckpts, "svamp_delta"]
        seed_pair_rank_corr[f"{a}-{b}"] = float(spearmanr(da, db)[0])

    # noise decomposition for the full-experiment power analysis
    between_var = float(df["svamp_delta_mean3"].var(ddof=1))
    within_var = float(df_long.groupby("ckpt")["svamp_delta"].var(ddof=1).mean())

    q_vars = [c for c in df.columns if c.startswith(("erank", "dormant", "aniso"))]
    outcomes = ["svamp_delta_mean3", "svamp_delta_legacy100_mean3", "svamp_before"]
    spear_rows = []
    for q in q_vars:
        row = {"q_variant": q}
        for outcome in outcomes:
            row[f"rho_{outcome}"] = spearmanr(df[q], df[outcome])[0]
        spear_rows.append(row)
    pd.DataFrame(spear_rows).set_index("q_variant").to_csv(
        out_dir / "spearman_table.csv")

    # ---- figures --------------------------------------------------------------
    colors = ["#2a78d6", "#1baf7a", "#eda100"]
    layer_colors = dict(zip(layers, colors))
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e8e7e2", "axes.axisbelow": True,
    })

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    for layer in layers:
        ax1.plot(df_q.index, df_q[f"erank_L{layer}"], "o-", lw=2,
                 color=layer_colors[layer], label=f"layer {layer}")
        ax2.plot(df_q.index, df_q[f"dormant100_L{layer}"], "o-", lw=2,
                 color=layer_colors[layer], label=f"τ=.1 L{layer}")
        ax2.plot(df_q.index, df_q[f"dormant025_L{layer}"], "o--", lw=1,
                 color=layer_colors[layer], alpha=.6, label=f"τ=.025 L{layer}")
    ax1.set(title=f"Q across GSM8K GRPO training (exp 1.5, lr {cfg['stage_a']['learning_rate']:g})",
            ylabel="effective rank")
    ax1.legend(frameon=False)
    ax2.set(xlabel="GRPO updates", ylabel="dormant fraction")
    ax2.set_xticks(list(df_q.index))
    ax2.legend(frameon=False, ncol=3, fontsize=8)
    fig.savefig(out_dir / "fig_a_q_vs_updates.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.8))
    for seed, marker in zip(seeds, ("o", "s", "^")):
        d = df_long[df_long.seed == seed].set_index("ckpt").loc[adapt_ckpts]
        ax.scatter(sub_q["erank_L12"], d["svamp_delta"], s=30, alpha=.45,
                   marker=marker, color="#8a8984", label=f"seed {seed}")
    ax.errorbar(sub_q["erank_L12"], df["svamp_delta_mean3"],
                yerr=df["svamp_delta_sd3"], fmt="o", ms=10, lw=0, elinewidth=1.6,
                capsize=4, color="#2a78d6", label="mean of 3 seeds ± sd")
    for n in adapt_ckpts:
        ax.annotate(f"ckpt {n}",
                    (sub_q.loc[n, "erank_L12"], df.loc[n, "svamp_delta_mean3"]),
                    xytext=(8, 4), textcoords="offset points", fontsize=9)
    ax.set(xlabel="effective rank @ layer 12",
           ylabel="SVAMP Δaccuracy on 300 q (50 updates)",
           title=f"exp 1.5 primary: ρ={rho:.2f} (p={p_value:.3f}, "
                 f"n={len(adapt_ckpts)}) — {'VALID' if mc2_pass else 'descriptive (MC2 fail)'}")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(out_dir / "fig_c_scatter_q_vs_svamp.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for seed, marker in zip(seeds, ("o", "s", "^")):
        d = df_long[df_long.seed == seed]
        ax.scatter(d["ckpt"], d["svamp_delta"], marker=marker, s=46, alpha=.75,
                   label=f"seed {seed}")
    ax.plot(list(df.index), df["svamp_delta_mean3"], "-", color="#52514e", lw=2,
            label="mean of seeds")
    ax.axhline(0, color="#b5b4ae", lw=1)
    ax.set(xlabel="Stage-A checkpoint (GRPO updates)",
           ylabel="SVAMP Δaccuracy (300 q)",
           title="Fixed-budget adaptation delta by checkpoint and seed")
    ax.set_xticks(adapt_ckpts)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(out_dir / "fig_d_delta_by_seed.png")
    plt.close(fig)

    summary = {
        "experiment": cfg["experiment"],
        "run_dir": str(run_dir),
        "n_checkpoints_measured": len(df_q),
        "n_checkpoints_adapted": len(adapt_ckpts),
        "n_seeds": len(seeds),
        "primary_q": "erank_L12",
        "primary_outcome": "svamp_delta_mean3",
        "spearman_rho": float(rho),
        "spearman_p": float(p_value),
        "per_seed_rho": per_seed,
        "seed_pair_delta_rank_corr": seed_pair_rank_corr,
        "manipulation_checks": {
            "mc1_erank_l12_rel_change": mc1_rel_change,
            "mc1_threshold": checks["mc1_erank_l12_rel_change_min"],
            "mc1_pass": bool(mc1_pass),
            "mc2_mean_delta_drops_vs_ckpt0": drops,
            "mc2_max_drop": mc2_max_drop,
            "mc2_threshold": checks["mc2_mean_delta_drop_vs_ckpt0_min"],
            "mc2_pass": bool(mc2_pass),
        },
        "rq1_primary_interpretable": bool(mc2_pass),
        "interpretation": cell,
        "delta_between_checkpoint_var": between_var,
        "delta_within_checkpoint_seed_var": within_var,
    }
    (out_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=1), encoding="utf-8")
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    args = parser.parse_args()
    print(json.dumps(run_exp15_analysis(Path(args.run_dir), lib.load_config()),
                     indent=1))
