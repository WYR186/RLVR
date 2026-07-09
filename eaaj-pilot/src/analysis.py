"""Phase-4 analysis shared by local scripts and notebooks."""
from __future__ import annotations

import json
from pathlib import Path


def run_analysis(run_dir, pilot_config: dict) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.stats import spearmanr

    run_dir = Path(run_dir)
    ckpts = pilot_config["stage_a"]["checkpoint_steps"]
    layers = pilot_config["measurement"]["layers"]
    measure_dir = run_dir / "measurements"
    adapt_root = run_dir / "adaptation"
    out_dir = run_dir / "analysis"
    out_dir.mkdir(exist_ok=True)

    eval_log = [json.loads(x) for x in (run_dir / "gsm8k_eval.jsonl").read_text().splitlines()]
    gsm_acc = {int(x["step"]): x["accuracy"] for x in eval_log}
    rows = []
    for n in ckpts:
        m = json.loads((measure_dir / f"metrics_ckpt{n}.json").read_text())
        a = json.loads((adapt_root / f"ckpt-{n}" / "summary.json").read_text())
        row = {"ckpt": n, "gsm8k_acc": gsm_acc.get(n, np.nan),
               "svamp_before": a["acc_before"], "svamp_after": a["acc_after"],
               "svamp_delta": a["delta_acc"]}
        for layer in layers:
            q = m["per_layer"][f"layer{layer}"]
            row.update({
                f"erank_L{layer}": q["erank"],
                f"erank_norm_L{layer}": q["erank_norm"],
                f"dormant025_L{layer}": q["dormant_frac_tau0.025"],
                f"dormant100_L{layer}": q["dormant_frac_tau0.1"],
                f"aniso_c_L{layer}": q["anisotropy_centered"],
            })
        rows.append(row)
    df = pd.DataFrame(rows).set_index("ckpt")
    df.to_csv(out_dir / "results_table.csv")

    colors = ["#2a78d6", "#1baf7a", "#eda100"]
    layer_colors = dict(zip(layers, colors))
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e8e7e2", "axes.axisbelow": True,
    })

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    for layer in layers:
        ax1.plot(df.index, df[f"erank_L{layer}"], "o-", lw=2,
                 color=layer_colors[layer], label=f"layer {layer}")
        ax2.plot(df.index, df[f"dormant100_L{layer}"], "o-", lw=2,
                 color=layer_colors[layer], label=f"τ=.1 L{layer}")
        ax2.plot(df.index, df[f"dormant025_L{layer}"], "o--", lw=1,
                 color=layer_colors[layer], alpha=.6, label=f"τ=.025 L{layer}")
    ax1.set(title="Q across GSM8K GRPO training", ylabel="effective rank")
    ax1.legend(frameon=False)
    ax2.set(xlabel="GRPO updates", ylabel="dormant fraction")
    ax2.set_xticks(ckpts)
    ax2.legend(frameon=False, ncol=3, fontsize=8)
    fig.savefig(out_dir / "fig_a_q_vs_updates.png")
    plt.close(fig)

    dash = pd.DataFrame([json.loads(x) for x in (run_dir / "dashboard.jsonl").read_text().splitlines()])
    # A resumed run replays steps since its last trainer checkpoint and appends
    # them again; file order is chronological, so keep the last row per step
    # (RNG replay makes replayed rows equivalent).
    dash = dash.drop_duplicates("step", keep="last").sort_values("step")
    reward_candidates = [c for c in dash.columns if c == "reward" or c.endswith("/reward")
                         or c.endswith("reward_mean") or c == "rewards/exact_answer_reward/mean"]
    if not reward_candidates:
        raise KeyError(f"no reward column in dashboard: {list(dash.columns)}")
    reward_col = reward_candidates[0]
    dash = dash.dropna(subset=[reward_col])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(dash.step, dash[reward_col], color="#b5b4ae", lw=1, alpha=.8)
    ax.plot(dash.step, dash[reward_col].rolling(10, min_periods=1).mean(),
            color="#52514e", lw=2.2, label="reward (10-step mean)")
    ax.set(xlabel="GRPO updates", ylabel="mean reward",
           title="Dashboard vs Q: reward with effective rank")
    axq = ax.twinx()
    axq.plot(df.index, df["erank_L12"], "o-", color="#2a78d6", lw=2.5,
             label="effective rank (layer 12)")
    axq.set_ylabel("effective rank, layer 12", color="#2a78d6")
    axq.spines.right.set_visible(True)
    fig.legend(loc="lower left", bbox_to_anchor=(.12, .14), frameon=False)
    fig.savefig(out_dir / "fig_b_reward_vs_q.png")
    plt.close(fig)

    rho, p_value = spearmanr(df["erank_L12"], df["svamp_delta"])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(df["erank_L12"], df["svamp_delta"], s=90, color="#2a78d6")
    for n in df.index:
        ax.annotate(f"ckpt {n}", (df.loc[n, "erank_L12"], df.loc[n, "svamp_delta"]),
                    xytext=(8, 4), textcoords="offset points", fontsize=9)
    ax.set(xlabel="effective rank @ layer 12", ylabel="SVAMP Δaccuracy (50 updates)",
           title=f"RQ1 pilot: Spearman ρ={rho:.2f} (p={p_value:.3f}, n=5)")
    fig.savefig(out_dir / "fig_c_scatter_q_vs_svamp.png")
    plt.close(fig)

    q_vars = [c for c in df.columns if c.startswith(("erank", "dormant", "aniso"))]
    outcomes = ["svamp_delta", "svamp_after", "gsm8k_acc"]
    spear_rows = []
    for q in q_vars:
        row = {"q_variant": q}
        for outcome in outcomes:
            row[f"rho_{outcome}"] = spearmanr(df[q], df[outcome])[0]
        spear_rows.append(row)
    spear = pd.DataFrame(spear_rows).set_index("q_variant")
    spear.to_csv(out_dir / "spearman_table.csv")
    summary = {"run_dir": str(run_dir), "n_checkpoints": len(df),
               "primary_q": "erank_L12", "primary_outcome": "svamp_delta",
               "spearman_rho": float(rho), "spearman_p": float(p_value)}
    (out_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=1))
    return summary
