"""Claim (D) — is the collapse pathway length explosion -> clip saturation ->
gradient starvation, and how much warning does each dashboard signal give?

Zero compute. Every input is a `dashboard.jsonl` already in the repository.

This is deliberately structured as a DETECTOR EVALUATION, not a narrative. A
signal that fires before collapse is only useful if it does not also fire on
runs that never collapse, so every signal is scored on both:

    lead time   steps between the signal firing and the reward collapse,
                measured only on runs that actually collapsed
    false alarm whether it fires on a run that trained to the end healthily

The bar these set is the bar Q has to clear. `PROJECT_OVERVIEW...md` Sec 3.2 (C)
notes grad_norm is logged as exactly 0.000 from step 35 in one collapse -- a
free, unambiguous "training is dead" flag already on the dashboard.

    python 11_collapse_pathway.py --out ../outputs/claim_d

All thresholds below are PRE-REGISTERED in the sense that matters here: they
are stated once, applied identically to collapsed and healthy runs, and the
false-alarm column is reported whether or not it is flattering.
"""
import argparse
import json
import math
from pathlib import Path

EXP2 = Path(__file__).resolve().parent.parent
ROOT = EXP2.parent

# --- pre-registered definitions -------------------------------------------

BASELINE_STEPS = 5      # reward baseline = mean of the first N logged steps
COLLAPSE_FRAC = 0.25    # collapsed = reward <= 25% of baseline ...
                        # ... and never recovers above it afterwards

SIGNALS = {
    "length_explosion": dict(
        field="completions/mean_length", op=">=", frac_of_cap=0.90,
        label="mean completion length >= 90% of the cap"),
    "clip_saturation": dict(
        field="completions/clipped_ratio", op=">=", value=0.50,
        label="clipped_ratio >= 0.50"),
    "no_reward_variance": dict(
        field="frac_reward_zero_std", op=">=", value=0.75,
        label="frac_reward_zero_std >= 0.75"),
    "grad_starvation": dict(
        field="grad_norm", op="==", value=0.0,
        label="grad_norm == 0.000 exactly"),
    "entropy_drop": dict(
        field="entropy", op="<=", frac_of_baseline=0.50,
        label="entropy <= 50% of its baseline (sign unstable, see Sec 3.3)"),
}


def load_run(path: Path):
    rows = []
    for line in path.read_text().splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "step" in d and "reward" in d:
            rows.append(d)
    rows.sort(key=lambda r: r["step"])
    return rows


def num(row, key):
    v = row.get(key)
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def collapse_step(rows):
    """First step at or after which reward stays <= COLLAPSE_FRAC x baseline.

    Requiring it to STAY down is what separates a collapse from a dip; a run
    that recovers is not a collapse and must not be scored as one.
    """
    r = [num(x, "reward") for x in rows]
    r = [x for x in r if x is not None]
    if len(r) < BASELINE_STEPS + 5:
        return None
    base = sum(r[:BASELINE_STEPS]) / BASELINE_STEPS
    if base <= 0:
        return None
    thr = COLLAPSE_FRAC * base
    for i in range(BASELINE_STEPS, len(r)):
        if all(x <= thr for x in r[i:]):
            return rows[i]["step"]
    return None


def signal_step(rows, spec, cap):
    """First step at which one signal fires, or None."""
    field = spec["field"]
    if "frac_of_cap" in spec:
        if not cap:
            return None, None
        thr = spec["frac_of_cap"] * cap
    elif "frac_of_baseline" in spec:
        vals = [num(x, field) for x in rows[:BASELINE_STEPS]]
        vals = [v for v in vals if v is not None]
        if not vals:
            return None, None
        thr = spec["frac_of_baseline"] * (sum(vals) / len(vals))
    else:
        thr = spec["value"]

    op = spec["op"]
    for row in rows:
        v = num(row, field)
        if v is None:
            continue
        hit = (v >= thr) if op == ">=" else (v <= thr) if op == "<=" else (v == thr)
        if hit:
            return row["step"], thr
    return None, thr


def group_variance_model(rows, n_gen: int):
    """Sec 3.2 (D)'s mechanism, as a falsifiable prediction.

    With group size G and per-sample success probability p, a group carries no
    gradient when every sample agrees. Under independence that is
    p^G + (1-p)^G. Comparing this against the LOGGED frac_reward_zero_std,
    using the logged mean reward as p, tests the stated mechanism instead of
    asserting it.
    """
    pts = []
    for row in rows:
        p = num(row, "reward")
        obs = num(row, "frac_reward_zero_std")
        if p is None or obs is None or not (0.0 <= p <= 1.0):
            continue
        pred = p ** n_gen + (1.0 - p) ** n_gen
        pts.append((p, pred, obs))
    if len(pts) < 5:
        return None
    resid = [o - pr for _, pr, o in pts]
    mean_abs = sum(abs(x) for x in resid) / len(resid)
    mean_signed = sum(resid) / len(resid)
    # p at which half of all groups are expected to be degenerate
    lo, hi = 0.0, 0.5
    for _ in range(60):
        mid = (lo + hi) / 2
        if mid ** n_gen + (1 - mid) ** n_gen > 0.5:
            lo = mid
        else:
            hi = mid
    return {"n_points": len(pts), "mean_abs_residual": mean_abs,
            "mean_signed_residual": mean_signed, "p_star_half": (lo + hi) / 2,
            "n_generations": n_gen}


def analyse(run_dir: Path, name: str):
    rows = load_run(run_dir / "dashboard.jsonl")
    if len(rows) < 10:
        return None
    cfg_path = run_dir / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.is_file() else {}
    # adaptation sub-runs inherit the parent run's config
    if not cfg:
        for parent in run_dir.parents:
            if (parent / "config.json").is_file():
                cfg = json.loads((parent / "config.json").read_text())
                break
    cap = cfg.get("max_completion_length")
    n_gen = int(cfg.get("num_generations") or 8)

    t_col = collapse_step(rows)
    sigs = {}
    for key, spec in SIGNALS.items():
        t, thr = signal_step(rows, spec, cap)
        sigs[key] = {
            "fired_at_step": t,
            "threshold": thr,
            "lead_steps": (t_col - t) if (t is not None and t_col is not None) else None,
        }
    r = [num(x, "reward") for x in rows if num(x, "reward") is not None]
    return {
        "run": name,
        "n_steps": len(rows),
        "learning_rate": cfg.get("learning_rate"),
        "num_generations": n_gen,
        "max_completion_length": cap,
        "beta": cfg.get("beta"),
        "reward_first": r[0], "reward_last": r[-1], "reward_min": min(r),
        "collapsed": t_col is not None,
        "collapse_step": t_col,
        "signals": sigs,
        "group_variance_model": group_variance_model(rows, n_gen),
    }


# --- reporting -------------------------------------------------------------

def summarise(results):
    """Lead time where it collapsed, false alarms where it did not."""
    collapsed = [r for r in results if r["collapsed"]]
    healthy = [r for r in results if not r["collapsed"]]
    table = {}
    for key in SIGNALS:
        leads = [r["signals"][key]["lead_steps"] for r in collapsed
                 if r["signals"][key]["lead_steps"] is not None]
        missed = sum(1 for r in collapsed
                     if r["signals"][key]["fired_at_step"] is None)
        alarms = [r["run"] for r in healthy
                  if r["signals"][key]["fired_at_step"] is not None]
        table[key] = {
            "label": SIGNALS[key]["label"],
            "n_collapsed_detected": len(leads),
            "n_collapsed_missed": missed,
            "lead_steps": sorted(leads),
            "median_lead": (sorted(leads)[len(leads) // 2] if leads else None),
            "min_lead": min(leads) if leads else None,
            "max_lead": max(leads) if leads else None,
            "n_false_alarms": len(alarms),
            "n_healthy": len(healthy),
            "false_alarm_runs": alarms[:6],
        }
    return {"n_collapsed": len(collapsed), "n_healthy": len(healthy),
            "signals": table}


def write_figure(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    collapsed = [r for r in results if r["collapsed"]]
    if not collapsed:
        return
    ex = max(collapsed, key=lambda r: r["n_steps"])
    rows = load_run(ROOT / "eaaj-pilot" / "outputs" / ex["run"] / "dashboard.jsonl")
    steps = [x["step"] for x in rows]
    cap = ex["max_completion_length"] or 512

    fig, axes = plt.subplots(4, 1, figsize=(8.2, 8.4), sharex=True)
    series = [
        ("completions/mean_length", "mean completion\nlength (tokens)", "#c0392b", cap),
        ("completions/clipped_ratio", "clipped ratio", "#d68910", None),
        ("frac_reward_zero_std", "frac groups with\nzero reward variance", "#7d3c98", None),
        ("grad_norm", "grad norm", "#1f4e79", None),
    ]
    for ax, (field, label, colour, hline) in zip(axes, series):
        ys = [num(x, field) for x in rows]
        ax.plot([s for s, y in zip(steps, ys) if y is not None],
                [y for y in ys if y is not None], lw=1.5, color=colour)
        ax.set_ylabel(label, fontsize=8.5)
        ax.grid(alpha=0.3)
        if hline:
            ax.axhline(hline, ls=":", lw=1, color="#7f8c8d")
            ax.annotate(f"cap {hline}", xy=(steps[0], hline), xytext=(4, -10),
                        textcoords="offset points", fontsize=7, color="#7f8c8d")
        if ex["collapse_step"] is not None:
            ax.axvline(ex["collapse_step"], ls="--", lw=1.4, color="#000000",
                       alpha=0.65)

    rw = [num(x, "reward") for x in rows]
    ax2 = axes[0].twinx()
    ax2.plot([s for s, y in zip(steps, rw) if y is not None],
             [y for y in rw if y is not None], lw=1.2, color="#148f77", alpha=0.75)
    ax2.set_ylabel("reward", fontsize=8, color="#148f77")
    ax2.tick_params(axis="y", labelcolor="#148f77", labelsize=7)

    axes[-1].set_xlabel("optimizer step")
    axes[0].set_title(
        f"Collapse pathway — {ex['run']}\n"
        f"lr={ex['learning_rate']}, {ex['num_generations']} generations, "
        f"dashed line = reward collapse at step {ex['collapse_step']}",
        fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    return ex["run"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs" / "claim_d"))
    ap.add_argument("--min-steps", type=int, default=50,
                    help="only score runs with at least this many logged steps; "
                         "short adaptation runs cannot show a lead time")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    results = []
    for dash in sorted((ROOT / "eaaj-pilot" / "outputs").rglob("dashboard.jsonl")):
        name = str(dash.parent.relative_to(ROOT / "eaaj-pilot" / "outputs"))
        r = analyse(dash.parent, name)
        if r and r["n_steps"] >= args.min_steps:
            results.append(r)

    summary = summarise(results)
    (out / "claim_d_results.json").write_text(
        json.dumps({"runs": results, "summary": summary}, indent=1))

    print(f"scored {len(results)} runs "
          f"({summary['n_collapsed']} collapsed, {summary['n_healthy']} healthy)\n")
    print("=== the natural experiment ===")
    print(f"{'run':<40} {'lr':>7} {'r0':>6} {'rEnd':>6} {'collapse@':>10}")
    for r in sorted(results, key=lambda r: (not r["collapsed"], str(r["learning_rate"]))):
        print(f"{r['run'][:40]:<40} {str(r['learning_rate']):>7} "
              f"{r['reward_first']:>6.3f} {r['reward_last']:>6.3f} "
              f"{(r['collapse_step'] if r['collapsed'] else '-'):>10}")

    print("\n=== signal evaluation ===")
    print(f"{'signal':<20} {'detected':>9} {'missed':>7} {'median lead':>12} "
          f"{'lead range':>14} {'false alarms':>13}")
    for key, s in summary["signals"].items():
        lr = (f"{s['min_lead']} .. {s['max_lead']}" if s["lead_steps"] else "-")
        print(f"{key:<20} {s['n_collapsed_detected']:>4}/{summary['n_collapsed']:<4} "
              f"{s['n_collapsed_missed']:>7} {str(s['median_lead']):>12} "
              f"{lr:>14} {s['n_false_alarms']:>6}/{s['n_healthy']:<6}")

    print("\n=== group-variance mechanism check ===")
    print("predicted frac_reward_zero_std = p^G + (1-p)^G, p = logged mean reward")
    print(f"{'run':<40} {'G':>3} {'mean|resid|':>12} {'signed':>9} {'p* (50%)':>9}")
    for r in results:
        m = r["group_variance_model"]
        if m:
            print(f"{r['run'][:40]:<40} {m['n_generations']:>3} "
                  f"{m['mean_abs_residual']:>12.4f} "
                  f"{m['mean_signed_residual']:>+9.4f} {m['p_star_half']:>9.4f}")

    fig = out / "claim_d_collapse_pathway.png"
    used = write_figure(results, fig)
    if used:
        print(f"\nwrote {fig}  (example run: {used})")
    print(f"wrote {out / 'claim_d_results.json'}")


if __name__ == "__main__":
    main()
