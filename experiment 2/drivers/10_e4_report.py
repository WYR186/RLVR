"""E4 reporting — turn the audited arm records into the headline table and figure.

Reads whatever `08_e4_ruler.py` and `07_e4_weight_dose.py` produced and emits:

  e4_summary.md     the table that goes into the finding / Research Doc
  e4_calibration.png  erank change vs relative weight dose, with E1's
                      measured LoRA change drawn as a horizontal reference

    python 10_e4_report.py --dir ../outputs/e4_small
    python 10_e4_report.py --dir ../outputs/e4_large --no-figure

Run `09_audit_e4_artifacts.py` first. This script reports; it does not verify.
"""
import argparse
import json
import math
import sys
from pathlib import Path

EXP2 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXP2))

from src import e4_calibration as e4  # noqa: E402

# E1's measured maxima, the numbers this whole experiment exists to place.
E1_MAX_RESID_PCT = 0.7227     # V5a, layer 16
E1_MAX_ANY_PCT = 0.7303       # V6b down_in, layer 14


def load_arms(d: Path) -> dict:
    skip = {"ruler_table.json", "probe_manifest.json", "probe_frozen.json",
            "arm_W_weight_dose.json", "audit_e4.json", "e4_summary.md"}
    return {p.stem: json.loads(p.read_text())
            for p in sorted(d.glob("*.json")) if p.name not in skip}


def ladder_rows(arms: dict, ref_label: str) -> list[dict]:
    ref = e4.erank_by_layer(arms[ref_label])
    rows = []
    for label, rec in arms.items():
        if label == ref_label:
            continue
        rel = e4.relative_change(ref, e4.erank_by_layer(rec))
        if not rel:
            continue
        peak = max(rel, key=lambda l: abs(rel[l]))
        pert = rec.get("perturbation")
        rows.append({
            "arm": label,
            "dose": pert["achieved_aggregate_dose"] if pert else None,
            "requested": pert["requested_dose"] if pert else None,
            "max_abs_pct": abs(rel[peak]),
            "signed_pct": rel[peak],
            "layer": peak,
            "kind": "ladder" if pert else "ruler",
        })
    rows.sort(key=lambda r: (r["kind"], r["dose"] if r["dose"] is not None else -1))
    return rows


def bracket(rows: list[dict], target_pct: float):
    """Which two ladder rungs does `target_pct` fall between?"""
    rungs = sorted((r for r in rows if r["kind"] == "ladder"),
                   key=lambda r: r["dose"])
    below = [r for r in rungs if r["max_abs_pct"] <= target_pct]
    above = [r for r in rungs if r["max_abs_pct"] > target_pct]
    return (below[-1] if below else None), (above[0] if above else None)


def write_figure(rows, arm_w, path: Path, scale: str, floor_pct=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rungs = sorted((r for r in rows if r["kind"] == "ladder"),
                   key=lambda r: r["dose"])
    if not rungs:
        print("no ladder rungs; figure skipped")
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot([r["dose"] for r in rungs], [r["max_abs_pct"] for r in rungs],
            marker="o", lw=1.8, color="#1f4e79",
            label="Arm N: isotropic weight noise")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("relative Frobenius weight dose  "
                  r"$\|\Delta W\|_F / \|W_0\|_F$")
    ax.set_ylabel(r"max $|\Delta$ effective rank$|$  (%)")

    ax.axhline(E1_MAX_ANY_PCT, ls="--", lw=1.4, color="#c0392b",
               label=f"E1: Stage-A LoRA moved erank {E1_MAX_ANY_PCT:.4f}%")

    for r in rows:
        if r["kind"] != "ruler":
            continue
        if r["max_abs_pct"] <= 0:
            # Both log axes; a zero line cannot be drawn. This is the
            # identical-model self-test, so say so instead of dropping it.
            ax.plot([], [], ls=":", lw=1.4, color="#7d3c98",
                    label="Arm R: 0.0000% (identical-model self-test)")
            continue
        ax.axhline(r["max_abs_pct"], ls=":", lw=1.4, color="#7d3c98",
                   label=f"Arm R: base vs Instruct ({r['max_abs_pct']:.2f}%)")

    if floor_pct and floor_pct > 0:
        # Everything under the cross-platform drift is unresolvable, so shade it
        # rather than letting a reader take a sub-floor rung as a measurement.
        ax.axhspan(ax.get_ylim()[0], floor_pct, color="#95a5a6", alpha=0.22,
                   zorder=0)
        ax.axhline(floor_pct, ls="-", lw=1.0, color="#7f8c8d")
        ax.annotate(f"cross-platform drift floor {floor_pct:.3f}%",
                    xy=(ax.get_xlim()[0], floor_pct), xytext=(6, 3),
                    textcoords="offset points", fontsize=7.5, color="#5d6d7e")

    if arm_w:
        # ckpt-50 and ckpt-100 differ by 7%, so their labels would collide;
        # stack them and give each its own leader height.
        live = sorted((c, d) for c, d in arm_w.items() if d and d > 0)
        y0, y1 = ax.get_ylim()
        for i, (ckpt, dose) in enumerate(live):
            ax.axvline(dose, ls="-.", lw=1.2, color="#148f77")
            # Sit the labels in the empty mid-band: the legend owns the top and
            # the shaded floor owns the bottom.
            frac = 0.52 - 0.09 * i
            y = 10 ** (math.log10(y0) + frac * (math.log10(y1) - math.log10(y0)))
            ax.annotate(f"our {ckpt}  {dose:.2e}", xy=(dose, y),
                        xytext=(7, 0), textcoords="offset points",
                        fontsize=7.5, color="#148f77", ha="left", va="center")

    ax.set_title(f"E4 detector calibration — {scale} scale\n"
                 "how large a weight change does effective rank register?",
                 fontsize=11)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--reference", default="R_instruct")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()
    d = Path(args.dir)

    arms = load_arms(d)
    if args.reference not in arms:
        raise SystemExit(f"reference arm {args.reference!r} not in {sorted(arms)}")
    rows = ladder_rows(arms, args.reference)
    scale = arms[args.reference]["measurement_contract"].get("e4_scale", "?")

    arm_w = {}
    wpath = d / "arm_W_weight_dose.json"
    if wpath.is_file():
        rel = json.loads(wpath.read_text()).get("relative_dose", {})
        if rel.get("status") != "not_run":
            arm_w = {c: r["aggregate_relative_dose"] for c, r in rel.items()}

    # Gate R2's cross-platform delta is the smallest change this measurement can
    # resolve; rungs below it are floor, not signal.
    floor_pct = None
    ref_rec = arms[args.reference]
    pr = ref_rec.get("platform_reproduction_vs_e1")
    if pr and math.isfinite(pr.get("max_abs_rel_delta_pct", float("nan"))):
        floor_pct = float(pr["max_abs_rel_delta_pct"])

    lo, hi = bracket(rows, E1_MAX_ANY_PCT)

    lines = [f"# E4 calibration summary — {scale} scale", "",
             f"Reference arm: `{args.reference}`. "
             f"Probe n={arms[args.reference]['meta']['n_probe']}, "
             f"layers {arms[args.reference]['meta']['layers']}, "
             f"dtype {arms[args.reference]['measurement_contract']['e4_dtype']}.",
             "",
             "| arm | requested dose | achieved dose | max \\|Δerank\\| | layer |",
             "|---|---|---|---|---|"]
    for r in rows:
        req = f"{r['requested']:.0e}" if r["requested"] is not None else "—"
        got = f"{r['dose']:.4e}" if r["dose"] is not None else "— (not a controlled dose)"
        lines.append(f"| {r['arm']} | {req} | {got} | "
                     f"{r['max_abs_pct']:.4f}% | {r['layer']} |")

    lines += ["", "## Where our Stage-A run falls", ""]
    if arm_w:
        for c, v in sorted(arm_w.items()):
            lines.append(f"- `{c}` aggregate relative dose: **{v:.4e}**")
    else:
        lines.append("- Arm W relative dose not computed "
                     "(base weight norms missing); only \\|ΔW\\|_F is known.")
    lines += ["",
              f"E1 measured the Stage-A LoRA moving erank by at most "
              f"**{E1_MAX_ANY_PCT:.4f}%** (down_in L14) / "
              f"{E1_MAX_RESID_PCT:.4f}% (resid L16)."]
    if lo and hi:
        lines.append(f"On this ladder that sits between the "
                     f"**{lo['requested']:.0e}** rung ({lo['max_abs_pct']:.4f}%) "
                     f"and the **{hi['requested']:.0e}** rung "
                     f"({hi['max_abs_pct']:.4f}%).")
    elif hi and not lo:
        lines.append(f"That is **below every rung measured** — the smallest "
                     f"({hi['requested']:.0e}) already moves erank "
                     f"{hi['max_abs_pct']:.4f}%.")
    elif lo and not hi:
        lines.append(f"That is **above every rung measured**; extend the ladder.")

    if floor_pct:
        below = [r for r in rows if r["kind"] == "ladder"
                 and r["max_abs_pct"] <= floor_pct]
        lines += ["",
                  f"**Resolution floor.** This platform reproduces E1's "
                  f"published ckpt-0 eranks to within **{floor_pct:.3f}%**, so a "
                  f"change smaller than that is not resolvable here."]
        if below:
            names = ", ".join(f"`{r['arm']}`" for r in sorted(
                below, key=lambda r: r["dose"]))
            lines.append(f"{len(below)} ladder rung(s) fall at or below it "
                         f"({names}) and must be read as floor, not as a "
                         f"measured response.")
    lines += ["", "## Reading rules", "",
              "- Arm R is not a controlled dose. It is an order-of-magnitude "
              "reference point between two released checkpoints separated by an "
              "undocumented pipeline. No causal language.",
              "- Arm N's noise is isotropic and full-rank; a real update is "
              "low-rank and structured. This calibrates the detector, not RLVR.",
              "- erank levels are never compared across scales or dtypes, only "
              "arms against their own scale's reference."]

    (d / "e4_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {d / 'e4_summary.md'}")

    if not args.no_figure:
        write_figure(rows, arm_w, d / "e4_calibration.png", scale,
                     floor_pct=floor_pct)


if __name__ == "__main__":
    main()
