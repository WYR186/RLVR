#!/usr/bin/env python3
"""Pre-registered lead-time forensics for experiment 1.5.1 (stall forensics).

Question: in the lr=1e-5 collapse regime, does Q (erank, dormant fraction)
deviate BEFORE the dashboard's acute precursor (entropy) and before the
event (terminal zero-reward-variance run)?

Pre-registered readouts (EXPERIMENT_1_5_1_PLAN_ZH.md §5; thresholds also
frozen in each exp1_5_1_config_seed*.json "analysis" block):

  Event E      first step of the terminal zero-reward-variance run that
               triggered the safety stop (trigger_step - patience + 1),
               cross-checked against dashboard.jsonl.
  Entropy lead E - onset of the final sustained entropy<0.10 run before E.
  SC1 (per-run)   some measured checkpoint c with c <= E-10 has
                  erank_L12(c)/erank_L12(0) - 1 <= -0.12
                  (beyond the healthy-run envelope, see below).
  SC2 (per-run)   any dormant fraction > 0 at a checkpoint <= E (any layer,
                  either tau) — healthy runs have measured 0.0 everywhere,
                  so any nonzero is a zero-false-positive-history signal.
  SC3 (per-run)   SC1's lead >= entropy lead + 5 steps (Q must beat the
                  best acute dashboard precursor by more than the 5-step
                  checkpoint grid resolution).
  Cross-replicate verdict: each SC "supported at pilot scale" iff it holds
  in >= 2 of the 3 pre-registered replicates. Everything else descriptive.

Healthy-run envelope behind SC1's -12%: worst healthy erank_L12 dip at
<=100 steps across the three completed lr=1e-6 runs is -8.6% (pilot WIN v2
ckpt-50; pilot CPU -5.3%; exp1.5 v3 -7.2%).

Runs that never collapse (capped at step 80) are reported as censored:
no event, Q trajectory still tabulated against the envelope.

Usage:
  python analysis_exp1_5_1.py RUN_DIR [RUN_DIR ...]
Writes forensics.json into each run dir (additive evidence) and prints the
combined verdict JSON. stdlib only — runs on any machine.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ENTROPY_ONSET_THRESHOLD = 0.10
SC1_ERANK_REL_THRESHOLD = -0.12
SC1_MIN_LEAD = 10
SC3_MARGIN_OVER_ENTROPY = 5
CLIP_PATHOLOGICAL = 0.90
HEALTHY_ENVELOPE = {
    "erank_L12_rel_worst_dip_at_le100_steps": -0.086,
    "sources": {
        "pilot_win_v2_ckpt50": -0.086,
        "exp1_5_v3_ckpt50": -0.072,
        "pilot_cpu_ckpt50": -0.053,
    },
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def dashboard_steps(run_dir: Path) -> dict[int, dict]:
    """Step -> row, keeping only training rows (must carry entropy)."""
    rows = {}
    for row in read_jsonl(run_dir / "dashboard.jsonl"):
        if "entropy" in row and "step" in row:
            rows[int(row["step"])] = row
    return rows


def find_event(run_dir: Path, rows: dict[int, dict]) -> dict:
    """Terminal zero-variance run start E, from safety_stop + dashboard."""
    stop_path = run_dir / "safety_stop.json"
    if not stop_path.exists():
        cap = run_dir / "hard_cap_stop.json"
        return {"collapsed": False,
                "censored_at": (json.loads(cap.read_text())["step"]
                                if cap.exists() else max(rows) if rows else None)}
    stop = json.loads(stop_path.read_text(encoding="utf-8"))
    trigger = int(stop["step"])
    run_cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    patience = int(run_cfg.get("safety", {}).get("zero_signal_patience", 5))
    event = trigger - patience + 1
    # Cross-check against the dashboard: walk back from the trigger while
    # reward_std stays exactly zero.
    walked = trigger
    while walked - 1 in rows and rows[walked - 1].get("reward_std") == 0.0:
        walked -= 1
    checked = {"event_from_patience": event, "event_from_dashboard": walked}
    if walked != event:
        checked["note"] = ("dashboard walk-back disagrees with patience "
                           "arithmetic; using the dashboard value")
        event = walked
    return {"collapsed": True, "trigger_step": trigger,
            "zero_signal_patience": patience, "event_step": event, **checked}


def entropy_onset(rows: dict[int, dict], event: int) -> dict:
    """First step of the final sustained entropy<threshold run before E."""
    prior = [s for s in sorted(rows) if s < event]
    onset = None
    for s in reversed(prior):
        if rows[s]["entropy"] < ENTROPY_ONSET_THRESHOLD:
            onset = s
        else:
            break
    if onset is None:
        return {"onset_step": None, "lead_steps": 0,
                "note": f"entropy never sustainedly < "
                        f"{ENTROPY_ONSET_THRESHOLD} before the event"}
    return {"onset_step": onset, "lead_steps": event - onset}


def clip_profile(rows: dict[int, dict]) -> dict:
    steps = sorted(rows)
    patho = [s for s in steps
             if rows[s].get("completions/clipped_ratio", 0.0) >= CLIP_PATHOLOGICAL]
    return {"threshold": CLIP_PATHOLOGICAL,
            "frac_steps_at_or_above": (len(patho) / len(steps)) if steps else None,
            "first_step_at_or_above": patho[0] if patho else None,
            "note": "chronic background flag, not a timing precursor "
                    "(pathological from the first steps at lr=1e-5)"}


def q_trajectory(run_dir: Path) -> list[dict]:
    out = []
    mdir = run_dir / "measurements"
    if not mdir.exists():
        return out
    files = sorted(mdir.glob("metrics_ckpt*.json"),
                   key=lambda p: int(p.stem.replace("metrics_ckpt", "")))
    base = None
    for p in files:
        m = json.loads(p.read_text(encoding="utf-8"))
        step = int(m["checkpoint"])
        row = {"step": step}
        for layer, vals in m["per_layer"].items():
            row[f"erank_{layer}"] = vals["erank"]
            for k, v in vals.items():
                if k.startswith("dormant_frac"):
                    row[f"{k}_{layer}"] = v
        if step == 0:
            base = row
        if base:
            for layer in ("layer4", "layer12", "layer22"):
                if f"erank_{layer}" in row and base.get(f"erank_{layer}"):
                    row[f"erank_{layer}_rel"] = (
                        row[f"erank_{layer}"] / base[f"erank_{layer}"] - 1.0)
        out.append(row)
    return out


def judge_run(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    rows = dashboard_steps(run_dir)
    event_info = find_event(run_dir, rows)
    q_rows = q_trajectory(run_dir)
    dormant_hits = []
    for row in q_rows:
        for k, v in row.items():
            if k.startswith("dormant_frac") and v and v > 0:
                dormant_hits.append({"step": row["step"], "metric": k, "value": v})
    result = {
        "run_dir": str(run_dir),
        "event": event_info,
        "clipping": clip_profile(rows),
        "q_trajectory": q_rows,
        "healthy_envelope": HEALTHY_ENVELOPE,
    }
    if not event_info.get("collapsed"):
        result["verdicts"] = {
            "censored": True,
            "note": "no collapse before the cap; SC1-SC3 not evaluable on "
                    "this replicate — the non-collapse itself is recorded "
                    "(collapse-hazard variance across seeds)",
            "sc2_dormant_nonzero_anywhere": bool(dormant_hits),
            "dormant_hits": dormant_hits,
        }
        return result
    event = event_info["event_step"]
    ent = entropy_onset(rows, event)
    sc1_hits = [r for r in q_rows
                if r["step"] <= event - SC1_MIN_LEAD
                and r.get("erank_layer12_rel") is not None
                and r["erank_layer12_rel"] <= SC1_ERANK_REL_THRESHOLD]
    sc1_pass = bool(sc1_hits)
    sc1_onset = min((r["step"] for r in sc1_hits), default=None)
    sc1_lead = (event - sc1_onset) if sc1_onset is not None else None
    pre_event_dormant = [h for h in dormant_hits if h["step"] <= event]
    sc2_pass = bool(pre_event_dormant)
    sc3_pass = (sc1_pass
                and sc1_lead is not None
                and sc1_lead >= ent["lead_steps"] + SC3_MARGIN_OVER_ENTROPY)
    result["entropy_precursor"] = ent
    result["verdicts"] = {
        "censored": False,
        "sc1_q_deviates_early": sc1_pass,
        "sc1_onset_step": sc1_onset,
        "sc1_lead_steps": sc1_lead,
        "sc1_threshold": SC1_ERANK_REL_THRESHOLD,
        "sc2_dormant_nonzero_pre_event": sc2_pass,
        "sc2_hits": pre_event_dormant,
        "sc3_q_beats_entropy_by_5": sc3_pass,
        "entropy_lead_steps": ent["lead_steps"],
    }
    return result


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    per_run = []
    for arg in sys.argv[1:]:
        res = judge_run(Path(arg))
        (Path(arg) / "forensics.json").write_text(
            json.dumps(res, indent=1), encoding="utf-8")
        per_run.append(res)
        v = res["verdicts"]
        e = res["event"]
        print(f"== {Path(arg).name}")
        if v.get("censored"):
            print(f"   censored (no collapse; capped at "
                  f"{e.get('censored_at')})")
        else:
            print(f"   event E={e['event_step']} (trigger {e['trigger_step']})"
                  f"  entropy lead={v['entropy_lead_steps']}"
                  f"  SC1={v['sc1_q_deviates_early']}"
                  f" (lead={v['sc1_lead_steps']})"
                  f"  SC2={v['sc2_dormant_nonzero_pre_event']}"
                  f"  SC3={v['sc3_q_beats_entropy_by_5']}")
    collapsed = [r for r in per_run if not r["verdicts"].get("censored")]
    combined = {
        "n_runs": len(per_run),
        "n_collapsed": len(collapsed),
        "cross_replicate": {
            sc: sum(1 for r in collapsed if r["verdicts"][key]) >= 2
            for sc, key in [
                ("sc1_supported", "sc1_q_deviates_early"),
                ("sc2_supported", "sc2_dormant_nonzero_pre_event"),
                ("sc3_supported", "sc3_q_beats_entropy_by_5")]
        } if len(collapsed) >= 2 else
        {"note": "needs >=2 collapsed replicates for the pre-registered "
                 ">=2-of-3 rule"},
    }
    print(json.dumps(combined, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
