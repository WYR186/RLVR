#!/usr/bin/env python3
"""Expansion-gate evaluation for experiment 1.6 (pre-registered).

Decides, from committed artifacts only, whether the endpoint-first phase 3
may be expanded to the full 6-checkpoint grid:

  G-A  late-window Q displacement: |mean(erank_L12 @ ckpt 300/400/500) /
       erank_L12 @ ckpt-0 - 1| >= 0.075
  G-B  endpoint outcome drop: mean-of-3-seed delta(ckpt-0) minus
       mean-of-3-seed delta(ckpt-500) >= 0.05

Verdict EXPAND iff G-A or G-B passes; STOP otherwise; INVESTIGATE while
inputs are missing. Writes exp16_gate_eval.json into the run dir (additive
evidence) and prints one unambiguous verdict line, matching the
exp15_gates.py convention. Thresholds live in exp1_6_config.json
analysis.expansion_gates — this script only reads them.

  python "experiment 1.5\\exp1_6_gate_eval.py"
  python "experiment 1.5\\exp1_6_gate_eval.py" --run-dir <dir>   (smoke)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR))

import exp1_5_lib as lib  # noqa: E402


def erank_l12(run_dir: Path, ckpt: int) -> float | None:
    p = run_dir / "measurements" / f"metrics_ckpt{ckpt}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))["per_layer"]["layer12"]["erank"]


def seed_deltas(run_dir: Path, ckpt: int, seeds: list[int]) -> list[float]:
    out = []
    for seed in seeds:
        p = run_dir / f"adaptation_seed{seed}" / f"ckpt-{ckpt}" / "summary.json"
        if p.exists():
            out.append(json.loads(p.read_text(encoding="utf-8"))["delta_acc"])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=EXP_DIR / "exp1_6_config.json")
    parser.add_argument("--run-dir", type=Path, default=None)
    args = parser.parse_args()
    cfg = lib.load_config(args.config)
    gates = cfg["analysis"]["expansion_gates"]
    if args.run_dir is not None:
        run_dir = args.run_dir
    else:
        from scripts.run_local_pipeline import EXECUTION_PROFILES
        run_dir, _ = lib.stage_a_run_dir(cfg, "cuda", EXECUTION_PROFILES["cuda"])

    late_ckpts = gates["ga_late_window_checkpoints"]
    ga_min = float(gates["ga_erank_l12_rel_change_min"])
    gb_min = float(gates["gb_endpoint_mean_delta_drop_min"])
    seeds = cfg["adaptation"]["seeds"]

    base = erank_l12(run_dir, 0)
    late = [erank_l12(run_dir, c) for c in late_ckpts]
    missing_q = [c for c, v in zip([0] + late_ckpts, [base] + late) if v is None]
    ga_rel = None
    ga_pass = None
    if not missing_q:
        ga_rel = sum(late) / len(late) / base - 1.0
        ga_pass = abs(ga_rel) >= ga_min
        print(f"G-A late-window mean erank_L12 rel change: {ga_rel:+.4f} "
              f"(|x| >= {ga_min} to pass) -> {'PASS' if ga_pass else 'FAIL'}")
    else:
        print(f"G-A: measurements missing for ckpts {missing_q} (run phase 2)")

    d0 = seed_deltas(run_dir, 0, seeds)
    d500 = seed_deltas(run_dir, 500, seeds)
    gb_drop = None
    gb_pass = None
    if len(d0) == len(seeds) and len(d500) == len(seeds):
        gb_drop = sum(d0) / len(d0) - sum(d500) / len(d500)
        gb_pass = gb_drop >= gb_min
        print(f"G-B endpoint mean-delta drop (ckpt0 - ckpt500): {gb_drop:+.4f} "
              f"(>= {gb_min} to pass) -> {'PASS' if gb_pass else 'FAIL'}")
    else:
        print(f"G-B: endpoint probe incomplete "
              f"(ckpt-0: {len(d0)}/{len(seeds)} seeds, "
              f"ckpt-500: {len(d500)}/{len(seeds)})")

    if ga_pass is None or gb_pass is None:
        verdict, code = "INVESTIGATE", 1
        detail = "inputs missing — finish phase 2 and the endpoint probe first"
    elif ga_pass or gb_pass:
        verdict, code = "EXPAND", 0
        detail = ("relaunch phase 3 with exp1_6_config_fullgrid.json "
                  "(same run dir) for the remaining checkpoints")
    else:
        verdict, code = "STOP", 2
        detail = ("neither gate passed: 3e-6 moved neither late-window Q nor "
                  "the endpoint outcome. Do NOT expand; commit artifacts and "
                  "report to the team (dose-response point recorded)")

    payload = {
        "evaluated_unix": time.time(),
        "config": str(args.config),
        "ga_late_window_rel_change": ga_rel,
        "ga_threshold": ga_min, "ga_pass": ga_pass,
        "gb_endpoint_mean_delta_drop": gb_drop,
        "gb_threshold": gb_min, "gb_pass": gb_pass,
        "seed_deltas_ckpt0": d0, "seed_deltas_ckpt500": d500,
        "verdict": verdict,
    }
    out = run_dir / "exp16_gate_eval.json"
    if run_dir.exists():
        out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"VERDICT: {verdict} — {detail}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
