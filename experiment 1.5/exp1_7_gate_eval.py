#!/usr/bin/env python3
"""Cross-trajectory expansion gate for experiment 1.7.

The three Stage-A trajectories are independent experimental replicates.
After Phase 2, each complete trajectory receives only the seed-42 endpoint
probe. Stage-B seeds 43/44 are unlocked iff the replicated Q gate or the
replicated endpoint-outcome gate passes.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR))

import exp1_5_lib as lib  # noqa: E402

DEFAULT_CONFIGS = [
    EXP_DIR / "exp1_7_config_seed42.json",
    EXP_DIR / "exp1_7_config_seed43.json",
    EXP_DIR / "exp1_7_config_seed44.json",
]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metric(run_dir: Path, checkpoint: int) -> float | None:
    path = run_dir / "measurements" / f"metrics_ckpt{checkpoint}.json"
    if not path.exists():
        return None
    return float(read_json(path)["per_layer"]["layer12"]["erank"])


def delta(run_dir: Path, checkpoint: int, seed: int) -> float | None:
    path = (run_dir / f"adaptation_seed{seed}" / f"ckpt-{checkpoint}"
            / "summary.json")
    if not path.exists():
        return None
    return float(read_json(path)["delta_acc"])


def same_sign_count(values: list[float], reference: float) -> int:
    if reference > 0:
        return sum(value > 0 for value in values)
    if reference < 0:
        return sum(value < 0 for value in values)
    return sum(value == 0 for value in values)


def run_dir_for(cfg: dict) -> Path:
    from scripts.run_local_pipeline import EXECUTION_PROFILES

    return lib.stage_a_run_dir(
        cfg, "cuda", EXECUTION_PROFILES["cuda"])[0]


def evaluate(config_paths: list[Path], run_dirs: list[Path] | None = None) -> dict:
    configs = [lib.load_config(path) for path in config_paths]
    if run_dirs is None:
        run_dirs = [run_dir_for(cfg) for cfg in configs]
    if len(run_dirs) != len(configs):
        raise ValueError("--run-dir count must equal --config count")

    gate_cfg = configs[0]["analysis"]["exp1_7_gates"]
    late_ckpts = [int(x) for x in gate_cfg["late_window_checkpoints"]]
    q_min = float(gate_cfg["q_median_abs_rel_change_min"])
    outcome_min = float(gate_cfg["endpoint_median_delta_drop_min"])
    probe_seed = int(gate_cfg["screen_adaptation_seed"])
    min_replicates = int(gate_cfg["min_complete_stage_a_replicates"])
    collapse_stop_n = int(gate_cfg["collapse_stop_replicates"])

    rows = []
    for path, cfg, run_dir in zip(config_paths, configs, run_dirs):
        seed = int(cfg["seed"])
        safety = run_dir / "safety_stop.json"
        hard_cap = run_dir / "hard_cap_stop.json"
        ckpt500 = run_dir / "ckpt-500" / "config.json"
        complete = ckpt500.exists() and (run_dir / "phase1_complete.json").exists()
        row = {
            "stage_a_seed": seed,
            "config": str(path),
            "run_dir": str(run_dir),
            "stage_a_complete": complete,
            "safety_stop": safety.exists(),
            "hard_cap_stop": hard_cap.exists(),
            "q_shift": None,
            "delta_ckpt0_seed42": None,
            "delta_ckpt500_seed42": None,
            "endpoint_drop": None,
        }
        if complete:
            base = metric(run_dir, 0)
            late = [metric(run_dir, ckpt) for ckpt in late_ckpts]
            if base is not None and all(value is not None for value in late):
                row["q_shift"] = statistics.mean(late) / base - 1.0
            d0 = delta(run_dir, 0, probe_seed)
            d500 = delta(run_dir, 500, probe_seed)
            row["delta_ckpt0_seed42"] = d0
            row["delta_ckpt500_seed42"] = d500
            if d0 is not None and d500 is not None:
                row["endpoint_drop"] = d0 - d500
        rows.append(row)

    complete_rows = [row for row in rows if row["stage_a_complete"]]
    safety_n = sum(row["safety_stop"] for row in rows)
    q_values = [row["q_shift"] for row in complete_rows
                if row["q_shift"] is not None]
    endpoint_values = [row["endpoint_drop"] for row in complete_rows
                       if row["endpoint_drop"] is not None]
    required_consistent = min_replicates

    q_median = statistics.median(q_values) if q_values else None
    q_consistent = (same_sign_count(q_values, q_median)
                    if q_median is not None else 0)
    q_pass = (
        len(q_values) == len(complete_rows)
        and len(q_values) >= min_replicates
        and abs(q_median) >= q_min
        and q_consistent >= required_consistent
    )

    outcome_median = (statistics.median(endpoint_values)
                      if endpoint_values else None)
    outcome_positive = sum(value > 0 for value in endpoint_values)
    outcome_pass = (
        len(endpoint_values) == len(complete_rows)
        and len(endpoint_values) >= min_replicates
        and outcome_median >= outcome_min
        and outcome_positive >= required_consistent
    )

    if safety_n >= collapse_stop_n:
        verdict = "STOP_COLLAPSE"
        code = 2
        detail = (f"{safety_n}/{len(rows)} Stage-A trajectories safety-stopped; "
                  "do not infer a ckpt-500 endpoint")
    elif len(complete_rows) < min_replicates:
        verdict = "INVESTIGATE"
        code = 1
        detail = (f"only {len(complete_rows)}/{len(rows)} Stage-A trajectories "
                  "are complete")
    elif len(q_values) < len(complete_rows):
        verdict = "INVESTIGATE"
        code = 1
        detail = "Phase-2 Q measurements are incomplete"
    elif len(endpoint_values) < len(complete_rows):
        verdict = "PROBE"
        code = 1
        detail = ("run the seed-42 ckpt-0/500 adaptation probe for every "
                  "complete Stage-A trajectory")
    elif q_pass or outcome_pass:
        verdict = "EXPAND"
        code = 0
        detail = "run Stage-B seeds 43/44 on both endpoints"
    else:
        verdict = "STOP"
        code = 2
        detail = "neither replicated gate passed; do not expand Stage B"

    return {
        "evaluated_unix": time.time(),
        "configs": [str(path) for path in config_paths],
        "replicates": rows,
        "complete_stage_a_replicates": len(complete_rows),
        "safety_stop_replicates": safety_n,
        "q_median_rel_change": q_median,
        "q_threshold": q_min,
        "q_same_sign_count": q_consistent,
        "q_pass": q_pass,
        "endpoint_median_delta_drop": outcome_median,
        "endpoint_threshold": outcome_min,
        "endpoint_positive_count": outcome_positive,
        "endpoint_pass": outcome_pass,
        "verdict": verdict,
        "detail": detail,
        "exit_code": code,
    }


def print_report(payload: dict) -> None:
    for row in payload["replicates"]:
        q = ("missing" if row["q_shift"] is None
             else f"{row['q_shift']:+.4f}")
        drop = ("missing" if row["endpoint_drop"] is None
                else f"{row['endpoint_drop']:+.4f}")
        print(
            f"Stage-A seed {row['stage_a_seed']}: "
            f"complete={row['stage_a_complete']} "
            f"safety_stop={row['safety_stop']} "
            f"q_shift={q} endpoint_drop={drop}"
        )
    if payload["q_median_rel_change"] is not None:
        print(
            "Q gate: median late-window rel change "
            f"{payload['q_median_rel_change']:+.4f}, "
            f"same-sign={payload['q_same_sign_count']}, "
            f"|median| >= {payload['q_threshold']:.4f} -> "
            f"{'PASS' if payload['q_pass'] else 'FAIL'}"
        )
    if payload["endpoint_median_delta_drop"] is not None:
        print(
            "Outcome gate: median endpoint delta drop "
            f"{payload['endpoint_median_delta_drop']:+.4f}, "
            f"positive={payload['endpoint_positive_count']}, "
            f"median >= {payload['endpoint_threshold']:.4f} -> "
            f"{'PASS' if payload['endpoint_pass'] else 'FAIL'}"
        )
    print(f"VERDICT: {payload['verdict']} - {payload['detail']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, action="append",
                        dest="configs")
    parser.add_argument("--run-dir", type=Path, action="append",
                        dest="run_dirs")
    parser.add_argument(
        "--output", type=Path,
        default=lib.PILOT / "outputs" / "exp1_7_gate_eval.json")
    args = parser.parse_args()

    config_paths = args.configs or DEFAULT_CONFIGS
    payload = evaluate(config_paths, args.run_dirs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print_report(payload)
    return int(payload["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
