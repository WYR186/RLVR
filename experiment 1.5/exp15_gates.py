#!/usr/bin/env python3
"""Go/no-go gates for experiment 1.5 on the Windows RTX 4070 box.

Each subcommand prints a single unambiguous verdict line (PASS / INVESTIGATE /
STOP) plus the evidence behind it, so the driving agent never has to eyeball
raw JSONL. Verdict semantics are defined in WIN4070_EXP15_GUIDE.md.

  python "experiment 1.5\\exp15_gates.py" rundir
  python "experiment 1.5\\exp15_gates.py" sentinel
  python "experiment 1.5\\exp15_gates.py" ckpt0
  python "experiment 1.5\\exp15_gates.py" bridge

All subcommands default to the pre-registered cuda run dir
(exp15_cuda_grpo_gsm8k_e73704296e47); pass --run-dir to override (smoke).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR))

import exp1_5_lib as lib  # noqa: E402

EXPECTED_RUN_NAME = "exp15_cuda_grpo_gsm8k_e73704296e47"
PILOT_V2_RUN = (lib.PILOT / "outputs" / "local_cuda_grpo_gsm8k_e9b0b52aab6c")

# Per-25-update sentinel bands scale with Stage-A learning rate. The completed
# fp32 pilot at lr=1e-6 moved roughly 5e-7..1e-6; the broken pure-bf16 run moved
# roughly 3e-9. The original v1.5 lr=1e-5 bands were 10x these base thresholds.
SENTINEL_BASE_LR = 1e-6
SENTINEL_BASE_STOP_BELOW = 1e-8
SENTINEL_BASE_INVESTIGATE_BELOW = 1e-7

CKPT0_ERANK_ATOL = 0.01             # same machine/dtype: pilot matched to 4 dp
BRIDGE_LEGACY_ATOL = 0.02           # informational band for legacy-100 ckpt-0


def _default_run_dir() -> Path:
    return lib.PILOT / "outputs" / EXPECTED_RUN_NAME


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def gate_rundir(run_dir: Path, cfg: dict) -> int:
    from scripts.run_local_pipeline import EXECUTION_PROFILES

    computed, _ = lib.stage_a_run_dir(cfg, "cuda", EXECUTION_PROFILES["cuda"])
    expected = cfg.get("expected_run_name", EXPECTED_RUN_NAME)
    print(f"computed run dir: {computed.name}")
    if computed.name == expected and run_dir.name == expected:
        print("VERDICT: PASS — matches the pre-registered run dir")
        return 0
    print(f"VERDICT: STOP — expected {expected}; config or execution "
          "profile drifted. Do not train; ask Aaron.")
    return 2


def gate_sentinel(run_dir: Path, cfg: dict) -> int:
    path = run_dir / "update_sentinel.jsonl"
    if not path.exists():
        print(f"VERDICT: INVESTIGATE — no sentinel file yet at {path} "
              "(first row lands at step 25, ~40 min in)")
        return 1
    rows = _read_jsonl(path)
    for row in rows:
        print(f"step {row['step']:>4}  rel_change_window={row['rel_change_window']:.3e}  "
              f"effective={row['updates_effective']}")
    last = rows[-1]
    learning_rate = float(cfg["stage_a"]["learning_rate"])
    scale = learning_rate / SENTINEL_BASE_LR
    stop_below = SENTINEL_BASE_STOP_BELOW * scale
    investigate_below = SENTINEL_BASE_INVESTIGATE_BELOW * scale
    expected_low = 1e-7 * scale
    expected_high = 1e-5 * scale
    rel = float(last["rel_change_window"])
    if not last["updates_effective"] or rel < stop_below:
        print(f"VERDICT: STOP — window {rel:.3e} < {stop_below:.0e} "
              "(v1 no-op territory). Ctrl+C, keep artifacts, report.")
        return 2
    if rel < investigate_below:
        print(f"VERDICT: INVESTIGATE — window {rel:.3e} is an order of "
              f"magnitude under the lr={learning_rate:g} expectation "
              f"({expected_low:.0e} .. {expected_high:.0e}). "
              "Pause after the current window and ask before continuing.")
        return 1
    print(f"VERDICT: PASS — window {rel:.3e} within/above the expected "
          f"lr={learning_rate:g} band ({expected_low:.0e} .. {expected_high:.0e})")
    return 0


def gate_ckpt0(run_dir: Path) -> int:
    mine_path = run_dir / "measurements" / "metrics_ckpt0.json"
    ref_path = PILOT_V2_RUN / "measurements" / "metrics_ckpt0.json"
    if not mine_path.exists():
        print(f"VERDICT: INVESTIGATE — {mine_path} not written yet (run phase 2)")
        return 1
    if not ref_path.exists():
        print(f"VERDICT: INVESTIGATE — pilot reference missing: {ref_path} "
              "(fresh clone without pilot artifacts? git pull)")
        return 1
    mine_payload = json.loads(mine_path.read_text(encoding="utf-8"))
    ref_payload = json.loads(ref_path.read_text(encoding="utf-8"))
    mine_contract = mine_payload.get("measurement_contract", {})
    ref_contract = ref_payload.get("measurement_contract", {})
    mine_dtype = mine_contract.get("model_dtype")
    ref_dtype = ref_contract.get("model_dtype")
    print(f"measurement dtype: exp1.5={mine_dtype}, pilot={ref_dtype}")
    if mine_dtype != ref_dtype:
        print("VERDICT: STOP — ckpt-0 measurement dtype differs from the "
              "pilot reference. Preserve these measurements, rerun Phase 2 "
              "with the pilot dtype, and do not continue to Phase 3.")
        return 2
    mine = mine_payload["per_layer"]
    ref = ref_payload["per_layer"]
    worst = 0.0
    for layer, ref_vals in ref.items():
        d = abs(mine[layer]["erank"] - ref_vals["erank"])
        worst = max(worst, d)
        print(f"{layer}: erank {mine[layer]['erank']:.4f} vs pilot "
              f"{ref_vals['erank']:.4f} (|Δ|={d:.4f}); dormant τ=.025 "
              f"{mine[layer]['dormant_frac_tau0.025']}")
    if worst <= CKPT0_ERANK_ATOL:
        print(f"VERDICT: PASS — ckpt-0 reproduces the pilot measurement "
              f"(max |Δerank| {worst:.4f} ≤ {CKPT0_ERANK_ATOL})")
        return 0
    print(f"VERDICT: STOP — ckpt-0 erank deviates {worst:.4f} > "
          f"{CKPT0_ERANK_ATOL} from the pilot's committed values. The probe/"
          "dtype/layer setup drifted; measurements would not be comparable. "
          "Do not continue to phase 3; report.")
    return 2


def gate_bridge(run_dir: Path) -> int:
    mine_path = run_dir / "adaptation_seed42" / "ckpt-0" / "baseline.json"
    ref_path = PILOT_V2_RUN / "adaptation" / "ckpt-0" / "baseline.json"
    if not mine_path.exists():
        print(f"VERDICT: INVESTIGATE — {mine_path} not written yet "
              "(runs during the first phase-3 job)")
        return 1
    mine = json.loads(mine_path.read_text(encoding="utf-8"))
    legacy = mine.get("acc_before_legacy100")
    print(f"exp1.5 ckpt-0: acc_before(300q)={mine['acc_before']:.4f}, "
          f"legacy-100 sub-score={legacy:.4f}")
    if not ref_path.exists():
        print("VERDICT: PASS — (pilot reference baseline not present locally; "
              "legacy comparison skipped)")
        return 0
    ref = json.loads(ref_path.read_text(encoding="utf-8"))["acc_before"]
    d = abs(legacy - ref)
    print(f"pilot ckpt-0 acc_before(100q)={ref:.4f}  |Δ|={d:.4f}")
    if d <= BRIDGE_LEGACY_ATOL:
        print(f"VERDICT: PASS — legacy-100 bridge within ±{BRIDGE_LEGACY_ATOL} "
              "of the pilot baseline (batching may shift a borderline item)")
        return 0
    print(f"VERDICT: INVESTIGATE — legacy-100 bridge off by {d:.4f} > "
          f"{BRIDGE_LEGACY_ATOL}; record it in compute_log notes and flag to "
          "Aaron before phase 4 (not a hard stop: the 300-q primary is "
          "self-consistent within exp 1.5).")
    return 1


def main() -> int:
    gates = {"rundir": gate_rundir, "sentinel": gate_sentinel,
             "ckpt0": gate_ckpt0, "bridge": gate_bridge}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gate", choices=sorted(gates))
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=lib.CONFIG_PATH)
    args = parser.parse_args()
    cfg = lib.load_config(args.config)
    if args.run_dir is not None:
        run_dir = args.run_dir
    else:
        from scripts.run_local_pipeline import EXECUTION_PROFILES
        run_dir, _ = lib.stage_a_run_dir(cfg, "cuda", EXECUTION_PROFILES["cuda"])
    if args.gate == "rundir":
        return gate_rundir(run_dir, cfg)
    if args.gate == "sentinel":
        return gate_sentinel(run_dir, cfg)
    return gates[args.gate](run_dir)


if __name__ == "__main__":
    raise SystemExit(main())
