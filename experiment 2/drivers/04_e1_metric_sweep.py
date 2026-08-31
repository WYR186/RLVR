"""E1 — metric re-measurement sweep over the existing 7B Stage-A checkpoints.

Implements `SPEC_E1_METRIC_REMEASUREMENT.md` §4 execution order. No training,
no Stage B, no team decision — it re-reads the three adapters already in Drive.

Run AFTER `00_restore_from_drive.py` (which itself follows cells 1-6 of
`colab/00_phase0_selfcontained.ipynb`). Steps 1-3 involve no generation and are
the bulk of the value; V1 (step 4/5) generates and is opt-in.

    python 04_e1_metric_sweep.py                 # steps 1-3
    python 04_e1_metric_sweep.py --v1a           # + V1a (frozen ckpt-0 continuations)
    python 04_e1_metric_sweep.py --v1a --v1b     # + V1b (on-policy, NOT comparable)

Spec §6 gate 1: if the reference arm does not reproduce §2, this stops. It does
not "accept and note it".
"""
import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

EXP2 = Path("/content/RLVR/experiment 2")
if not EXP2.is_dir():                      # local dev / dry run
    EXP2 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXP2))

import numpy as np  # noqa: E402

import src.guru_data as guru_data  # noqa: E402
import src.pipeline as pipeline  # noqa: E402
from src import e1_sweep  # noqa: E402

RUN = "exp2_colab_guru_math7b_instruct_group8_e33527592dd9"
OUT = Path("/content/outputs") / RUN / "measurements" / "e1_sweep"
CKPTS = Path("/content/ckpts")
CONFIG_NAME = "exp2_colab_config_mvp_instruct.json"
CONFIG_SHA12 = "e33527592dd9"
SPLITS_NAME = "exp2_colab_splits_instruct.json"
SPLITS_SHA16 = {
    "stage_a_train_ids": ("a06fe6b80e7d40ca", 54251),
    "stage_b_train_ids": ("df8623cf009e0690", 1132),
    "stage_b_eval_ids": ("8ee975c7089dc72a", 300),
    "probe_stage_a_topup_ids": ("1e61252e7b54793e", 4096),
}

# Spec §2 — the reference arm must reproduce these exactly or the sweep stops.
REFERENCE = {
    0:   {"layer5": {"erank": 1127.4155}, "layer14": {"erank": 1281.0450},
          "layer26": {"erank": 1426.0597}},
    50:  {"layer5": {"erank": 1128.2271}, "layer14": {"erank": 1287.8799},
          "layer26": {"erank": 1433.8730}},
    100: {"layer5": {"erank": 1128.1812}, "layer14": {"erank": 1287.8093},
          "layer26": {"erank": 1432.5480}},
}
REFERENCE_SCORE_MIN = {0: {"layer5": 0.1604, "layer14": 0.4148, "layer26": 0.1606}}


def sha16(ids):
    return hashlib.sha256(
        json.dumps(sorted(ids), separators=(",", ":")).encode()).hexdigest()[:16]


def load_provenance():
    """Spec §6 gate 3 — assert config hash and all four split hashes at load."""
    cfg_path = EXP2 / CONFIG_NAME
    got = hashlib.sha256(cfg_path.read_text().encode()).hexdigest()[:12]
    if got != CONFIG_SHA12:
        raise SystemExit(
            f"config hash mismatch: {cfg_path} is {got}, expected {CONFIG_SHA12}. "
            "Run 00_restore_from_drive.py first — the Drive copy is authoritative "
            "and the committed copy is NOT the one this run used.")
    config = json.loads(cfg_path.read_text())

    splits = json.loads((EXP2 / "data" / SPLITS_NAME).read_text())
    for key, (want, n) in SPLITS_SHA16.items():
        if sha16(splits[key]) != want or len(splits[key]) != n:
            raise SystemExit(f"split {key} failed its hash/length check")
    print(f"provenance OK: config {got}, 4/4 split hashes verified")
    return config, splits


def probe_prompts(config, splits):
    rows = guru_data.dataset_rows_for(
        "probe", None, splits, config["model_id"], config.get("model_revision"),
        config["dataset"]["revision"])
    prompts = [r["prompt"] for r in rows]
    print(f"probe: {len(prompts)} prompts")
    return prompts


# ---------------------------------------------------------------------------
# step 1 — reference arm (the gate)
# ---------------------------------------------------------------------------

def run_reference_arm(config, prompts, steps):
    m = config["measurement"]
    verdicts, results = {}, {}
    for step in steps:
        print(f"\n[reference arm] ckpt-{step} ...", flush=True)
        q = pipeline.measure_checkpoint_q(
            config["model_id"], config["peft"], str(CKPTS / f"ckpt-{step}"),
            prompts, layers=tuple(m["layers"]),
            revision=config.get("model_revision"), batch_size=m["batch_size"])
        results[step] = q
        v = e1_sweep.check_reference_arm(q, REFERENCE[step])
        verdicts[step] = v
        for layer, row in v["per_layer"].items():
            print(f"  {layer}: erank {row['erank_measured']:.6f} "
                  f"(expected {row['erank_expected']:.4f}, "
                  f"delta {row['erank_delta']:.2e}) "
                  f"{'OK' if row['erank_ok'] else 'DRIFT'}")
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"reference_arm_ckpt{step}.json").write_text(
            json.dumps({"checkpoint": step, "metrics": q, "gate": v},
                       indent=1, default=str))

    if not all(v["passed"] for v in verdicts.values()):
        raise SystemExit(
            "\nREFERENCE-ARM GATE FAILED (spec §6 gate 1).\n"
            "The environment drifted from the run that produced "
            "FINDING_Q_METRICS_7B_INSTRUCT.md, so nothing downstream is "
            "interpretable. Investigate before proceeding; do not accept and note it.\n"
            f"{json.dumps(verdicts, indent=1)}")
    print("\nreference-arm gate PASSED at every checkpoint")
    return results


# ---------------------------------------------------------------------------
# steps 2-3 — V2 + V3 + V5 + V6 on one instrumented sweep, then V4 post-processing
# ---------------------------------------------------------------------------

def run_sweep(config, prompts, steps, *, all_layers: bool,
              continuation_probe=None, variant_label="V2V3V4V5V6",
              contract_overrides=None):
    import torch

    from src.pipeline import build_peft_model, unwrap_for_hooks

    m = config["measurement"]
    ref_layers = tuple(m["layers"])
    n_blocks = 28
    layers = tuple(range(n_blocks)) if all_layers else ref_layers
    # V5a is a depth profile, not a request to repeat every expensive V5b/V5c/V6
    # spectrum at 28 layers.  It retains residual/last at every block, while
    # streaming reference dormancy at every block and the full V2/V3 tensors at
    # the three registered layers (also regenerating legacy-overwritten vectors).
    spectrum_layers = () if variant_label == "V5a" else ref_layers
    depth_profile_layers = layers if variant_label == "V5a" else ()
    probe_prefixes = () if variant_label == "V5a" else e1_sweep.PROBE_PREFIXES

    pilot = pipeline._pilot()
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_pilot_metrics_e1", EXP2.parent / "eaaj-pilot" / "src" / "metrics.py")
    pm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pm)
    del pilot

    records = {}
    for step in steps:
        print(f"\n[{variant_label}] ckpt-{step} ...", flush=True)
        model, tokenizer = build_peft_model(
            config["model_id"], config["peft"],
            "cuda" if torch.cuda.is_available() else "cpu",
            revision=config.get("model_revision"),
            adapter_path=str(CKPTS / f"ckpt-{step}"))
        unwrapped = unwrap_for_hooks(model)

        this_prompts = prompts
        starts = None
        if continuation_probe is not None:
            conts = continuation_probe(model, tokenizer, step)
            base = prompts[:e1_sweep.V1_N_PROMPTS]
            this_prompts = e1_sweep.build_v1_probe(base, conts)
            starts = e1_sweep.continuation_start_indices(
                tokenizer, base, this_prompts, max_length=m.get("max_length", 512))

        pooled, dormancy, meta = e1_sweep.collect_e1_activations(
            unwrapped, tokenizer, this_prompts, layers=layers,
            batch_size=m["batch_size"], max_length=512,
            spectrum_layers=spectrum_layers,
            depth_profile_layers=depth_profile_layers,
            per_prompt_layers=ref_layers,
            full_variant_layers=ref_layers, continuation_starts=starts)

        rec = e1_sweep.build_variant_records(
            pooled, dormancy, meta, pm.spectrum_metrics, pm.anisotropy_metrics,
            probe_prefixes=probe_prefixes, score_dir=OUT / "scores",
            checkpoint=step, variant_label=variant_label)
        rec["variant"] = variant_label
        rec["checkpoint"] = step
        rec["measurement_contract"] = e1_sweep.measurement_contract(
            model_dtype="bfloat16_base_float32_lora_adapter",
            max_length=512, batch_size=m["batch_size"],
            n_probe=meta["n_probe"], layers=layers,
            overrides=contract_overrides)
        rec["provenance"] = {
            "config_hash": CONFIG_SHA12,
            "splits_sha16": {k: v[0] for k, v in SPLITS_SHA16.items()},
            "adapter_path": str(CKPTS / f"ckpt-{step}"),
            "reference_arm_gate": "pass",
        }
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"{variant_label}_ckpt{step}.json").write_text(
            json.dumps(rec, indent=1, default=str))
        records[step] = rec
        print(f"  wrote {variant_label}_ckpt{step}.json "
              f"(hook check max abs err {meta['gated_mlp_hook_check_max_abs_err']:.2e})")

        del model, tokenizer, unwrapped, pooled, dormancy
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1a", action="store_true",
                    help="V1a: frozen ckpt-0 continuations (comparable)")
    ap.add_argument("--v1b", action="store_true",
                    help="V1b: on-policy continuations (NOT comparable across ckpts)")
    ap.add_argument("--all-layers", action="store_true",
                    help="V5a: depth profile over all 28 blocks")
    ap.add_argument("--only-v5a", action="store_true",
                    help="run the reference gate and corrected all-28-layer V5a "
                         "supplement only; also regenerates score vectors under "
                         "variant-scoped filenames")
    ap.add_argument("--steps", type=int, nargs="+", default=[0, 50, 100])
    ap.add_argument("--skip-base", action="store_true",
                    help="skip steps 1-3 (reference arm + V2/V3/V4/V5/V6) because a "
                         "previous invocation in this session already produced them; "
                         "only legitimate when the gate PASSED in that same runtime")
    args = ap.parse_args()

    if args.skip_base and not (args.v1a or args.v1b):
        raise SystemExit("--skip-base only makes sense together with --v1a/--v1b")
    if args.only_v5a and (args.skip_base or args.v1a or args.v1b):
        raise SystemExit("--only-v5a cannot be combined with --skip-base/--v1a/--v1b")

    config, splits = load_provenance()
    prompts = probe_prompts(config, splits)

    records = {}
    if args.only_v5a:
        run_reference_arm(config, prompts, args.steps)
        records = run_sweep(config, prompts, args.steps, all_layers=True,
                            variant_label="V5a",
                            contract_overrides={
                                "depth_profile": "all_28_decoder_blocks",
                                "spectrum_tensor": "residual_stream_output",
                                "spectrum_pooling": "last_non_padding_token",
                                "probe_size_sweep": "not_run_not_part_of_V5a",
                            })
        csv_path = e1_sweep.write_summary_csv(records, OUT / "summary_v5a.csv")
        print(f"\nV5a supplement complete. Outputs under {OUT}")
        print(f"  summary: {csv_path}")
        print(f"  variant-scoped score vectors: {OUT / 'scores'}")
        return
    elif args.skip_base:
        # Spec §6 gate 1 still has to have been satisfied — just not twice in one
        # session. Refuse unless this run's own gate artifacts are on disk.
        missing = [s for s in args.steps
                   if not (OUT / f"reference_arm_ckpt{s}.json").exists()]
        if missing:
            raise SystemExit(
                f"--skip-base but no reference_arm_ckpt{missing} in {OUT}: the "
                "gate has not run here, so nothing downstream is interpretable.")
        for s in args.steps:
            v = json.loads((OUT / f"reference_arm_ckpt{s}.json").read_text())["gate"]
            if not v["passed"]:
                raise SystemExit(f"--skip-base but the ckpt-{s} gate did not pass")
        print(f"--skip-base: reference-arm gate already PASSED for {args.steps}")
    else:
        run_reference_arm(config, prompts, args.steps)      # step 1 — GATE
        records = run_sweep(config, prompts, args.steps,    # steps 2-3
                            all_layers=args.all_layers)

    if args.v1a:                                            # step 4
        frozen = {}

        def ckpt0_continuations(model, tokenizer, step):
            if not frozen:
                print("  generating frozen ckpt-0 continuations ...", flush=True)
                frozen["c"] = e1_sweep.generate_continuations(
                    model, tokenizer, prompts[:e1_sweep.V1_N_PROMPTS])
            return frozen["c"]

        if args.steps[0] != 0:
            raise SystemExit("V1a needs ckpt-0 first to freeze its continuations")
        records.update(run_sweep(
            config, prompts, args.steps, all_layers=False,
            continuation_probe=ckpt0_continuations, variant_label="V1a",
            contract_overrides=e1_sweep.v1_contract_overrides(
                on_policy=False, n_probe=e1_sweep.V1_N_PROMPTS, hidden_size=3584)))

    if args.v1b:                                            # step 5
        def own_continuations(model, tokenizer, step):
            print(f"  generating on-policy continuations for ckpt-{step} ...",
                  flush=True)
            return e1_sweep.generate_continuations(
                model, tokenizer, prompts[:e1_sweep.V1_N_PROMPTS])

        run_sweep(config, prompts, args.steps, all_layers=False,
                  continuation_probe=own_continuations, variant_label="V1b",
                  contract_overrides=e1_sweep.v1_contract_overrides(
                      on_policy=True, n_probe=e1_sweep.V1_N_PROMPTS,
                      hidden_size=3584))

    csv_path = e1_sweep.write_summary_csv(records, OUT / "summary.csv")
    print(f"\nE1 sweep complete. Outputs under {OUT}")
    print(f"  summary: {csv_path}")
    print(f"  per-unit score vectors: {OUT / 'scores'}")


if __name__ == "__main__":
    main()
