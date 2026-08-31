"""E4 Arms R and N — the ruler and the calibration ladder.

Arm R  Q(base) vs Q(Instruct) on the frozen probe text. The intervention
       separating the two is a full instruction-tuning + RLHF pipeline, i.e.
       enormously larger than 100 rank-16 LoRA updates. It is an
       order-of-magnitude reference point, NOT a controlled dose.
Arm N  Q(Instruct + isotropic noise) at an exact relative Frobenius dose,
       swept over a ladder. This is the controlled axis: it turns "how big
       must a weight change be before Q moves" into a curve, and Arm W's
       number places our Stage-A run on it.

Every arm reuses E1's reductions verbatim (`src/e1_sweep.collect_e1_activations`
plus eaaj-pilot's `spectrum_metrics`), so nothing about the metric changes
between E1 and E4 except which weights are in the model.

Device-agnostic by design. The same command measures Qwen2.5-0.5B on a 16 GB
Mac (free, and a real second scale) and Qwen2.5-7B on an A100.

    # 0.5B, runs on Apple Silicon
    python 08_e4_ruler.py --scale small --probe ../outputs/e4/probe_frozen.json \
        --out ../outputs/e4_small --arms R N

    # 7B, needs a 40 GB+ accelerator
    python 08_e4_ruler.py --scale large --probe ../outputs/e4/probe_frozen.json \
        --out ../outputs/e4_large --arms R N

Each arm writes its own JSON and is skipped if that JSON already exists, so an
interrupted session resumes by re-running the same command.
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

EXP2 = Path("/content/RLVR/experiment 2")
if not EXP2.is_dir():
    EXP2 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXP2))

import importlib.util  # noqa: E402

from src import e1_sweep  # noqa: E402
from src import e4_calibration as e4  # noqa: E402

SCALES = {
    "large": {
        "base": "Qwen/Qwen2.5-7B",
        "instruct": "Qwen/Qwen2.5-7B-Instruct",
        "n_blocks": 28,
        "ref_layers": (5, 14, 26),
        "approx_gb_bf16": 15.2,
    },
    "small": {
        "base": "Qwen/Qwen2.5-0.5B",
        "instruct": "Qwen/Qwen2.5-0.5B-Instruct",
        "n_blocks": 24,
        # Depth-matched to the 7B reference layers (5/28, 14/28, 26/28).
        "ref_layers": (4, 12, 22),
        "approx_gb_bf16": 1.0,
    },
}

# E1's published ckpt-0 residual eranks, for the platform-reproduction record.
E1_CKPT0_ERANK = {"layer5": 1127.4155, "layer14": 1281.0450, "layer26": 1426.0597}


def load_pilot_metrics():
    spec = importlib.util.spec_from_file_location(
        "_pilot_metrics_e4", EXP2.parent / "eaaj-pilot" / "src" / "metrics.py")
    pm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pm)
    return pm


def resolve_device(requested: str):
    import torch
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(requested: str, device: str):
    import torch
    if requested != "auto":
        return getattr(torch, requested)
    # bf16 matches E1's contract on CUDA. On MPS/CPU float32 is the
    # better-tested kernel path and costs nothing at 0.5B; E4-small never
    # compares erank LEVELS with E1, only arms against each other.
    return torch.bfloat16 if device == "cuda" else torch.float32


def memory_gate(scale: dict, device: str, dtype) -> dict:
    """Refuse before downloading 15 GB if the machine plainly cannot hold it."""
    import torch

    bytes_per = 2 if dtype in (torch.bfloat16, torch.float16) else 4
    need_gb = scale["approx_gb_bf16"] * bytes_per / 2
    info = {"device": device, "dtype": str(dtype), "weights_gb": round(need_gb, 2)}

    if device == "cuda":
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        info["accelerator_total_gb"] = round(total, 1)
        info["accelerator_name"] = torch.cuda.get_device_name(0)
        headroom = total - need_gb
    else:
        import shutil  # noqa: F401  (kept for parity with the disk note below)
        try:
            import psutil
            total = psutil.virtual_memory().total / 1e9
        except ImportError:
            import subprocess
            total = int(subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"]).strip()) / 1e9
        info["host_ram_gb"] = round(total, 1)
        headroom = total - need_gb
    info["headroom_gb"] = round(headroom, 1)

    # Activations, the probe matrices and the framework itself need room too.
    if headroom < 6.0:
        raise SystemExit(
            f"MEMORY GATE FAILED.\n"
            f"  weights need ~{need_gb:.1f} GB in {dtype}, "
            f"available ~{info.get('accelerator_total_gb') or info.get('host_ram_gb')} GB, "
            f"headroom {headroom:.1f} GB.\n"
            f"  At least ~6 GB of headroom is needed for activations, the "
            f"pooled probe matrices and the framework.\n"
            f"  Options: --scale small (Qwen2.5-0.5B, ~1 GB) on this machine, "
            f"or run --scale large on an A100/H100 session.\n"
            f"  Do NOT reach for 4-bit quantization here: it perturbs exactly "
            f"the activation statistics this experiment measures.")
    print(f"memory gate OK: {json.dumps(info)}")
    return info


def load_plain_model(model_id: str, device: str, dtype, revision=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, dtype=dtype)
    model.to(device)
    model.eval()
    return model, tok


def tokenizer_identity_gate(tok_a, tok_b, prompts, n_check: int = 64) -> dict:
    """Both Arm R models must turn the frozen text into identical token ids.

    If they do not, the arms differ in their input as well as their weights and
    the comparison is not controlled. That is a stop, not a note.
    """
    sample = prompts[:n_check]
    ids_a = tok_a(sample, add_special_tokens=False)["input_ids"]
    ids_b = tok_b(sample, add_special_tokens=False)["input_ids"]
    mismatched = [i for i, (x, y) in enumerate(zip(ids_a, ids_b)) if x != y]
    same_vocab = tok_a.get_vocab() == tok_b.get_vocab()
    verdict = {
        "n_checked": len(sample),
        "n_mismatched": len(mismatched),
        "identical_vocab": bool(same_vocab),
        "passed": bool(not mismatched and same_vocab),
    }
    if not verdict["passed"]:
        raise SystemExit(
            "TOKENIZER IDENTITY GATE FAILED.\n"
            f"  {len(mismatched)}/{len(sample)} frozen prompts tokenize "
            f"differently; identical vocab: {same_vocab}.\n"
            "  Arm R would then differ in its INPUT as well as its weights, so "
            "any erank difference would be uninterpretable. Stopping.")
    print(f"tokenizer identity gate OK: {json.dumps(verdict)}")
    return verdict


def measure(model, tok, prompts, layers, *, batch_size, pm, label,
            max_length=512):
    t0 = time.time()
    pooled, dormancy, meta = e1_sweep.collect_e1_activations(
        model, tok, prompts, layers=layers, batch_size=batch_size,
        max_length=max_length, spectrum_layers=layers,
        per_prompt_layers=layers, full_variant_layers=layers)
    rec = e1_sweep.build_variant_records(
        pooled, dormancy, meta, pm.spectrum_metrics, pm.anisotropy_metrics,
        probe_prefixes=(), score_dir=None, checkpoint=label,
        variant_label=label)
    rec["wall_seconds"] = round(time.time() - t0, 1)
    del pooled, dormancy
    gc.collect()
    return rec


def free_model(model):
    import torch
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", choices=sorted(SCALES), required=True)
    ap.add_argument("--probe", required=True, help="probe_frozen.json from step 1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", nargs="+", default=["R", "N"], choices=["R", "N"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto",
                    choices=["auto", "bfloat16", "float16", "float32"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--layers", default="auto",
                    help="'auto' uses the scale's depth-matched reference "
                         "layers; or a comma list; or 'all'")
    ap.add_argument("--doses", default=",".join(str(d) for d in e4.DOSE_LADDER))
    ap.add_argument("--force", action="store_true", help="recompute existing arms")
    ap.add_argument("--base-model", default=None,
                    help="override the scale's base model id (a local snapshot "
                         "path works too); pointing both overrides at the SAME "
                         "model is the pipeline smoke test - Arm R must then "
                         "report exactly 0.0000%% change")
    ap.add_argument("--instruct-model", default=None,
                    help="override the scale's instruct model id")
    args = ap.parse_args()

    scale = dict(SCALES[args.scale])
    if args.base_model:
        scale["base"] = args.base_model
    if args.instruct_model:
        scale["instruct"] = args.instruct_model
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    mem = memory_gate(scale, device, dtype)

    if args.layers == "auto":
        layers = tuple(scale["ref_layers"])
    elif args.layers == "all":
        layers = tuple(range(scale["n_blocks"]))
    else:
        layers = tuple(int(x) for x in args.layers.split(","))

    probe = json.loads(Path(args.probe).read_text())
    prompts = probe["prompts"]
    print(f"probe: {len(prompts)} frozen prompts, layers {list(layers)}, "
          f"device {device}, dtype {dtype}")

    pm = load_pilot_metrics()
    contract_common = {
        "e4_scale": args.scale,
        "e4_device": device,
        "e4_dtype": str(dtype),
        "probe_source": "frozen rendered text from 06_e4_freeze_probe.py",
        "adapter": "none - bare model weights",
    }

    def write(label, rec, extra):
        rec["arm"] = label
        rec["measurement_contract"] = e1_sweep.measurement_contract(
            model_dtype=str(dtype), max_length=args.max_length,
            batch_size=args.batch_size, n_probe=rec["meta"]["n_probe"],
            layers=layers, overrides={**contract_common, **extra})
        p = out / f"{label}.json"
        p.write_text(json.dumps(rec, indent=1, default=str))
        print(f"  wrote {p}  ({rec['wall_seconds']} s)")
        return rec

    records = {}

    def existing(label):
        p = out / f"{label}.json"
        if p.is_file() and not args.force:
            print(f"[{label}] already present, skipping (use --force to redo)")
            records[label] = json.loads(p.read_text())
            return True
        return False

    # -- Arm R ------------------------------------------------------------
    if "R" in args.arms:
        if not existing("R_instruct") or not existing("R_base"):
            m_i, t_i = load_plain_model(scale["instruct"], device, dtype)
            m_b, t_b = load_plain_model(scale["base"], device, dtype)
            gate = tokenizer_identity_gate(t_i, t_b, prompts)
            free_model(m_b)

            print(f"\n[R_instruct] {scale['instruct']} ...", flush=True)
            rec = measure(m_i, t_i, prompts, layers,
                          batch_size=args.batch_size, pm=pm, label="R_instruct",
                          max_length=args.max_length)
            rec["tokenizer_identity_gate"] = gate
            rec["model_id"] = scale["instruct"]
            if args.scale == "large" and not args.instruct_model:
                rec["platform_reproduction_vs_e1"] = e4.platform_reproduction_delta(
                    e4.erank_by_layer(rec), E1_CKPT0_ERANK)
            records["R_instruct"] = write("R_instruct", rec, {
                "arm_role": "reference - the model E1's ckpt-0 adapter sits on"})
            free_model(m_i)

            m_b, t_b = load_plain_model(scale["base"], device, dtype)
            print(f"\n[R_base] {scale['base']} ...", flush=True)
            rec = measure(m_b, t_b, prompts, layers,
                          batch_size=args.batch_size, pm=pm, label="R_base",
                          max_length=args.max_length)
            rec["model_id"] = scale["base"]
            records["R_base"] = write("R_base", rec, {
                "arm_role": "ruler - separated from the reference by a full "
                            "instruction-tuning + RLHF pipeline",
                "caveat": "NOT a controlled dose; an order-of-magnitude "
                          "reference point only."})
            free_model(m_b)

    # -- Arm N ------------------------------------------------------------
    if "N" in args.arms:
        doses = [float(x) for x in args.doses.split(",") if x.strip()]
        for dose in doses:
            label = f"N_dose_{dose:.0e}".replace("-0", "-")
            if existing(label):
                continue
            print(f"\n[{label}] reloading {scale['instruct']} "
                  f"and perturbing at relative dose {dose:g} ...", flush=True)
            m, t = load_plain_model(scale["instruct"], device, dtype)
            pert = e4.perturb_model_(m, dose)
            print(f"  achieved aggregate dose "
                  f"{pert['achieved_aggregate_dose']:.6e} "
                  f"over {pert['n_modules_perturbed']} modules")
            rec = measure(m, t, prompts, layers, batch_size=args.batch_size,
                          pm=pm, label=label, max_length=args.max_length)
            # Keep the summary; the full per-module list is large and its
            # aggregate is what the ladder uses.
            rec["perturbation"] = {k: v for k, v in pert.items()
                                   if k != "per_module"}
            rec["perturbation"]["per_module_n"] = len(pert["per_module"])
            rec["model_id"] = scale["instruct"]
            records[label] = write(label, rec, {
                "arm_role": "calibration ladder rung",
                "requested_relative_dose": dose,
                "achieved_relative_dose": pert["achieved_aggregate_dose"]})
            free_model(m)

    # -- assembly ---------------------------------------------------------
    if "R_instruct" in records:
        table = e4.ruler_table(records, "R_instruct")
        table["hardware"] = mem
        table["scale"] = args.scale
        table["layers"] = list(layers)
        (out / "ruler_table.json").write_text(json.dumps(table, indent=1))
        print(f"\nwrote {out / 'ruler_table.json'}")
        print(f"\n{'arm':>18}  {'max |d erank|':>14}  {'at layer':>8}")
        for label, row in sorted(table["arms"].items()):
            if "max_abs_change_pct" not in row:
                print(f"{label:>18}  {row.get('status', 'n/a'):>14}")
                continue
            print(f"{label:>18}  {row['max_abs_change_pct']:>13.4f}%  "
                  f"{row['max_abs_change_layer']:>8}")
        print("\nE1 reference for comparison: the Stage-A LoRA moved erank by "
              "at most 0.7303% (down_in L14) / 0.7227% (resid L16).")
    else:
        print("\nArm R reference not measured in this session; "
              "ruler_table.json not written.")


if __name__ == "__main__":
    main()
