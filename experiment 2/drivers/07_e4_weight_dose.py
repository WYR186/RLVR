"""E4 Arm W — the Stage-A LoRA's dose in weight space. CPU only, RAM-light.

Answers, in a number, what "our intervention was small" means: the relative
Frobenius change ||B A (alpha/r)||_F / ||W_0||_F that the three Stage-A
adapters actually applied to the base weights. That number is the x-position
of our run on Arm N's calibration ladder.

Every tensor is streamed one at a time through safetensors, so this runs on a
16 GB machine even though the base model is 15 GB on disk.

    # adapter side only - no base model needed, seconds
    python 07_e4_weight_dose.py --adapters ../../eaaj-pilot/outputs/<run>/stage_a \
        --out ../outputs/e4 --skip-base-norms

    # full relative dose - needs the base weights on disk (15.2 GB)
    python 07_e4_weight_dose.py --adapters <dir> --out ../outputs/e4 --download

`--download` fetches only `*.safetensors` + the index from the Hub. Point
`--base-dir` at an existing snapshot to skip the download entirely.
"""
import argparse
import json
import sys
from pathlib import Path

EXP2 = Path("/content/RLVR/experiment 2")
if not EXP2.is_dir():
    EXP2 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXP2))

from src import e4_calibration as e4  # noqa: E402

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
CKPTS = ("ckpt-0", "ckpt-50", "ckpt-100")


def resolve_base_dir(args) -> Path | None:
    if args.base_dir:
        return Path(args.base_dir)
    if not args.download:
        return None
    from huggingface_hub import snapshot_download
    print(f"downloading {args.base_model} weight shards "
          "(~15.2 GB, disk only - never fully loaded into RAM) ...", flush=True)
    return Path(snapshot_download(
        args.base_model, revision=args.base_revision,
        allow_patterns=["*.safetensors", "*.safetensors.index.json"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", required=True,
                    help="directory containing ckpt-0/ ckpt-50/ ckpt-100/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--base-revision", default=None)
    ap.add_argument("--base-dir", default=None,
                    help="existing local snapshot of the base weights")
    ap.add_argument("--download", action="store_true",
                    help="fetch the base weight shards from the Hub")
    ap.add_argument("--skip-base-norms", action="store_true",
                    help="emit ||dW||_F only; relative dose is left not-run "
                         "with a reason, to be completed later")
    ap.add_argument("--checkpoints", nargs="+", default=list(CKPTS))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    adapters = Path(args.adapters)

    missing = [c for c in args.checkpoints
               if not (adapters / c / "adapter_model.safetensors").is_file()]
    if missing:
        raise SystemExit(
            f"missing adapter weights for {missing} under {adapters}.\n"
            "The .safetensors files are gitignored (154-165 MB each); see "
            "experiment 2/WEIGHTS.md for where to get them.")

    deltas = {}
    for ckpt in args.checkpoints:
        d = e4.lora_delta_norms(adapters / ckpt)
        deltas[ckpt] = d
        total = sum(v["delta_fro"] ** 2 for v in d.values()) ** 0.5
        print(f"{ckpt}: {len(d)} LoRA modules, ||dW||_F (all modules) = {total:.6g}")

    record = {
        "arm": "W",
        "base_model": args.base_model,
        "adapters_dir": str(adapters),
        "delta_norms": deltas,
    }

    base_dir = None if args.skip_base_norms else resolve_base_dir(args)
    if base_dir is None:
        record["relative_dose"] = {
            "status": "not_run",
            "reason": ("base weight norms were not computed "
                       + ("(--skip-base-norms)" if args.skip_base_norms
                          else "(no --base-dir and no --download)")
                       + "; ||dW||_F above is complete and the ratio can be "
                         "formed later without re-reading the adapters"),
        }
        print("\nrelative dose NOT computed - recorded as not-run with a reason.")
    else:
        wanted = set()
        for d in deltas.values():
            wanted |= set(d)
        shards = sorted(Path(base_dir).glob("*.safetensors"))
        if not shards:
            raise SystemExit(f"no .safetensors shards under {base_dir}")
        print(f"\nstreaming ||W_0||_F from {len(shards)} shards "
              f"for {len(wanted)} target weights ...", flush=True)
        base_norms = e4.base_weight_norms(shards, wanted=wanted)
        print(f"  read {len(base_norms)}/{len(wanted)} target weights")
        record["base_norms_n"] = len(base_norms)
        record["base_dir"] = str(base_dir)
        record["relative_dose"] = {
            ckpt: e4.relative_dose(d, base_norms) for ckpt, d in deltas.items()
        }
        print()
        for ckpt in args.checkpoints:
            r = record["relative_dose"][ckpt]
            print(f"{ckpt}: aggregate relative dose = "
                  f"{r['aggregate_relative_dose']:.6e}  "
                  f"(per-module min {r['min_per_module_relative_dose']:.3e}, "
                  f"max {r['max_per_module_relative_dose']:.3e}, "
                  f"n={r['n_modules']})")

    path = out / "arm_W_weight_dose.json"
    path.write_text(json.dumps(record, indent=1))
    print(f"\nwrote {path}")
    print("\nckpt-0 is the pre-update adapter: LoRA initialises B=0, so its "
          "dose MUST be exactly 0. A nonzero value there means the wrong "
          "checkpoint was read.")


if __name__ == "__main__":
    main()
