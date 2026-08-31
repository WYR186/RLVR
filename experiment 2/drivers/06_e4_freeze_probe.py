"""E4 step 1 — freeze the probe TEXT so both arms see byte-identical input.

Arm R compares Qwen2.5-7B against Qwen2.5-7B-Instruct. Those two models do not
share a chat template, so rendering the probe through "the model's own
template" would feed them different text and confound the weights difference
with a prompt difference. This driver renders the probe ONCE through the
Instruct tokenizer — exactly as E1 did — and writes the rendered strings to
disk. Every later arm reads that file and never re-renders.

It needs only a tokenizer and the GURU parquet, no model weights, so it runs
on a 16 GB machine in minutes. Run it there, commit the manifest, and let the
GPU session consume the frozen file.

    python 06_e4_freeze_probe.py --out ../outputs/e4
    python 06_e4_freeze_probe.py --out ../outputs/e4 --n-probe 512   # smoke

Gate: the probe id set must hash to the value E1 recorded. If it does not,
this is not E1's probe and no E4 number would be comparable to E1's, so it
stops rather than writing a lookalike.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

EXP2 = Path("/content/RLVR/experiment 2")
if not EXP2.is_dir():
    EXP2 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXP2))

import src.guru_data as guru_data  # noqa: E402

SPLITS_NAME = "exp2_colab_splits_instruct.json"
# E1's frozen probe: 4096 stage-A math prompts, hash from 04_e1_metric_sweep.py.
PROBE_SHA16 = "1e61252e7b54793e"
PROBE_N = 4096


def sha16(ids):
    return hashlib.sha256(
        json.dumps(sorted(ids), separators=(",", ":")).encode()).hexdigest()[:16]


def sha256_text(strings) -> str:
    h = hashlib.sha256()
    for s in strings:
        h.update(s.encode())
        h.update(b"\x00")
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--splits", default=str(EXP2 / "data" / SPLITS_NAME))
    ap.add_argument("--n-probe", type=int, default=PROBE_N,
                    help="prefix of the frozen probe to write; < %d marks the "
                         "manifest sample_truncated and its erank LEVELS are "
                         "not comparable to E1's" % PROBE_N)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    splits = json.loads(Path(args.splits).read_text())
    got = sha16(splits["probe_stage_a_topup_ids"])
    n_ids = len(splits["probe_stage_a_topup_ids"])
    if got != PROBE_SHA16 or n_ids != PROBE_N:
        raise SystemExit(
            f"probe id set is {got} (n={n_ids}), expected {PROBE_SHA16} "
            f"(n={PROBE_N}). This is not E1's frozen probe, so nothing measured "
            "on it would be comparable to E1. Refusing to write a lookalike.")
    print(f"probe id gate OK: {got}, n={n_ids}")

    model_id = splits["model_id"]
    print(f"rendering through {model_id} chat template "
          f"(downloads the GURU parquet on first run) ...", flush=True)
    rows = guru_data.dataset_rows_for(
        "probe", None, splits, model_id, splits.get("model_revision"),
        splits["dataset_revision"])
    prompts = [r["prompt"] for r in rows]
    ids = [r["id"] for r in rows]
    if len(prompts) != PROBE_N:
        raise SystemExit(
            f"rendered {len(prompts)} prompts but the frozen probe has {PROBE_N}. "
            "The dataset revision or the id set has drifted.")

    full_text_sha = sha256_text(prompts)
    if args.n_probe < len(prompts):
        prompts = prompts[:args.n_probe]
        ids = ids[:args.n_probe]

    payload = {
        "prompts": prompts,
        "ids": ids,
    }
    manifest = {
        "n_probe": len(prompts),
        "n_probe_full": PROBE_N,
        "sample_truncated_vs_e1": len(prompts) < PROBE_N,
        "probe_ids_sha16": got,
        "rendered_text_sha256": sha256_text(prompts),
        "rendered_text_sha256_full_4096": full_text_sha,
        "rendered_with_model_id": model_id,
        "rendered_with_model_revision": splits.get("model_revision"),
        "dataset_revision": splits["dataset_revision"],
        "chat_template_applied": True,
        "add_generation_prompt": True,
        "note": ("Both Arm R models consume THIS text verbatim. Do not "
                 "re-render per model: Qwen2.5-7B and Qwen2.5-7B-Instruct do "
                 "not share a chat template, and re-rendering would confound "
                 "the weight difference with a prompt difference."),
    }
    (out / "probe_frozen.json").write_text(json.dumps(payload))
    (out / "probe_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"\nwrote {out / 'probe_frozen.json'}  ({len(prompts)} prompts)")
    print(f"      {out / 'probe_manifest.json'}")
    print(f"rendered text sha256: {manifest['rendered_text_sha256']}")
    if manifest["sample_truncated_vs_e1"]:
        print("\nWARNING: n_probe < 4096. Cross-checkpoint comparisons stay "
              "valid, but erank LEVELS are sample-truncated and must not be "
              "compared with E1's published values.")


if __name__ == "__main__":
    main()
