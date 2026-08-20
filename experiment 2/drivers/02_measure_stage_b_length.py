"""Stage-B completion-length measurement — sizes the completion cap by measurement.

VERBATIM as executed 2026-08-19 (the surrounding cell wrote this to
/content/measure_stage_b.py and launched it detached).

Generation only: no optimizer step, no registered variable changed by the probe
itself. Measured on the two ACTUAL Stage-B starting policies rather than at init,
because the Stage-A cap was sized at init and under-predicted in-training clipping.

Result: truncation at the registered cap of 640 is 9.38% (ckpt-0) / 8.33% (ckpt-100)
BEFORE training starts, against a stop that fires at >10% for 5 consecutive updates.
The sizing rule selects 2048. See FINDING_STAGE_B_CAP_SIZING.md.
"""
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import torch

EXP2 = "/content/RLVR/experiment 2"
sys.path.insert(0, EXP2)
import src.guru_data as guru_data          # noqa: E402
import src.pipeline as pipeline            # noqa: E402

CONFIG = json.load(open(EXP2 + "/exp2_colab_config_mvp_instruct.json"))
MODEL_ID = CONFIG["model_id"]
sa, sb = CONFIG["stage_a"], CONFIG["stage_b"]
DRIVE = Path("/content/drive/MyDrive/eaaj-exp2-checkpoints")

# Re-freeze the splits and PROVE they are the Stage-A run's splits.
splits = guru_data.build_exp2_splits(
    MODEL_ID, CONFIG["model_revision"],
    sa["token_filter_max"], sb["token_filter_max"],
    sb["eval_questions"], CONFIG["measurement"]["probe_questions"],
    stage_a_prompt_suffix=None,
    dataset_revision=CONFIG["dataset"]["revision"],
    seed=CONFIG["seed"], out_name=CONFIG["mvp"]["splits_file"])

sha16 = lambda L: hashlib.sha256(
    json.dumps(sorted(L), separators=(",", ":")).encode()).hexdigest()[:16]
EXPECT = {"stage_a_train_ids": ("a06fe6b80e7d40ca", 54251),
          "stage_b_train_ids": ("df8623cf009e0690", 1132),
          "stage_b_eval_ids": ("8ee975c7089dc72a", 300),
          "probe_stage_a_topup_ids": ("1e61252e7b54793e", 4096)}
for k, (exp, n) in EXPECT.items():
    got = sha16(splits[k])
    print("%-24s n=%-6d sha=%s %s" % (k, len(splits[k]), got,
          "OK" if got == exp else "MISMATCH expected " + exp), flush=True)
    assert got == exp and len(splits[k]) == n, k
print("SPLITS_VERIFIED_IDENTICAL_TO_STAGE_A_RUN", flush=True)
(DRIVE / CONFIG["mvp"]["splits_file"]).write_text(json.dumps(splits, indent=1))

rows = guru_data.dataset_rows_for("b", "train", splits, MODEL_ID,
                                  CONFIG["model_revision"],
                                  CONFIG["dataset"]["revision"])
print("stage-B train rows:", len(rows), flush=True)
random.Random(CONFIG["seed"]).shuffle(rows)
rows = rows[:24]

CAP = 2048
CANDIDATES = (640, 768, 896, 1024, 1280, 1536, 2048)
ARMS = [("ckpt-0", "/content/ckpts/ckpt-0"),
        ("ckpt-100", "/content/ckpts/ckpt-100")]
out = {"cap_probe": CAP, "n_prompts": len(rows),
       "num_generations": sb["num_generations"], "arms": {}}

for label, adapter in ARMS:
    model, tok = pipeline.build_peft_model(MODEL_ID, CONFIG["peft"], "cuda",
                                           adapter_path=adapter)
    model.eval()
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    lengths, t0 = [], time.time()
    for i in range(0, len(rows), 4):
        batch = rows[i:i + 4]
        enc = tok([r["prompt"] for r in batch], return_tensors="pt",
                  padding=True, truncation=True,
                  max_length=sb["max_prompt_length"]).to("cuda")
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=CAP, do_sample=True,
                               temperature=sb["temperature"], top_p=sb["top_p"],
                               num_return_sequences=sb["num_generations"],
                               pad_token_id=tok.pad_token_id)
        gen = g[:, enc["input_ids"].shape[1]:]
        for seq in gen:
            nz = (seq != tok.pad_token_id).nonzero()
            lengths.append(int(nz[-1]) + 1 if len(nz) else 0)
        print("  %s batch %d/%d n=%d %.0fs" % (label, i // 4 + 1,
              (len(rows) + 3) // 4, len(lengths), time.time() - t0), flush=True)
    lengths.sort()
    n = len(lengths)
    q = lambda p: lengths[min(n - 1, int(p * n))]
    print("%s LENGTHS n=%d mean=%.1f p50=%d p90=%d p95=%d p99=%d max=%d"
          % (label, n, sum(lengths) / n, q(.5), q(.9), q(.95), q(.99),
             lengths[-1]), flush=True)
    trunc = {}
    for cap in CANDIDATES:
        tr = sum(1 for L in lengths if L >= cap) / n
        trunc[cap] = tr
        print("  cap %5d -> truncation %.4f%s" % (cap, tr,
              "   <= 2.34pct stage-A survived level" if tr <= 0.0234 else ""),
              flush=True)
    out["arms"][label] = {"n": n, "mean": sum(lengths) / n, "p50": q(.5),
                          "p95": q(.95), "p99": q(.99), "max": lengths[-1],
                          "lengths": lengths,
                          "truncation": {str(k): v for k, v in trunc.items()}}
    del model
    torch.cuda.empty_cache()

# The sizing rule: smallest cap whose truncation on the WORST arm is <= 2.34%,
# the init-policy figure of the Stage-A recipe that completed 100/100 updates.
worst = {c: max(out["arms"][a]["truncation"][str(c)] for a, _ in ARMS)
         for c in CANDIDATES}
chosen = None
for c in CANDIDATES:
    if worst[c] <= 0.0234:
        chosen = c
        break
out["worst_arm_truncation"] = {str(k): v for k, v in worst.items()}
out["sizing_rule"] = "smallest cap with worst-arm truncation <= 0.0234"
out["chosen_cap"] = chosen
out["projected_in_training_clip"] = (worst[chosen] * 2.79) if chosen else None
print("WORST-ARM TRUNCATION:", json.dumps(out["worst_arm_truncation"]), flush=True)
print("CHOSEN STAGE-B CAP:", chosen, flush=True)
if chosen is None:
    print("NO CANDIDATE PASSES - escalate, do not silently pick the largest", flush=True)
else:
    print("projected in-training clipped_ratio ~ %.4f (gate is 0.10)"
          % out["projected_in_training_clip"], flush=True)
Path("/content/stage_b_length_measurement.json").write_text(json.dumps(out))
(DRIVE / "stage_b_length_measurement.json").write_text(json.dumps(out))
print("STAGE_B_MEASUREMENT_DONE", flush=True)
