"""Stage B — fixed-budget GRPO on GURU Simulation from each Stage-A checkpoint.

VERBATIM as executed 2026-08-19. Produces the Delta-R headline.

Three arms, 30 updates each, identical budget — that identity is the whole point of
the comparison. Runs under the measured cap of 2048 (config bd99ddd2817f), not the
registered 640, which measured 9.38% truncation before training started and killed
an earlier attempt at update 26.

Arm order is [0, 100, 50] on purpose: if the session dies mid-run, the two endpoints
give a readable Delta-R contrast on their own and ckpt-50 only adds the midpoint.
Each arm mirrors to Drive on completion, and an arm whose summary.json already
exists is skipped, so a second session resumes rather than repeats.
"""
import json
import sys
import time
import traceback
from pathlib import Path

EXP2 = "/content/RLVR/experiment 2"
sys.path.insert(0, EXP2)
import src.guru_data as guru_data          # noqa: E402
import src.pipeline as pipeline            # noqa: E402

CONFIG = json.load(open(EXP2 + "/exp2_colab_config_mvp_instruct_stageb_v2.json"))
MODEL_ID = CONFIG["model_id"]
sb = CONFIG["stage_b"]
RUN = "exp2_colab_guru_math7b_instruct_group8_e33527592dd9"
OUT = Path("/content/outputs") / RUN / "stage_b_v2"
DRIVE = Path("/content/drive/MyDrive/eaaj-exp2-checkpoints") / "stage_b_v2"

splits = json.load(open(EXP2 + "/data/" + CONFIG["mvp"]["splits_file"]))
train_rows = guru_data.dataset_rows_for("b", "train", splits, MODEL_ID,
        CONFIG["model_revision"], CONFIG["dataset"]["revision"])
eval_rows = guru_data.dataset_rows_for("b", "eval", splits, MODEL_ID,
        CONFIG["model_revision"], CONFIG["dataset"]["revision"])
print("train %d  eval %d" % (len(train_rows), len(eval_rows)), flush=True)
train_ds = guru_data.to_hf_dataset(train_rows)

ORDER = [0, 100, 50]
results = {}
for k in ORDER:
    out_dir = OUT / ("ckpt-%d" % k)
    print("\n===== STAGE B ARM ckpt-%d =====" % k, flush=True)
    t0 = time.time()
    try:
        s = pipeline.run_stage_b_adaptation(
            MODEL_ID, CONFIG["peft"], "/content/ckpts/ckpt-%d" % k,
            train_ds, eval_rows, out_dir,
            budget_updates=sb["budget_updates"],   # 30, identical for all arms
            eval_every=30,                         # de-scoped from 10; see §7
            reward_mode=sb["reward_mode"],         # exact — no format bonus
            learning_rate=sb["learning_rate"],
            per_device_batch=sb["per_device_train_batch_size"],
            grad_accum=sb["gradient_accumulation_steps"],
            num_generations=sb["num_generations"], beta=sb["beta"],
            temperature=sb["temperature"], top_p=sb["top_p"],
            max_completion_length=sb["max_completion_length"],   # 2048, measured
            revision=CONFIG["model_revision"], seed=CONFIG["seed"],
            device="cuda", drive_backup_dir=DRIVE / ("ckpt-%d" % k))
        results[k] = s
        print("ARM ckpt-%d DONE  before=%.4f after=%.4f delta=%+.4f  %.0fs"
              % (k, s["acc_before"], s["acc_after"], s["delta_acc"],
                 time.time() - t0), flush=True)
    except Exception as e:
        # A safety stop raises. Record it and move to the next arm rather than
        # losing the arms that would still have run.
        print("ARM ckpt-%d STOPPED: %r" % (k, e), flush=True)
        traceback.print_exc()
        results[k] = {"stopped": repr(e)}
    DRIVE.mkdir(parents=True, exist_ok=True)
    (DRIVE / "delta_r_partial.json").write_text(
        json.dumps(results, indent=1, default=str))

print("\n===== DELTA-R =====", flush=True)
for k in [0, 50, 100]:
    r = results.get(k, {})
    if "delta_acc" in r:
        print("ckpt-%-4d before %.4f  after %.4f  deltaR %+.4f"
              % (k, r["acc_before"], r["acc_after"], r["delta_acc"]), flush=True)
    else:
        print("ckpt-%-4d %s" % (k, r.get("stopped", "not run")), flush=True)
(DRIVE / "delta_r.json").write_text(json.dumps(results, indent=1, default=str))
print("STAGE_B_V2_DONE", flush=True)
