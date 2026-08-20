"""Stage A — GRPO on GURU Math, Qwen2.5-7B-Instruct + LoRA, 100 updates.

RECONSTRUCTED. This is an equivalent call, not the byte-identical file that ran on
2026-08-18. Its print statements match `stage_a/stage_a_instruct.log` line for line
(that log is committed), and every parameter is read from the registered config
rather than restated here, so there is nothing to drift.

Executed as a detached subprocess from a copy of colab/00_phase0_selfcontained.ipynb:

    p = subprocess.Popen([sys.executable, '-u', '/content/run_stage_a_instruct.py'],
                         stdout=open('/content/stage_a_instruct.log', 'w'),
                         stderr=subprocess.STDOUT, start_new_session=True)

Detached so the notebook kernel stays free: re-running a small tail cell is both the
progress check and the browser keep-alive Colab needs to not reclaim the runtime.
"""
import json
import sys
from pathlib import Path

EXP2 = "/content/RLVR/experiment 2"
sys.path.insert(0, EXP2)
import src.guru_data as guru_data          # noqa: E402
import src.pipeline as pipeline            # noqa: E402

CONFIG = json.load(open(EXP2 + "/exp2_colab_config_mvp_instruct.json"))
MODEL_ID = CONFIG["model_id"]
sa, sb = CONFIG["stage_a"], CONFIG["stage_b"]
RUN = "exp2_colab_guru_math7b_instruct_group8_e33527592dd9"
OUT = Path("/content/outputs") / RUN / "stage_a"
DRIVE = Path("/content/drive/MyDrive/eaaj-exp2-checkpoints") / "stage_a"

# Freeze the splits. Deterministic under CONFIG["seed"]; build_exp2_splits refuses
# to overwrite an existing frozen file that disagrees with a fresh recomputation.
splits = guru_data.build_exp2_splits(
    MODEL_ID, CONFIG["model_revision"],
    sa["token_filter_max"], sb["token_filter_max"],
    sb["eval_questions"], CONFIG["measurement"]["probe_questions"],
    stage_a_prompt_suffix=None,
    dataset_revision=CONFIG["dataset"]["revision"],
    seed=CONFIG["seed"], out_name=CONFIG["mvp"]["splits_file"])
print("splits frozen: stage_a_train %d | stage_b_train %d | stage_b_eval %d"
      % (len(splits["stage_a_train_ids"]), len(splits["stage_b_train_ids"]),
         len(splits["stage_b_eval_ids"])), flush=True)

rows = guru_data.dataset_rows_for("a", "train", splits, MODEL_ID,
                                  CONFIG["model_revision"],
                                  CONFIG["dataset"]["revision"])
print("stage-A train rows:", len(rows), flush=True)
train_ds = guru_data.to_hf_dataset(rows)

summary = pipeline.run_stage_a_grpo(
    MODEL_ID, CONFIG["peft"], train_ds, OUT,
    checkpoint_steps=sa["checkpoint_steps"],       # [0, 50, 100]
    max_steps=sa["max_steps"],                     # 100
    reward_mode=sa["reward_mode"],                 # exact_plus_boxed_format_0.1
    learning_rate=sa["learning_rate"],             # 2e-5
    per_device_batch=sa["per_device_train_batch_size"],
    grad_accum=sa["gradient_accumulation_steps"],
    num_generations=sa["num_generations"],         # 8
    beta=sa["beta"],                               # 0.0 — no KL term
    temperature=sa["temperature"], top_p=sa["top_p"],
    max_completion_length=sa["max_completion_length"],   # 1536
    revision=CONFIG["model_revision"], seed=CONFIG["seed"],
    device="cuda", eval_every=sa["eval_every"],
    drive_backup_dir=DRIVE)

print("STAGE_A_SUMMARY " + json.dumps(summary), flush=True)
