"""Offline checks for frozen experimental inputs and the pilot contract."""
import json
from pathlib import Path

from src.data import GSM8K_REVISION, SVAMP_REVISION


ROOT = Path(__file__).resolve().parent.parent


def test_frozen_split_sizes_and_disjointness():
    gsm = json.loads((ROOT / "data/gsm8k_splits.json").read_text())
    svamp = json.loads((ROOT / "data/svamp_splits.json").read_text())
    probe = json.loads((ROOT / "data/probe_set_ids.json").read_text())
    assert len(gsm["gsm8k_train_idx"]) == 512
    assert len(gsm["gsm8k_eval_idx"]) == 64
    assert len(gsm["probe_idx"]) == 512
    assert not set(gsm["gsm8k_eval_idx"]) & set(gsm["probe_idx"])
    assert len(svamp["svamp_train_idx"]) == 256
    assert len(svamp["svamp_eval_idx"]) == 100
    assert len(probe["probe_prompts"]) == 512
    assert len(probe["probe_big_prompts"]) == 2048
    assert probe["probe_prompts"] == probe["probe_big_prompts"][:512]


def test_pre_registered_pilot_budget_and_checkpoints():
    cfg = json.loads((ROOT / "pilot_config.json").read_text())
    assert cfg["model_id"] == "Qwen/Qwen2.5-0.5B"
    assert cfg["dataset_revisions"]["openai/gsm8k"] == GSM8K_REVISION
    assert cfg["dataset_revisions"]["ChilleD/SVAMP"] == SVAMP_REVISION
    assert cfg["stage_a"]["max_steps"] == 200
    assert cfg["stage_a"]["checkpoint_steps"] == [0, 25, 50, 100, 200]
    assert cfg["measurement"]["probe_questions"] == 512
    assert cfg["adaptation"]["budget_updates"] == 50
    assert cfg["adaptation"]["train_questions"] == 256
    assert cfg["adaptation"]["eval_questions"] == 100


def test_training_paths_do_not_pass_removed_trl_prompt_length_arg():
    # TRL 1.6 removed GRPOConfig.max_prompt_length. Keep the value in the
    # pre-registered config for preflight/eval truncation, but never pass it to
    # GRPOConfig (a real one-step benchmark caught this API drift).
    notebook = json.loads(
        (ROOT / "01_grpo_gsm8k.ipynb").read_text(encoding="utf-8"))
    notebook_code = "\n".join(
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    runner = (ROOT / "scripts/run_local_pipeline.py").read_text(encoding="utf-8")
    adaptation = (ROOT / "src/adaptation.py").read_text(encoding="utf-8")
    assert 'max_prompt_length=CONFIG["max_prompt_length"]' not in notebook_code
    assert 'max_prompt_length=cfg["max_prompt_length"]' not in runner
    grpo_call = adaptation.split("cfg = GRPOConfig(", 1)[1].split("trainer =", 1)[0]
    assert "max_prompt_length=" not in grpo_call
