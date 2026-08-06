import json
from pathlib import Path

import pytest
from transformers import TrainerCallback

from run_exp2_4070_poststop import (
    CodeIOEvalCallback,
    poststop_run_dir,
    transfer_table,
    validate_stage_b_cell,
)


HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "exp2_config_4070_instruct_v8_poststop.json"


def _config():
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    value["_config_path"] = str(CONFIG_PATH)
    return value


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8")


def test_poststop_contract_is_truncated_and_isolated():
    cfg = _config()
    assert cfg["source"]["required_checkpoints"] == [0, 25, 50, 100]
    assert cfg["stage_b"]["checkpoints"] == [0, 50, 100]
    assert cfg["stage_b"]["seeds"] == [42, 43, 44]
    assert cfg["source"]["forbidden_as_complete"] == [150, 200]
    assert cfg["source"]["safety_stop_step"] == 110
    assert poststop_run_dir(cfg).name.startswith("exp2_4070_v8_poststop_")
    assert poststop_run_dir(cfg).name != cfg["source"]["run_name"]


def test_source_identity_contract_covers_every_main_checkpoint():
    cfg = _config()
    hashes = cfg["source"]["sha256"]
    for checkpoint in cfg["source"]["required_checkpoints"]:
        value = hashes[f"ckpt-{checkpoint}/model.safetensors"]
        assert len(value) == 64
        int(value, 16)
    assert len(cfg["phase2b"]["probe_sha256"]) == 64


def test_phase2_and_stageb_keep_registered_fixed_budget_semantics():
    cfg = _config()
    assert cfg["phase2_zero_shot"]["checkpoints"] == [0, 25, 50, 100]
    assert cfg["phase2_zero_shot"]["decode"] == "greedy"
    assert cfg["phase2b"]["probe_questions"] == 2048
    assert cfg["phase2b"]["layers"] == [4, 12, 22]
    assert cfg["stage_b"]["budget_updates"] == 50
    assert cfg["stage_b"]["eval_at_updates"] == [0, 10, 20, 30, 40, 50]
    assert cfg["stage_b"]["reward_mode"] == "exact"


def test_transfer_table_uses_checkpoint_zero_baseline():
    cfg = _config()
    results = {
        0: {"score": 0.20, "n_eval": 300},
        25: {"score": 0.22, "n_eval": 300},
        50: {"score": 0.18, "n_eval": 300},
        100: {"score": 0.25, "n_eval": 300},
    }
    table = transfer_table(cfg, results)
    assert table["baseline_score"] == pytest.approx(0.20)
    assert [row["T_t"] for row in table["rows"]] == pytest.approx(
        [0.0, 0.02, -0.02, 0.05])


def test_transfer_table_rejects_missing_checkpoint():
    cfg = _config()
    with pytest.raises(ValueError, match="do not cover"):
        transfer_table(cfg, {
            0: {"score": 0.2, "n_eval": 300},
            50: {"score": 0.2, "n_eval": 300},
            100: {"score": 0.2, "n_eval": 300},
        })


def _complete_cell(path, checkpoint=0, seed=42):
    cfg = _config()
    path.mkdir(parents=True)
    (path / "summary.json").write_text(json.dumps({
        "checkpoint": checkpoint,
        "seed": seed,
        "actual_updates": 50,
        "completion_status": "complete",
    }), encoding="utf-8")
    _write_jsonl(path / "codeio_eval_curve.jsonl", [
        {"step": step, "score": 0.1} for step in [0, 10, 20, 30, 40, 50]
    ])
    _write_jsonl(path / "update_sentinel.jsonl", [
        {"step": step, "updates_effective": True}
        for step in [10, 20, 30, 40, 50]
    ])
    _write_jsonl(path / "dashboard.jsonl", [
        {"step": step, "loss": 0.1, "grad_norm": 1.0}
        for step in range(1, 51)
    ])
    return cfg


def test_stage_b_validator_accepts_exact_complete_contract(tmp_path):
    cfg = _complete_cell(tmp_path / "cell")
    result = validate_stage_b_cell(tmp_path / "cell", cfg, 0, 42)
    assert result["actual_updates"] == 50


def test_stage_b_validator_rejects_safety_stop(tmp_path):
    cfg = _complete_cell(tmp_path / "cell")
    (tmp_path / "cell" / "safety_stop.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="safety stop"):
        validate_stage_b_cell(tmp_path / "cell", cfg, 0, 42)


def test_stage_b_validator_rejects_missing_update(tmp_path):
    cell = tmp_path / "cell"
    cfg = _complete_cell(cell)
    _write_jsonl(cell / "dashboard.jsonl", [
        {"step": step, "loss": 0.1, "grad_norm": 1.0}
        for step in range(1, 50)
    ])
    with pytest.raises(RuntimeError, match="all 50 updates"):
        validate_stage_b_cell(cell, cfg, 0, 42)


def test_codeio_callback_uses_transformers_noop_surface():
    assert issubclass(CodeIOEvalCallback, TrainerCallback)
