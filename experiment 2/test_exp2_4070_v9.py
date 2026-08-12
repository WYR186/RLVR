from __future__ import annotations

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest


HERE = Path(__file__).resolve().parent
RUNNER_PATH = HERE / "run_exp2_4070_v9.py"
CONFIG_PATH = HERE / "exp2_config_4070_instruct_v9.json"


def load_runner():
    name = "test_exp2_4070_v9_runner"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_v9_contract_accepts_only_linked_group_geometry():
    runner = load_runner()
    result = runner.validate_contract(load_config(), CONFIG_PATH)
    assert result["status"] == "contract_valid_candidate_not_promoted"
    assert result["stage_a_group_size"] == 8
    assert result["stage_a_device_batch"] == 8
    assert result["stage_a_unique_prompts_per_update"] == 8
    assert result["formal_stage_a_eligible"] is False


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("stage_a", "learning_rate", 1e-5),
        ("stage_a", "reward_mode", "exact"),
        ("gates", "zero_variance_streak_stop", 6),
        ("execution", "optimizer", "adamw_torch"),
    ],
)
def test_v9_contract_rejects_unregistered_changes(section, field, value):
    runner = load_runner()
    cfg = deepcopy(load_config())
    cfg[section][field] = value
    with pytest.raises(RuntimeError):
        runner.validate_contract(cfg, CONFIG_PATH)


def test_existing_preflight_requires_registered_identity_and_gate():
    runner = load_runner()
    valid = {
        "runner_variant": "4070_instruct_v9",
        "n_prompts": 16,
        "num_generations": 8,
        "minimum_combined_variable_groups": 2,
        "gate_pass": True,
    }
    assert runner._validate_existing_preflight(valid, 16, 2) is valid
    invalid = dict(valid, gate_pass=False)
    with pytest.raises(RuntimeError):
        runner._validate_existing_preflight(invalid, 16, 2)


def test_safety_stop_saves_diagnostic_only_snapshot(tmp_path):
    runner = load_runner()

    class DummyModel:
        def save_pretrained(self, path, safe_serialization):
            assert safe_serialization is True
            Path(path, "model.safetensors").write_bytes(b"diagnostic")

    callback = runner.SnapshottingLocalSafetyCallback(
        tmp_path / "safety_stop.json", signal_patience=1)
    state = SimpleNamespace(global_step=7)
    control = SimpleNamespace(should_training_stop=False)
    callback.on_log(
        None, state, control,
        logs={"frac_reward_zero_std": 1.0, "loss": 0.0, "grad_norm": 0.0},
        model=DummyModel())
    marker = json.loads((
        tmp_path / "safety-stop-weights-step-7" / "DIAGNOSTIC_ONLY.json"
    ).read_text(encoding="utf-8"))
    assert control.should_training_stop is True
    assert marker["resume_allowed"] is False
    assert marker["registered_checkpoint"] is False
