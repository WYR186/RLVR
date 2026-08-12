import json
from pathlib import Path

from exp2_4070_data import _render_prompt
from exp2_4070_reward import boxed_format_score, score_completion


HERE = Path(__file__).resolve().parent


def test_variant_keeps_validated_total_sequence_budget():
    cfg = json.loads((HERE / "exp2_config_4070.json").read_text())
    assert cfg["stage_a"]["max_prompt_length"] + cfg["stage_a"]["max_completion_length"] == 1024
    assert cfg["stage_b"]["max_prompt_length"] + cfg["stage_b"]["max_completion_length"] == 1024
    assert cfg["variant_limits"]["prompt_truncation"] is False


def test_instruct_rescue_is_separate_and_same_size():
    cfg = json.loads((HERE / "exp2_config_4070_instruct.json").read_text())
    assert cfg["model_id"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert cfg["stage_b"]["train_questions"] == 1132
    assert cfg["stage_b"]["max_prompt_length"] + cfg["stage_b"]["max_completion_length"] == 1024
    assert cfg["variant_limits"]["not_equivalent_to_original_exp2"] is True


def test_instruct_v2_reallocates_without_growing_sequence_budget():
    cfg = json.loads((HERE / "exp2_config_4070_instruct_v2.json").read_text())
    assert cfg["stage_a"]["max_prompt_length"] == 256
    assert cfg["stage_a"]["max_completion_length"] == 768
    assert cfg["stage_a"]["max_prompt_length"] + cfg["stage_a"]["max_completion_length"] == 1024
    assert cfg["stage_b"]["max_prompt_length"] + cfg["stage_b"]["max_completion_length"] == 1024


def test_instruct_v3_uses_concise_prompt_without_growing_sequence_budget():
    cfg = json.loads((HERE / "exp2_config_4070_instruct_v3.json").read_text())
    assert cfg["variant_tag"] == "4070_instruct_v3"
    assert "boxed" in cfg["stage_a"]["prompt_suffix"].lower()
    assert cfg["stage_a"]["max_prompt_length"] + cfg["stage_a"]["max_completion_length"] == 1024
    assert cfg["stage_b"]["max_prompt_length"] + cfg["stage_b"]["max_completion_length"] == 1024
    assert cfg["gates"]["phase0_stage_a_smoke_max_clip_ratio_each_update"] == 0.10


def test_concise_suffix_is_added_without_mutating_source_messages():
    class TokenizerStub:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            assert tokenize is False
            assert add_generation_prompt is True
            return messages[-1]["content"]

    messages = [{"role": "user", "content": "Solve 1+1."}]
    rendered = _render_prompt(TokenizerStub(), messages, "Finish inside \\boxed{}.")
    assert rendered == "Solve 1+1.\n\nFinish inside \\boxed{}."
    assert messages == [{"role": "user", "content": "Solve 1+1."}]


def test_instruct_v4_trades_group_size_for_completion_budget_on_4070():
    cfg = json.loads((HERE / "exp2_config_4070_instruct_v4.json").read_text())
    stage = cfg["stage_a"]
    assert cfg["variant_tag"] == "4070_instruct_v4"
    assert stage["max_prompt_length"] + stage["max_completion_length"] == 1536
    assert stage["per_device_train_batch_size"] == 4
    assert stage["num_generations"] == 4
    assert stage["per_device_train_batch_size"] * stage["gradient_accumulation_steps"] // stage["num_generations"] == 8
    assert cfg["gates"]["phase0_stage_a_smoke_max_clip_ratio_each_update"] == 0.10


def test_instruct_v5_preserves_unique_prompt_batch_with_three_generations():
    cfg = json.loads((HERE / "exp2_config_4070_instruct_v5.json").read_text())
    stage = cfg["stage_a"]
    assert cfg["variant_tag"] == "4070_instruct_v5"
    assert stage["max_prompt_length"] + stage["max_completion_length"] == 1792
    assert stage["per_device_train_batch_size"] == 3
    assert stage["num_generations"] == 3
    assert stage["per_device_train_batch_size"] * stage["gradient_accumulation_steps"] // stage["num_generations"] == 8
    assert cfg["gates"]["phase0_stage_a_smoke_max_clip_ratio_each_update"] == 0.10


def test_instruct_v6_reallocates_prompt_budget_and_requires_real_updates():
    cfg = json.loads((HERE / "exp2_config_4070_instruct_v6.json").read_text())
    stage = cfg["stage_a"]
    assert cfg["variant_tag"] == "4070_instruct_v6"
    assert stage["max_prompt_length"] + stage["max_completion_length"] == 1536
    assert stage["per_device_train_batch_size"] == stage["num_generations"] == 4
    assert stage["per_device_train_batch_size"] * stage["gradient_accumulation_steps"] // stage["num_generations"] == 8
    assert cfg["gates"]["phase0_stage_a_smoke_min_nonzero_reward_updates"] == 1


def test_instruct_v7_registers_small_math_format_reward_only():
    cfg = json.loads((HERE / "exp2_config_4070_instruct_v7.json").read_text())
    assert cfg["variant_tag"] == "4070_instruct_v7"
    assert cfg["stage_a"]["reward_mode"] == "exact_plus_boxed_format_0.1"
    assert cfg["stage_b"].get("reward_mode", "exact") == "exact"


def test_boxed_format_score_requires_balanced_nonempty_math_box():
    assert boxed_format_score(r"reasoning \boxed{\frac{1}{2}}", "math__deepscaler_preview") == 1.0
    assert boxed_format_score(r"reasoning \boxed{}", "math__deepscaler_preview") == 0.0
    assert boxed_format_score(r"reasoning \boxed{42", "math__deepscaler_preview") == 0.0
    assert boxed_format_score(r"\boxed{42}", "simulation__codeio") == 0.0


def test_instruct_v8_combines_length_safe_geometry_with_shaped_signal():
    cfg = json.loads((HERE / "exp2_config_4070_instruct_v8.json").read_text())
    stage = cfg["stage_a"]
    assert cfg["variant_tag"] == "4070_instruct_v8"
    assert stage["reward_mode"] == "exact_plus_boxed_format_0.1"
    assert stage["per_device_train_batch_size"] == stage["num_generations"] == 3
    assert stage["max_prompt_length"] + stage["max_completion_length"] == 1792
    assert stage["per_device_train_batch_size"] * stage["gradient_accumulation_steps"] // stage["num_generations"] == 8


def test_math_reward_uses_last_boxed_answer():
    assert score_completion(r"work \\boxed{26}", "26", "math__deepscaler_preview", {}) == 1.0
    assert score_completion(r"work \\boxed{25}", "26", "math__deepscaler_preview", {}) == 0.0


def test_codeio_reward_structural_json_equality():
    truth = '"input": {"start_value": 808}'
    good = 'reasoning\n```json\n{"input": {"start_value": 808}}\n```'
    bad = '```json\n{"input": {"start_value": 404}}\n```'
    assert score_completion(good, truth, "simulation__codeio", {}) == 1.0
    assert score_completion(bad, truth, "simulation__codeio", {}) == 0.0
