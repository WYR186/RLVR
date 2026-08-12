"""Unit tests for the thin GURU reward wrapper (src/guru_reward.py).

These exercise the REAL vendored verifier (vendor/reasoning360_reward_score),
not a hand-rolled approximation — the point of switching to the vendored
code was to stop guessing at the answer format, so these tests should fail
loudly if the wrapper's plumbing (dict-vs-float unwrapping, exception
fallback, data_source routing) is wrong, not re-implement the verifier's own
correctness (that's Reasoning360's test surface, not ours).
"""
from src.guru_reward import (boxed_format_score, exact_correct, guru_reward,
                             guru_reward_boxed_01, score_completion,
                             select_reward_fn)

MATH_SOURCE = "math__deepscaler_preview"
SIM_SOURCE = "simulation__codeio"


class TestMathScoring:
    def test_correct_boxed_answer(self):
        assert score_completion(r"work work \boxed{42}", "42", MATH_SOURCE) == 1.0

    def test_incorrect_boxed_answer(self):
        assert score_completion(r"work work \boxed{7}", "42", MATH_SOURCE) == 0.0

    def test_last_boxed_wins(self):
        assert score_completion(r"\boxed{41} correction \boxed{42}", "42", MATH_SOURCE) == 1.0

    def test_no_boxed_answer_scores_zero(self):
        assert score_completion("I cannot solve this.", "42", MATH_SOURCE) == 0.0


class TestSimulationScoring:
    def test_matching_json_output(self):
        completion = '```json\n{"output": {"result": 42}}\n```'
        gold = '"output": {"result": 42}'
        assert score_completion(completion, gold, SIM_SOURCE) == 1.0

    def test_mismatched_json_output(self):
        completion = '```json\n{"output": {"result": 43}}\n```'
        gold = '"output": {"result": 42}'
        assert score_completion(completion, gold, SIM_SOURCE) == 0.0

    def test_malformed_json_scores_zero_not_crash(self):
        assert score_completion("not json at all {{{", "42", SIM_SOURCE) == 0.0


class TestUnsupportedDataSource:
    def test_unknown_source_scores_zero(self):
        # score_completion catches the ValueError and returns 0.0 — a
        # malformed completion/config must never crash GRPO mid-training.
        assert score_completion("anything", "42", "table__hitab") == 0.0


class TestBoxedFormatBonus:
    def test_well_formed_boxed_on_math(self):
        assert boxed_format_score(r"\boxed{42}", MATH_SOURCE) == 1.0

    def test_empty_boxed_scores_zero(self):
        assert boxed_format_score(r"\boxed{}", MATH_SOURCE) == 0.0

    def test_no_boxed_scores_zero(self):
        assert boxed_format_score("42", MATH_SOURCE) == 0.0

    def test_never_applies_to_non_math(self):
        assert boxed_format_score(r"\boxed{42}", SIM_SOURCE) == 0.0


class TestRewardFunctionContracts:
    def test_guru_reward_math_batch(self):
        completions = [r"\boxed{42}", r"\boxed{7}"]
        r = guru_reward(completions=completions, ground_truth=["42", "42"],
                        data_source=[MATH_SOURCE, MATH_SOURCE])
        assert r == [1.0, 0.0]

    def test_guru_reward_no_shaping_on_math(self):
        # plain guru_reward must NOT add the boxed bonus — that's
        # guru_reward_boxed_01's job (Stage A only, per reward_mode dispatch)
        completions = [r"\boxed{7}"]
        r = guru_reward(completions=completions, ground_truth=["42"], data_source=[MATH_SOURCE])
        assert r == [0.0]

    def test_guru_reward_boxed_01_adds_bonus_on_wrong_but_formatted(self):
        completions = [r"\boxed{7}"]
        r = guru_reward_boxed_01(completions=completions, ground_truth=["42"], data_source=[MATH_SOURCE])
        assert r == [0.1]  # 0.0 exact + 0.1 format bonus

    def test_guru_reward_boxed_01_on_correct_answer(self):
        completions = [r"\boxed{42}"]
        r = guru_reward_boxed_01(completions=completions, ground_truth=["42"], data_source=[MATH_SOURCE])
        assert r == [1.1]

    def test_chat_format_completion(self):
        completions = [[{"role": "assistant", "content": r"\boxed{5}"}]]
        r = guru_reward(completions=completions, ground_truth=["5"], data_source=[MATH_SOURCE])
        assert r == [1.0]

    def test_extra_info_defaults_to_empty_dicts(self):
        # extra_info=None must not crash the zip(..., strict=True)
        r = guru_reward(completions=[r"\boxed{1}"], ground_truth=["1"], data_source=[MATH_SOURCE])
        assert r == [1.0]


class TestExactCorrect:
    def test_math_correct(self):
        assert exact_correct(r"\boxed{9}", "9", MATH_SOURCE) is True

    def test_math_incorrect(self):
        assert exact_correct(r"\boxed{8}", "9", MATH_SOURCE) is False

    def test_simulation_correct(self):
        completion = '```json\n{"a": 1}\n```'
        assert exact_correct(completion, '{"a": 1}', SIM_SOURCE) is True


class TestDispatch:
    def test_select_reward_fn_exact(self):
        assert select_reward_fn("exact") is guru_reward

    def test_select_reward_fn_boxed(self):
        assert select_reward_fn("exact_plus_boxed_format_0.1") is guru_reward_boxed_01

    def test_select_reward_fn_unknown_raises(self):
        try:
            select_reward_fn("nonexistent_mode")
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for an unconfigured reward_mode")
