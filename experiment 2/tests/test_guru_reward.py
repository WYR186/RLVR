"""Unit tests for exp2's domain-dispatched reward/extraction (Phase 0 step 3
"the reward function is written from the discovered format, not a guess" —
these tests fix the behavior of the *current* best-effort extractors so a
future edit against real GURU examples has a regression baseline to edit
against, not a claim that the format is already verified)."""
from src.guru_reward import (answers_match_numeric, code_output_exact_reward,
                             extract_boxed, extract_gold_numeric,
                             extract_pred_numeric, extract_predicted_output,
                             normalize_code_output, numeric_exact_reward,
                             select_eval_fns, select_reward_fn)


class TestBoxedExtraction:
    def test_simple_boxed(self):
        assert extract_boxed(r"work work \boxed{42}") == "42"

    def test_last_boxed_wins(self):
        assert extract_boxed(r"\boxed{41} correction \boxed{42}") == "42"

    def test_nested_braces(self):
        assert extract_boxed(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"

    def test_no_boxed(self):
        assert extract_boxed("no box here") is None

    def test_unbalanced_boxed(self):
        assert extract_boxed(r"\boxed{42") is None


class TestPredNumeric:
    def test_boxed_wins_over_trailing_text(self):
        assert extract_pred_numeric(r"so \boxed{42} is my answer, see step 3") == 42.0

    def test_boxed_with_embedded_number_fallback(self):
        # non-numeric boxed content falls back to the last number inside the box
        assert extract_pred_numeric(r"\boxed{x=42}") == 42.0

    def test_hash_convention_when_no_boxed(self):
        assert extract_pred_numeric("She has 7 apples.\n#### 7") == 7.0

    def test_last_number_fallback(self):
        assert extract_pred_numeric("3 + 4 = 7. The answer is 7.") == 7.0

    def test_no_number(self):
        assert extract_pred_numeric("I cannot solve this.") is None


class TestGoldNumeric:
    def test_bare_float(self):
        assert extract_gold_numeric(42.0) == 42.0

    def test_bare_int(self):
        assert extract_gold_numeric(42) == 42.0

    def test_boxed_string(self):
        assert extract_gold_numeric(r"\boxed{-5}") == -5.0

    def test_plain_numeric_string(self):
        assert extract_gold_numeric("17") == 17.0


class TestNumericMatch:
    def test_exact(self):
        assert answers_match_numeric(7.0, 7.0)

    def test_int_float_equivalence(self):
        assert answers_match_numeric(7.0, 7)

    def test_mismatch(self):
        assert not answers_match_numeric(7.0, 8.0)

    def test_none_never_matches(self):
        assert not answers_match_numeric(None, 7.0)
        assert not answers_match_numeric(7.0, None)

    def test_no_relative_slack_on_large_answers(self):
        assert not answers_match_numeric(10000.5, 10000.0)


class TestNumericReward:
    def test_reward_contract(self):
        completions = [r"\boxed{42}", "the answer is 43"]
        answer = [42.0, 42.0]
        assert numeric_exact_reward(completions=completions, answer=answer) == [1.0, 0.0]

    def test_chat_format_completion(self):
        completions = [[{"role": "assistant", "content": r"\boxed{5}"}]]
        assert numeric_exact_reward(completions=completions, answer=[5.0]) == [1.0]


class TestCodeOutputNormalization:
    def test_strips_whitespace(self):
        assert normalize_code_output("  42  ") == "42"

    def test_strips_matching_quotes(self):
        assert normalize_code_output('"hello"') == "hello"

    def test_case_sensitive(self):
        assert normalize_code_output("True") != normalize_code_output("true")

    def test_list_literal_canonicalization(self):
        assert normalize_code_output("[1, 2, 3]") == normalize_code_output("[1,2,3]")

    def test_dict_literal_spacing_canonicalized(self):
        assert normalize_code_output('{"a": 1}') == normalize_code_output("{'a':1}")

    def test_dict_key_order_not_canonicalized(self):
        # documented limitation: repr() preserves insertion order, so two
        # dicts that are equal as objects but differ in key order do NOT
        # normalize to the same string. Revisit in Phase 0 if the real
        # CodeI/O verifier needs order-independent dict comparison.
        a = normalize_code_output('{"a": 1, "b": 2}')
        b = normalize_code_output('{"b": 2, "a": 1}')
        assert a != b

    def test_non_literal_passthrough(self):
        assert normalize_code_output("hello   world") == "hello world"


class TestPredictedOutputExtraction:
    def test_output_marker(self):
        text = "reasoning here\nOutput: 42"
        assert extract_predicted_output(text) == "42"

    def test_last_output_marker_wins(self):
        text = "Output: 41\nwait, recompute\nOutput: 42"
        assert extract_predicted_output(text) == "42"

    def test_code_fence_fallback(self):
        text = "```\n[1, 2, 3]\n```"
        assert extract_predicted_output(text) == "[1, 2, 3]"

    def test_whole_text_fallback(self):
        assert extract_predicted_output("just 42") == "just 42"


class TestCodeOutputReward:
    def test_exact_match_after_normalization(self):
        completions = ["Output: [1, 2, 3]"]
        answer = ["[1,2,3]"]
        assert code_output_exact_reward(completions=completions, answer=answer) == [1.0]

    def test_mismatch(self):
        completions = ["Output: 42"]
        answer = ["43"]
        assert code_output_exact_reward(completions=completions, answer=answer) == [0.0]


class TestDispatch:
    def test_select_reward_fn_math(self):
        assert select_reward_fn("Math") is numeric_exact_reward

    def test_select_reward_fn_simulation(self):
        assert select_reward_fn("Simulation") is code_output_exact_reward

    def test_select_reward_fn_unknown_raises(self):
        try:
            select_reward_fn("Table")
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for an unconfigured domain")

    def test_select_eval_fns_math_roundtrip(self):
        extract_pred, extract_gold, matches = select_eval_fns("Math")
        assert matches(extract_pred(r"\boxed{9}"), extract_gold(9.0))

    def test_select_eval_fns_simulation_roundtrip(self):
        extract_pred, extract_gold, matches = select_eval_fns("Simulation")
        assert matches(extract_pred("Output: 9"), extract_gold("9"))
