"""Unit tests for exact-answer reward parsing (briefing §6 Phase 0)."""
import pytest

from src.reward import (answers_match, exact_answer_reward,
                        extract_gold_answer, extract_pred_answer)


class TestGoldExtraction:
    def test_gsm8k_format(self):
        assert extract_gold_answer("She has 3+4=7 apples.\n#### 7") == 7.0

    def test_gsm8k_with_commas(self):
        assert extract_gold_answer("... total.\n#### 1,234") == 1234.0

    def test_gsm8k_negative(self):
        assert extract_gold_answer("#### -5") == -5.0

    def test_bare_number_svamp_style(self):
        assert extract_gold_answer("32.0") == 32.0


class TestPredExtraction:
    def test_last_number_wins(self):
        assert extract_pred_answer("3 + 4 = 7. The answer is 7.") == 7.0

    def test_hash_convention_wins_over_later_text(self):
        assert extract_pred_answer("so #### 42 is my answer, see page 3") == 42.0

    def test_last_hash_convention_wins(self):
        assert extract_pred_answer("first #### 41; correction #### 42") == 42.0

    def test_dollar_and_commas(self):
        assert extract_pred_answer("The total cost is $1,250.50") == 1250.5

    def test_no_number(self):
        assert extract_pred_answer("I cannot solve this.") is None

    def test_trailing_period(self):
        assert extract_pred_answer("The answer is 12.") == 12.0

    def test_decimal(self):
        assert extract_pred_answer("= 0.5 in the end") == 0.5


class TestMatching:
    def test_int_float_equivalence(self):
        assert answers_match(7.0, 7)
        assert answers_match(1234.0, 1234.0)

    def test_close_but_wrong(self):
        assert not answers_match(7.0, 8.0)

    def test_large_answer_does_not_get_relative_slack(self):
        assert not answers_match(10000.5, 10000.0)

    def test_none_never_matches(self):
        assert not answers_match(None, 7.0)
        assert not answers_match(7.0, None)


class TestTRLInterface:
    def test_string_completions(self):
        rewards = exact_answer_reward(
            completions=["I think #### 7", "the answer is 9"],
            answer=[7.0, 8.0])
        assert rewards == [1.0, 0.0]

    def test_conversational_completions(self):
        comps = [[{"role": "assistant", "content": "The answer is 7."}]]
        assert exact_answer_reward(completions=comps, answer=[7.0]) == [1.0]

    def test_raw_gsm8k_gold_strings(self):
        rewards = exact_answer_reward(
            completions=["#### 7"], answer=["3+4=7\n#### 7"])
        assert rewards == [1.0]

    def test_extra_kwargs_ignored(self):
        rewards = exact_answer_reward(
            completions=["#### 1"], answer=[1.0],
            prompts=["q"], trainer_state=None)
        assert rewards == [1.0]
