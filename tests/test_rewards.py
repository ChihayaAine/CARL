from carl.rewards.math import math_exact_match
from carl.rewards.qa import qa_token_f1


def test_math_exact_match_boxed():
    assert math_exact_match("The answer is \\boxed{42}.", "42") == 1.0
    assert math_exact_match("Answer: 41", "42") == 0.0


def test_qa_f1_overlap():
    assert qa_token_f1("the quick brown fox", "quick brown fox") > 0.7
    assert qa_token_f1("totally unrelated", "the quick brown fox") < 0.2
